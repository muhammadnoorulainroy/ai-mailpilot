/**
 * Phase 1 discovery-foundation tests: v18 backfill and non-null canonical_key, stable identity
 * with no silent deletes or retires, the local-only discovery guard on both flows, aliases, the
 * dormant seed taxonomy, and deterministic sampling.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import BetterSqlite3, { type Database } from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { openDatabase } from '../src/db/database.js';
import { runMigrations } from '../src/db/migrations.js';
import { migrations, EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { CategoryAliasRepository } from '../src/repositories/category-alias-repository.js';
import { DiscoveryAuditRepository } from '../src/repositories/discovery-audit-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { TopicDiscoveryService } from '../src/services/topic-discovery-service.js';
import { CategoryImprovementService } from '../src/services/category-improvement-service.js';
import { discoveryProvider, assertDiscoveryLocal } from '../src/services/discovery-guard.js';
import { SEED_TAXONOMY, getSeedSuggestions } from '../src/services/seed-taxonomy.js';
import { seededShuffle } from '../src/util/rand.js';
import { vectorToBuffer, bufferToVector } from '../src/util/vector.js';
import { LlmConfigSchema, type LlmConfig } from '../src/config/schema.js';
import type { LlmClient } from '../src/llm/client.js';

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as never;
const config = (over: Partial<LlmConfig> = {}): LlmConfig => ({
  ...LlmConfigSchema.parse({}),
  ...over,
});

/** A stub LLM client whose chat/embed are spies; discovery guard tests never let these run. */
function fakeLlm(): LlmClient {
  return {
    health: vi.fn(async () => ({ ok: true, models: [] })),
    embed: vi.fn(async () => [1, 0]),
    embedBatch: vi.fn(async () => [[1, 0]]),
    chat: vi.fn(async () => '{"topics":[]}'),
    chatStream: vi.fn(),
  } as unknown as LlmClient;
}

/** A minimal non-zero embedding vector for tests. */
function vec(): Float32Array {
  const a = new Float32Array(EMBEDDING_DIM);
  a[0] = 1;
  return a;
}

describe('v18 backfill and canonical_key', () => {
  it('backfills existing categories to active with a unique canonical_key and first_seen_at', () => {
    const db = new BetterSqlite3(':memory:');
    sqliteVec.load(db);
    runMigrations(
      db,
      migrations.filter((m) => m.version <= 17),
    );
    db.prepare('INSERT INTO accounts (id, address, kind, created_at) VALUES (?, ?, ?, ?)').run(
      'acc',
      'a@b.com',
      'work',
      1000,
    );
    db.prepare(
      'INSERT INTO categories (id, account_id, label, description, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
    ).run('c1', 'acc', 'Bills', null, 'auto', 1000, 1000);

    runMigrations(db, migrations); // applies only v18

    const row = db
      .prepare('SELECT status, canonical_key, first_seen_at FROM categories WHERE id = ?')
      .get('c1') as {
      status: string;
      canonical_key: string;
      first_seen_at: number;
    };
    expect(row.status).toBe('active');
    expect(row.canonical_key).toBe('bills');
    expect(row.first_seen_at).toBe(1000);
    db.close();
  });

  it('gives colliding labels distinct deterministic keys', () => {
    const db = new BetterSqlite3(':memory:');
    sqliteVec.load(db);
    runMigrations(
      db,
      migrations.filter((m) => m.version <= 17),
    );
    db.prepare('INSERT INTO accounts (id, address, kind, created_at) VALUES (?, ?, ?, ?)').run(
      'acc',
      'a@b.com',
      'work',
      1000,
    );
    const ins = db.prepare(
      'INSERT INTO categories (id, account_id, label, description, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
    );
    ins.run('c1', 'acc', 'Bills', null, 'auto', 1000, 1000);
    ins.run('c2', 'acc', 'bills', null, 'auto', 1001, 1001);
    runMigrations(db, migrations);
    const keys = (
      db.prepare('SELECT canonical_key FROM categories ORDER BY created_at').all() as Array<{
        canonical_key: string;
      }>
    ).map((r) => r.canonical_key);
    expect(keys).toEqual(['bills', 'bills_2']);
    db.close();
  });

  it('rejects inserting a category with a null canonical_key', () => {
    const db = openDatabase(':memory:');
    const acc = new AccountRepository(db).create({ address: 'a@b.com', kind: 'work' });
    expect(() =>
      db
        .prepare(
          "INSERT INTO categories (id, account_id, label, source, status, created_at, updated_at) VALUES ('x', ?, 'NoKey', 'auto', 'active', 1, 1)",
        )
        .run(acc.id),
    ).toThrow(/canonical_key/);
    db.close();
  });
});

describe('v21 category proposal suggested_key backfill', () => {
  it('repairs DBs that applied v19 before suggested_key existed', () => {
    const db = new BetterSqlite3(':memory:');
    db.exec(`
      CREATE TABLE migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at INTEGER NOT NULL
      );
      CREATE TABLE category_proposals (
        id TEXT PRIMARY KEY,
        account_id TEXT NOT NULL,
        category_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        cluster_index INTEGER NOT NULL,
        label TEXT NOT NULL,
        description TEXT NOT NULL,
        canonical_key TEXT NOT NULL,
        embedding_model_id TEXT NOT NULL,
        centroid BLOB NOT NULL,
        member_ids TEXT NOT NULL,
        proposed_count INTEGER NOT NULL,
        cohesion REAL NOT NULL,
        separation REAL NOT NULL,
        confidence REAL NOT NULL,
        evidence TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'pending',
        created_at INTEGER NOT NULL,
        applied_at INTEGER,
        dismissed_at INTEGER,
        kind TEXT NOT NULL DEFAULT 'new_category',
        source_category_id TEXT,
        suppression_key TEXT NOT NULL DEFAULT ''
      );
    `);
    const migrationInsert = db.prepare(
      'INSERT INTO migrations (version, name, applied_at) VALUES (?, ?, ?)',
    );
    for (const m of migrations.filter((migration) => migration.version <= 20)) {
      migrationInsert.run(m.version, m.name, 1000 + m.version);
    }

    const proposalInsert = db.prepare(`
      INSERT INTO category_proposals (
        id, account_id, category_id, run_id, cluster_index, label, description, canonical_key,
        embedding_model_id, centroid, member_ids, proposed_count, cohesion, separation,
        confidence, evidence, status, created_at, kind, source_category_id, suppression_key
      ) VALUES (?, 'acc', ?, 'run', 0, ?, 'desc', ?, 'bge-m3', X'0000', '[]', 0, 0, 0, 0, '[]', ?, 1, ?, ?, ?)
    `);
    proposalInsert.run(
      'new-proposal',
      'suggested-cat',
      'Invoices',
      'finance_invoices',
      'dismissed',
      'new_category',
      null,
      '',
    );
    proposalInsert.run(
      'merge-proposal',
      'target-cat',
      'Merge',
      'developer',
      'dismissed',
      'merge',
      'source-cat',
      'merge:source-cat:target-cat',
    );

    expect(
      runMigrations(
        db,
        migrations.filter((m) => m.version <= 21),
      ),
    ).toEqual([21]);

    const cols = db.prepare('PRAGMA table_info(category_proposals)').all() as Array<{
      name: string;
    }>;
    expect(cols.map((c) => c.name)).toContain('suggested_key');
    const rows = db
      .prepare('SELECT id, suggested_key FROM category_proposals ORDER BY id')
      .all() as Array<{ id: string; suggested_key: string }>;
    expect(rows).toEqual([
      { id: 'merge-proposal', suggested_key: '' },
      { id: 'new-proposal', suggested_key: 'finance_invoices' },
    ]);
    db.close();
  });
});

describe('v22 personal discovery opt-in default', () => {
  it('excludes already-synced personal accounts without changing work accounts', () => {
    const db = new BetterSqlite3(':memory:');
    db.exec(`
      CREATE TABLE migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at INTEGER NOT NULL
      );
      CREATE TABLE accounts (
        id TEXT PRIMARY KEY,
        address TEXT NOT NULL,
        display_name TEXT,
        kind TEXT NOT NULL,
        exclude_from_discovery INTEGER NOT NULL DEFAULT 0,
        created_at INTEGER NOT NULL
      );
      INSERT INTO accounts
        (id, address, kind, exclude_from_discovery, created_at)
      VALUES
        ('personal', 'p@example.com', 'personal', 0, 1),
        ('work', 'w@example.com', 'work', 0, 1),
        ('already-off', 'off@example.com', 'personal', 1, 1);
    `);
    const migrationInsert = db.prepare(
      'INSERT INTO migrations (version, name, applied_at) VALUES (?, ?, ?)',
    );
    for (const m of migrations.filter((migration) => migration.version <= 21)) {
      migrationInsert.run(m.version, m.name, 1000 + m.version);
    }

    expect(
      runMigrations(
        db,
        migrations.filter((m) => m.version <= 22),
      ),
    ).toEqual([22]);

    const rows = db
      .prepare('SELECT id, exclude_from_discovery FROM accounts ORDER BY id')
      .all() as Array<{ id: string; exclude_from_discovery: number }>;
    expect(rows).toEqual([
      { id: 'already-off', exclude_from_discovery: 1 },
      { id: 'personal', exclude_from_discovery: 1 },
      { id: 'work', exclude_from_discovery: 0 },
    ]);
    db.close();
  });
});

describe('v23 category prototype index', () => {
  const MODEL = 'bge-m3';
  /** A distinctive centroid vector so byte preservation across the rebuild is checkable. */
  function centroidVec(seed: number): Float32Array {
    const v = new Float32Array(EMBEDDING_DIM);
    for (let i = 0; i < 8; i++) v[i] = Math.sin(seed + i) * 0.5;
    return v;
  }

  /** Build a v22-state DB (old category_embedding_index shape + vec0) with migrations 1..22 recorded. */
  function buildV22Db(): Database {
    const db = new BetterSqlite3(':memory:');
    db.pragma('foreign_keys = ON');
    sqliteVec.load(db);
    db.exec(`
      CREATE TABLE migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at INTEGER NOT NULL
      );
      CREATE TABLE categories (id TEXT PRIMARY KEY, label TEXT NOT NULL);
      CREATE VIRTUAL TABLE category_embeddings USING vec0(embedding FLOAT[${EMBEDDING_DIM}]);
      CREATE TABLE category_embedding_index (
        rowid INTEGER PRIMARY KEY,
        category_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
        model_id TEXT NOT NULL,
        email_count INTEGER NOT NULL DEFAULT 0,
        updated_at INTEGER NOT NULL,
        UNIQUE (category_id, model_id)
      );
      CREATE TRIGGER trg_category_embedding_index_delete
      AFTER DELETE ON category_embedding_index
      BEGIN
        DELETE FROM category_embeddings WHERE rowid = OLD.rowid;
      END;
    `);
    const migrationInsert = db.prepare(
      'INSERT INTO migrations (version, name, applied_at) VALUES (?, ?, ?)',
    );
    for (const m of migrations.filter((migration) => migration.version <= 22)) {
      migrationInsert.run(m.version, m.name, 1000 + m.version);
    }
    return db;
  }

  /** Insert a vec0 embedding (vec0 auto-assigns the rowid) and return that rowid. */
  function insertVec(db: Database, vec: Float32Array): number {
    const info = db
      .prepare('INSERT INTO category_embeddings (embedding) VALUES (?)')
      .run(vectorToBuffer(vec));
    return Number(info.lastInsertRowid);
  }

  /** Seed one category and its single (pre-Phase-4) centroid; returns the shared rowid. */
  function seed(db: Database, categoryId: string, vec: Float32Array): number {
    db.prepare('INSERT OR IGNORE INTO categories (id, label) VALUES (?, ?)').run(
      categoryId,
      categoryId,
    );
    const rowid = insertVec(db, vec);
    db.prepare(
      'INSERT INTO category_embedding_index (rowid, category_id, model_id, email_count, updated_at) VALUES (?, ?, ?, ?, ?)',
    ).run(rowid, categoryId, MODEL, 5, 1000);
    return rowid;
  }

  /** Add a sub-prototype (index >= 1) to an already-migrated DB; returns its rowid. */
  function addSubPrototype(
    db: Database,
    categoryId: string,
    prototypeIndex: number,
    vec: Float32Array,
  ): number {
    const rowid = insertVec(db, vec);
    db.prepare(
      'INSERT INTO category_embedding_index (rowid, category_id, model_id, prototype_index, email_count, updated_at) VALUES (?, ?, ?, ?, ?, ?)',
    ).run(rowid, categoryId, MODEL, prototypeIndex, 3, 1);
    return rowid;
  }

  function readVec(db: Database, categoryId: string, prototypeIndex: number): Float32Array | null {
    const row = db
      .prepare(
        `SELECT ce.embedding AS embedding FROM category_embeddings ce
           JOIN category_embedding_index cei ON cei.rowid = ce.rowid
          WHERE cei.category_id = ? AND cei.prototype_index = ?`,
      )
      .get(categoryId, prototypeIndex) as { embedding: Buffer } | undefined;
    return row ? bufferToVector(row.embedding) : null;
  }

  it('preserves legacy single centroids as prototype_index=0 with rowid and vector bytes intact', () => {
    const db = buildV22Db();
    const vecA = centroidVec(1);
    const vecB = centroidVec(2);
    const rowidA = seed(db, 'cat-a', vecA);
    const rowidB = seed(db, 'cat-b', vecB);

    expect(runMigrations(db, migrations)).toEqual([23]);

    // Column added; existing rows are prototype_index=0 with their original rowid preserved.
    const cols = db.prepare('PRAGMA table_info(category_embedding_index)').all() as Array<{
      name: string;
    }>;
    expect(cols.some((c) => c.name === 'prototype_index')).toBe(true);
    const rows = db
      .prepare(
        'SELECT rowid, category_id, prototype_index FROM category_embedding_index ORDER BY category_id',
      )
      .all() as Array<{ rowid: number; category_id: string; prototype_index: number }>;
    expect(rows).toEqual([
      { rowid: rowidA, category_id: 'cat-a', prototype_index: 0 },
      { rowid: rowidB, category_id: 'cat-b', prototype_index: 0 },
    ]);

    // The vec rows still resolve through the preserved rowid, byte-for-byte.
    expect(Array.from(readVec(db, 'cat-a', 0)!)).toEqual(Array.from(vecA));
    expect(Array.from(readVec(db, 'cat-b', 0)!)).toEqual(Array.from(vecB));
    db.close();
  });

  it('enforces UNIQUE(category_id, model_id, prototype_index): rejects a duplicate aggregate, accepts a sub-prototype', () => {
    const db = buildV22Db();
    seed(db, 'cat-a', centroidVec(1));
    expect(runMigrations(db, migrations)).toEqual([23]);

    // A second prototype_index=0 for the same (category, model) is rejected.
    const dupRowid = insertVec(db, centroidVec(9));
    expect(() =>
      db
        .prepare(
          'INSERT INTO category_embedding_index (rowid, category_id, model_id, prototype_index, email_count, updated_at) VALUES (?, ?, ?, ?, ?, ?)',
        )
        .run(dupRowid, 'cat-a', MODEL, 0, 1, 1),
    ).toThrow();

    // A sub-prototype (index 1) for the same (category, model) is accepted.
    expect(() => addSubPrototype(db, 'cat-a', 1, centroidVec(3))).not.toThrow();
    db.close();
  });

  it('delete trigger removes the vec row for a deleted prototype but leaves siblings intact', () => {
    const db = buildV22Db();
    seed(db, 'cat-a', centroidVec(1));
    expect(runMigrations(db, migrations)).toEqual([23]);
    const subRowid = addSubPrototype(db, 'cat-a', 1, centroidVec(3));

    // Delete only the sub-prototype's index row.
    db.prepare(
      'DELETE FROM category_embedding_index WHERE category_id = ? AND prototype_index = 1',
    ).run('cat-a');

    // Its vec row is gone (trigger); the aggregate (index 0) vec row survives.
    expect(
      (
        db
          .prepare('SELECT COUNT(*) AS n FROM category_embeddings WHERE rowid = ?')
          .get(subRowid) as { n: number }
      ).n,
    ).toBe(0);
    expect(readVec(db, 'cat-a', 0)).not.toBeNull();
    db.close();
  });

  it('deleting a category cascades all its prototypes and their vec rows', () => {
    const db = buildV22Db();
    seed(db, 'cat-a', centroidVec(1));
    expect(runMigrations(db, migrations)).toEqual([23]);
    addSubPrototype(db, 'cat-a', 1, centroidVec(3));

    db.prepare('DELETE FROM categories WHERE id = ?').run('cat-a');

    // FK cascade removed both index rows; the trigger removed both vec rows.
    expect(
      (db.prepare('SELECT COUNT(*) AS n FROM category_embedding_index').get() as { n: number }).n,
    ).toBe(0);
    expect(
      (db.prepare('SELECT COUNT(*) AS n FROM category_embeddings').get() as { n: number }).n,
    ).toBe(0);
    db.close();
  });

  it('is forward-only: a second run applies nothing', () => {
    const db = buildV22Db();
    seed(db, 'cat-a', centroidVec(1));
    expect(runMigrations(db, migrations)).toEqual([23]);
    expect(runMigrations(db, migrations)).toEqual([]);
    db.close();
  });
});

describe('stable identity and no silent edits', () => {
  let db: Database;
  let accounts: AccountRepository;
  let categories: CategoryRepository;
  beforeEach(() => {
    db = openDatabase(':memory:');
    accounts = new AccountRepository(db);
    categories = new CategoryRepository(db);
  });
  afterEach(() => db.close());

  it('reconcile keeps an obsolete unused auto category active and reports it as omitted', () => {
    const acc = accounts.create({ address: 'a@b.com', kind: 'work' });
    const cat = categories.create({ accountId: acc.id, label: 'Old Auto', source: 'auto' });
    const result = categories.reconcileAutoCategories(acc.id, 'bge-m3', []);
    expect(categories.findById(cat.id)?.status).toBe('active');
    expect(result.omitted).toContain(cat.id);
  });

  it('retire() sets retired status and removes the category from listActive', () => {
    const acc = accounts.create({ address: 'a@b.com', kind: 'work' });
    const cat = categories.create({ accountId: acc.id, label: 'Gone', source: 'auto' });
    categories.retire(cat.id);
    expect(categories.findById(cat.id)?.status).toBe('retired');
    expect(categories.findById(cat.id)?.retiredAt).not.toBeNull();
    expect(categories.listActive(acc.id).map((c) => c.id)).not.toContain(cat.id);
    expect(categories.listRetired(acc.id).map((c) => c.id)).toContain(cat.id);
    expect(categories.listAll(acc.id).map((c) => c.id)).toContain(cat.id);
  });

  it('freezes canonical_key: creating gives distinct keys, update does not change it', () => {
    const acc = accounts.create({ address: 'a@b.com', kind: 'work' });
    const a = categories.create({ accountId: acc.id, label: 'Invoices', source: 'user' });
    expect(a.canonicalKey).toBe('invoices');
    categories.update(a.id, { description: 'changed' });
    expect(categories.findById(a.id)?.canonicalKey).toBe('invoices');
  });
});

describe('local-only discovery guard', () => {
  it('discoveryProvider is main when allowCloudDiscovery is false, ignoring categorizeUseChatProvider', () => {
    expect(discoveryProvider(config({ allowCloudDiscovery: false, chatBaseUrl: 'http://x' }))).toBe(
      'main',
    );
    // categorizeUseChatProvider (Refine's flag) must never route discovery/improve to cloud.
    expect(
      discoveryProvider(
        config({
          allowCloudDiscovery: false,
          categorizeUseChatProvider: true,
          chatBaseUrl: 'http://x',
        }),
      ),
    ).toBe('main');
    expect(discoveryProvider(config({ allowCloudDiscovery: true, chatBaseUrl: 'http://x' }))).toBe(
      'chat',
    );
  });

  it('assertDiscoveryLocal throws for a cloud provider when not opted in', () => {
    expect(() => assertDiscoveryLocal(config({ allowCloudDiscovery: false }), 'chat')).toThrow(
      /local-only/,
    );
    expect(() =>
      assertDiscoveryLocal(config({ allowCloudDiscovery: false }), 'main'),
    ).not.toThrow();
  });

  it('topic discovery skips a personal account, writes a skipped audit row, and does not call the model', async () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const audit = new DiscoveryAuditRepository(db);
    const llm = fakeLlm();
    const acc = accounts.create({ address: 'p@x.com', kind: 'personal' });
    const svc = new TopicDiscoveryService(
      llm,
      new EmailRepository(db),
      new EmbeddingRepository(db),
      new CategoryRepository(db),
      silentLogger,
      accounts,
      audit,
      () => config(),
    );
    const res = await svc.discover(acc.id, 'bge-m3', 'qwen');
    expect(res.topicsCreated).toBe(0);
    expect(llm.chat).not.toHaveBeenCalled();
    const rows = audit.listForAccount(acc.id);
    expect(rows[0]?.status).toBe('skipped');
    expect(rows[0]?.flow).toBe('topic_discovery');
    db.close();
  });

  it('topic discovery audits insufficient when an eligible account has no emails', async () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const audit = new DiscoveryAuditRepository(db);
    const acc = accounts.create({ address: 'w2@x.com', kind: 'work' });
    const svc = new TopicDiscoveryService(
      fakeLlm(),
      new EmailRepository(db),
      new EmbeddingRepository(db),
      new CategoryRepository(db),
      silentLogger,
      accounts,
      audit,
      () => config(),
    );
    const res = await svc.discover(acc.id, 'bge-m3', 'qwen');
    expect(res.topicsCreated).toBe(0);
    const rows = audit.listForAccount(acc.id);
    expect(rows[0]?.status).toBe('insufficient');
    expect(rows[0]?.flow).toBe('topic_discovery');
    db.close();
  });

  it('category improvement blocks a cloud provider when not opted in and audits it', async () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const audit = new DiscoveryAuditRepository(db);
    const llm = fakeLlm();
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    const svc = new CategoryImprovementService(
      db,
      llm,
      new EmailRepository(db),
      new EmbeddingRepository(db),
      new CategoryRepository(db),
      silentLogger,
      accounts,
      audit,
      () => config({ allowCloudDiscovery: false }),
    );
    await expect(svc.suggest(acc.id, 'bge-m3', 'gpt-4o', 'chat')).rejects.toThrow(/local-only/);
    expect(llm.chat).not.toHaveBeenCalled();
    const rows = audit.listForAccount(acc.id);
    expect(rows[0]?.status).toBe('blocked');
    expect(rows[0]?.flow).toBe('improve_categories');
    db.close();
  });
});

describe('aliases', () => {
  let db: Database;
  let accounts: AccountRepository;
  let categories: CategoryRepository;
  let aliases: CategoryAliasRepository;
  beforeEach(() => {
    db = openDatabase(':memory:');
    accounts = new AccountRepository(db);
    categories = new CategoryRepository(db);
    aliases = new CategoryAliasRepository(db);
  });
  afterEach(() => db.close());

  it('resolves an alias across case and accents to the same category', () => {
    const acc = accounts.create({ address: 'a@b.com', kind: 'work' });
    const cat = categories.create({ accountId: acc.id, label: 'Invoices', source: 'user' });
    aliases.addAlias(acc.id, cat.id, 'Factures', 'user');
    expect(aliases.findByAlias(acc.id, 'factures')?.id).toBe(cat.id);
    expect(aliases.findByAlias(acc.id, 'FACTURES')?.id).toBe(cat.id);
    expect(aliases.findByAlias(acc.id, 'unknown')).toBeNull();
  });

  it('ignores a duplicate normalized alias and drops a removed one', () => {
    const acc = accounts.create({ address: 'a@b.com', kind: 'work' });
    const cat = categories.create({ accountId: acc.id, label: 'Invoices', source: 'user' });
    aliases.addAlias(acc.id, cat.id, 'Factures');
    aliases.addAlias(acc.id, cat.id, 'factures'); // same normalized, ignored
    expect(aliases.listForCategory(cat.id)).toHaveLength(1);
    aliases.removeAlias(acc.id, 'Factures');
    expect(aliases.findByAlias(acc.id, 'factures')).toBeNull();
  });

  it('does not resolve an alias to a retired or suggested category', () => {
    const acc = accounts.create({ address: 'a@b.com', kind: 'work' });
    const cat = categories.create({ accountId: acc.id, label: 'Invoices', source: 'user' });
    aliases.addAlias(acc.id, cat.id, 'Factures');
    expect(aliases.findByAlias(acc.id, 'factures')?.id).toBe(cat.id);
    categories.retire(cat.id);
    expect(aliases.findByAlias(acc.id, 'factures')).toBeNull();

    const suggested = categories.create({
      accountId: acc.id,
      label: 'Crypto',
      source: 'auto',
      status: 'suggested',
    });
    aliases.addAlias(acc.id, suggested.id, 'Trading');
    expect(aliases.findByAlias(acc.id, 'trading')).toBeNull();
  });

  it('never resolves or stores an alias across account boundaries', () => {
    const a = accounts.create({ address: 'a@x.com', kind: 'work' });
    const b = accounts.create({ address: 'b@x.com', kind: 'work' });
    const bCat = categories.create({ accountId: b.id, label: 'B Category', source: 'user' });

    // addAlias refuses to store an alias for account A pointing at account B's category.
    aliases.addAlias(a.id, bCat.id, 'CrossAlias');
    expect(aliases.findByAlias(a.id, 'crossalias')).toBeNull();
    expect(aliases.listForCategory(bCat.id)).toHaveLength(0);

    // Even a raw cross-account alias row does not resolve, because the join is account-scoped.
    db.prepare(
      `INSERT INTO category_aliases (account_id, category_id, alias, normalized_alias, source, created_at)
       VALUES (?, ?, 'Raw', 'raw', 'auto', 1)`,
    ).run(a.id, bCat.id);
    expect(aliases.findByAlias(a.id, 'raw')).toBeNull();
  });
});

describe('seed taxonomy is dormant', () => {
  it('offers all seeds for a fresh account and filters ones already represented', () => {
    expect(getSeedSuggestions(new Set())).toHaveLength(SEED_TAXONOMY.length);
    const filtered = getSeedSuggestions(new Set(['finance.banking']));
    expect(filtered.map((s) => s.canonicalKey)).not.toContain('finance.banking');
    expect(filtered).toHaveLength(SEED_TAXONOMY.length - 1);
  });
});

describe('deterministic sampling', () => {
  it('seededShuffle is stable for the same seed and differs for a different seed', () => {
    const arr = Array.from({ length: 20 }, (_, i) => i);
    expect(seededShuffle(arr, 123)).toEqual(seededShuffle(arr, 123));
    expect(seededShuffle(arr, 123)).not.toEqual(seededShuffle(arr, 999));
    expect([...arr]).toEqual(Array.from({ length: 20 }, (_, i) => i)); // input untouched
  });
});

describe('improve categories respects category status', () => {
  it('apply does not assign an expansion to a retired category', async () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const categories = new CategoryRepository(db);
    const emails = new EmailRepository(db);
    const embeddings = new EmbeddingRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
    const cat = categories.create({ accountId: acc.id, label: 'Old', source: 'auto' });
    categories.retire(cat.id);
    emails.upsertBatch([{ messageId: 'e1', accountId: acc.id, folder: 'INBOX' }]);
    embeddings.saveEmbedding({ messageId: 'e1', accountId: acc.id, modelId: 'bge-m3' }, vec());

    const svc = new CategoryImprovementService(
      db,
      fakeLlm(),
      emails,
      embeddings,
      categories,
      silentLogger,
    );
    await svc.apply(acc.id, 'bge-m3', {
      existingCategoryExpansions: [{ categoryId: cat.id, messageIds: ['e1'] }],
      newCategories: [],
      merges: [],
    });
    const count = (
      db
        .prepare('SELECT COUNT(*) AS c FROM email_categories WHERE category_id = ?')
        .get(cat.id) as {
        c: number;
      }
    ).c;
    expect(count).toBe(0);
    db.close();
  });
});
