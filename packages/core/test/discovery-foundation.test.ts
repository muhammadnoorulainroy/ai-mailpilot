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
import { migrations } from '../src/db/schema.js';
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
import { LlmConfigSchema, type LlmConfig } from '../src/config/schema.js';
import type { LlmClient } from '../src/llm/client.js';

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as never;
const config = (over: Partial<LlmConfig> = {}): LlmConfig => ({ ...LlmConfigSchema.parse({}), ...over });

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

describe('v18 backfill and canonical_key', () => {
  it('backfills existing categories to active with a unique canonical_key and first_seen_at', () => {
    const db = new BetterSqlite3(':memory:');
    sqliteVec.load(db);
    runMigrations(db, migrations.filter((m) => m.version <= 17));
    db.prepare('INSERT INTO accounts (id, address, kind, created_at) VALUES (?, ?, ?, ?)').run(
      'acc', 'a@b.com', 'work', 1000,
    );
    db.prepare(
      'INSERT INTO categories (id, account_id, label, description, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
    ).run('c1', 'acc', 'Bills', null, 'auto', 1000, 1000);

    runMigrations(db, migrations); // applies only v18

    const row = db.prepare('SELECT status, canonical_key, first_seen_at FROM categories WHERE id = ?').get('c1') as {
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
    runMigrations(db, migrations.filter((m) => m.version <= 17));
    db.prepare('INSERT INTO accounts (id, address, kind, created_at) VALUES (?, ?, ?, ?)').run('acc', 'a@b.com', 'work', 1000);
    const ins = db.prepare(
      'INSERT INTO categories (id, account_id, label, description, source, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
    );
    ins.run('c1', 'acc', 'Bills', null, 'auto', 1000, 1000);
    ins.run('c2', 'acc', 'bills', null, 'auto', 1001, 1001);
    runMigrations(db, migrations);
    const keys = (db.prepare('SELECT canonical_key FROM categories ORDER BY created_at').all() as Array<{ canonical_key: string }>).map((r) => r.canonical_key);
    expect(keys).toEqual(['bills', 'bills_2']);
    db.close();
  });

  it('rejects inserting a category with a null canonical_key', () => {
    const db = openDatabase(':memory:');
    const acc = new AccountRepository(db).create({ address: 'a@b.com', kind: 'work' });
    expect(() =>
      db.prepare(
        "INSERT INTO categories (id, account_id, label, source, status, created_at, updated_at) VALUES ('x', ?, 'NoKey', 'auto', 'active', 1, 1)",
      ).run(acc.id),
    ).toThrow(/canonical_key/);
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
  it('discoveryProvider is main when allowCloudDiscovery is false', () => {
    expect(discoveryProvider(config({ allowCloudDiscovery: false, chatBaseUrl: 'http://x' }))).toBe('main');
    expect(discoveryProvider(config({ allowCloudDiscovery: true, chatBaseUrl: 'http://x' }))).toBe('chat');
  });

  it('assertDiscoveryLocal throws for a cloud provider when not opted in', () => {
    expect(() => assertDiscoveryLocal(config({ allowCloudDiscovery: false }), 'chat')).toThrow(/local-only/);
    expect(() => assertDiscoveryLocal(config({ allowCloudDiscovery: false }), 'main')).not.toThrow();
  });

  it('topic discovery skips a personal account, writes a skipped audit row, and does not call the model', async () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const audit = new DiscoveryAuditRepository(db);
    const llm = fakeLlm();
    const acc = accounts.create({ address: 'p@x.com', kind: 'personal' });
    const svc = new TopicDiscoveryService(
      llm, new EmailRepository(db), new EmbeddingRepository(db), new CategoryRepository(db),
      silentLogger, accounts, audit, () => config(),
    );
    const res = await svc.discover(acc.id, 'bge-m3', 'qwen');
    expect(res.topicsCreated).toBe(0);
    expect(llm.chat).not.toHaveBeenCalled();
    const rows = audit.listForAccount(acc.id);
    expect(rows[0]?.status).toBe('skipped');
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
      db, llm, new EmailRepository(db), new EmbeddingRepository(db), new CategoryRepository(db),
      silentLogger, accounts, audit, () => config({ allowCloudDiscovery: false }),
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
