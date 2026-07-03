/**
 * Integration tests for the core SQLite repositories and the services built on
 * them, covering categorization, embeddings, FTS keyword search, conversation
 * persistence, triage, priority briefing, and the email assistant.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import BetterSqlite3, { type Database } from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { openDatabase } from '../src/db/database.js';
import { runMigrations } from '../src/db/migrations.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { AttachmentRepository } from '../src/repositories/attachment-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmailAssistantRepository } from '../src/repositories/email-assistant-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { ConversationRepository } from '../src/repositories/conversation-repository.js';
import { TriageRepository, type TriageBucket } from '../src/repositories/triage-repository.js';
import { CategorizeJobRepository } from '../src/repositories/categorize-job-repository.js';
import type { LlmCategorizeProgress } from '../src/services/llm-categorize-orchestrator.js';
import { DashboardService } from '../src/services/dashboard-service.js';
import { PriorityService } from '../src/services/priority-service.js';
import { EmailAssistantService } from '../src/services/email-assistant-service.js';
import type { TriageMetadata } from '@ai-mailpilot/shared';
import { EMBEDDING_DIM, migrations } from '../src/db/schema.js';

let db: Database;
let accounts: AccountRepository;
let emails: EmailRepository;
let embeddings: EmbeddingRepository;
let categories: CategoryRepository;
const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as never;

/** Builds a test embedding vector with the first slot pinned and the second slot tunable. */
function vec(second = 0): Float32Array {
  const arr = new Float32Array(EMBEDDING_DIM);
  arr[0] = 1;
  arr[1] = second;
  return arr;
}

/** Reads the stored assignment method for one (email, category) directly (getEmailCategories omits it). */
function methodOf(messageId: string, accountId: string, categoryId: string): string | null {
  const row = db
    .prepare(
      'SELECT method FROM email_categories WHERE message_id = ? AND account_id = ? AND category_id = ?',
    )
    .get(messageId, accountId, categoryId) as { method: string | null } | undefined;
  return row?.method ?? null;
}

beforeEach(() => {
  db = openDatabase(':memory:');
  accounts = new AccountRepository(db);
  emails = new EmailRepository(db);
  embeddings = new EmbeddingRepository(db);
  categories = new CategoryRepository(db);
});

afterEach(() => {
  db.close();
});

describe('CategorizeJobRepository (restart recovery)', () => {
  /** Builds a categorize progress record with consistent counts derived from processed. */
  const progress = (
    status: LlmCategorizeProgress['status'],
    processed: number,
  ): LlmCategorizeProgress => ({
    status,
    accountId: 'a',
    modelId: 'qwen',
    total: 100,
    processed,
    assigned: processed,
    uncategorized: 0,
    failed: 0,
    clusters: 50,
    clustersProcessed: processed,
    gatedClusters: 7,
    llmCalls: 9,
    startedAt: 1000,
  });

  it('marks a running job interrupted on restart instead of resetting to idle', () => {
    new CategorizeJobRepository(db).save('a', progress('running', 42), 2000);
    const restarted = new CategorizeJobRepository(db);
    expect(restarted.markRunningInterrupted(3000)).toBe(1);
    const recovered = restarted.getMostRecent();
    expect(recovered?.status).toBe('interrupted');
    expect(recovered?.processed).toBe(42);
    expect(recovered?.gatedClusters).toBe(7);
    expect(recovered?.llmCalls).toBe(9);
  });

  it('leaves a completed job untouched on restart', () => {
    const jobs = new CategorizeJobRepository(db);
    jobs.save('a', progress('completed', 100), 2000);
    expect(jobs.markRunningInterrupted(3000)).toBe(0);
    expect(jobs.getMostRecent()?.status).toBe('completed');
  });
});

describe('reconcileAutoCategories durable taxonomy', () => {
  const omit = [{ label: 'Other', description: 'd', centroid: vec(), emailCount: 5 }];

  it('preserves an existing auto category with assignments even when a run omits it', () => {
    const acct = accounts.create({ address: 'rec1@x.y', kind: 'work' });
    emails.upsertBatch(
      ['g1', 'g2', 'g3'].map((id) => ({ messageId: id, accountId: acct.id, folder: 'INBOX' })),
    );
    const grades = categories.create({
      accountId: acct.id,
      label: 'Course Grades',
      source: 'auto',
    });
    for (const id of ['g1', 'g2', 'g3']) {
      categories.replaceEmailAssignments(id, acct.id, [
        {
          messageId: id,
          accountId: acct.id,
          categoryId: grades.id,
          confidence: 0.7,
          assignedBy: 'auto',
          assignedAt: 1000,
          method: 'llm',
        },
      ]);
    }
    categories.reconcileAutoCategories(acct.id, 'bge-m3', omit);
    expect(categories.listForAccount(acct.id).map((c) => c.label)).toContain('Course Grades');
  });

  it('keeps an empty obsolete auto category active and reports it as omitted (no silent delete)', () => {
    const acct = accounts.create({ address: 'rec2@x.y', kind: 'work' });
    const unused = categories.create({ accountId: acct.id, label: 'Unused', source: 'auto' });
    const result = categories.reconcileAutoCategories(acct.id, 'bge-m3', omit);
    expect(categories.findById(unused.id)?.status).toBe('active');
    expect(categories.listForAccount(acct.id).map((c) => c.label)).toContain('Unused');
    expect(result.omitted).toContain(unused.id);
  });

  it('preserves an auto category a user filed mail into', () => {
    const acct = accounts.create({ address: 'rec3@x.y', kind: 'work' });
    emails.upsertBatch([{ messageId: 'u1', accountId: acct.id, folder: 'INBOX' }]);
    const cat = categories.create({ accountId: acct.id, label: 'UserFiled', source: 'auto' });
    categories.replaceEmailAssignments('u1', acct.id, [
      {
        messageId: 'u1',
        accountId: acct.id,
        categoryId: cat.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: 1000,
        method: null,
      },
    ]);
    categories.reconcileAutoCategories(acct.id, 'bge-m3', omit);
    expect(categories.listForAccount(acct.id).map((c) => c.label)).toContain('UserFiled');
  });
});

describe('EmailRepository.listSummariesByDomain', () => {
  it('anchors on @ so a domain does not match a superstring domain', () => {
    const acct = accounts.create({ address: 'd@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'g1', accountId: acct.id, folder: 'INBOX', fromAddr: 'Name <a@gmail.com>' },
      { messageId: 'h1', accountId: acct.id, folder: 'INBOX', fromAddr: 'b@github.com' },
    ]);

    expect(emails.listSummariesByDomain(acct.id, 'mail.com', 10)).toHaveLength(0);
    expect(emails.listSummariesByDomain(acct.id, 'gmail.com', 10).map((e) => e.messageId)).toEqual([
      'g1',
    ]);
    expect(emails.listSummariesByDomain(acct.id, 'github.com', 10).map((e) => e.messageId)).toEqual(
      ['h1'],
    );
  });
});

describe('EmbeddingRepository model-id canonicalization (L2 / original mismatch bug)', () => {
  it('writes under :latest and reads back under bare and upper-case tags', () => {
    const acct = accounts.create({ address: 'a1@x.y', kind: 'work' });
    emails.upsertBatch([{ messageId: 'm1', accountId: acct.id, folder: 'INBOX' }]);
    embeddings.saveEmbedding(
      { messageId: 'm1', accountId: acct.id, modelId: 'bge-m3:latest' },
      vec(),
    );

    expect(
      embeddings.getEmbedding({ messageId: 'm1', accountId: acct.id, modelId: 'bge-m3' }),
    ).not.toBeNull();
    expect(
      embeddings.getEmbedding({ messageId: 'm1', accountId: acct.id, modelId: 'bge-m3:LATEST' }),
    ).not.toBeNull();
    expect(embeddings.countForModel(acct.id, 'bge-m3')).toBe(1);
    expect(embeddings.listForAccount(acct.id, 'bge-m3:latest')).toHaveLength(1);
  });
});

describe('EmbeddingRepository.search recall under account skew (M1)', () => {
  it('returns k hits for a small account even when a large account dominates the index', () => {
    const big = accounts.create({ address: 'big@x.y', kind: 'work' });
    const small = accounts.create({ address: 'small@x.y', kind: 'work' });
    emails.upsertBatch(
      Array.from({ length: 70 }, (_, i) => ({
        messageId: `big${i}`,
        accountId: big.id,
        folder: 'INBOX',
      })),
    );
    emails.upsertBatch(
      Array.from({ length: 5 }, (_, j) => ({
        messageId: `small${j}`,
        accountId: small.id,
        folder: 'INBOX',
      })),
    );

    for (let i = 0; i < 70; i++) {
      embeddings.saveEmbedding(
        { messageId: `big${i}`, accountId: big.id, modelId: 'bge-m3' },
        vec(0.001 * i),
      );
    }
    for (let j = 0; j < 5; j++) {
      embeddings.saveEmbedding(
        { messageId: `small${j}`, accountId: small.id, modelId: 'bge-m3' },
        vec(0.5 + 0.001 * j),
      );
    }

    const hits = embeddings.search(small.id, 'bge-m3', vec(0), 5);
    expect(hits).toHaveLength(5);
    expect(hits.every((h) => h.accountId === small.id)).toBe(true);
  });
});

describe('CategoryRepository.mergeInto provenance (M2)', () => {
  /** Creates one email assigned to a source and target category with the given provenance for merge tests. */
  function setupEmailInTwoCategories(
    targetConf: number,
    targetBy: 'user' | 'auto',
    sourceConf: number,
    sourceBy: 'user' | 'auto',
  ): { sourceId: string; targetId: string } {
    const acct = accounts.create({ address: 'x@y.z', kind: 'work' });
    emails.upsertBatch([{ messageId: 'msg', accountId: acct.id, folder: 'INBOX' }]);
    const target = categories.create({ accountId: acct.id, label: 'Target', source: 'auto' });
    const source = categories.create({ accountId: acct.id, label: 'Source', source: 'auto' });
    const now = Date.now();
    categories.replaceEmailAssignments('msg', acct.id, [
      {
        messageId: 'msg',
        accountId: acct.id,
        categoryId: target.id,
        confidence: targetConf,
        assignedBy: targetBy,
        assignedAt: now,
      },
      {
        messageId: 'msg',
        accountId: acct.id,
        categoryId: source.id,
        confidence: sourceConf,
        assignedBy: sourceBy,
        assignedAt: now,
      },
    ]);
    return { sourceId: source.id, targetId: target.id };
  }

  it('keeps the winning row provenance when the destination confidence is higher', () => {
    const { sourceId, targetId } = setupEmailInTwoCategories(0.9, 'user', 0.4, 'auto');
    const acctId = accounts.list()[0]!.id;
    categories.mergeInto(sourceId, targetId);

    const remaining = categories.getEmailCategories('msg', acctId);
    expect(remaining).toHaveLength(1);
    expect(remaining[0]!.categoryId).toBe(targetId);
    expect(remaining[0]!.confidence).toBeCloseTo(0.9);
    expect(remaining[0]!.assignedBy).toBe('user');
  });

  it('keeps user provenance even when the source auto confidence is higher', () => {
    // Target is a user assignment; a higher-confidence auto source must NOT overwrite it.
    const { sourceId, targetId } = setupEmailInTwoCategories(0.5, 'user', 0.95, 'auto');
    const acctId = accounts.list()[0]!.id;
    categories.mergeInto(sourceId, targetId);

    const remaining = categories.getEmailCategories('msg', acctId);
    expect(remaining).toHaveLength(1);
    expect(remaining[0]!.assignedBy).toBe('user');
    expect(remaining[0]!.confidence).toBeCloseTo(0.95); // confidence is the max
  });

  it('keeps user provenance on an equal-confidence tie against auto (the fixed hole)', () => {
    // Source user 1.0 vs target auto 1.0: the user row must win the tie.
    const { sourceId, targetId } = setupEmailInTwoCategories(1.0, 'auto', 1.0, 'user');
    const acctId = accounts.list()[0]!.id;
    categories.mergeInto(sourceId, targetId);

    const remaining = categories.getEmailCategories('msg', acctId);
    expect(remaining).toHaveLength(1);
    expect(remaining[0]!.assignedBy).toBe('user');
    expect(remaining[0]!.confidence).toBeCloseTo(1.0);
  });

  it('adopts the source user provenance over a higher-confidence target auto row', () => {
    // Source user 0.8 vs target auto 1.0: user still wins; confidence is the max.
    const { sourceId, targetId } = setupEmailInTwoCategories(1.0, 'auto', 0.8, 'user');
    const acctId = accounts.list()[0]!.id;
    categories.mergeInto(sourceId, targetId);

    const remaining = categories.getEmailCategories('msg', acctId);
    expect(remaining).toHaveLength(1);
    expect(remaining[0]!.assignedBy).toBe('user');
    expect(remaining[0]!.confidence).toBeCloseTo(1.0);
  });

  it('keeps the higher-confidence row and its method among same-provenance auto rows', () => {
    const acct = accounts.create({ address: 'mm@y.z', kind: 'work' });
    emails.upsertBatch([{ messageId: 'msg', accountId: acct.id, folder: 'INBOX' }]);
    const target = categories.create({ accountId: acct.id, label: 'T', source: 'auto' });
    const source = categories.create({ accountId: acct.id, label: 'S', source: 'auto' });
    categories.replaceEmailAssignments('msg', acct.id, [
      {
        messageId: 'msg',
        accountId: acct.id,
        categoryId: target.id,
        confidence: 0.6,
        assignedBy: 'auto',
        assignedAt: 1,
        method: 'embed',
      },
      {
        messageId: 'msg',
        accountId: acct.id,
        categoryId: source.id,
        confidence: 0.9,
        assignedBy: 'auto',
        assignedAt: 2,
        method: 'gate',
      },
    ]);
    categories.mergeInto(source.id, target.id);

    const remaining = categories.getEmailCategories('msg', acct.id);
    expect(remaining).toHaveLength(1);
    expect(remaining[0]!.assignedBy).toBe('auto');
    expect(remaining[0]!.confidence).toBeCloseTo(0.9);
    expect(methodOf('msg', acct.id, target.id)).toBe('gate'); // the winning row's method
  });

  it('moves all source rows to the target and deletes the source category', () => {
    const acct = accounts.create({ address: 'am@y.z', kind: 'work' });
    const target = categories.create({ accountId: acct.id, label: 'T', source: 'auto' });
    const source = categories.create({ accountId: acct.id, label: 'S', source: 'auto' });
    for (const id of ['a', 'b', 'c']) {
      emails.upsertBatch([{ messageId: id, accountId: acct.id, folder: 'INBOX' }]);
      categories.replaceEmailAssignments(id, acct.id, [
        {
          messageId: id,
          accountId: acct.id,
          categoryId: source.id,
          confidence: 0.7,
          assignedBy: 'auto',
          assignedAt: 1,
          method: 'embed',
        },
      ]);
    }
    const { reassigned } = categories.mergeInto(source.id, target.id);

    expect(reassigned).toBe(3);
    expect(categories.findById(source.id)).toBeNull();
    for (const id of ['a', 'b', 'c']) {
      expect(categories.getEmailCategories(id, acct.id).map((r) => r.categoryId)).toEqual([
        target.id,
      ]);
    }
  });

  it('leaves an email membership in an unrelated category untouched', () => {
    const acct = accounts.create({ address: 'un@y.z', kind: 'work' });
    const target = categories.create({ accountId: acct.id, label: 'T', source: 'auto' });
    const source = categories.create({ accountId: acct.id, label: 'S', source: 'auto' });
    const other = categories.create({ accountId: acct.id, label: 'O', source: 'auto' });
    emails.upsertBatch([{ messageId: 'm1', accountId: acct.id, folder: 'INBOX' }]);
    // m1 is multi-labeled: in the source AND an unrelated category.
    categories.replaceEmailAssignments('m1', acct.id, [
      {
        messageId: 'm1',
        accountId: acct.id,
        categoryId: source.id,
        confidence: 0.7,
        assignedBy: 'auto',
        assignedAt: 1,
        method: 'embed',
      },
      {
        messageId: 'm1',
        accountId: acct.id,
        categoryId: other.id,
        confidence: 0.9,
        assignedBy: 'auto',
        assignedAt: 1,
        method: 'gate',
      },
    ]);
    categories.mergeInto(source.id, target.id);

    const m1 = categories.getEmailCategories('m1', acct.id);
    expect(m1.map((r) => r.categoryId).sort()).toEqual([other.id, target.id].sort());
    const otherRow = m1.find((r) => r.categoryId === other.id)!;
    expect(otherRow.confidence).toBeCloseTo(0.9);
    expect(methodOf('m1', acct.id, other.id)).toBe('gate');
    expect(categories.findById(other.id)).not.toBeNull();
  });
});

describe('CategoryRepository.listEmailsForCategory pagination (M11)', () => {
  it('honors limit and offset', () => {
    const acct = accounts.create({ address: 'p@q.r', kind: 'work' });
    const cat = categories.create({ accountId: acct.id, label: 'C', source: 'auto' });
    const now = Date.now();
    for (let i = 0; i < 3; i++) {
      emails.upsertBatch([
        { messageId: `e${i}`, accountId: acct.id, folder: 'INBOX', date: now + i },
      ]);
      categories.replaceEmailAssignments(`e${i}`, acct.id, [
        {
          messageId: `e${i}`,
          accountId: acct.id,
          categoryId: cat.id,
          confidence: 0.8,
          assignedBy: 'auto',
          assignedAt: now,
        },
      ]);
    }

    expect(categories.listEmailsForCategory(cat.id, 2, 0)).toHaveLength(2);
    expect(categories.listEmailsForCategory(cat.id, 2, 2)).toHaveLength(1);
    expect(categories.countEmails(cat.id)).toBe(3);
  });

  it('returns each email full multi-label membership', () => {
    const acct = accounts.create({ address: 'm@q.r', kind: 'work' });
    const catA = categories.create({ accountId: acct.id, label: 'A', source: 'auto' });
    const catB = categories.create({ accountId: acct.id, label: 'B', source: 'auto' });
    const now = Date.now();
    emails.upsertBatch([{ messageId: 'e0', accountId: acct.id, folder: 'INBOX', date: now }]);
    categories.replaceEmailAssignments('e0', acct.id, [
      {
        messageId: 'e0',
        accountId: acct.id,
        categoryId: catA.id,
        confidence: 0.82,
        assignedBy: 'auto',
        assignedAt: now,
        method: 'embed',
      },
      {
        messageId: 'e0',
        accountId: acct.id,
        categoryId: catB.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: now,
      },
    ]);

    const rows = categories.listEmailsForCategory(catA.id, 10, 0);
    expect(rows).toHaveLength(1);
    const chips = rows[0]!.categories;
    expect(chips.map((c) => c.id).sort()).toEqual([catA.id, catB.id].sort());
    const a = chips.find((c) => c.id === catA.id)!;
    const b = chips.find((c) => c.id === catB.id)!;
    expect(a.assignedBy).toBe('auto');
    expect(a.method).toBe('embed');
    expect(a.confidence).toBeCloseTo(0.82);
    expect(b.assignedBy).toBe('user');
  });
});

describe('CategoryRepository.getPrimaryCategoryPerEmail (folders)', () => {
  /** Creates one email in two categories with the given confidence and provenance for primary-category tests. */
  function emailInTwoCats(
    confA: number,
    byA: 'user' | 'auto',
    confB: number,
    byB: 'user' | 'auto',
  ): { acctId: string; catA: string; catB: string } {
    const acct = accounts.create({ address: 'pri@x.y', kind: 'work' });
    const catA = categories.create({ accountId: acct.id, label: 'A', source: 'auto' });
    const catB = categories.create({ accountId: acct.id, label: 'B', source: 'auto' });
    const now = Date.now();
    emails.upsertBatch([{ messageId: 'e0', accountId: acct.id, folder: 'INBOX' }]);
    categories.replaceEmailAssignments('e0', acct.id, [
      {
        messageId: 'e0',
        accountId: acct.id,
        categoryId: catA.id,
        confidence: confA,
        assignedBy: byA,
        assignedAt: now,
      },
      {
        messageId: 'e0',
        accountId: acct.id,
        categoryId: catB.id,
        confidence: confB,
        assignedBy: byB,
        assignedAt: now,
      },
    ]);
    return { acctId: acct.id, catA: catA.id, catB: catB.id };
  }

  it('picks the highest-confidence category', () => {
    const { acctId, catB } = emailInTwoCats(0.4, 'auto', 0.8, 'auto');
    const primary = categories.getPrimaryCategoryPerEmail(acctId);
    expect(primary).toHaveLength(1);
    expect(primary[0]!.categoryId).toBe(catB);
  });

  it('breaks ties in favor of a user assignment', () => {
    const { acctId, catB } = emailInTwoCats(0.8, 'auto', 0.8, 'user');
    const primary = categories.getPrimaryCategoryPerEmail(acctId);
    expect(primary[0]!.categoryId).toBe(catB);
  });

  it('ignores a retired category and selects the active one, even at lower confidence', () => {
    const acct = accounts.create({ address: 'pr2@x.y', kind: 'work' });
    const active = categories.create({ accountId: acct.id, label: 'Active', source: 'auto' });
    const retired = categories.create({
      accountId: acct.id,
      label: 'Retired',
      source: 'auto',
      status: 'retired',
    });
    emails.upsertBatch([{ messageId: 'e0', accountId: acct.id, folder: 'INBOX' }]);
    categories.replaceEmailAssignments('e0', acct.id, [
      {
        messageId: 'e0',
        accountId: acct.id,
        categoryId: retired.id,
        confidence: 0.99,
        assignedBy: 'auto',
        assignedAt: 1,
      },
      {
        messageId: 'e0',
        accountId: acct.id,
        categoryId: active.id,
        confidence: 0.5,
        assignedBy: 'auto',
        assignedAt: 1,
      },
    ]);
    expect(categories.getPrimaryCategoryPerEmail(acct.id)).toEqual([
      { messageId: 'e0', categoryId: active.id },
    ]);
  });

  it('returns no primary when the only assignment is to a retired category', () => {
    const acct = accounts.create({ address: 'pr3@x.y', kind: 'work' });
    const retired = categories.create({
      accountId: acct.id,
      label: 'Retired',
      source: 'auto',
      status: 'retired',
    });
    emails.upsertBatch([{ messageId: 'e0', accountId: acct.id, folder: 'INBOX' }]);
    categories.replaceEmailAssignments('e0', acct.id, [
      {
        messageId: 'e0',
        accountId: acct.id,
        categoryId: retired.id,
        confidence: 0.9,
        assignedBy: 'auto',
        assignedAt: 1,
      },
    ]);
    expect(categories.getPrimaryCategoryPerEmail(acct.id)).toEqual([]);
  });

  it('returns no primary when the only assignment is to a suggested category', () => {
    const acct = accounts.create({ address: 'pr4@x.y', kind: 'work' });
    const suggested = categories.create({
      accountId: acct.id,
      label: 'Suggested',
      source: 'auto',
      status: 'suggested',
    });
    emails.upsertBatch([{ messageId: 'e0', accountId: acct.id, folder: 'INBOX' }]);
    categories.replaceEmailAssignments('e0', acct.id, [
      {
        messageId: 'e0',
        accountId: acct.id,
        categoryId: suggested.id,
        confidence: 0.9,
        assignedBy: 'auto',
        assignedAt: 1,
      },
    ]);
    expect(categories.getPrimaryCategoryPerEmail(acct.id)).toEqual([]);
  });
});

describe('CategoryRepository.update description clearing (L1)', () => {
  it('clears to null on explicit null and keeps on undefined', () => {
    const acct = accounts.create({ address: 'd@e.f', kind: 'work' });
    const cat = categories.create({
      accountId: acct.id,
      label: 'C',
      description: 'original',
      source: 'user',
    });

    const keep = categories.update(cat.id, { label: 'Renamed' });
    expect(keep?.description).toBe('original');

    const cleared = categories.update(cat.id, { description: null });
    expect(cleared?.description).toBeNull();
    expect(categories.findById(cat.id)?.description).toBeNull();
  });
});

describe('EmailRepository.selectNeedFetch (resync diff)', () => {
  it('returns absent and failed-fetch ids, omits already-synced ones', () => {
    const acct = accounts.create({ address: 's@y.z', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'synced', accountId: acct.id, folder: 'INBOX', body: 'hi', bodyFetched: true },
    ]);
    emails.upsertBatch([
      { messageId: 'partial', accountId: acct.id, folder: 'INBOX', bodyFetched: false },
    ]);

    const need = emails.selectNeedFetch(acct.id, ['absent', 'synced', 'partial']);
    expect(need).toEqual(['absent', 'partial']);
  });

  it('treats a genuinely body-less but successfully-fetched email as synced', () => {
    const acct = accounts.create({ address: 'e@y.z', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'empty', accountId: acct.id, folder: 'INBOX', bodyFetched: true },
    ]);
    expect(emails.selectNeedFetch(acct.id, ['empty'])).toEqual([]);
  });

  it('stops returning an id once a failed fetch is repaired', () => {
    const acct = accounts.create({ address: 'r@y.z', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'm', accountId: acct.id, folder: 'INBOX', bodyFetched: false },
    ]);
    expect(emails.selectNeedFetch(acct.id, ['m'])).toEqual(['m']);

    emails.upsertBatch([
      { messageId: 'm', accountId: acct.id, folder: 'INBOX', body: 'now here', bodyFetched: true },
    ]);
    expect(emails.selectNeedFetch(acct.id, ['m'])).toEqual([]);
  });
});

describe('EmailRepository body-change embedding invalidation', () => {
  /** Creates an account with one fetched email that already has a saved embedding, returning the account id. */
  function embeddedEmail(body: string): string {
    const acct = accounts.create({ address: 'b@y.z', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'm', accountId: acct.id, folder: 'INBOX', body, bodyFetched: true },
    ]);
    embeddings.saveEmbedding({ messageId: 'm', accountId: acct.id, modelId: 'bge-m3' }, vec());
    return acct.id;
  }

  it('drops the stale embedding when the body changes', () => {
    const acctId = embeddedEmail('first version');
    emails.upsertBatch([
      {
        messageId: 'm',
        accountId: acctId,
        folder: 'INBOX',
        body: 'second version',
        bodyFetched: true,
      },
    ]);
    expect(
      embeddings.getEmbedding({ messageId: 'm', accountId: acctId, modelId: 'bge-m3' }),
    ).toBeNull();
  });

  it('keeps the embedding when the body is unchanged', () => {
    const acctId = embeddedEmail('same text');
    emails.upsertBatch([
      { messageId: 'm', accountId: acctId, folder: 'INBOX', body: 'same text', bodyFetched: true },
    ]);
    expect(
      embeddings.getEmbedding({ messageId: 'm', accountId: acctId, modelId: 'bge-m3' }),
    ).not.toBeNull();
  });

  it('keeps the embedding when a re-push omits the body (metadata-only update)', () => {
    const acctId = embeddedEmail('keep me');
    emails.upsertBatch([
      { messageId: 'm', accountId: acctId, folder: 'ARCHIVE', bodyFetched: true },
    ]);
    expect(
      embeddings.getEmbedding({ messageId: 'm', accountId: acctId, modelId: 'bge-m3' }),
    ).not.toBeNull();
  });
});

describe('DashboardService coverage counts (triage vs category wiring bug)', () => {
  it('reports uncategorized from category assignments, independent of triage', () => {
    const acct = accounts.create({ address: 'dash@x.y', kind: 'work' });
    const triage = new TriageRepository(db);
    const dash = new DashboardService(emails, triage, categories);
    const cat = categories.create({ accountId: acct.id, label: 'C', source: 'auto' });
    const now = Date.now();

    for (let i = 0; i < 3; i++) {
      emails.upsertBatch([
        {
          messageId: `e${i}`,
          accountId: acct.id,
          folder: 'INBOX',
          body: 'x',
          bodyFetched: true,
          date: now + i,
        },
      ]);
    }
    for (const id of ['e0', 'e1']) {
      categories.replaceEmailAssignments(id, acct.id, [
        {
          messageId: id,
          accountId: acct.id,
          categoryId: cat.id,
          confidence: 0.9,
          assignedBy: 'auto',
          assignedAt: now,
        },
      ]);
    }
    triage.upsert({ messageId: 'e2', accountId: acct.id, bucket: 'urgent' });

    const d = dash.build(acct.id);
    expect(d.emails.total).toBe(3);
    expect(d.emails.uncategorized).toBe(1);
    expect(d.emails.unclassified).toBe(2);
  });
});

describe('CategoryRepository provenance skip sets (incremental categorization)', () => {
  it('distinguishes any-assigned from llm-protected by method + assigned_by', () => {
    const acct = accounts.create({ address: 'p@x.y', kind: 'work' });
    const c1 = categories.create({ accountId: acct.id, label: 'C1', source: 'auto' });
    const c2 = categories.create({ accountId: acct.id, label: 'C2', source: 'auto' });
    const now = Date.now();
    for (const id of ['mUser', 'mEmbed', 'mLlm', 'mNone']) {
      emails.upsertBatch([{ messageId: id, accountId: acct.id, folder: 'INBOX' }]);
    }
    categories.replaceEmailAssignments('mUser', acct.id, [
      {
        messageId: 'mUser',
        accountId: acct.id,
        categoryId: c1.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: now,
      },
    ]);
    categories.addAutoAssignments(acct.id, [
      {
        messageId: 'mEmbed',
        accountId: acct.id,
        categoryId: c1.id,
        confidence: 0.9,
        assignedBy: 'auto',
        assignedAt: now,
        method: 'embed',
      },
    ]);
    categories.bulkReplaceForCluster(acct.id, ['mLlm'], [c2.id], 'llm');

    expect(categories.getAssignedMessageIds(acct.id)).toEqual(new Set(['mUser', 'mEmbed', 'mLlm']));
    expect(categories.getLlmProtectedMessageIds(acct.id)).toEqual(new Set(['mUser', 'mLlm']));
  });

  it('addAutoAssignments adds without wiping existing assignments', () => {
    const acct = accounts.create({ address: 'add@x.y', kind: 'work' });
    const c1 = categories.create({ accountId: acct.id, label: 'C1', source: 'auto' });
    const now = Date.now();
    emails.upsertBatch([
      { messageId: 'a', accountId: acct.id, folder: 'INBOX' },
      { messageId: 'b', accountId: acct.id, folder: 'INBOX' },
    ]);
    categories.addAutoAssignments(acct.id, [
      {
        messageId: 'a',
        accountId: acct.id,
        categoryId: c1.id,
        confidence: 0.9,
        assignedBy: 'auto',
        assignedAt: now,
        method: 'embed',
      },
    ]);
    categories.addAutoAssignments(acct.id, [
      {
        messageId: 'b',
        accountId: acct.id,
        categoryId: c1.id,
        confidence: 0.9,
        assignedBy: 'auto',
        assignedAt: now,
        method: 'embed',
      },
    ]);
    expect(categories.getAssignedMessageIds(acct.id)).toEqual(new Set(['a', 'b']));
  });
});

describe('CategoryRepository.reconcileAutoCategories (stable identities)', () => {
  it('keeps matching labels (stable id), preserves user-corrected categories, keeps obsolete active', () => {
    const acct = accounts.create({ address: 'rec@x.y', kind: 'work' });
    const centroid = vec(0.2);
    categories.reconcileAutoCategories(acct.id, 'bge-m3', [
      { label: 'Job Alerts', description: 'd', centroid, emailCount: 1 },
      { label: 'Old Topic', description: 'd', centroid, emailCount: 1 },
      { label: 'Stale', description: 'd', centroid, emailCount: 1 },
    ]);
    const seeded = categories.listForAccount(acct.id);
    const jobId = seeded.find((c) => c.label === 'Job Alerts')!.id;
    const oldId = seeded.find((c) => c.label === 'Old Topic')!.id;

    emails.upsertBatch([{ messageId: 'u', accountId: acct.id, folder: 'INBOX' }]);
    categories.replaceEmailAssignments('u', acct.id, [
      {
        messageId: 'u',
        accountId: acct.id,
        categoryId: oldId,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: Date.now(),
      },
    ]);

    categories.reconcileAutoCategories(acct.id, 'bge-m3', [
      { label: 'Job Alerts', description: 'updated', centroid, emailCount: 5 },
      { label: 'New', description: 'd', centroid, emailCount: 2 },
    ]);

    const after = categories.listForAccount(acct.id);
    expect(after.map((c) => c.label).sort()).toEqual(['Job Alerts', 'New', 'Old Topic', 'Stale']);
    expect(after.find((c) => c.label === 'Job Alerts')!.id).toBe(jobId);
    expect(after.find((c) => c.label === 'Old Topic')!.id).toBe(oldId);
    expect(categories.getUserAssignedMessageIds(acct.id)).toEqual(new Set(['u']));
  });
});

describe('ConversationRepository (chat record persistence)', () => {
  it('creates, appends turns, reloads, lists, and deletes a conversation', () => {
    const conversations = new ConversationRepository(db);
    const acct = accounts.create({ address: 'c@x.y', kind: 'work' });

    const convo = conversations.create(acct.id);
    expect(conversations.get(convo.id)?.turns).toEqual([]);

    conversations.append(convo.id, [
      { role: 'user', content: 'when is the deadline?', at: 1 },
      {
        role: 'assistant',
        content: 'Friday [1].',
        at: 2,
        sources: [{ messageId: 'm1', subject: 'Deadline', fromAddr: 'p@x.y', date: 1, score: 0.7 }],
      },
    ]);
    conversations.append(convo.id, [{ role: 'user', content: 'and the time?', at: 3 }]);

    const loaded = conversations.get(convo.id);
    expect(loaded?.turns).toHaveLength(3);
    expect(loaded?.turns[1]?.sources?.[0]?.messageId).toBe('m1');

    const list = conversations.listForAccount(acct.id);
    expect(list).toHaveLength(1);
    expect(list[0]?.preview).toBe('when is the deadline?');

    expect(conversations.delete(convo.id)).toBe(true);
    expect(conversations.get(convo.id)).toBeNull();
  });
});

describe('CategoryRepository.replaceAutoCategories (retired)', () => {
  it('throws, because it destroyed category identity; reconcileAutoCategories replaces it', () => {
    expect(() => categories.replaceAutoCategories()).toThrow(/retired/);
  });
});

describe('ConversationRepository summary-buffer persistence', () => {
  it('persists and preserves the rolling summary across append', () => {
    const acct = accounts.create({ address: 'conv@x.y', kind: 'work' });
    const convos = new ConversationRepository(db);
    const c = convos.create(acct.id);
    expect(c.summary).toBe('');
    expect(c.summarizedCount).toBe(0);

    const now = Date.now();
    convos.append(c.id, [{ role: 'user', content: 'q1', at: now }]);
    convos.updateSummary(c.id, 'rolling summary', 1);

    const afterSummary = convos.get(c.id)!;
    expect(afterSummary.summary).toBe('rolling summary');
    expect(afterSummary.summarizedCount).toBe(1);

    convos.append(c.id, [{ role: 'assistant', content: 'a1', at: now + 1 }]);
    const afterAppend = convos.get(c.id)!;
    expect(afterAppend.summary).toBe('rolling summary');
    expect(afterAppend.summarizedCount).toBe(1);
    expect(afterAppend.turns).toHaveLength(2);
  });

  it('clamps a corrupt summarizedCount to the number of turns', () => {
    const acct = accounts.create({ address: 'clamp@x.y', kind: 'work' });
    const convos = new ConversationRepository(db);
    const c = convos.create(acct.id);
    convos.append(c.id, [{ role: 'user', content: 'only one', at: Date.now() }]);
    convos.updateSummary(c.id, 's', 99);
    expect(convos.get(c.id)!.summarizedCount).toBe(1);
  });

  it('updateSummary is monotonic: a smaller count cannot regress the summary', () => {
    const acct = accounts.create({ address: 'mono@x.y', kind: 'work' });
    const convos = new ConversationRepository(db);
    const c = convos.create(acct.id);
    convos.append(c.id, [
      { role: 'user', content: 'a', at: 1 },
      { role: 'assistant', content: 'b', at: 2 },
      { role: 'user', content: 'c', at: 3 },
    ]);
    convos.updateSummary(c.id, 'advanced', 2);
    convos.updateSummary(c.id, 'stale', 1);
    const after = convos.get(c.id)!;
    expect(after.summary).toBe('advanced');
    expect(after.summarizedCount).toBe(2);
  });
});

describe('EmailRepository.keywordSearch (FTS5 hybrid retrieval)', () => {
  it('finds emails by exact term in subject or body, scoped to the account', () => {
    const acct = accounts.create({ address: 'k@x.y', kind: 'work' });
    const other = accounts.create({ address: 'o@x.y', kind: 'work' });
    emails.upsertBatch([
      {
        messageId: 'viva',
        accountId: acct.id,
        folder: 'INBOX',
        subject: 'CSC101 project vivas',
        body: 'Your viva is scheduled.',
      },
      {
        messageId: 'bank',
        accountId: acct.id,
        folder: 'INBOX',
        subject: 'Statement',
        body: 'Your monthly balance is ready.',
      },
      {
        messageId: 'xacct',
        accountId: other.id,
        folder: 'INBOX',
        subject: 'viva elsewhere',
        body: 'viva',
      },
    ]);

    const hits = emails.keywordSearch(acct.id, 'when was my viva?');
    expect(hits).toContain('viva');
    expect(hits).not.toContain('bank');
    expect(hits).not.toContain('xacct');
  });

  it('matches across accents (remove_diacritics) for French queries', () => {
    const acct = accounts.create({ address: 'fr@x.y', kind: 'work' });
    emails.upsertBatch([
      {
        messageId: 'm',
        accountId: acct.id,
        folder: 'INBOX',
        subject: 'Résumé du stage',
        body: 'Voici le résumé.',
      },
    ]);
    expect(emails.keywordSearch(acct.id, 'resume')).toContain('m');
  });

  it('returns [] for a punctuation-only query and never throws on FTS operators', () => {
    const acct = accounts.create({ address: 'q@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'm', accountId: acct.id, folder: 'INBOX', subject: 'hello', body: 'world' },
    ]);
    expect(emails.keywordSearch(acct.id, '???')).toEqual([]);
    expect(() => emails.keywordSearch(acct.id, 'hello AND "world" OR (x*')).not.toThrow();
  });

  it('keeps the index in sync when a body changes or an email is deleted', () => {
    const acct = accounts.create({ address: 'sync@x.y', kind: 'work' });
    emails.upsertBatch([
      {
        messageId: 'm',
        accountId: acct.id,
        folder: 'INBOX',
        subject: 'about zebras',
        body: 'zebra facts',
        bodyFetched: true,
      },
    ]);
    expect(emails.keywordSearch(acct.id, 'zebra')).toContain('m');

    emails.upsertBatch([
      {
        messageId: 'm',
        accountId: acct.id,
        folder: 'INBOX',
        subject: 'about giraffes',
        body: 'giraffe facts',
        bodyFetched: true,
      },
    ]);
    expect(emails.keywordSearch(acct.id, 'zebra')).not.toContain('m');
    expect(emails.keywordSearch(acct.id, 'giraffe')).toContain('m');

    emails.upsertBatch([
      {
        messageId: 'm',
        accountId: acct.id,
        folder: 'ARCHIVE',
        subject: 'about giraffes',
        bodyFetched: true,
      },
    ]);
    expect(emails.keywordSearch(acct.id, 'giraffe')).toContain('m');

    emails.delete('m', acct.id);
    expect(emails.keywordSearch(acct.id, 'giraffe')).toEqual([]);
  });

  it('lists ids within a date range newest first, excluding dateless emails', () => {
    const acct = accounts.create({ address: 'range@x.y', kind: 'work' });
    const day = 86_400_000;
    const base = Date.UTC(2026, 5, 1);
    emails.upsertBatch([
      { messageId: 'd0', accountId: acct.id, folder: 'INBOX', date: base },
      { messageId: 'd1', accountId: acct.id, folder: 'INBOX', date: base + day },
      { messageId: 'd2', accountId: acct.id, folder: 'INBOX', date: base + 2 * day },
      { messageId: 'far', accountId: acct.id, folder: 'INBOX', date: base + 100 * day },
      { messageId: 'nodate', accountId: acct.id, folder: 'INBOX' },
    ]);

    const ids = emails.listIdsInRange(acct.id, base, base + 3 * day);
    expect(ids).toEqual(['d2', 'd1', 'd0']);
    expect(emails.listIdsInRange(acct.id, base, base + 3 * day, 2)).toEqual(['d2', 'd1']);
  });

  it('backfills emails that existed before the FTS migration ran', () => {
    const raw = new BetterSqlite3(':memory:');
    raw.pragma('foreign_keys = ON');
    sqliteVec.load(raw);
    runMigrations(
      raw,
      migrations.filter((m) => m.version <= 7),
    );
    const now = Date.now();
    raw
      .prepare('INSERT INTO accounts (id, address, kind, created_at) VALUES (?, ?, ?, ?)')
      .run('acct', 'a@x.y', 'work', now);
    raw
      .prepare(
        `INSERT INTO emails (message_id, account_id, folder, subject, from_addr, date, has_attachments, body, body_format, body_fetched, indexed_at)
         VALUES (?, ?, 'INBOX', ?, 's@x.y', ?, 0, ?, 'text', 1, ?)`,
      )
      .run('old', 'acct', 'Old viva email', now, 'Your CSC101 viva is scheduled', now);

    runMigrations(raw, migrations);

    const repo = new EmailRepository(raw);
    expect(repo.keywordSearch('acct', 'viva')).toContain('old');
    raw.close();
  });
});

describe("PriorityService + TriageRepository (Today's Focus)", () => {
  const DAY = 24 * 60 * 60 * 1000;
  const dayStart = 1_750_000_000_000;
  const now = dayStart + 10 * 3_600_000;
  let triage: TriageRepository;
  let priority: PriorityService;
  let acct: { id: string };

  /** Builds triage metadata with neutral defaults overridable per test. */
  const meta = (m: Partial<TriageMetadata> = {}): TriageMetadata => ({
    actionRequired: false,
    needsReply: false,
    deadlineAt: null,
    importanceScore: 50,
    suggestedAction: null,
    shortSummary: null,
    confidence: null,
    ...m,
  });

  /** Inserts an email and its triage row for the current account in one step. */
  function seed(
    messageId: string,
    dateMs: number,
    bucket: TriageBucket,
    m?: Partial<TriageMetadata>,
  ): void {
    emails.upsertBatch([
      { messageId, accountId: acct.id, folder: 'INBOX', subject: messageId, date: dateMs },
    ]);
    triage.upsert({ messageId, accountId: acct.id, bucket, reasoning: 'r', metadata: meta(m) });
  }

  beforeEach(() => {
    triage = new TriageRepository(db);
    priority = new PriorityService(triage);
    acct = accounts.create({ address: 'focus@x.y', kind: 'work' });
  });

  it('today returns only today emails, with older unresolved action as carryover', () => {
    seed('today1', dayStart + 3 * 3_600_000, 'urgent', { actionRequired: true });
    seed('older3d', dayStart - 3 * DAY, 'urgent', { actionRequired: true });
    seed('old20d', dayStart - 20 * DAY, 'urgent', { actionRequired: true });

    const p = priority.build(acct.id, { range: 'today', dayStartMs: dayStart, now });
    expect(p.counts.needsAction).toBe(1);
    expect(p.needsAction.map((e) => e.messageId)).toEqual(['today1']);
    expect(p.carryover.map((e) => e.messageId)).toEqual(['older3d']);
  });

  it('last 7 days includes the recent window', () => {
    seed('today1', dayStart + 3_600_000, 'urgent', { actionRequired: true });
    seed('d3', dayStart - 3 * DAY, 'urgent', { actionRequired: true });
    seed('d10', dayStart - 10 * DAY, 'urgent', { actionRequired: true });

    const p = priority.build(acct.id, { range: 'week', dayStartMs: dayStart, now });
    expect(p.counts.needsAction).toBe(2);
    expect(new Set(p.needsAction.map((e) => e.messageId))).toEqual(new Set(['today1', 'd3']));
    expect(p.carryover.map((e) => e.messageId)).toEqual(['d10']);
  });

  it('all returns everything and no carryover', () => {
    seed('today1', dayStart, 'urgent', { actionRequired: true });
    seed('d30', dayStart - 30 * DAY, 'urgent', { actionRequired: true });
    const p = priority.build(acct.id, { range: 'all', dayStartMs: dayStart, now });
    expect(p.counts.needsAction).toBe(2);
    expect(p.carryover).toEqual([]);
  });

  it('respects the local-timezone day boundary', () => {
    seed('justBefore', dayStart - 1, 'urgent', { actionRequired: true });
    seed('atStart', dayStart, 'urgent', { actionRequired: true });
    seed('tomorrow', dayStart + DAY + 1, 'urgent', { actionRequired: true });
    emails.upsertBatch([
      {
        messageId: 'unclassifiedTomorrow',
        accountId: acct.id,
        folder: 'INBOX',
        date: dayStart + DAY + 2,
      },
    ]);
    const p = priority.build(acct.id, { range: 'today', dayStartMs: dayStart, now });
    expect(p.needsAction.map((e) => e.messageId)).toEqual(['atStart']);
    expect(p.carryover.map((e) => e.messageId)).toEqual(['justBefore']);
    expect(p.counts.unclassified).toBe(0);
  });

  it('last 7 days ends at tomorrow local midnight, not the infinite future', () => {
    seed('d3', dayStart - 3 * DAY, 'urgent', { actionRequired: true });
    seed('tomorrow', dayStart + DAY + 1, 'urgent', { actionRequired: true });

    const p = priority.build(acct.id, { range: 'week', dayStartMs: dayStart, now });
    expect(p.needsAction.map((e) => e.messageId)).toEqual(['d3']);
  });

  it('carryover excludes emails the user resolved (done)', () => {
    seed('d3done', dayStart - 3 * DAY, 'urgent', { actionRequired: true });
    triage.setResolution(acct.id, 'd3done', null, now, null);
    const p = priority.build(acct.id, { range: 'today', dayStartMs: dayStart, now });
    expect(p.carryover).toEqual([]);
  });

  it('hides a snoozed email until its snooze passes', () => {
    seed('snoozy', dayStart + 1, 'urgent', { actionRequired: true });
    triage.setResolution(acct.id, 'snoozy', null, null, dayStart + 2 * DAY);
    expect(
      priority.build(acct.id, { range: 'today', dayStartMs: dayStart, now }).counts.needsAction,
    ).toBe(0);
    const later = dayStart + 3 * DAY;
    expect(
      priority.build(acct.id, { range: 'all', dayStartMs: dayStart, now: later }).counts
        .needsAction,
    ).toBe(1);
  });

  it('partitions important vs summaries by importance, and spam to low priority', () => {
    seed('imp', dayStart + 1, 'summarize', { importanceScore: 70 });
    seed('digest', dayStart + 2, 'summarize', { importanceScore: 30 });
    seed('junk', dayStart + 3, 'spam', { actionRequired: true });
    const p = priority.build(acct.id, { range: 'today', dayStartMs: dayStart, now });
    expect(p.important.map((e) => e.messageId)).toContain('imp');
    expect(p.summaries.map((e) => e.messageId)).toContain('digest');
    expect(p.lowPriority.map((e) => e.messageId)).toEqual(['junk']);
    expect(p.counts.needsAction).toBe(0);
  });

  it('reports a range-aware unclassified count', () => {
    emails.upsertBatch([
      { messageId: 'u1', accountId: acct.id, folder: 'INBOX', date: dayStart + 1 },
      { messageId: 'u2', accountId: acct.id, folder: 'INBOX', date: dayStart + 2 },
      { messageId: 'uOld', accountId: acct.id, folder: 'INBOX', date: dayStart - 10 * DAY },
    ]);
    expect(
      priority.build(acct.id, { range: 'today', dayStartMs: dayStart, now }).counts.unclassified,
    ).toBe(2);
    expect(
      priority.build(acct.id, { range: 'all', dayStartMs: dayStart, now }).counts.unclassified,
    ).toBe(3);
  });

  it('renders old rows with null metadata (pre-migration) without crashing', () => {
    emails.upsertBatch([
      {
        messageId: 'oldrow',
        accountId: acct.id,
        folder: 'INBOX',
        subject: 'Old',
        date: dayStart + 1,
      },
    ]);
    triage.upsert({
      messageId: 'oldrow',
      accountId: acct.id,
      bucket: 'summarize',
      reasoning: 'old',
    });
    const p = priority.build(acct.id, { range: 'today', dayStartMs: dayStart, now });
    const row = p.summaries.find((e) => e.messageId === 'oldrow');
    expect(row).toBeDefined();
    expect(row!.importanceScore).toBe(35);
    expect(row!.actionRequired).toBe(false);
  });

  it('treats never-triaged and null-metadata legacy rows as pending priority work', () => {
    emails.upsertBatch([
      { messageId: 'new', accountId: acct.id, folder: 'INBOX', subject: 'New', date: dayStart + 3 },
      {
        messageId: 'legacy',
        accountId: acct.id,
        folder: 'INBOX',
        subject: 'Legacy',
        date: dayStart + 2,
      },
      {
        messageId: 'done',
        accountId: acct.id,
        folder: 'INBOX',
        subject: 'Done',
        date: dayStart + 1,
      },
    ]);
    triage.upsert({ messageId: 'legacy', accountId: acct.id, bucket: 'urgent', reasoning: 'old' });
    triage.upsert({
      messageId: 'done',
      accountId: acct.id,
      bucket: 'summarize',
      reasoning: 'done',
      metadata: meta(),
    });

    expect(triage.countPendingTriage(acct.id)).toBe(2);
    expect(triage.findPendingTriageEmails(acct.id).map((e) => e.messageId)).toEqual([
      'new',
      'legacy',
    ]);
    expect(triage.countPendingTriage(acct.id, true)).toBe(3);
  });
});

describe('EmailAssistantService', () => {
  it('generates, caches, and invalidates an opened-email summary with attachment status', async () => {
    const acct = accounts.create({ address: 'assistant@x.y', kind: 'work' });
    emails.upsert({
      messageId: 'm1',
      accountId: acct.id,
      folder: 'INBOX',
      subject: 'Project update',
      fromAddr: 'Ada <ada@example.com>',
      date: 1_700_000_000_000,
      body: 'Please review the attached project plan by Friday.',
      hasAttachments: true,
    });
    const attachments = new AttachmentRepository(db);
    const att = attachments.upsertAttachment({
      accountId: acct.id,
      messageId: 'm1',
      filename: 'plan.txt',
      partName: '1.2',
    });
    attachments.replaceChunks(att.id, 'm1', acct.id, ['Milestone one is due Friday.']);
    attachments.setStatus(att.id, 'extracted', 28);

    let calls = 0;
    const llm = {
      chat: async () => {
        calls += 1;
        return JSON.stringify({
          summary: 'Ada asks the user to review the project plan.',
          keyPoints: ['Review requested', 'Plan attachment was included'],
          actionRequired: true,
          needsReply: false,
          deadline: 'Friday',
          suggestedAction: 'Review the project plan',
          attachmentSummary: 'The plan mentions a Friday milestone.',
        });
      },
    };
    const svc = new EmailAssistantService(
      llm as never,
      accounts,
      emails,
      attachments,
      new EmailAssistantRepository(db),
      silentLogger,
    );

    const first = await svc.summarize(acct.id, 'm1', { modelId: 'gpt-4o-mini', provider: 'cloud' });
    expect(first.cached).toBe(false);
    expect(first.attachments).toEqual([
      { filename: 'plan.txt', status: 'extracted', included: true },
    ]);
    expect(calls).toBe(1);

    const second = await svc.summarize(acct.id, 'm1', {
      modelId: 'gpt-4o-mini',
      provider: 'cloud',
    });
    expect(second.cached).toBe(true);
    expect(calls).toBe(1);

    emails.upsert({
      messageId: 'm1',
      accountId: acct.id,
      folder: 'INBOX',
      subject: 'Project update',
      body: 'Please review the updated project plan by Monday.',
      hasAttachments: true,
    });
    await svc.summarize(acct.id, 'm1', { modelId: 'gpt-4o-mini', provider: 'cloud' });
    expect(calls).toBe(2);
  });

  it('degrades instead of throwing when the model omits the summary field, and does not cache it', async () => {
    const acct = accounts.create({ address: 'nosum@x.y', kind: 'work' });
    emails.upsert({
      messageId: 'm-nosum',
      accountId: acct.id,
      folder: 'INBOX',
      subject: 'Permit renewal',
      fromAddr: 'Office <office@example.com>',
      body: 'It is important not to wait until the last minute to avoid delays.',
      hasAttachments: false,
    });

    let calls = 0;
    const llm = {
      chat: async () => {
        calls += 1;
        return JSON.stringify({
          keyPoints: ['Renew the residence permit early', 'Avoid last-minute delays'],
          actionRequired: true,
          deadline: 'before it expires',
        });
      },
    };
    const svc = new EmailAssistantService(
      llm as never,
      accounts,
      emails,
      new AttachmentRepository(db),
      new EmailAssistantRepository(db),
      silentLogger,
    );

    const out = await svc.summarize(acct.id, 'm-nosum', { modelId: 'qwen3:8b', provider: 'local' });
    expect(out.summary.length).toBeGreaterThan(0);
    expect(out.actionRequired).toBe(true);
    expect(out.keyPoints).toContain('Renew the residence permit early');

    await svc.summarize(acct.id, 'm-nosum', { modelId: 'qwen3:8b', provider: 'local' });
    expect(calls).toBe(2);
  });

  it('degrades instead of throwing when the model returns non-JSON', async () => {
    const acct = accounts.create({ address: 'badjson@x.y', kind: 'work' });
    emails.upsert({
      messageId: 'm-badjson',
      accountId: acct.id,
      folder: 'INBOX',
      subject: 'Hi',
      body: 'Some body text.',
      hasAttachments: false,
    });
    const llm = { chat: async () => 'Sorry, I cannot produce JSON for this.' };
    const svc = new EmailAssistantService(
      llm as never,
      accounts,
      emails,
      new AttachmentRepository(db),
      new EmailAssistantRepository(db),
      silentLogger,
    );
    const out = await svc.summarize(acct.id, 'm-badjson', {
      modelId: 'qwen3:8b',
      provider: 'local',
    });
    expect(out.summary.length).toBeGreaterThan(0);
  });

  it('dedupes concurrent opened-email summary generation for the same model and content', async () => {
    const acct = accounts.create({ address: 'dedupe@x.y', kind: 'work' });
    emails.upsert({
      messageId: 'm-dedupe',
      accountId: acct.id,
      folder: 'INBOX',
      subject: 'Slow local summary',
      fromAddr: 'Ada <ada@example.com>',
      body: 'This summary is intentionally slow.',
    });

    let calls = 0;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const llm = {
      chat: async () => {
        calls += 1;
        await gate;
        return JSON.stringify({
          summary: 'Ada sent a slow test email.',
          keyPoints: ['Slow summary path'],
          actionRequired: false,
          needsReply: false,
          deadline: null,
          suggestedAction: null,
          attachmentSummary: null,
        });
      },
    };
    const svc = new EmailAssistantService(
      llm as never,
      accounts,
      emails,
      new AttachmentRepository(db),
      new EmailAssistantRepository(db),
      silentLogger,
    );

    const first = svc.summarize(acct.id, 'm-dedupe', { modelId: 'qwen3', provider: 'local' });
    const second = svc.summarize(acct.id, 'm-dedupe', { modelId: 'qwen3', provider: 'local' });
    expect(calls).toBe(1);

    release();
    const [a, b] = await Promise.all([first, second]);
    expect(a.cached).toBe(false);
    expect(b.summary).toBe(a.summary);
    expect(calls).toBe(1);

    const cached = await svc.summarize(acct.id, 'm-dedupe', {
      modelId: 'qwen3',
      provider: 'local',
    });
    expect(cached.cached).toBe(true);
    expect(calls).toBe(1);
  });

  it('drafts through the selected chat provider and omits Ollama-only fields on cloud', async () => {
    const acct = accounts.create({ address: 'draft@x.y', displayName: 'Noor', kind: 'work' });
    emails.upsert({
      messageId: 'm2',
      accountId: acct.id,
      folder: 'INBOX',
      subject: 'Meeting',
      fromAddr: 'Max <max@example.com>',
      body: 'Can you meet tomorrow?',
    });
    let req: unknown;
    const llm = {
      chat: async (opts: unknown) => {
        req = opts;
        return 'Hi Max,\n\nTomorrow works for me.\n\nBest,';
      },
    };
    const svc = new EmailAssistantService(
      llm as never,
      accounts,
      emails,
      new AttachmentRepository(db),
      new EmailAssistantRepository(db),
      silentLogger,
    );

    const out = await svc.draftReply(acct.id, 'm2', 'Keep it short.', {
      modelId: 'gpt-4o-mini',
      provider: 'cloud',
    });
    expect(out.draft).toContain('Tomorrow works');
    expect(out.draft).toBe('Hi Max,\n\nTomorrow works for me.\n\nBest,\nNoor');
    expect(req).toMatchObject({
      model: 'gpt-4o-mini',
      provider: 'chat',
      think: undefined,
    });
  });

  it('repairs a one-paragraph model draft into professional email format', async () => {
    const acct = accounts.create({
      address: 'noor@example.com',
      displayName: 'Noor',
      kind: 'work',
    });
    emails.upsert({
      messageId: 'm3',
      accountId: acct.id,
      folder: 'INBOX',
      subject: 'Application update',
      fromAddr: 'Claire <claire@example.com>',
      body: 'Thank you for applying. We will contact you after the review process.',
    });
    const llm = {
      chat: async () =>
        'Thank you for the update, Claire. I appreciate the information regarding the review process. Best regards,',
    };
    const svc = new EmailAssistantService(
      llm as never,
      accounts,
      emails,
      new AttachmentRepository(db),
      new EmailAssistantRepository(db),
      silentLogger,
    );

    const out = await svc.draftReply(acct.id, 'm3', undefined, {
      modelId: 'gpt-4o-mini',
      provider: 'cloud',
    });
    expect(out.draft).toBe(
      'Hello Claire,\n\nThank you for the update, Claire. I appreciate the information regarding the review process.\n\nBest regards,\nNoor',
    );
  });
});
