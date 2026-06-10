/**
 * Tests for the user correction and learning loop, covering manual category
 * assignment with centroid nudging, idempotency and dedup of corrections,
 * auto-categorization respecting user locks, and approved category expansions.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import type { Logger } from 'pino';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { CategorizationService } from '../src/services/categorization-service.js';
import { CorrectionService } from '../src/services/correction-service.js';
import { CategoryImprovementService } from '../src/services/category-improvement-service.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { l2Distance } from '../src/util/vector.js';

let db: Database;
let accounts: AccountRepository;
let emails: EmailRepository;
let embeddings: EmbeddingRepository;
let categories: CategoryRepository;
let correction: CorrectionService;

const MODEL = 'bge-m3';
const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

/** Builds a test embedding with a fixed first dimension and the given second value. */
function vec(second = 0): Float32Array {
  const arr = new Float32Array(EMBEDDING_DIM);
  arr[0] = 1;
  arr[1] = second;
  return arr;
}

beforeEach(() => {
  db = openDatabase(':memory:');
  accounts = new AccountRepository(db);
  emails = new EmailRepository(db);
  embeddings = new EmbeddingRepository(db);
  categories = new CategoryRepository(db);
  correction = new CorrectionService(db, categories, embeddings);
});

afterEach(() => db.close());

/** Inserts an email and its embedding so categorization tests have data to act on. */
function seedEmail(accountId: string, messageId: string, embeddingSecond: number): void {
  emails.upsertBatch([{ messageId, accountId, folder: 'INBOX' }]);
  embeddings.saveEmbedding({ messageId, accountId, modelId: MODEL }, vec(embeddingSecond));
}

describe('CorrectionService.setUserCategories (corrections + learning loop)', () => {
  it('files the email as a user assignment, locks it, and nudges the centroid toward it', () => {
    const acct = accounts.create({ address: 'u@x.y', kind: 'work' });
    seedEmail(acct.id, 'msg', 0.5);
    const cat = categories.create({ accountId: acct.id, label: 'Target', source: 'auto' });
    categories.saveCentroid(cat.id, MODEL, vec(0), 3);

    const before = categories.getCentroid(cat.id, MODEL)!;
    correction.setUserCategories(acct.id, 'msg', [cat.id], MODEL);

    const assigned = categories.getEmailCategoriesWithLabels('msg', acct.id);
    expect(assigned).toHaveLength(1);
    expect(assigned[0]!.categoryId).toBe(cat.id);
    expect(assigned[0]!.assignedBy).toBe('user');
    expect(assigned[0]!.confidence).toBeCloseTo(1);

    expect(categories.getUserAssignedMessageIds(acct.id).has('msg')).toBe(true);

    const after = categories.getCentroid(cat.id, MODEL)!;
    expect(after.emailCount).toBe(4);
    expect(after.vector[1]).toBeGreaterThan(before.vector[1]);
    expect(l2Distance(after.vector, vec(0.5))).toBeLessThan(l2Distance(before.vector, vec(0.5)));
  });

  it('is idempotent: re-filing the same email does not re-nudge or inflate the count', () => {
    const acct = accounts.create({ address: 'idem@x.y', kind: 'work' });
    seedEmail(acct.id, 'msg', 0.4);
    const cat = categories.create({ accountId: acct.id, label: 'C', source: 'auto' });
    categories.saveCentroid(cat.id, MODEL, vec(0), 2);

    const r1 = correction.setUserCategories(acct.id, 'msg', [cat.id], MODEL);
    expect(r1.centroidsUpdated).toBe(1);
    const afterFirst = categories.getCentroid(cat.id, MODEL)!;
    expect(afterFirst.emailCount).toBe(3);

    const r2 = correction.setUserCategories(acct.id, 'msg', [cat.id], MODEL);
    expect(r2.centroidsUpdated).toBe(0);
    const afterSecond = categories.getCentroid(cat.id, MODEL)!;
    expect(afterSecond.emailCount).toBe(3);
    expect(afterSecond.vector[1]).toBeCloseTo(afterFirst.vector[1], 6);
  });

  it('deduplicates category ids so a centroid is nudged at most once', () => {
    const acct = accounts.create({ address: 'dup@x.y', kind: 'work' });
    seedEmail(acct.id, 'msg', 0.4);
    const cat = categories.create({ accountId: acct.id, label: 'C', source: 'auto' });
    categories.saveCentroid(cat.id, MODEL, vec(0), 1);

    const r = correction.setUserCategories(acct.id, 'msg', [cat.id, cat.id], MODEL);
    expect(r.applied).toBe(1);
    expect(r.centroidsUpdated).toBe(1);
    expect(categories.getCentroid(cat.id, MODEL)!.emailCount).toBe(2);
  });

  it('seeds a centroid from the email when the category has none', () => {
    const acct = accounts.create({ address: 'u2@x.y', kind: 'work' });
    seedEmail(acct.id, 'msg', 0.3);
    const cat = categories.create({ accountId: acct.id, label: 'New', source: 'user' });
    expect(categories.getCentroid(cat.id, MODEL)).toBeNull();

    correction.setUserCategories(acct.id, 'msg', [cat.id], MODEL);

    const centroid = categories.getCentroid(cat.id, MODEL);
    expect(centroid).not.toBeNull();
    expect(centroid!.emailCount).toBe(1);
  });
});

describe('auto-categorize respects user locks (categorizeBatch)', () => {
  it('skips user-locked emails so corrections are never overwritten', () => {
    const acct = accounts.create({ address: 'u3@x.y', kind: 'work' });
    seedEmail(acct.id, 'locked', 0.0);
    seedEmail(acct.id, 'free', 0.0);
    const cat = categories.create({ accountId: acct.id, label: 'C', source: 'auto' });
    categories.saveCentroid(cat.id, MODEL, vec(0), 1);

    const svc = new CategorizationService(categories, embeddings);

    const open = svc.categorizeBatch(acct.id, MODEL);
    expect(open.scored).toBe(2);
    expect(open.matches.has('locked')).toBe(true);

    const locked = svc.categorizeBatch(acct.id, MODEL, {}, new Set(['locked']));
    expect(locked.scored).toBe(1);
    expect(locked.matches.has('locked')).toBe(false);
    expect(locked.matches.has('free')).toBe(true);
  });
});

describe('CategoryImprovementService.apply existing-category expansions', () => {
  it('files only still-uncategorized approved emails and updates the target centroid', async () => {
    const acct = accounts.create({ address: 'improve@x.y', kind: 'work' });
    seedEmail(acct.id, 'free', 0.6);
    seedEmail(acct.id, 'already-filed', 0.8);

    const target = categories.create({ accountId: acct.id, label: 'Job Opportunities', source: 'auto' });
    const other = categories.create({ accountId: acct.id, label: 'Other', source: 'auto' });
    categories.saveCentroid(target.id, MODEL, vec(0), 1);
    categories.replaceEmailAssignments('already-filed', acct.id, [
      {
        messageId: 'already-filed',
        accountId: acct.id,
        categoryId: other.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: Date.now(),
      },
    ]);

    const before = categories.getCentroid(target.id, MODEL)!;
    const svc = new CategoryImprovementService(
      db,
      { embed: vi.fn() } as never,
      emails,
      embeddings,
      categories,
      silentLogger,
    );

    const out = await svc.apply(acct.id, MODEL, {
      existingCategoryExpansions: [
        { categoryId: target.id, messageIds: ['free', 'already-filed', 'missing'] },
      ],
      newCategories: [],
      merges: [],
    });

    expect(out.expanded).toBe(1);
    expect(categories.getEmailCategoriesWithLabels('free', acct.id)).toMatchObject([
      { categoryId: target.id, assignedBy: 'auto', method: 'llm' },
    ]);
    expect(categories.getEmailCategoriesWithLabels('already-filed', acct.id)).toMatchObject([
      { categoryId: other.id, assignedBy: 'user' },
    ]);

    const after = categories.getCentroid(target.id, MODEL)!;
    expect(after.emailCount).toBe(2);
    expect(after.vector[1]).toBeGreaterThan(before.vector[1]);
  });

  it('can add multiple approved auto tags to the same email without overwriting user labels', async () => {
    const acct = accounts.create({ address: 'multi@x.y', kind: 'work' });
    seedEmail(acct.id, 'multi', 0.5);
    seedEmail(acct.id, 'locked', 0.7);

    const banking = categories.create({ accountId: acct.id, label: 'Banking', source: 'auto' });
    const security = categories.create({ accountId: acct.id, label: 'Security', source: 'auto' });
    const manual = categories.create({ accountId: acct.id, label: 'Manual', source: 'user' });
    categories.saveCentroid(banking.id, MODEL, vec(0), 1);
    categories.saveCentroid(security.id, MODEL, vec(0), 1);
    categories.replaceEmailAssignments('locked', acct.id, [
      {
        messageId: 'locked',
        accountId: acct.id,
        categoryId: manual.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: Date.now(),
      },
    ]);

    const svc = new CategoryImprovementService(
      db,
      { embed: vi.fn() } as never,
      emails,
      embeddings,
      categories,
      silentLogger,
    );

    const out = await svc.apply(acct.id, MODEL, {
      existingCategoryExpansions: [
        { categoryId: banking.id, messageIds: ['multi', 'locked'] },
        { categoryId: security.id, messageIds: ['multi', 'locked'] },
      ],
      newCategories: [],
      merges: [],
    });

    expect(out.expanded).toBe(2);
    expect(categories.getEmailCategoriesWithLabels('multi', acct.id).map((c) => c.label).sort()).toEqual([
      'Banking',
      'Security',
    ]);
    expect(categories.getEmailCategoriesWithLabels('locked', acct.id).map((c) => c.label)).toEqual([
      'Manual',
    ]);
  });
});
