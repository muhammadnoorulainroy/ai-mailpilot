/**
 * Phase 3.3 detection tests: StructuralProposalService turns read-only health metrics into safe
 * merge/retire review-queue proposals. It proposes a merge only for near-duplicate active auto
 * categories, a retire only for empty active auto categories, never touches user-created categories,
 * dedups on suppressionKey (pending, applied, or dismissed), and writes only proposal rows.
 */
import { describe, it, expect } from 'vitest';
import type { Logger } from 'pino';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { CategoryProposalRepository } from '../src/repositories/category-proposal-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { CategoryHealthService } from '../src/services/category-health-service.js';
import { StructuralProposalService } from '../src/services/structural-proposal-service.js';

const MODEL = 'bge-m3';
const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

/** A unit vector on dims 0 and 1 whose cosine to axis(0) is exactly v0 (with v0^2 + v1^2 = 1). */
function tiltedFromAxis0(v0: number, v1: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[0] = v0;
  v[1] = v1;
  return v;
}

function harness() {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const categories = new CategoryRepository(db);
  const proposals = new CategoryProposalRepository(db);
  const embeddings = new EmbeddingRepository(db);
  const emails = new EmailRepository(db);
  const health = new CategoryHealthService(categories, embeddings);
  const service = new StructuralProposalService(categories, proposals, health, silentLogger);
  const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
  return { db, categories, proposals, embeddings, emails, service, accountId: acc.id };
}

type Harness = ReturnType<typeof harness>;

/** Create an active category, optionally seeding a stored centroid and N auto members. */
function makeCategory(
  h: Harness,
  label: string,
  key: string,
  opts: {
    source?: 'auto' | 'user';
    centroid?: Float32Array;
    members?: number;
    memberDim?: number;
  } = {},
) {
  const cat = h.categories.create({
    accountId: h.accountId,
    label,
    source: opts.source ?? 'auto',
    status: 'active',
    canonicalKey: key,
  });
  if (opts.centroid) {
    h.categories.saveCentroid(cat.id, MODEL, opts.centroid, opts.members ?? 1);
  }
  for (let i = 0; i < (opts.members ?? 0); i++) {
    const messageId = `${key}-m${i}`;
    h.emails.upsertBatch([
      { messageId, accountId: h.accountId, folder: 'INBOX', subject: 's', fromAddr: 'a@b.com' },
    ]);
    h.embeddings.saveEmbedding(
      { messageId, accountId: h.accountId, modelId: MODEL },
      axis(opts.memberDim ?? 0),
    );
    h.categories.addAutoAssignments(h.accountId, [
      {
        messageId,
        accountId: h.accountId,
        categoryId: cat.id,
        confidence: 0.5,
        assignedBy: 'auto',
        assignedAt: Date.now(),
        method: 'embed',
      },
    ]);
  }
  return cat;
}

function tableCounts(db: Database) {
  const n = (t: string) => (db.prepare(`SELECT COUNT(*) AS n FROM ${t}`).get() as { n: number }).n;
  return {
    categories: n('categories'),
    emailCategories: n('email_categories'),
    centroids: n('category_embedding_index'),
    audit: n('discovery_audit'),
    proposals: n('category_proposals'),
  };
}

describe('StructuralProposalService merge detection', () => {
  it('creates one merge proposal for two clearly overlapping active auto categories', () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Bills', 'bills', { centroid: axis(0), members: 5, memberDim: 0 });

    const result = h.service.generate(h.accountId, MODEL);

    const merges = result.created.filter((c) => c.kind === 'merge');
    expect(merges).toHaveLength(1);
    const bills = h.categories.findByLabel(h.accountId, 'Bills')!;
    const invoices = h.categories.findByLabel(h.accountId, 'Invoices')!;
    // The larger category (Bills, 5) survives as the target; the smaller (Invoices, 3) is the source.
    expect(merges[0]!.categoryId).toBe(bills.id);
    expect(merges[0]!.sourceCategoryId).toBe(invoices.id);
    expect(merges[0]!.suppressionKey).toBe('merge:bills|invoices');
    // It is stored as a real pending merge proposal via createStructural.
    const pending = h.proposals.listPending(h.accountId).filter((p) => p.kind === 'merge');
    expect(pending).toHaveLength(1);
    expect(pending[0]!.categoryId).toBe(bills.id);
    expect(pending[0]!.sourceCategoryId).toBe(invoices.id);
  });

  it('does not create a merge proposal for well-separated categories', () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Travel', 'travel', { centroid: axis(1), members: 3, memberDim: 1 });

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
  });

  it('does not create a merge for a moderate overlap below the threshold', () => {
    const h = harness();
    // Two auto categories with a genuine overlap of 0.8, which is below MERGE_MIN_OVERLAP (0.9).
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Bills', 'bills', {
      centroid: tiltedFromAxis0(0.8, 0.6),
      members: 3,
      memberDim: 0,
    });

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
  });

  it('creates a merge for an overlap just above the threshold', () => {
    const h = harness();
    // Overlap 0.95: above MERGE_MIN_OVERLAP (0.9) but well short of the identical-centroid case.
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Bills', 'bills', {
      centroid: tiltedFromAxis0(0.95, 0.3122498999199199),
      members: 5,
      memberDim: 0,
    });

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(1);
  });

  it('does not create a merge when the nearest category is user-created', () => {
    const h = harness();
    // An active AUTO category whose nearest stored centroid is an active USER category: no merge,
    // because a user-created category must never be entangled in an applicable structural proposal.
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'User Bills', 'user_bills', {
      source: 'user',
      centroid: axis(0),
      members: 3,
      memberDim: 0,
    });

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
  });
});

describe('StructuralProposalService retire detection', () => {
  it('creates a retire proposal for an empty active auto category', () => {
    const h = harness();
    const empty = makeCategory(h, 'Empty', 'empty');

    const result = h.service.generate(h.accountId, MODEL);

    const retires = result.created.filter((c) => c.kind === 'retire');
    expect(retires).toHaveLength(1);
    expect(retires[0]!.categoryId).toBe(empty.id);
    expect(retires[0]!.suppressionKey).toBe('retire:empty');
    expect(h.proposals.listPending(h.accountId).some((p) => p.kind === 'retire')).toBe(true);
  });

  it('does not create a retire proposal for a category that still has assignments', () => {
    const h = harness();
    makeCategory(h, 'Busy', 'busy', { members: 2, memberDim: 0 });

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'retire')).toHaveLength(0);
    expect(result.retireCandidates).toBe(0);
  });

  it('does not recreate a retire proposal that is already pending', () => {
    const h = harness();
    makeCategory(h, 'Empty', 'empty');

    const first = h.service.generate(h.accountId, MODEL);
    expect(first.created.filter((c) => c.kind === 'retire')).toHaveLength(1);

    const second = h.service.generate(h.accountId, MODEL);
    expect(second.created.filter((c) => c.kind === 'retire')).toHaveLength(0);
    expect(second.skippedExisting).toBeGreaterThanOrEqual(1);
    // Still only one pending retire proposal.
    expect(h.proposals.listPending(h.accountId).filter((p) => p.kind === 'retire')).toHaveLength(1);
  });
});

describe('StructuralProposalService safety and dedup', () => {
  it('does not create any proposal for user-created categories', () => {
    const h = harness();
    // An empty user category (retire candidate shape) and two overlapping user categories (merge shape).
    makeCategory(h, 'User Empty', 'user_empty', { source: 'user' });
    makeCategory(h, 'User A', 'user_a', {
      source: 'user',
      centroid: axis(0),
      members: 3,
      memberDim: 0,
    });
    makeCategory(h, 'User B', 'user_b', {
      source: 'user',
      centroid: axis(0),
      members: 3,
      memberDim: 0,
    });

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
    expect(result.retireCandidates).toBe(0);
  });

  it('does not duplicate an equivalent pending structural proposal', () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Bills', 'bills', { centroid: axis(0), members: 5, memberDim: 0 });

    const first = h.service.generate(h.accountId, MODEL);
    expect(first.created.filter((c) => c.kind === 'merge')).toHaveLength(1);

    const second = h.service.generate(h.accountId, MODEL);
    expect(second.created).toHaveLength(0);
    expect(second.skippedExisting).toBeGreaterThanOrEqual(1);
    // Still only one pending merge proposal.
    expect(h.proposals.listPending(h.accountId).filter((p) => p.kind === 'merge')).toHaveLength(1);
  });

  it('does not recreate a dismissed proposal with the same suppression key', () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Bills', 'bills', { centroid: axis(0), members: 5, memberDim: 0 });

    const first = h.service.generate(h.accountId, MODEL);
    const mergeId = first.created.find((c) => c.kind === 'merge')!.id;
    h.proposals.markDismissed(mergeId);

    const second = h.service.generate(h.accountId, MODEL);
    expect(second.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(second.skippedExisting).toBeGreaterThanOrEqual(1);
  });

  it('writes nothing but proposal rows', () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Bills', 'bills', { centroid: axis(0), members: 5, memberDim: 0 });
    makeCategory(h, 'Empty', 'empty');

    const before = tableCounts(h.db);
    const result = h.service.generate(h.accountId, MODEL);
    const after = tableCounts(h.db);

    // At least one proposal was created; only category_proposals grew.
    expect(result.created.length).toBeGreaterThan(0);
    expect(after.proposals).toBe(before.proposals + result.created.length);
    expect(after.categories).toBe(before.categories);
    expect(after.emailCategories).toBe(before.emailCategories);
    expect(after.centroids).toBe(before.centroids);
    expect(after.audit).toBe(before.audit);
    // No category was retired or otherwise mutated: all three are still active.
    expect(h.categories.listActive(h.accountId)).toHaveLength(3);
  });
});
