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
  const service = new StructuralProposalService(
    categories,
    proposals,
    health,
    embeddings,
    emails,
    silentLogger,
  );
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

/**
 * Create an active auto category whose auto members fall into the given (dim, subject, count) groups,
 * so a test can shape a category that is coherent, multi-modal, or too thin to split. Each group's
 * members share one embedding axis and one subject, so the groups form distinct, well-separated
 * subclusters with deterministic class-TF-IDF keyphrases.
 */
function makeGroupedCategory(
  h: Harness,
  label: string,
  key: string,
  groups: Array<{ dim: number; subject: string; count: number }>,
) {
  const cat = h.categories.create({
    accountId: h.accountId,
    label,
    source: 'auto',
    status: 'active',
    canonicalKey: key,
  });
  let idx = 0;
  for (const g of groups) {
    for (let i = 0; i < g.count; i++) {
      const messageId = `${key}-m${idx++}`;
      h.emails.upsertBatch([
        {
          messageId,
          accountId: h.accountId,
          folder: 'INBOX',
          subject: g.subject,
          fromAddr: 'a@b.com',
        },
      ]);
      h.embeddings.saveEmbedding(
        { messageId, accountId: h.accountId, modelId: MODEL },
        axis(g.dim),
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
  it('creates one merge proposal for two clearly overlapping same-purpose active auto categories', () => {
    const h = harness();
    // Invoices and Receipts both map to the same known purpose group, so a high-overlap merge is safe.
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Receipts', 'receipts', { centroid: axis(0), members: 5, memberDim: 0 });

    const result = h.service.generate(h.accountId, MODEL);

    const merges = result.created.filter((c) => c.kind === 'merge');
    expect(merges).toHaveLength(1);
    const receipts = h.categories.findByLabel(h.accountId, 'Receipts')!;
    const invoices = h.categories.findByLabel(h.accountId, 'Invoices')!;
    // The larger category (Receipts, 5) survives as the target; the smaller (Invoices, 3) is the source.
    expect(merges[0]!.categoryId).toBe(receipts.id);
    expect(merges[0]!.sourceCategoryId).toBe(invoices.id);
    expect(merges[0]!.suppressionKey).toBe('merge:invoices|receipts');
    // It is stored as a real pending merge proposal via createStructural.
    const pending = h.proposals.listPending(h.accountId).filter((p) => p.kind === 'merge');
    expect(pending).toHaveLength(1);
    expect(pending[0]!.categoryId).toBe(receipts.id);
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
    makeCategory(h, 'Receipts', 'receipts', {
      centroid: tiltedFromAxis0(0.95, 0.3122498999199199),
      members: 5,
      memberDim: 0,
    });

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(1);
  });

  it('does not merge two distinct-purpose categories even at very high overlap', () => {
    const h = harness();
    // Two transactional categories that collapse in embedding space (identical centroid axis) but map
    // to DIFFERENT known purposes: invoices vs shipping. A raw-overlap merge here would be wrong.
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 4, memberDim: 0 });
    makeCategory(h, 'Shipping Updates', 'shipping_updates', {
      centroid: axis(0),
      members: 4,
      memberDim: 0,
    });

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    // The pair was overlap-eligible (counted as a candidate) but rejected on purpose, not filtered
    // out earlier by the overlap threshold.
    expect(result.mergeCandidates).toBeGreaterThanOrEqual(1);
  });

  it('merges near-duplicate-labelled categories that match no known purpose', () => {
    const h = harness();
    // Neither label maps to a known purpose group, but the labels are near-duplicate, so a merge is
    // still a safe suggestion.
    makeCategory(h, 'Client Portal', 'client_portal', {
      centroid: axis(0),
      members: 3,
      memberDim: 0,
    });
    makeCategory(h, 'Client Portal Access', 'client_portal_access', {
      centroid: axis(0),
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

  it('never merges an empty category that carries a stale overlapping centroid; only retires it', () => {
    const h = harness();
    // Empty auto category with a saved centroid but zero assignments (a stale centroid), overlapping
    // a non-empty auto category. Without the guard the merge loop would also propose merging it.
    const empty = makeCategory(h, 'Empty', 'empty', { centroid: axis(0), members: 0 });
    makeCategory(h, 'Full', 'full', { centroid: axis(0), members: 3, memberDim: 0 });

    const result = h.service.generate(h.accountId, MODEL);

    // Retire is the only structural proposal for the empty category.
    const retires = result.created.filter((c) => c.kind === 'retire');
    expect(retires).toHaveLength(1);
    expect(retires[0]!.categoryId).toBe(empty.id);
    // No merge proposal is created, and none references the empty category on either side.
    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
    expect(
      result.created.some(
        (c) =>
          c.kind === 'merge' && (c.categoryId === empty.id || c.sourceCategoryId === empty.id),
      ),
    ).toBe(false);
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
    makeCategory(h, 'Receipts', 'receipts', { centroid: axis(0), members: 5, memberDim: 0 });

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
    makeCategory(h, 'Receipts', 'receipts', { centroid: axis(0), members: 5, memberDim: 0 });

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
    makeCategory(h, 'Receipts', 'receipts', { centroid: axis(0), members: 5, memberDim: 0 });
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

describe('StructuralProposalService split detection', () => {
  it('proposes a split for a loose category with two separable, well-labelled subclusters', () => {
    const h = harness();
    // 16 auto members: 8 "invoices" on one axis and 8 "travel" on an orthogonal axis. The category is
    // loose overall (cohesion ~0.71) but each half is tight and clearly separated.
    const source = makeGroupedCategory(h, 'Big Bucket', 'big', [
      { dim: 0, subject: 'invoices', count: 8 },
      { dim: 5, subject: 'travel', count: 8 },
    ]);

    const result = h.service.generate(h.accountId, MODEL);

    const splits = result.created.filter((c) => c.kind === 'split');
    expect(splits).toHaveLength(1);
    expect(splits[0]!.categoryId).toBe(source.id);
    expect(splits[0]!.sourceCategoryId).toBeNull();
    expect(splits[0]!.suppressionKey).toBe('split:big');

    const pending = h.proposals.listPending(h.accountId).filter((p) => p.kind === 'split');
    expect(pending).toHaveLength(1);
    const children = h.proposals.listChildren(pending[0]!.id);
    expect(children).toHaveLength(2);
    // Labels come from deterministic class-TF-IDF keyphrases of each subcluster's subjects.
    expect(children.map((c) => c.label).sort()).toEqual(['Invoices', 'Travel']);
    // Each child claims its 8 auto members and none is shared.
    expect(children.every((c) => c.proposedCount === 8)).toBe(true);
    const allChildMembers = children.flatMap((c) => c.memberIds);
    expect(new Set(allChildMembers).size).toBe(16);
  });

  it('writes only proposal and child rows, mutating no category or assignment', () => {
    const h = harness();
    makeGroupedCategory(h, 'Big Bucket', 'big', [
      { dim: 0, subject: 'invoices', count: 8 },
      { dim: 5, subject: 'travel', count: 8 },
    ]);

    const before = tableCounts(h.db);
    const result = h.service.generate(h.accountId, MODEL);
    const after = tableCounts(h.db);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(1);
    expect(after.categories).toBe(before.categories);
    expect(after.emailCategories).toBe(before.emailCategories);
    expect(after.centroids).toBe(before.centroids);
    // The source category is untouched: still active with all 16 members.
    expect(h.categories.listActive(h.accountId)).toHaveLength(1);
  });

  it('does not split a coherent single-mode category', () => {
    const h = harness();
    // 16 auto members all on one axis: tight (cohesion ~1.0), so it is excluded before analysis.
    makeGroupedCategory(h, 'Tidy', 'tidy', [{ dim: 0, subject: 'invoices', count: 16 }]);

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    expect(result.splitCandidates).toBe(0);
  });

  it('does not split when the subclusters are too thin to be viable child categories', () => {
    const h = harness();
    // 12 members across three orthogonal axes of 4 each: loose enough to analyze, but every
    // subcluster is below SPLIT_MIN_CHILD_SIZE, so no split is proposed.
    makeGroupedCategory(h, 'Thin Groups', 'thin', [
      { dim: 0, subject: 'invoices', count: 4 },
      { dim: 5, subject: 'travel', count: 4 },
      { dim: 9, subject: 'newsletters', count: 4 },
    ]);

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    // It cleared the cheap pre-filter (loose and large enough) but failed the child-size gate.
    expect(result.splitCandidates).toBeGreaterThanOrEqual(1);
  });

  it('does not consider a category below the minimum split size', () => {
    const h = harness();
    // Two clean, separable halves of 5 each, but only 10 members total: below SPLIT_MIN_SIZE.
    makeGroupedCategory(h, 'Small', 'small', [
      { dim: 0, subject: 'invoices', count: 5 },
      { dim: 5, subject: 'travel', count: 5 },
    ]);

    const result = h.service.generate(h.accountId, MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    expect(result.splitCandidates).toBe(0);
  });

  it('leaves user-confirmed members off the proposed children', () => {
    const h = harness();
    const source = makeGroupedCategory(h, 'Big Bucket', 'big', [
      { dim: 0, subject: 'invoices', count: 8 },
      { dim: 5, subject: 'travel', count: 8 },
    ]);
    // Add a user-confirmed member on a third axis. Split moves only auto rows, so it must not appear
    // in any child's member list.
    h.emails.upsertBatch([
      {
        messageId: 'user-msg',
        accountId: h.accountId,
        folder: 'INBOX',
        subject: 'personal note',
        fromAddr: 'me@x.com',
      },
    ]);
    h.embeddings.saveEmbedding(
      { messageId: 'user-msg', accountId: h.accountId, modelId: MODEL },
      axis(9),
    );
    h.categories.addAutoAssignments(h.accountId, [
      {
        messageId: 'user-msg',
        accountId: h.accountId,
        categoryId: source.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: Date.now(),
        method: 'llm',
      },
    ]);

    const result = h.service.generate(h.accountId, MODEL);

    const pending = h.proposals.listPending(h.accountId).filter((p) => p.kind === 'split');
    expect(pending).toHaveLength(1);
    const allChildMembers = h.proposals.listChildren(pending[0]!.id).flatMap((c) => c.memberIds);
    expect(allChildMembers).not.toContain('user-msg');
  });

  it('does not re-propose a split whose suppression key is already pending or dismissed', () => {
    const h = harness();
    makeGroupedCategory(h, 'Big Bucket', 'big', [
      { dim: 0, subject: 'invoices', count: 8 },
      { dim: 5, subject: 'travel', count: 8 },
    ]);

    const first = h.service.generate(h.accountId, MODEL);
    const split = first.created.find((c) => c.kind === 'split')!;
    expect(split).toBeDefined();

    // A second run with the split still pending proposes nothing new for it.
    const second = h.service.generate(h.accountId, MODEL);
    expect(second.created.filter((c) => c.kind === 'split')).toHaveLength(0);

    // Still true after the split is dismissed: the suppression key blocks a re-propose.
    h.proposals.markDismissed(split.id);
    const third = h.service.generate(h.accountId, MODEL);
    expect(third.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    expect(h.proposals.listPending(h.accountId).filter((p) => p.kind === 'split')).toHaveLength(0);
  });

  it('is deterministic: identical input yields the same child labels and keys across runs', () => {
    const groups = [
      { dim: 0, subject: 'invoices', count: 8 },
      { dim: 5, subject: 'travel', count: 8 },
    ];
    const a = harness();
    makeGroupedCategory(a, 'Big Bucket', 'big', groups);
    const b = harness();
    makeGroupedCategory(b, 'Big Bucket', 'big', groups);

    const ra = a.service.generate(a.accountId, MODEL);
    const rb = b.service.generate(b.accountId, MODEL);

    const childrenA = a.proposals.listChildren(ra.created.find((c) => c.kind === 'split')!.id);
    const childrenB = b.proposals.listChildren(rb.created.find((c) => c.kind === 'split')!.id);
    expect(childrenA.map((c) => `${c.label}:${c.canonicalKey}:${c.proposedCount}`)).toEqual(
      childrenB.map((c) => `${c.label}:${c.canonicalKey}:${c.proposedCount}`),
    );
  });
});
