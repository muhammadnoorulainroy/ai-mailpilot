/**
 * Phase 3.3 detection tests: StructuralProposalService turns read-only health metrics into safe
 * merge/retire/split review-queue proposals. It proposes a merge only for near-duplicate active auto
 * categories, a retire only for empty active auto categories, and a split only when a loose category's
 * subclusters can be named by the local model as distinct, non-overlapping purposes that survive the
 * deterministic validation gate. It never touches user-created categories, dedups on suppressionKey,
 * and writes only proposal rows.
 */
import { describe, it, expect } from 'vitest';
import type { Logger } from 'pino';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import type { LlmClient } from '../src/llm/client.js';
import type { LlmConfig } from '../src/config/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { CategoryProposalRepository } from '../src/repositories/category-proposal-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { CategoryHealthService } from '../src/services/category-health-service.js';
import {
  StructuralProposalService,
  MERGE_MIN_OVERLAP,
} from '../src/services/structural-proposal-service.js';

const MODEL = 'bge-m3';
const GEN_MODEL = 'qwen';
const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

/** A local-model naming answer that names cluster 0 "Invoices Received" and cluster 1 "Flight Bookings". */
const DEFAULT_NAMING = JSON.stringify({
  clusters: [
    {
      clusterIndex: 0,
      action: 'new_category',
      label: 'Invoices Received',
      description: 'Payment invoices.',
      suggestedKey: 'finance.invoices',
    },
    {
      clusterIndex: 1,
      action: 'new_category',
      label: 'Flight Bookings',
      description: 'Flight reservations.',
      suggestedKey: 'travel.flights',
    },
  ],
});

/**
 * A naming answer that leaves both clusters uncategorized. The labels are otherwise VALID (they would
 * pass the gate under `new_category`), so the only reason these children are rejected is the
 * `leave_uncategorized` action itself; this keeps the test honest about the action gate.
 */
const UNCATEGORIZED_NAMING = JSON.stringify({
  clusters: [
    {
      clusterIndex: 0,
      action: 'leave_uncategorized',
      label: 'Invoices Received',
      description: 'Payment invoices.',
      suggestedKey: 'finance.invoices',
    },
    {
      clusterIndex: 1,
      action: 'leave_uncategorized',
      label: 'Flight Bookings',
      description: 'Flight reservations.',
      suggestedKey: 'travel.flights',
    },
  ],
});

/** Names both subclusters as sub-variants of the SAME shipping purpose (valid labels, no distinct purpose). */
const SHIPPING_NAMING = JSON.stringify({
  clusters: [
    {
      clusterIndex: 0,
      action: 'new_category',
      label: 'Package Tracking',
      description: 'Package tracking updates.',
      suggestedKey: 'shipping.tracking',
    },
    {
      clusterIndex: 1,
      action: 'new_category',
      label: 'Delivery Dispatch',
      description: 'Delivery dispatch notices.',
      suggestedKey: 'shipping.dispatch',
    },
  ],
});

/** Names both subclusters as promotional fragments (rejected by the marketing/low-value gate). */
const MARKETING_NAMING = JSON.stringify({
  clusters: [
    {
      clusterIndex: 0,
      action: 'new_category',
      label: 'Sale',
      description: '',
      suggestedKey: 'sale',
    },
    {
      clusterIndex: 1,
      action: 'new_category',
      label: 'Deals',
      description: '',
      suggestedKey: 'deals',
    },
  ],
});

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

/** Builds the harness. `chat` is the local model's naming response (or a throw), only used for splits. */
function harness(chat: () => Promise<string> = async () => DEFAULT_NAMING, multiPrototype = false) {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const categories = new CategoryRepository(db);
  const proposals = new CategoryProposalRepository(db);
  const embeddings = new EmbeddingRepository(db);
  const emails = new EmailRepository(db);
  const health = new CategoryHealthService(categories, embeddings);
  const llm = {
    chat,
    async embed() {
      return [];
    },
    async embedBatch() {
      return [];
    },
    async health() {
      return { ok: true, models: [] };
    },
    chatStream() {
      return (async function* () {})();
    },
  } as unknown as LlmClient;
  const getConfig = () => ({ allowCloudDiscovery: false }) as unknown as LlmConfig;
  const service = new StructuralProposalService(
    categories,
    proposals,
    health,
    embeddings,
    emails,
    llm,
    getConfig,
    silentLogger,
    () => multiPrototype,
  );
  const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
  return { db, categories, proposals, embeddings, emails, health, service, accountId: acc.id };
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
 * subclusters.
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

/** A Big Bucket category: 8 invoice members on axis 0 and 8 travel members on axis 5 (a clean split). */
function makeSplittable(h: Harness) {
  return makeGroupedCategory(h, 'Big Bucket', 'big', [
    { dim: 0, subject: 'invoices', count: 8 },
    { dim: 5, subject: 'travel', count: 8 },
  ]);
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
  it('creates one merge proposal for two clearly overlapping same-purpose active auto categories', async () => {
    const h = harness();
    // Invoices and Receipts both map to the same known purpose group, so a high-overlap merge is safe.
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Receipts', 'receipts', { centroid: axis(0), members: 5, memberDim: 0 });

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

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

  it('does not create a merge proposal for well-separated categories', async () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Travel', 'travel', { centroid: axis(1), members: 3, memberDim: 1 });

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
  });

  it('does not create a merge for a moderate overlap below the threshold', async () => {
    const h = harness();
    // Two auto categories with a genuine overlap of 0.8, which is below MERGE_MIN_OVERLAP (0.9).
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Bills', 'bills', {
      centroid: tiltedFromAxis0(0.8, 0.6),
      members: 3,
      memberDim: 0,
    });

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
  });

  it('creates a merge for an overlap just above the threshold', async () => {
    const h = harness();
    // Overlap 0.95: above MERGE_MIN_OVERLAP (0.9) but well short of the identical-centroid case.
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Receipts', 'receipts', {
      centroid: tiltedFromAxis0(0.95, 0.3122498999199199),
      members: 5,
      memberDim: 0,
    });

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(1);
  });

  it('does not merge two distinct-purpose categories even at very high overlap', async () => {
    const h = harness();
    // Two transactional categories that collapse in embedding space (identical centroid axis) but map
    // to DIFFERENT known purposes: invoices vs shipping. A raw-overlap merge here would be wrong.
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 4, memberDim: 0 });
    makeCategory(h, 'Shipping Updates', 'shipping_updates', {
      centroid: axis(0),
      members: 4,
      memberDim: 0,
    });

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    // The pair was overlap-eligible (counted as a candidate) but rejected on purpose, not filtered
    // out earlier by the overlap threshold.
    expect(result.mergeCandidates).toBeGreaterThanOrEqual(1);
  });

  it('merges near-duplicate-labelled categories that match no known purpose', async () => {
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

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(1);
  });

  it('does not create a merge when the nearest category is user-created', async () => {
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

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
  });

  it('never merges an empty category that carries a stale overlapping centroid; only retires it', async () => {
    const h = harness();
    // Empty auto category with a saved centroid but zero assignments (a stale centroid), overlapping
    // a non-empty auto category. Without the guard the merge loop would also propose merging it.
    const empty = makeCategory(h, 'Empty', 'empty', { centroid: axis(0), members: 0 });
    makeCategory(h, 'Full', 'full', { centroid: axis(0), members: 3, memberDim: 0 });

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    // Retire is the only structural proposal for the empty category.
    const retires = result.created.filter((c) => c.kind === 'retire');
    expect(retires).toHaveLength(1);
    expect(retires[0]!.categoryId).toBe(empty.id);
    // No merge proposal is created, and none references the empty category on either side.
    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
    expect(
      result.created.some(
        (c) => c.kind === 'merge' && (c.categoryId === empty.id || c.sourceCategoryId === empty.id),
      ),
    ).toBe(false);
  });
});

describe('StructuralProposalService retire detection', () => {
  it('creates a retire proposal for an empty active auto category', async () => {
    const h = harness();
    const empty = makeCategory(h, 'Empty', 'empty');

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    const retires = result.created.filter((c) => c.kind === 'retire');
    expect(retires).toHaveLength(1);
    expect(retires[0]!.categoryId).toBe(empty.id);
    expect(retires[0]!.suppressionKey).toBe('retire:empty');
    expect(h.proposals.listPending(h.accountId).some((p) => p.kind === 'retire')).toBe(true);
  });

  it('does not create a retire proposal for a category that still has assignments', async () => {
    const h = harness();
    makeCategory(h, 'Busy', 'busy', { members: 2, memberDim: 0 });

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'retire')).toHaveLength(0);
    expect(result.retireCandidates).toBe(0);
  });

  it('does not recreate a retire proposal that is already pending', async () => {
    const h = harness();
    makeCategory(h, 'Empty', 'empty');

    const first = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    expect(first.created.filter((c) => c.kind === 'retire')).toHaveLength(1);

    const second = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    expect(second.created.filter((c) => c.kind === 'retire')).toHaveLength(0);
    expect(second.skippedExisting).toBeGreaterThanOrEqual(1);
    // Still only one pending retire proposal.
    expect(h.proposals.listPending(h.accountId).filter((p) => p.kind === 'retire')).toHaveLength(1);
  });
});

describe('StructuralProposalService safety and dedup', () => {
  it('does not create any proposal for user-created categories', async () => {
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

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created).toHaveLength(0);
    expect(result.mergeCandidates).toBe(0);
    expect(result.retireCandidates).toBe(0);
  });

  it('does not duplicate an equivalent pending structural proposal', async () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Receipts', 'receipts', { centroid: axis(0), members: 5, memberDim: 0 });

    const first = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    expect(first.created.filter((c) => c.kind === 'merge')).toHaveLength(1);

    const second = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    expect(second.created).toHaveLength(0);
    expect(second.skippedExisting).toBeGreaterThanOrEqual(1);
    // Still only one pending merge proposal.
    expect(h.proposals.listPending(h.accountId).filter((p) => p.kind === 'merge')).toHaveLength(1);
  });

  it('does not recreate a dismissed proposal with the same suppression key', async () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Receipts', 'receipts', { centroid: axis(0), members: 5, memberDim: 0 });

    const first = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    const mergeId = first.created.find((c) => c.kind === 'merge')!.id;
    h.proposals.markDismissed(mergeId);

    const second = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    expect(second.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    expect(second.skippedExisting).toBeGreaterThanOrEqual(1);
  });

  it('writes nothing but proposal rows', async () => {
    const h = harness();
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeCategory(h, 'Receipts', 'receipts', { centroid: axis(0), members: 5, memberDim: 0 });
    makeCategory(h, 'Empty', 'empty');

    const before = tableCounts(h.db);
    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
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
  it('proposes a split whose children carry the local-model names that survive the gate', async () => {
    const h = harness();
    // 16 auto members: 8 "invoices" on one axis and 8 "travel" on an orthogonal axis. The model names
    // the two subclusters as distinct purposes.
    const source = makeSplittable(h);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    const splits = result.created.filter((c) => c.kind === 'split');
    expect(splits).toHaveLength(1);
    expect(splits[0]!.categoryId).toBe(source.id);
    expect(splits[0]!.sourceCategoryId).toBeNull();
    expect(splits[0]!.suppressionKey).toBe('split:big');

    const pending = h.proposals.listPending(h.accountId).filter((p) => p.kind === 'split');
    expect(pending).toHaveLength(1);
    const children = h.proposals.listChildren(pending[0]!.id);
    expect(children).toHaveLength(2);
    // Labels come from the local model (two words each), not from a raw single keyphrase.
    expect(children.map((c) => c.label).sort()).toEqual(['Flight Bookings', 'Invoices Received']);
    expect(children.every((c) => c.proposedCount === 8)).toBe(true);
    const allChildMembers = children.flatMap((c) => c.memberIds);
    expect(new Set(allChildMembers).size).toBe(16);
  });

  it('writes only proposal and child rows, mutating no category or assignment', async () => {
    const h = harness();
    makeSplittable(h);

    const before = tableCounts(h.db);
    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    const after = tableCounts(h.db);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(1);
    expect(after.categories).toBe(before.categories);
    expect(after.emailCategories).toBe(before.emailCategories);
    expect(after.centroids).toBe(before.centroids);
    // The source category is untouched: still active with all 16 members.
    expect(h.categories.listActive(h.accountId)).toHaveLength(1);
  });

  it('abandons the split when the model marks the subclusters uncategorized', async () => {
    const h = harness(async () => UNCATEGORIZED_NAMING);
    makeSplittable(h);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    // It was analyzed (structurally multi-modal) but produced no nameable children.
    expect(result.splitCandidates).toBeGreaterThanOrEqual(1);
  });

  it('abandons the split when a child would duplicate an existing active category', async () => {
    const h = harness();
    // An existing "Invoices" category makes the "Invoices Received" child overlap it, so only the
    // travel child survives the gate; with fewer than two children the split is abandoned.
    makeCategory(h, 'Invoices', 'invoices', { centroid: axis(0), members: 3, memberDim: 0 });
    makeSplittable(h);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    expect(h.categories.findByLabel(h.accountId, 'Invoices')).not.toBeNull();
  });

  it('abandons the split but still proposes retire when the naming call fails', async () => {
    const h = harness(async () => {
      throw new Error('local model unavailable');
    });
    makeSplittable(h);
    const empty = makeCategory(h, 'Empty', 'empty');

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    // No split (naming failed) but the retire is unaffected: naming never aborts the whole run.
    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    const retires = result.created.filter((c) => c.kind === 'retire');
    expect(retires).toHaveLength(1);
    expect(retires[0]!.categoryId).toBe(empty.id);
  });

  it('does not split a coherent single-mode category', async () => {
    const h = harness();
    // 16 auto members all on one axis: tight (cohesion ~1.0), so it is excluded before naming.
    makeGroupedCategory(h, 'Tidy', 'tidy', [{ dim: 0, subject: 'invoices', count: 16 }]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    expect(result.splitCandidates).toBe(0);
  });

  it('does not split when the subclusters are too thin to be viable child categories', async () => {
    const h = harness();
    // 12 members across three orthogonal axes of 4 each: loose enough to analyze, but every
    // subcluster is below SPLIT_MIN_CHILD_SIZE, so no split is proposed and naming is never called.
    makeGroupedCategory(h, 'Thin Groups', 'thin', [
      { dim: 0, subject: 'invoices', count: 4 },
      { dim: 5, subject: 'travel', count: 4 },
      { dim: 9, subject: 'newsletters', count: 4 },
    ]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    expect(result.splitCandidates).toBeGreaterThanOrEqual(1);
  });

  it('does not consider a category below the minimum split size', async () => {
    const h = harness();
    // Two clean, separable halves of 5 each, but only 10 members total: below SPLIT_MIN_SIZE.
    makeGroupedCategory(h, 'Small', 'small', [
      { dim: 0, subject: 'invoices', count: 5 },
      { dim: 5, subject: 'travel', count: 5 },
    ]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    expect(result.splitCandidates).toBe(0);
  });

  it('leaves user-confirmed members off the proposed children', async () => {
    const h = harness();
    const source = makeSplittable(h);
    // A user-confirmed member on a third axis. Split children carry only auto members, so it must not
    // appear in any child's member list.
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

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    const pending = h.proposals.listPending(h.accountId).filter((p) => p.kind === 'split');
    expect(pending).toHaveLength(1);
    const allChildMembers = h.proposals.listChildren(pending[0]!.id).flatMap((c) => c.memberIds);
    expect(allChildMembers).not.toContain('user-msg');
  });

  it('does not re-propose a split whose suppression key is already pending or dismissed', async () => {
    const h = harness();
    makeSplittable(h);

    const first = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    const split = first.created.find((c) => c.kind === 'split')!;
    expect(split).toBeDefined();

    const second = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    expect(second.created.filter((c) => c.kind === 'split')).toHaveLength(0);

    h.proposals.markDismissed(split.id);
    const third = await h.service.generate(h.accountId, MODEL, GEN_MODEL);
    expect(third.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    expect(h.proposals.listPending(h.accountId).filter((p) => p.kind === 'split')).toHaveLength(0);
  });

  it('is deterministic: identical input and naming yield the same children across runs', async () => {
    const groups = [
      { dim: 0, subject: 'invoices', count: 8 },
      { dim: 5, subject: 'travel', count: 8 },
    ];
    const a = harness();
    makeGroupedCategory(a, 'Big Bucket', 'big', groups);
    const b = harness();
    makeGroupedCategory(b, 'Big Bucket', 'big', groups);

    const ra = await a.service.generate(a.accountId, MODEL, GEN_MODEL);
    const rb = await b.service.generate(b.accountId, MODEL, GEN_MODEL);

    const childrenA = a.proposals.listChildren(ra.created.find((c) => c.kind === 'split')!.id);
    const childrenB = b.proposals.listChildren(rb.created.find((c) => c.kind === 'split')!.id);
    expect(childrenA.map((c) => `${c.label}:${c.canonicalKey}:${c.proposedCount}`)).toEqual(
      childrenB.map((c) => `${c.label}:${c.canonicalKey}:${c.proposedCount}`),
    );
  });
});

describe('StructuralProposalService split parent-purpose gate', () => {
  it('rejects splitting a mature purpose category into same-purpose sub-variants', async () => {
    // Shipping -> Package Tracking / Delivery Dispatch: both children are valid labels that PASS the
    // validation gate, but both map to the parent's own shipping purpose, so the parent-purpose gate
    // is the sole reason the split is abandoned.
    const h = harness(async () => SHIPPING_NAMING);
    makeGroupedCategory(h, 'Shipping and Deliveries', 'shipping_and_deliveries', [
      { dim: 0, subject: 'package tracking', count: 8 },
      { dim: 5, subject: 'delivery dispatch', count: 8 },
    ]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
    // It was analyzed as structurally multi-modal, but rejected as same-purpose fragmentation.
    expect(result.splitCandidates).toBeGreaterThanOrEqual(1);
  });

  it('rejects splitting a marketing category into promotional fragments', async () => {
    // Marketing -> Sale / Deals: promotional labels are rejected by the marketing/low-value gate, so
    // fewer than two children survive and the split is abandoned.
    const h = harness(async () => MARKETING_NAMING);
    makeGroupedCategory(h, 'Marketing Promotions', 'marketing_promotions', [
      { dim: 0, subject: 'sale offer', count: 8 },
      { dim: 5, subject: 'deals discount', count: 8 },
    ]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'split')).toHaveLength(0);
  });

  it('allows splitting a generic/mixed category into genuinely distinct purposes', async () => {
    // "Mixed" has no purpose signature, so the parent-purpose gate does not fire; its two children map
    // to distinct purposes (invoices, travel) and the split is proposed.
    const h = harness();
    makeGroupedCategory(h, 'Mixed', 'mixed', [
      { dim: 0, subject: 'invoices', count: 8 },
      { dim: 5, subject: 'travel', count: 8 },
    ]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    const splits = result.created.filter((c) => c.kind === 'split');
    expect(splits).toHaveLength(1);
    const children = h.proposals.listChildren(splits[0]!.id);
    expect(children.map((c) => c.label).sort()).toEqual(['Flight Bookings', 'Invoices Received']);
  });

  it('allows splitting a mature purpose category when a child introduces a different purpose', async () => {
    // A "Banking Transactions" bucket that also hides travel mail: one child stays in the bank purpose,
    // the other (Flight Bookings) is a distinct purpose, so the split is allowed.
    const h = harness(async () =>
      JSON.stringify({
        clusters: [
          {
            clusterIndex: 0,
            action: 'new_category',
            label: 'Card Statements',
            description: 'Bank card statements.',
            suggestedKey: 'bank.statements',
          },
          {
            clusterIndex: 1,
            action: 'new_category',
            label: 'Flight Bookings',
            description: 'Flight reservations.',
            suggestedKey: 'travel.flights',
          },
        ],
      }),
    );
    makeGroupedCategory(h, 'Banking Transactions', 'banking_transactions', [
      { dim: 0, subject: 'card statement', count: 8 },
      { dim: 5, subject: 'flight booking', count: 8 },
    ]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    const splits = result.created.filter((c) => c.kind === 'split');
    expect(splits).toHaveLength(1);
    const children = h.proposals.listChildren(splits[0]!.id);
    expect(children.map((c) => c.label).sort()).toEqual(['Card Statements', 'Flight Bookings']);
  });
});

describe('StructuralProposalService multi-prototype conservatism (Phase 4)', () => {
  it('does not merge two categories that only share a sub-prototype (merge stays on aggregate)', async () => {
    // Flag ON. Two auto categories whose AGGREGATE centroids are orthogonal (overlap 0) but which each
    // carry a sub-prototype on the same axis 5. Merge must ignore the sub-prototypes: a naive
    // max-pairwise-over-prototypes overlap would report 1.0 and wrongly merge distinct purposes.
    const h = harness(async () => DEFAULT_NAMING, true);
    const alpha = makeCategory(h, 'Alpha', 'alpha', { centroid: axis(0), members: 3, memberDim: 0 });
    const beta = makeCategory(h, 'Beta', 'beta', { centroid: axis(1), members: 3, memberDim: 1 });
    h.categories.savePrototypeSet(alpha.id, MODEL, axis(0), 3, [{ vector: axis(5), emailCount: 3 }]);
    h.categories.savePrototypeSet(beta.id, MODEL, axis(1), 3, [{ vector: axis(5), emailCount: 3 }]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    expect(result.created.filter((c) => c.kind === 'merge')).toHaveLength(0);
    // The aggregate overlap gate rejects the pair before it is even counted as a merge candidate.
    expect(result.mergeCandidates).toBe(0);
    // Health overlap is measured on the aggregate centroids only, so it stays far below the threshold.
    const metrics = h.health.metricsForAccount(h.accountId, MODEL);
    for (const m of metrics) {
      expect(m.overlap ?? 0).toBeLessThan(MERGE_MIN_OVERLAP);
    }
  });

  it('still merges a genuine same-purpose near-duplicate with the flag on and sub-prototypes present', async () => {
    // Flag ON with divergent sub-prototypes on both sides. The aggregate centroids are identical, so the
    // merge decision is unchanged from flag-off: sub-prototypes neither block nor are needed for it.
    const h = harness(async () => DEFAULT_NAMING, true);
    const invoices = makeCategory(h, 'Invoices', 'invoices', {
      centroid: axis(0),
      members: 3,
      memberDim: 0,
    });
    const receipts = makeCategory(h, 'Receipts', 'receipts', {
      centroid: axis(0),
      members: 5,
      memberDim: 0,
    });
    h.categories.savePrototypeSet(invoices.id, MODEL, axis(0), 3, [{ vector: axis(3), emailCount: 3 }]);
    h.categories.savePrototypeSet(receipts.id, MODEL, axis(0), 5, [{ vector: axis(7), emailCount: 5 }]);

    const result = await h.service.generate(h.accountId, MODEL, GEN_MODEL);

    const merges = result.created.filter((c) => c.kind === 'merge');
    expect(merges).toHaveLength(1);
    // The larger category (Receipts) survives as the target; sub-prototypes did not alter the outcome.
    expect(merges[0]!.categoryId).toBe(receipts.id);
    expect(merges[0]!.sourceCategoryId).toBe(invoices.id);
  });

  it('rejects a split child that overlaps a sub-prototype of another active category (flag on only)', async () => {
    // Big Bucket splits cleanly into an invoices child (axis 0) and a travel child (axis 5). A separate
    // active category "Newsletters" has an aggregate on axis 9 but a sub-prototype on axis 5. With the
    // flag ON, the travel child overlaps that sub-prototype and is dropped, leaving a single survivor and
    // no split. With the flag OFF, only the axis-9 aggregate is visible, so both children survive.
    function build(multiPrototype: boolean): Harness {
      const h = harness(async () => DEFAULT_NAMING, multiPrototype);
      makeSplittable(h);
      const news = makeCategory(h, 'Newsletters', 'news', {
        centroid: axis(9),
        members: 4,
        memberDim: 9,
      });
      h.categories.savePrototypeSet(news.id, MODEL, axis(9), 4, [{ vector: axis(5), emailCount: 4 }]);
      return h;
    }

    const offH = build(false);
    const offResult = await offH.service.generate(offH.accountId, MODEL, GEN_MODEL);
    expect(offResult.created.filter((c) => c.kind === 'split')).toHaveLength(1);

    const onH = build(true);
    const onResult = await onH.service.generate(onH.accountId, MODEL, GEN_MODEL);
    expect(onResult.created.filter((c) => c.kind === 'split')).toHaveLength(0);
  });
});
