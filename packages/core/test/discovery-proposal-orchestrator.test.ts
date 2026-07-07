/**
 * Phase 2c tests: the proposal orchestrator. Generate persists a review queue of suggested
 * categories (never active, never assigned, never a centroid) and dedups on re-run. Apply promotes,
 * seeds the centroid, and assigns only still-uncategorized members with proposal provenance, never
 * touching user-assigned or already auto-assigned mail. Dismiss is soft and suppresses re-proposal.
 */
import { describe, it, expect } from 'vitest';
import type { Logger } from 'pino';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { CategoryProposalRepository } from '../src/repositories/category-proposal-repository.js';
import { CategoryAliasRepository } from '../src/repositories/category-alias-repository.js';
import { DiscoveryAuditRepository } from '../src/repositories/discovery-audit-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import type { LlmClient } from '../src/llm/client.js';
import type { LlmConfig } from '../src/config/schema.js';
import { ResidualDiscoveryService } from '../src/services/residual-discovery-service.js';
import { DiscoveryProposalService } from '../src/services/discovery-proposal-service.js';
import {
  DiscoveryProposalOrchestrator,
  ProposalApplyError,
} from '../src/services/discovery-proposal-orchestrator.js';
import { CategoryCentroidRebuildService } from '../src/services/category-centroid-rebuild-service.js';

const MODEL = 'bge-m3';
const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

const NAMING = JSON.stringify({
  clusters: [
    {
      clusterIndex: 0,
      action: 'new_category',
      label: 'Receipts & Invoices',
      description: 'Payment confirmations and invoices.',
      suggestedKey: 'finance.invoices',
    },
    {
      clusterIndex: 1,
      action: 'new_category',
      label: 'Flight Bookings',
      description: 'Flight reservations and itineraries.',
      suggestedKey: 'travel.flights',
    },
  ],
});

function harness(answer: string = NAMING, centroidRebuildOverride?: CategoryCentroidRebuildService) {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const categories = new CategoryRepository(db);
  const proposals = new CategoryProposalRepository(db);
  const audit = new DiscoveryAuditRepository(db);
  const emails = new EmailRepository(db);
  const embeddings = new EmbeddingRepository(db);
  const acc = accounts.create({ address: 'w@x.com', kind: 'work' });

  const blobs = [
    { dim: 0, prefix: 'inv', subject: 'Invoice paid', from: 'billing@acme.com' },
    { dim: 1, prefix: 'fly', subject: 'Flight itinerary', from: 'trips@fly.com' },
  ];
  for (const b of blobs) {
    const rows = Array.from({ length: 6 }, (_, i) => ({
      messageId: `${b.prefix}-${i}`,
      accountId: acc.id,
      folder: 'INBOX',
      subject: `${b.subject} ${i}`,
      fromAddr: b.from,
    }));
    emails.upsertBatch(rows);
    for (const r of rows) {
      embeddings.saveEmbedding(
        { messageId: r.messageId, accountId: acc.id, modelId: MODEL },
        axis(b.dim),
      );
    }
  }

  let currentAnswer = answer;
  const llm = {
    async chat() {
      return currentAnswer;
    },
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
  const residual = new ResidualDiscoveryService(embeddings, categories);
  const proposalService = new DiscoveryProposalService(
    residual,
    emails,
    categories,
    llm,
    getConfig,
    silentLogger,
  );
  const centroidRebuild =
    centroidRebuildOverride ??
    new CategoryCentroidRebuildService(categories, embeddings, silentLogger);
  const aliases = new CategoryAliasRepository(db);
  const orchestrator = new DiscoveryProposalOrchestrator(
    db,
    proposals,
    categories,
    emails,
    proposalService,
    accounts,
    audit,
    centroidRebuild,
    aliases,
    getConfig,
    silentLogger,
  );

  return {
    db,
    accounts,
    categories,
    proposals,
    audit,
    emails,
    embeddings,
    centroidRebuild,
    aliases,
    orchestrator,
    accountId: acc.id,
    setAnswer: (a: string) => {
      currentAnswer = a;
    },
  };
}

function assignmentsFor(db: Database, accountId: string, messageId: string) {
  return db
    .prepare(
      'SELECT category_id, assigned_by, method FROM email_categories WHERE account_id = ? AND message_id = ?',
    )
    .all(accountId, messageId) as Array<{
    category_id: string;
    assigned_by: string;
    method: string | null;
  }>;
}

/** The single assignment row for an email, including the confidence and assigned_at that a move must preserve. */
function fullAssignmentFor(db: Database, accountId: string, messageId: string) {
  return db
    .prepare(
      'SELECT category_id, assigned_by, method, confidence, assigned_at FROM email_categories WHERE account_id = ? AND message_id = ?',
    )
    .get(accountId, messageId) as {
    category_id: string;
    assigned_by: string;
    method: string | null;
    confidence: number;
    assigned_at: number;
  };
}

describe('DiscoveryProposalOrchestrator.generate', () => {
  it('persists suggested categories with no active category, no assignment, and no centroid', async () => {
    const h = harness();
    const result = await h.orchestrator.generate(h.accountId, MODEL, 'qwen');

    expect(result.created).toHaveLength(2);
    expect(h.categories.listSuggested(h.accountId)).toHaveLength(2);
    expect(h.categories.listActive(h.accountId)).toHaveLength(0);
    // No mail assigned and no centroid until a proposal is applied.
    expect(h.categories.getAssignedMessageIds(h.accountId).size).toBe(0);
    expect(h.categories.getCentroidEntries(h.accountId, MODEL)).toHaveLength(0);
    // The pending review queue shows both, strongest first.
    expect(h.orchestrator.listPending(h.accountId)).toHaveLength(2);

    const audits = h.audit.listForAccount(h.accountId);
    expect(audits[0]).toMatchObject({
      flow: 'discovery_proposal',
      status: 'ok',
      provider: 'local',
    });
  });

  it('dedups on re-run so the same purpose is not proposed twice', async () => {
    const h = harness();
    await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    const second = await h.orchestrator.generate(h.accountId, MODEL, 'qwen');

    // The gate already rejects a candidate that duplicates a still-suggested proposal, so no new
    // rows are written and the queue is unchanged.
    expect(second.created).toHaveLength(0);
    expect(h.categories.listSuggested(h.accountId)).toHaveLength(2);
    expect(h.proposals.listPending(h.accountId)).toHaveLength(2);
  });

  it('does not re-propose a dismissed purpose that the model relabeled (suggested-key suppression)', async () => {
    const h = harness();
    await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    const inv = h.orchestrator
      .listPending(h.accountId)
      .find((p) => p.label === 'Receipts & Invoices')!;
    h.orchestrator.dismiss(h.accountId, inv.id);

    // Re-run: the model gives the same cluster a different label but the same suggested key.
    h.setAnswer(
      JSON.stringify({
        clusters: [
          {
            clusterIndex: 0,
            action: 'new_category',
            label: 'Billing Statements',
            description: 'x',
            suggestedKey: 'finance.invoices',
          },
          {
            clusterIndex: 1,
            action: 'new_category',
            label: 'Flight Bookings',
            description: 'y',
            suggestedKey: 'travel.flights',
          },
        ],
      }),
    );
    const rerun = await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    expect(rerun.created.map((c) => c.label)).not.toContain('Billing Statements');
    expect(rerun.skippedDuplicates).toBeGreaterThanOrEqual(1);
  });

  it('skips a candidate whose label exists under a different canonical key without aborting the run', async () => {
    const h = harness();
    // A pre-existing retired category with the same label but a different canonical key: invisible to
    // the gate (retired), so the candidate reaches persistence, where the label guard must skip it
    // rather than let the UNIQUE(account_id, label) insert abort the whole run.
    h.categories.create({
      accountId: h.accountId,
      label: 'Flight Bookings',
      source: 'user',
      status: 'retired',
      canonicalKey: 'flight_bookings_legacy',
    });
    const result = await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    expect(result.created.map((c) => c.label)).toContain('Receipts & Invoices');
    expect(result.created.map((c) => c.label)).not.toContain('Flight Bookings');
    expect(result.skippedDuplicates).toBeGreaterThanOrEqual(1);
  });
});

describe('DiscoveryProposalOrchestrator.apply', () => {
  it('promotes to active, seeds the centroid, and assigns only still-uncategorized members', async () => {
    const h = harness();
    await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    const pending = h.orchestrator.listPending(h.accountId);
    const target = pending[0]!;
    const stored = h.proposals.findById(target.id)!;
    const members = stored.memberIds;
    expect(members.length).toBeGreaterThanOrEqual(3);

    // Pre-assign one member by the user and one by a prior auto pass to a different category.
    const other = h.categories.create({ accountId: h.accountId, label: 'Other', source: 'user' });
    const now = Date.now();
    h.categories.replaceEmailAssignments(members[0]!, h.accountId, [
      {
        messageId: members[0]!,
        accountId: h.accountId,
        categoryId: other.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: now,
        method: null,
      },
    ]);
    h.categories.addAutoAssignments(h.accountId, [
      {
        messageId: members[1]!,
        accountId: h.accountId,
        categoryId: other.id,
        confidence: 0.4,
        assignedBy: 'auto',
        assignedAt: now,
        method: 'embed',
      },
    ]);

    const result = h.orchestrator.apply(h.accountId, target.id);
    expect(result.kind).toBe('new_category');

    // The category is now active with a centroid; the proposal is applied.
    const active = h.categories.listActive(h.accountId).find((c) => c.id === target.categoryId);
    expect(active).toBeDefined();
    expect(
      h.categories
        .getCentroidEntries(h.accountId, MODEL)
        .some((c) => c.categoryId === target.categoryId),
    ).toBe(true);
    expect(h.proposals.findById(target.id)!.status).toBe('applied');

    // The user-assigned and already auto-assigned members keep their original labels, untouched.
    expect(assignmentsFor(h.db, h.accountId, members[0]!)).toEqual([
      { category_id: other.id, assigned_by: 'user', method: null },
    ]);
    expect(assignmentsFor(h.db, h.accountId, members[1]!)).toEqual([
      { category_id: other.id, assigned_by: 'auto', method: 'embed' },
    ]);
    // Every remaining member is assigned to the promoted category with proposal provenance.
    expect(result.assigned).toBe(members.length - 2);
    for (const id of members.slice(2)) {
      expect(assignmentsFor(h.db, h.accountId, id)).toEqual([
        { category_id: target.categoryId, assigned_by: 'auto', method: 'proposal' },
      ]);
    }
  });

  it('refuses to apply a proposal that is not pending', async () => {
    const h = harness();
    await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    const target = h.orchestrator.listPending(h.accountId)[0]!;
    h.orchestrator.apply(h.accountId, target.id);
    expect(() => h.orchestrator.apply(h.accountId, target.id)).toThrow(/not pending/);
  });

  it('skips a member deleted between generate and apply instead of failing the whole apply', async () => {
    const h = harness();
    await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    const target = h.orchestrator.listPending(h.accountId)[0]!;
    const members = h.proposals.findById(target.id)!.memberIds;
    // Delete one member's email after the proposal was generated; its email_categories cascade away.
    h.db
      .prepare('DELETE FROM emails WHERE account_id = ? AND message_id = ?')
      .run(h.accountId, members[0]!);

    const result = h.orchestrator.apply(h.accountId, target.id);

    expect(h.proposals.findById(target.id)!.status).toBe('applied');
    expect(h.categories.listActive(h.accountId).some((c) => c.id === target.categoryId)).toBe(true);
    expect(result.assigned).toBe(members.length - 1);
    expect(assignmentsFor(h.db, h.accountId, members[0]!)).toHaveLength(0);
  });
});

describe('DiscoveryProposalOrchestrator.dismiss', () => {
  it('retires the suggested category, keeps the record, and suppresses re-proposal', async () => {
    const h = harness();
    await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    const target = h.orchestrator.listPending(h.accountId)[0]!;

    h.orchestrator.dismiss(h.accountId, target.id);

    // The category is retired, not active, not suggested; the proposal row is kept as dismissed.
    expect(h.categories.listActive(h.accountId).some((c) => c.id === target.categoryId)).toBe(
      false,
    );
    expect(h.categories.listSuggested(h.accountId).some((c) => c.id === target.categoryId)).toBe(
      false,
    );
    expect(h.categories.listRetired(h.accountId).some((c) => c.id === target.categoryId)).toBe(
      true,
    );
    expect(h.proposals.findById(target.id)!.status).toBe('dismissed');
    expect(h.orchestrator.listPending(h.accountId)).toHaveLength(1);

    // A re-run does not re-propose the dismissed purpose. The gate no longer sees it (it is retired,
    // not suggested), so the candidate reaches persistence, where findByCanonicalKey suppresses it.
    const rerun = await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    expect(rerun.created.map((c) => c.label)).not.toContain(target.label);
    expect(rerun.skippedDuplicates).toBeGreaterThanOrEqual(1);
  });
});

type Harness = ReturnType<typeof harness>;

function activeCategory(h: Harness, label: string, key: string) {
  return h.categories.create({
    accountId: h.accountId,
    label,
    source: 'auto',
    status: 'active',
    canonicalKey: key,
  });
}

function seedEmail(h: Harness, messageId: string) {
  h.emails.upsertBatch([
    {
      messageId,
      accountId: h.accountId,
      folder: 'INBOX',
      subject: `s ${messageId}`,
      fromAddr: 'a@b.com',
    },
  ]);
}

function seedEmailWithEmbedding(h: Harness, messageId: string, dim: number) {
  seedEmail(h, messageId);
  h.embeddings.saveEmbedding({ messageId, accountId: h.accountId, modelId: MODEL }, axis(dim));
}

function assignUser(h: Harness, messageId: string, categoryId: string) {
  h.categories.replaceEmailAssignments(messageId, h.accountId, [
    {
      messageId,
      accountId: h.accountId,
      categoryId,
      confidence: 1,
      assignedBy: 'user',
      assignedAt: Date.now(),
      method: null,
    },
  ]);
}

function retireProposal(h: Harness, categoryId: string, key: string) {
  return h.proposals.createStructural({
    accountId: h.accountId,
    kind: 'retire',
    categoryId,
    sourceCategoryId: null,
    runId: 'run-retire',
    label: 'Retire',
    description: '',
    canonicalKey: key,
    suppressionKey: `retire:${key}`,
    embeddingModelId: MODEL,
    confidence: 0.9,
    evidence: [],
  });
}

function seedAutoOnSource(
  h: Harness,
  messageId: string,
  categoryId: string,
  method: 'embed' | 'gate' | 'llm' | 'proposal' = 'embed',
  confidence = 0.5,
) {
  seedEmailWithEmbedding(h, messageId, 2);
  h.categories.addAutoAssignments(h.accountId, [
    {
      messageId,
      accountId: h.accountId,
      categoryId,
      confidence,
      assignedBy: 'auto',
      assignedAt: Date.now(),
      method,
    },
  ]);
}

function splitProposal(h: Harness, sourceId: string, key: string) {
  return h.proposals.createStructural({
    accountId: h.accountId,
    kind: 'split',
    categoryId: sourceId,
    sourceCategoryId: null,
    runId: 'run-split',
    label: 'Split',
    description: '',
    canonicalKey: key,
    suppressionKey: `split:${key}`,
    embeddingModelId: MODEL,
    confidence: 0.9,
    evidence: [],
  });
}

function addChild(
  h: Harness,
  proposalId: string,
  label: string,
  key: string,
  memberIds: string[],
  centroid: Float32Array = axis(0),
) {
  return h.proposals.createChild({
    proposalId,
    label,
    description: '',
    canonicalKey: key,
    embeddingModelId: MODEL,
    centroid,
    memberIds,
    proposedCount: memberIds.length,
    cohesion: 0,
    separation: 0,
    confidence: 0,
  });
}

function mergeProposal(
  h: Harness,
  sourceId: string,
  targetId: string,
  sourceKey: string,
  targetKey: string,
) {
  return h.proposals.createStructural({
    accountId: h.accountId,
    kind: 'merge',
    categoryId: targetId,
    sourceCategoryId: sourceId,
    runId: 'run-merge',
    label: 'Merge',
    description: '',
    canonicalKey: targetKey,
    suppressionKey: `merge:${[sourceKey, targetKey].sort().join('|')}`,
    embeddingModelId: MODEL,
    confidence: 0.9,
    evidence: [],
  });
}

describe('DiscoveryProposalOrchestrator.apply (structural kinds)', () => {
  it('retire applies for an empty active category and marks it applied', () => {
    const h = harness();
    const cat = activeCategory(h, 'Noise', 'noise');
    const p = retireProposal(h, cat.id, 'noise');

    const result = h.orchestrator.apply(h.accountId, p.id);

    expect(result.kind).toBe('retire');
    expect(result.assigned).toBe(0);
    expect(h.categories.findById(cat.id)!.status).toBe('retired');
    expect(h.proposals.findById(p.id)!.status).toBe('applied');
  });

  it('retire blocks when the target gained auto mail after the proposal was generated', () => {
    const h = harness();
    const cat = activeCategory(h, 'Was Empty', 'was_empty');
    // The retire proposal is generated while the category is empty.
    const p = retireProposal(h, cat.id, 'was_empty');
    // A categorize/refine run auto-assigns mail into it before the user approves.
    seedEmail(h, 'auto-1');
    h.categories.addAutoAssignments(h.accountId, [
      {
        messageId: 'auto-1',
        accountId: h.accountId,
        categoryId: cat.id,
        confidence: 0.5,
        assignedBy: 'auto',
        assignedAt: Date.now(),
        method: 'embed',
      },
    ]);

    let thrown: unknown;
    try {
      h.orchestrator.apply(h.accountId, p.id);
    } catch (err) {
      thrown = err;
    }
    expect(thrown).toBeInstanceOf(ProposalApplyError);
    expect((thrown as ProposalApplyError).httpStatus).toBe(409);
    expect((thrown as Error).message).toMatch(/no longer empty/i);
    // Zero writes: the category stays active, the proposal stays pending, the auto assignment is kept.
    expect(h.categories.findById(cat.id)!.status).toBe('active');
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
    expect(assignmentsFor(h.db, h.accountId, 'auto-1')).toEqual([
      { category_id: cat.id, assigned_by: 'auto', method: 'embed' },
    ]);
  });

  it('retire blocks when the target has user-confirmed members and writes nothing', () => {
    const h = harness();
    const cat = activeCategory(h, 'Kept', 'kept');
    seedEmail(h, 'k-1');
    h.categories.replaceEmailAssignments('k-1', h.accountId, [
      {
        messageId: 'k-1',
        accountId: h.accountId,
        categoryId: cat.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: Date.now(),
        method: null,
      },
    ]);
    const p = retireProposal(h, cat.id, 'kept');

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/user-confirmed/i);
    // Nothing changed: the category stays active and the proposal stays pending.
    expect(h.categories.findById(cat.id)!.status).toBe('active');
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
    expect(assignmentsFor(h.db, h.accountId, 'k-1')).toEqual([
      { category_id: cat.id, assigned_by: 'user', method: null },
    ]);
  });

  it('retire blocks when the target is not active', () => {
    const h = harness();
    const cat = h.categories.create({
      accountId: h.accountId,
      label: 'Already',
      source: 'auto',
      status: 'retired',
      canonicalKey: 'already',
    });
    const p = retireProposal(h, cat.id, 'already');
    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/not active/i);
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
  });

  it('merge moves source rows to the target, keeps user provenance over an equal-confidence auto row, and applies', () => {
    const h = harness();
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    const now = Date.now();
    seedEmail(h, 'm-1');
    seedEmail(h, 'm-2');
    // m-1: a user row on the source and an equal-confidence auto (gate) row already on the target.
    h.categories.replaceEmailAssignments('m-1', h.accountId, [
      {
        messageId: 'm-1',
        accountId: h.accountId,
        categoryId: source.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: now,
        method: null,
      },
      {
        messageId: 'm-1',
        accountId: h.accountId,
        categoryId: target.id,
        confidence: 1,
        assignedBy: 'auto',
        assignedAt: now,
        method: 'gate',
      },
    ]);
    // m-2: an auto row on the source only.
    h.categories.addAutoAssignments(h.accountId, [
      {
        messageId: 'm-2',
        accountId: h.accountId,
        categoryId: source.id,
        confidence: 0.5,
        assignedBy: 'auto',
        assignedAt: now,
        method: 'embed',
      },
    ]);
    const p = mergeProposal(h, source.id, target.id, 'src', 'dst');

    const result = h.orchestrator.apply(h.accountId, p.id);

    expect(result.kind).toBe('merge');
    expect(h.categories.findById(source.id)).toBeNull(); // source absorbed and deleted
    expect(h.proposals.findById(p.id)!.status).toBe('applied');
    // m-1 on the target stays user (provenance dominates the equal-confidence auto row).
    expect(assignmentsFor(h.db, h.accountId, 'm-1')).toEqual([
      { category_id: target.id, assigned_by: 'user', method: null },
    ]);
    // m-2 moved to the target keeping its auto provenance and method.
    expect(assignmentsFor(h.db, h.accountId, 'm-2')).toEqual([
      { category_id: target.id, assigned_by: 'auto', method: 'embed' },
    ]);
  });

  it('merge rebuilds the target centroid from the merged members', () => {
    const h = harness();
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    // Seed the target with a centroid on a different axis so a real rebuild is observable.
    h.categories.saveCentroid(target.id, MODEL, axis(0), 1);
    // Three user-confirmed source members whose embeddings sit on axis 3.
    for (const id of ['g-1', 'g-2', 'g-3']) {
      seedEmailWithEmbedding(h, id, 3);
      assignUser(h, id, source.id);
    }
    const p = mergeProposal(h, source.id, target.id, 'src', 'dst');

    const result = h.orchestrator.apply(h.accountId, p.id);

    expect(result.kind).toBe('merge');
    expect(h.categories.findById(source.id)).toBeNull();
    // The three user members now belong to the target.
    expect(h.categories.listCategoryMemberIds(h.accountId, target.id, 'user').sort()).toEqual([
      'g-1',
      'g-2',
      'g-3',
    ]);
    // The centroid was rebuilt from those members: it points along axis 3 now, not the old axis 0.
    const centroid = h.categories.getCentroid(target.id, MODEL)!;
    expect(centroid.vector[3]).toBeGreaterThan(0.9);
    expect(centroid.vector[0]).toBeLessThan(0.1);
    expect(centroid.emailCount).toBe(3);
    expect(h.proposals.findById(p.id)!.status).toBe('applied');
  });

  it('merge with no usable member embeddings does not crash and keeps the existing target centroid', () => {
    const h = harness();
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    // Existing target centroid on axis 0; the merged members have NO embeddings, so a rebuild has no
    // usable vectors and must fall back to leaving this centroid untouched.
    h.categories.saveCentroid(target.id, MODEL, axis(0), 1);
    for (const id of ['h-1', 'h-2', 'h-3']) {
      seedEmail(h, id); // no embedding saved
      assignUser(h, id, source.id);
    }
    const p = mergeProposal(h, source.id, target.id, 'src', 'dst');

    const result = h.orchestrator.apply(h.accountId, p.id);

    expect(result.kind).toBe('merge');
    expect(h.proposals.findById(p.id)!.status).toBe('applied');
    // Insufficient trusted data: the target keeps its existing centroid unchanged (safe fallback).
    const centroid = h.categories.getCentroid(target.id, MODEL)!;
    expect(centroid.vector[0]).toBeGreaterThan(0.9);
    expect(centroid.emailCount).toBe(1);
    // Members still moved to the target.
    expect(h.categories.listCategoryMemberIds(h.accountId, target.id, 'user').sort()).toEqual([
      'h-1',
      'h-2',
      'h-3',
    ]);
  });

  it('rolls back the whole merge and leaves the proposal pending when the centroid rebuild throws', () => {
    const throwing = {
      rebuild() {
        throw new Error('rebuild boom');
      },
    } as unknown as CategoryCentroidRebuildService;
    const h = harness(NAMING, throwing);
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    seedEmail(h, 'r-1');
    h.categories.addAutoAssignments(h.accountId, [
      {
        messageId: 'r-1',
        accountId: h.accountId,
        categoryId: source.id,
        confidence: 0.5,
        assignedBy: 'auto',
        assignedAt: Date.now(),
        method: 'embed',
      },
    ]);
    const p = mergeProposal(h, source.id, target.id, 'src', 'dst');

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/rebuild boom/);
    // The whole merge rolled back: the source still exists, its assignment is intact, and the
    // proposal is still pending (never marked applied).
    expect(h.categories.findById(source.id)).not.toBeNull();
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
    expect(assignmentsFor(h.db, h.accountId, 'r-1')).toEqual([
      { category_id: source.id, assigned_by: 'auto', method: 'embed' },
    ]);
  });

  it('transfers the source label and canonical key to the target as aliases on merge', () => {
    const h = harness();
    const source = activeCategory(h, 'Old Receipts', 'old_receipts');
    const target = activeCategory(h, 'Receipts', 'receipts');
    const p = mergeProposal(h, source.id, target.id, 'old_receipts', 'receipts');

    h.orchestrator.apply(h.accountId, p.id);

    expect(h.categories.findById(source.id)).toBeNull();
    // The surviving target now answers to the absorbed source's label and canonical key.
    expect(h.aliases.findByAlias(h.accountId, 'Old Receipts')!.id).toBe(target.id);
    expect(h.aliases.findByAlias(h.accountId, 'old_receipts')!.id).toBe(target.id);
  });

  it('transfers a canonical key that normalizes differently from the label as its own alias', () => {
    const h = harness();
    // The key 'receipts_2' normalizes to 'receipts 2', distinct from the label 'Receipts'
    // ('receipts'), so this pins the canonical-key half of the transfer independent of the label.
    const source = activeCategory(h, 'Receipts', 'receipts_2');
    const target = activeCategory(h, 'Dst', 'dst');
    const p = mergeProposal(h, source.id, target.id, 'receipts_2', 'dst');

    h.orchestrator.apply(h.accountId, p.id);

    expect(h.categories.findById(source.id)).toBeNull();
    expect(h.aliases.findByAlias(h.accountId, 'Receipts')!.id).toBe(target.id);
    // The distinct canonical key resolves to the target on its own alias row.
    expect(h.aliases.findByAlias(h.accountId, 'receipts_2')!.id).toBe(target.id);
  });

  it('re-points the source existing aliases to the target on merge', () => {
    const h = harness();
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    // A French alias the user gave the source before the merge.
    h.aliases.addAlias(h.accountId, source.id, 'Factures', 'user');
    const p = mergeProposal(h, source.id, target.id, 'src', 'dst');

    h.orchestrator.apply(h.accountId, p.id);

    expect(h.aliases.findByAlias(h.accountId, 'Factures')!.id).toBe(target.id);
    expect(h.aliases.listForCategory(target.id)).toContain('Factures');
  });

  it('does not create duplicate aliases when the source label and key normalize the same', () => {
    const h = harness();
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    const p = mergeProposal(h, source.id, target.id, 'src', 'dst');

    h.orchestrator.apply(h.accountId, p.id);

    // 'Src' (label) and 'src' (canonical key) normalize to the same alias, so only one row is kept.
    const srcAliases = h.aliases
      .listForCategory(target.id)
      .filter((a) => a.toLowerCase() === 'src');
    expect(srcAliases).toHaveLength(1);
    expect(h.aliases.findByAlias(h.accountId, 'src')!.id).toBe(target.id);
  });

  it('skips a conflicting alias without failing the merge', () => {
    const h = harness();
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    // A third active category already owns the source label as an alias.
    const other = activeCategory(h, 'Other', 'other');
    h.aliases.addAlias(h.accountId, other.id, 'Src', 'user');
    const p = mergeProposal(h, source.id, target.id, 'src', 'dst');

    const result = h.orchestrator.apply(h.accountId, p.id);

    // The merge still applies; the conflicting alias keeps its original owner (not corrupted).
    expect(result.kind).toBe('merge');
    expect(h.categories.findById(source.id)).toBeNull();
    expect(h.proposals.findById(p.id)!.status).toBe('applied');
    expect(h.aliases.findByAlias(h.accountId, 'Src')!.id).toBe(other.id);
  });

  it('rolls back the alias transfer when the merge fails', () => {
    const throwing = {
      rebuild() {
        throw new Error('rebuild boom');
      },
    } as unknown as CategoryCentroidRebuildService;
    const h = harness(NAMING, throwing);
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    h.aliases.addAlias(h.accountId, source.id, 'Factures', 'user');
    const p = mergeProposal(h, source.id, target.id, 'src', 'dst');

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/rebuild boom/);
    // Everything rolled back: the source still exists, its alias still points at the source, and the
    // target gained no transferred alias.
    expect(h.categories.findById(source.id)).not.toBeNull();
    expect(h.aliases.findByAlias(h.accountId, 'Factures')!.id).toBe(source.id);
    expect(h.aliases.listForCategory(target.id)).not.toContain('Factures');
    expect(h.aliases.listForCategory(target.id)).not.toContain('Src');
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
  });

  it('dismisses sibling merge proposals that shared the deleted source on merge apply', () => {
    const h = harness();
    const s = activeCategory(h, 'S', 's');
    const t = activeCategory(h, 'T', 't');
    const u = activeCategory(h, 'U', 'u');
    const mST = mergeProposal(h, s.id, t.id, 's', 't');
    const mSU = mergeProposal(h, s.id, u.id, 's', 'u');

    h.orchestrator.apply(h.accountId, mST.id);

    expect(h.categories.findById(s.id)).toBeNull();
    expect(h.proposals.findById(mST.id)!.status).toBe('applied');
    // The sibling merge that also named S as its source is dismissed, not left as a failing card.
    expect(h.proposals.findById(mSU.id)!.status).toBe('dismissed');
    expect(h.orchestrator.listPending(h.accountId).some((p) => p.id === mSU.id)).toBe(false);
  });

  it('does not dismiss a merge with a different source, and a merge targeting S is cascade-removed not source-dismissed', () => {
    const h = harness();
    const s = activeCategory(h, 'S', 's');
    const t = activeCategory(h, 'T', 't');
    const u = activeCategory(h, 'U', 'u');
    const x = activeCategory(h, 'X', 'x');
    const y = activeCategory(h, 'Y', 'y');
    const mST = mergeProposal(h, s.id, t.id, 's', 't');
    const mXY = mergeProposal(h, x.id, y.id, 'x', 'y'); // unrelated source
    const mUS = mergeProposal(h, u.id, s.id, 'u', 's'); // S is the TARGET (category_id = s)

    h.orchestrator.apply(h.accountId, mST.id);

    // A pending merge with an unrelated source is untouched.
    expect(h.proposals.findById(mXY.id)!.status).toBe('pending');
    // The merge whose target was S is removed by ON DELETE CASCADE (category_id), not by this cleanup.
    expect(h.proposals.findById(mUS.id)).toBeNull();
  });

  it('rolls back the sibling dismissal when the merge fails', () => {
    const throwing = {
      rebuild() {
        throw new Error('rebuild boom');
      },
    } as unknown as CategoryCentroidRebuildService;
    const h = harness(NAMING, throwing);
    const s = activeCategory(h, 'S', 's');
    const t = activeCategory(h, 'T', 't');
    const u = activeCategory(h, 'U', 'u');
    const mST = mergeProposal(h, s.id, t.id, 's', 't');
    const mSU = mergeProposal(h, s.id, u.id, 's', 'u');

    expect(() => h.orchestrator.apply(h.accountId, mST.id)).toThrow(/rebuild boom/);
    // All-or-nothing: source still exists, both merges still pending.
    expect(h.categories.findById(s.id)).not.toBeNull();
    expect(h.proposals.findById(mST.id)!.status).toBe('pending');
    expect(h.proposals.findById(mSU.id)!.status).toBe('pending');
  });

  it('merge blocks when the source category is missing and writes nothing', () => {
    const h = harness();
    const target = activeCategory(h, 'Dst', 'dst');
    const p = h.proposals.createStructural({
      accountId: h.accountId,
      kind: 'merge',
      categoryId: target.id,
      sourceCategoryId: 'ghost-source',
      runId: 'run-merge',
      label: 'Merge',
      description: '',
      canonicalKey: 'dst',
      suppressionKey: 'merge:dst|ghost',
      embeddingModelId: MODEL,
      confidence: 0.9,
      evidence: [],
    });

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/source category not found/i);
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
    expect(h.categories.findById(target.id)!.status).toBe('active');
  });

  it('split creates child categories, seeds their centroids, and moves eligible auto members', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    // Two auto members on the source, one destined for each child.
    seedAutoOnSource(h, 'a-1', source.id, 'gate', 0.7);
    seedAutoOnSource(h, 'b-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', ['b-1'], axis(5));

    const result = h.orchestrator.apply(h.accountId, p.id);

    expect(result.kind).toBe('split');
    expect(result.assigned).toBe(2);
    const childA = h.categories.findByLabel(h.accountId, 'Child A')!;
    const childB = h.categories.findByLabel(h.accountId, 'Child B')!;
    expect(childA.status).toBe('active');
    expect(childB.status).toBe('active');
    // The auto members moved to their child, preserving provenance and method.
    expect(assignmentsFor(h.db, h.accountId, 'a-1')).toEqual([
      { category_id: childA.id, assigned_by: 'auto', method: 'gate' },
    ]);
    expect(assignmentsFor(h.db, h.accountId, 'b-1')).toEqual([
      { category_id: childB.id, assigned_by: 'auto', method: 'embed' },
    ]);
    // Each child centroid was seeded from the child proposal data.
    expect(h.categories.getCentroid(childA.id, MODEL)!.vector[4]).toBeGreaterThan(0.9);
    expect(h.categories.getCentroid(childB.id, MODEL)!.vector[5]).toBeGreaterThan(0.9);
    // The source is empty after the moves, so it is retired.
    expect(h.categories.findById(source.id)!.status).toBe('retired');
    expect(h.proposals.findById(p.id)!.status).toBe('applied');
  });

  it('split leaves user assignments on the source and keeps the source active', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    // A user member and an auto member; both are listed by a child.
    seedEmailWithEmbedding(h, 'u-1', 2);
    assignUser(h, 'u-1', source.id);
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1', 'u-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', ['b-unknown'], axis(5));

    const result = h.orchestrator.apply(h.accountId, p.id);

    expect(result.assigned).toBe(1); // only the auto member moved
    const childA = h.categories.findByLabel(h.accountId, 'Child A')!;
    // The user member is untouched: still on the source, still user.
    expect(assignmentsFor(h.db, h.accountId, 'u-1')).toEqual([
      { category_id: source.id, assigned_by: 'user', method: null },
    ]);
    // The auto member moved to its child.
    expect(assignmentsFor(h.db, h.accountId, 'a-1')).toEqual([
      { category_id: childA.id, assigned_by: 'auto', method: 'embed' },
    ]);
    // A user assignment remains, so the source stays active.
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(h.proposals.findById(p.id)!.status).toBe('applied');
  });

  it('split does not move a member listed by more than one child', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    seedAutoOnSource(h, 'dup', source.id, 'embed', 0.6); // listed by both children
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6); // listed by one child
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['dup', 'a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', ['dup'], axis(5));

    const result = h.orchestrator.apply(h.accountId, p.id);

    expect(result.assigned).toBe(1); // only a-1 moved; dup is ambiguous
    const childA = h.categories.findByLabel(h.accountId, 'Child A')!;
    expect(assignmentsFor(h.db, h.accountId, 'a-1')).toEqual([
      { category_id: childA.id, assigned_by: 'auto', method: 'embed' },
    ]);
    // The ambiguous member stays on the source, so the source stays active.
    expect(assignmentsFor(h.db, h.accountId, 'dup')).toEqual([
      { category_id: source.id, assigned_by: 'auto', method: 'embed' },
    ]);
    expect(h.categories.findById(source.id)!.status).toBe('active');
  });

  it('split preserves a moved auto member confidence, method, and assigned_at exactly', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    seedEmailWithEmbedding(h, 'a-1', 2);
    h.categories.addAutoAssignments(h.accountId, [
      {
        messageId: 'a-1',
        accountId: h.accountId,
        categoryId: source.id,
        confidence: 0.73,
        assignedBy: 'auto',
        assignedAt: 111222333,
        method: 'gate',
      },
    ]);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', [], axis(5));

    h.orchestrator.apply(h.accountId, p.id);

    const childA = h.categories.findByLabel(h.accountId, 'Child A')!;
    const row = fullAssignmentFor(h.db, h.accountId, 'a-1');
    expect(row).toEqual({
      category_id: childA.id,
      assigned_by: 'auto',
      method: 'gate',
      confidence: 0.73,
      assigned_at: 111222333,
    });
  });

  it('split blocks on a label-only collision with an existing category and writes nothing', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    // An existing category shares only the LABEL a child would take (its key differs).
    activeCategory(h, 'Taken Label', 'taken_label_key');
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Taken Label', 'child_a', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', [], axis(5));

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/collides/i);
    // Zero writes: no child created, the source keeps its member and stays active, proposal pending.
    expect(h.categories.findByLabel(h.accountId, 'Child B')).toBeNull();
    expect(h.categories.findByCanonicalKey(h.accountId, 'child_a')).toBeNull();
    expect(assignmentsFor(h.db, h.accountId, 'a-1')).toEqual([
      { category_id: source.id, assigned_by: 'auto', method: 'embed' },
    ]);
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
  });

  it('split blocks on a key-only collision with an existing category and writes nothing', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    // An existing category shares only the canonical KEY a child would take (its label differs).
    activeCategory(h, 'Other Label', 'dup_key');
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'dup_key', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', [], axis(5));

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/collides/i);
    expect(h.categories.findByLabel(h.accountId, 'Child A')).toBeNull();
    expect(h.categories.findByLabel(h.accountId, 'Child B')).toBeNull();
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
  });

  it('split blocks when two children share a canonical key and writes nothing', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    // Two siblings collide on the key (distinct labels), so no pre-existing category is involved.
    addChild(h, p.id, 'Child A', 'dup_key', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'dup_key', [], axis(5));

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/collides/i);
    expect(h.categories.findByLabel(h.accountId, 'Child A')).toBeNull();
    expect(h.categories.findByLabel(h.accountId, 'Child B')).toBeNull();
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
  });

  it('split blocks when two children share a label and writes nothing', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    // Two siblings collide on the label (distinct keys).
    addChild(h, p.id, 'Same Label', 'key_a', ['a-1'], axis(4));
    addChild(h, p.id, 'Same Label', 'key_b', [], axis(5));

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow(/collides/i);
    expect(h.categories.findByCanonicalKey(h.accountId, 'key_a')).toBeNull();
    expect(h.categories.findByCanonicalKey(h.accountId, 'key_b')).toBeNull();
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
  });

  it('rolls back the whole split and leaves the proposal pending when a centroid save fails', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1'], axis(4));
    // Child B has a malformed (wrong-dimension) centroid, so saveCentroid throws mid-transaction.
    addChild(h, p.id, 'Child B', 'child_b', [], new Float32Array(1));

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow();
    // Everything rolled back: neither child exists, the member is still on the source, and the
    // source and proposal are untouched.
    expect(h.categories.findByLabel(h.accountId, 'Child A')).toBeNull();
    expect(h.categories.findByLabel(h.accountId, 'Child B')).toBeNull();
    expect(assignmentsFor(h.db, h.accountId, 'a-1')).toEqual([
      { category_id: source.id, assigned_by: 'auto', method: 'embed' },
    ]);
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
  });

  it('rebuilds the source centroid from the remaining user members when the source stays active', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    // Old broad (mixed) centroid on axis 0 - the stale centroid the split must not leave behind.
    h.categories.saveCentroid(source.id, MODEL, axis(0), 5);
    // Three user-confirmed members on axis 6 remain on the source; one auto member moves to a child.
    for (const id of ['u-1', 'u-2', 'u-3']) {
      seedEmailWithEmbedding(h, id, 6);
      assignUser(h, id, source.id);
    }
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', ['b-unknown'], axis(5));

    h.orchestrator.apply(h.accountId, p.id);

    // The source stays active (3 user members) and its centroid was rebuilt from them (axis 6),
    // no longer the old broad axis-0 centroid.
    expect(h.categories.findById(source.id)!.status).toBe('active');
    const centroid = h.categories.getCentroid(source.id, MODEL)!;
    expect(centroid.vector[6]).toBeGreaterThan(0.9);
    expect(centroid.vector[0]).toBeLessThan(0.1);
    // The user members are untouched on the source.
    expect(assignmentsFor(h.db, h.accountId, 'u-1')).toEqual([
      { category_id: source.id, assigned_by: 'user', method: null },
    ]);
  });

  it('rebuilds the source centroid from leftover ambiguous auto members', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    h.categories.saveCentroid(source.id, MODEL, axis(0), 5);
    // Three ambiguous auto members on axis 7, listed by BOTH children so they never move.
    for (const id of ['x-1', 'x-2', 'x-3']) {
      seedEmailWithEmbedding(h, id, 7);
      h.categories.addAutoAssignments(h.accountId, [
        {
          messageId: id,
          accountId: h.accountId,
          categoryId: source.id,
          confidence: 0.5,
          assignedBy: 'auto',
          assignedAt: Date.now(),
          method: 'embed',
        },
      ]);
    }
    // One unambiguous auto member that does move to Child A.
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1', 'x-1', 'x-2', 'x-3'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', ['x-1', 'x-2', 'x-3'], axis(5));

    h.orchestrator.apply(h.accountId, p.id);

    // The ambiguous members stay on the source, which stays active with a centroid rebuilt from them
    // (axis 7), not the old mixed centroid.
    expect(h.categories.findById(source.id)!.status).toBe('active');
    const centroid = h.categories.getCentroid(source.id, MODEL)!;
    expect(centroid.vector[7]).toBeGreaterThan(0.9);
    expect(centroid.vector[0]).toBeLessThan(0.1);
  });

  it('drops the stale source centroid when too few members remain to rebuild', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    h.categories.saveCentroid(source.id, MODEL, axis(0), 5);
    // One user member remains: below MIN_TRUSTED_REBUILD and it blocks the auto fallback.
    seedEmailWithEmbedding(h, 'u-1', 6);
    assignUser(h, 'u-1', source.id);
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', ['b-unknown'], axis(5));

    h.orchestrator.apply(h.accountId, p.id);

    // The source stays active but there is too little data to rebuild, so the stale centroid is removed
    // entirely rather than left to keep attracting mail to the old broad bucket.
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(h.categories.getCentroid(source.id, MODEL)).toBeNull();
  });

  it('rolls back the whole split when the source centroid refresh fails', () => {
    const boom = {
      rebuild() {
        throw new Error('rebuild failed');
      },
    } as unknown as CategoryCentroidRebuildService;
    const h = harness(NAMING, boom);
    const source = activeCategory(h, 'Big', 'big');
    h.categories.saveCentroid(source.id, MODEL, axis(0), 5);
    // A user member keeps the source active after the move, so the source-centroid refresh runs.
    seedEmailWithEmbedding(h, 'u-1', 6);
    assignUser(h, 'u-1', source.id);
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', ['b-unknown'], axis(5));

    expect(() => h.orchestrator.apply(h.accountId, p.id)).toThrow();

    // Nothing persisted: no children, the auto member stayed put, the source keeps its old centroid,
    // and the proposal is still pending.
    expect(h.categories.findByLabel(h.accountId, 'Child A')).toBeNull();
    expect(h.categories.findByLabel(h.accountId, 'Child B')).toBeNull();
    expect(assignmentsFor(h.db, h.accountId, 'a-1')).toEqual([
      { category_id: source.id, assigned_by: 'auto', method: 'embed' },
    ]);
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(h.categories.getCentroid(source.id, MODEL)!.vector[0]).toBeGreaterThan(0.9);
    expect(h.proposals.findById(p.id)!.status).toBe('pending');
  });
});

describe('DiscoveryProposalOrchestrator.dismiss (structural kinds)', () => {
  it('records the dismissal, preserves the suppression key, and never touches the live categories', () => {
    const h = harness();
    const source = activeCategory(h, 'S', 's');
    const target = activeCategory(h, 'T', 't');
    const p = mergeProposal(h, source.id, target.id, 's', 't');

    const res = h.orchestrator.dismiss(h.accountId, p.id);

    expect(res.dismissed).toBe(true);
    expect(h.proposals.findById(p.id)!.status).toBe('dismissed');
    // Neither the merge source nor the target is retired or deleted by a dismiss.
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(h.categories.findById(target.id)!.status).toBe('active');
    // The suppression key survives so a re-run does not re-propose the same merge.
    expect(h.proposals.resolvedStructuralSuppressionKeys(h.accountId).has(p.suppressionKey)).toBe(
      true,
    );
  });

  it('dismissing a split proposal only records the dismissal and touches no category or assignment', () => {
    const h = harness();
    const source = activeCategory(h, 'Big', 'big');
    seedAutoOnSource(h, 'a-1', source.id, 'embed', 0.6);
    const p = splitProposal(h, source.id, 'big');
    addChild(h, p.id, 'Child A', 'child_a', ['a-1'], axis(4));
    addChild(h, p.id, 'Child B', 'child_b', [], axis(5));

    const res = h.orchestrator.dismiss(h.accountId, p.id);

    expect(res.dismissed).toBe(true);
    expect(h.proposals.findById(p.id)!.status).toBe('dismissed');
    // No child was created, the source stays active, and its member is untouched.
    expect(h.categories.findByLabel(h.accountId, 'Child A')).toBeNull();
    expect(h.categories.findByLabel(h.accountId, 'Child B')).toBeNull();
    expect(h.categories.findById(source.id)!.status).toBe('active');
    expect(assignmentsFor(h.db, h.accountId, 'a-1')).toEqual([
      { category_id: source.id, assigned_by: 'auto', method: 'embed' },
    ]);
    expect(h.proposals.resolvedStructuralSuppressionKeys(h.accountId).has(p.suppressionKey)).toBe(
      true,
    );
  });
});

describe('DiscoveryProposalOrchestrator.listPending kind-awareness', () => {
  it('reports kind and sourceCategoryId for new_category, merge, and retire proposals', async () => {
    const h = harness();
    // new_category proposals from a normal generate run.
    await h.orchestrator.generate(h.accountId, MODEL, 'qwen');
    // A structural merge (target survives, source absorbed) and a retire, constructed directly.
    const source = activeCategory(h, 'Src', 'src');
    const target = activeCategory(h, 'Dst', 'dst');
    const merge = mergeProposal(h, source.id, target.id, 'src', 'dst');
    const empty = activeCategory(h, 'Empty', 'empty');
    const retire = retireProposal(h, empty.id, 'empty');

    const pending = h.orchestrator.listPending(h.accountId);
    const byId = new Map(pending.map((p) => [p.id, p]));

    // new_category rows carry kind='new_category' and no source.
    const newCats = pending.filter((p) => p.kind === 'new_category');
    expect(newCats.length).toBeGreaterThanOrEqual(2);
    expect(newCats.every((p) => p.sourceCategoryId === null)).toBe(true);

    // merge carries kind='merge' with sourceCategoryId set to the absorbed category.
    expect(byId.get(merge.id)).toMatchObject({ kind: 'merge', sourceCategoryId: source.id });
    // retire carries kind='retire' with a null source.
    expect(byId.get(retire.id)).toMatchObject({ kind: 'retire', sourceCategoryId: null });
  });

  it('enriches structural proposals with live affected counts, user impact, and split children', () => {
    const h = harness();

    // A merge whose source holds one auto member and one user member.
    const mSrc = activeCategory(h, 'Merge Src', 'msrc');
    const mDst = activeCategory(h, 'Merge Dst', 'mdst');
    seedAutoOnSource(h, 'm-auto', mSrc.id);
    seedEmailWithEmbedding(h, 'm-user', 3);
    assignUser(h, 'm-user', mSrc.id);
    const merge = mergeProposal(h, mSrc.id, mDst.id, 'msrc', 'mdst');

    // A split whose source holds two auto members, each destined for one child.
    const sSrc = activeCategory(h, 'Split Src', 'ssrc');
    seedAutoOnSource(h, 's-a', sSrc.id);
    seedAutoOnSource(h, 's-b', sSrc.id);
    const split = splitProposal(h, sSrc.id, 'ssrc');
    addChild(h, split.id, 'Child A', 'child_a', ['s-a'], axis(4));
    addChild(h, split.id, 'Child B', 'child_b', ['s-b'], axis(5));

    const byId = new Map(h.orchestrator.listPending(h.accountId).map((p) => [p.id, p]));

    // The merge reports the source's live counts (2 assigned, 1 of them user-confirmed).
    const mergeView = byId.get(merge.id)!;
    expect(mergeView.affectedCount).toBe(2);
    expect(mergeView.userImpactCount).toBe(1);
    expect(mergeView.children).toBeUndefined();

    // The split reports its source's live count and its child categories.
    const splitView = byId.get(split.id)!;
    expect(splitView.affectedCount).toBe(2);
    expect(splitView.userImpactCount).toBe(0);
    expect(splitView.children).toHaveLength(2);
    expect(splitView.children!.map((c) => c.label).sort()).toEqual(['Child A', 'Child B']);
    expect(splitView.children!.find((c) => c.label === 'Child A')!.sampleSubjects).toEqual([
      's s-a',
    ]);
    expect(splitView.children!.find((c) => c.label === 'Child B')!.sampleSubjects).toEqual([
      's s-b',
    ]);
  });
});
