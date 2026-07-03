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
import { DiscoveryAuditRepository } from '../src/repositories/discovery-audit-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import type { LlmClient } from '../src/llm/client.js';
import type { LlmConfig } from '../src/config/schema.js';
import { ResidualDiscoveryService } from '../src/services/residual-discovery-service.js';
import { DiscoveryProposalService } from '../src/services/discovery-proposal-service.js';
import { DiscoveryProposalOrchestrator } from '../src/services/discovery-proposal-orchestrator.js';

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

function harness(answer: string = NAMING) {
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
  const orchestrator = new DiscoveryProposalOrchestrator(
    db,
    proposals,
    categories,
    emails,
    proposalService,
    accounts,
    audit,
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
