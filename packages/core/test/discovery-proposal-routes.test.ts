/**
 * Phase 2c tests: the proposal HTTP routes. Drives the generate, list, apply, and dismiss endpoints
 * through Fastify inject against a real orchestrator and in-memory database, checking status codes
 * and that ownership and lifecycle guards return the right errors.
 */
import { describe, it, expect } from 'vitest';
import Fastify from 'fastify';
import type { Logger } from 'pino';
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
import type { AppContext } from '../src/context.js';
import { ResidualDiscoveryService } from '../src/services/residual-discovery-service.js';
import { DiscoveryProposalService } from '../src/services/discovery-proposal-service.js';
import { DiscoveryProposalOrchestrator } from '../src/services/discovery-proposal-orchestrator.js';
import { CategoryCentroidRebuildService } from '../src/services/category-centroid-rebuild-service.js';
import { CategoryHealthService } from '../src/services/category-health-service.js';
import { StructuralProposalService } from '../src/services/structural-proposal-service.js';
import { registerCategoryRoutes } from '../src/routes/categories.js';

const MODEL = 'bge-m3';
const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

const NAMING = JSON.stringify({
  clusters: [
    {
      clusterIndex: 0,
      action: 'new_category',
      label: 'Receipts & Invoices',
      description: 'Invoices.',
      suggestedKey: 'finance.invoices',
    },
    {
      clusterIndex: 1,
      action: 'new_category',
      label: 'Flight Bookings',
      description: 'Flights.',
      suggestedKey: 'travel.flights',
    },
  ],
});

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

async function buildApp() {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const categories = new CategoryRepository(db);
  const proposals = new CategoryProposalRepository(db);
  const audit = new DiscoveryAuditRepository(db);
  const emails = new EmailRepository(db);
  const embeddings = new EmbeddingRepository(db);
  const acc = accounts.create({ address: 'w@x.com', kind: 'work' });

  for (const b of [
    { dim: 0, prefix: 'inv', from: 'billing@acme.com' },
    { dim: 1, prefix: 'fly', from: 'trips@fly.com' },
  ]) {
    const rows = Array.from({ length: 6 }, (_, i) => ({
      messageId: `${b.prefix}-${i}`,
      accountId: acc.id,
      folder: 'INBOX',
      subject: `Subject ${b.prefix} ${i}`,
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

  const llm = {
    async chat() {
      return NAMING;
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
  const centroidRebuild = new CategoryCentroidRebuildService(categories, embeddings, silentLogger);
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
  const health = new CategoryHealthService(categories, embeddings);
  const structural = new StructuralProposalService(categories, proposals, health, silentLogger);

  const ctx = {
    config: { llm: { embeddingModel: MODEL, generationModel: 'qwen' } },
    db,
    llm,
    logger: silentLogger,
    repos: {
      accounts,
      categories,
      emails,
      embeddings,
      categoryProposals: proposals,
      discoveryAudit: audit,
    },
    services: { discoveryProposal: orchestrator, structuralProposal: structural },
  } as unknown as AppContext;

  const app = Fastify();
  await registerCategoryRoutes(app, ctx);
  await app.ready();
  return { app, accountId: acc.id, accounts, categories, emails };
}

describe('discovery proposal routes', () => {
  it('runs the full generate, list, apply, dismiss flow over HTTP', async () => {
    const { app, accountId, categories } = await buildApp();

    const gen = await app.inject({
      method: 'POST',
      url: '/categories/proposals/generate',
      payload: { accountId },
    });
    expect(gen.statusCode).toBe(200);
    expect(gen.json().created).toHaveLength(2);

    const list = await app.inject({
      method: 'GET',
      url: `/categories/proposals?accountId=${accountId}`,
    });
    expect(list.statusCode).toBe(200);
    const pending = list.json().proposals as Array<{ id: string; categoryId: string }>;
    expect(pending).toHaveLength(2);

    const apply = await app.inject({
      method: 'POST',
      url: `/categories/proposals/${pending[0]!.id}/apply`,
      payload: { accountId },
    });
    expect(apply.statusCode).toBe(200);
    expect(categories.listActive(accountId).some((c) => c.id === pending[0]!.categoryId)).toBe(
      true,
    );

    const dismiss = await app.inject({
      method: 'POST',
      url: `/categories/proposals/${pending[1]!.id}/dismiss`,
      payload: { accountId },
    });
    expect(dismiss.statusCode).toBe(200);
    expect(dismiss.json()).toMatchObject({ dismissed: true });

    // Both are resolved, so the queue is empty.
    const after = await app.inject({
      method: 'GET',
      url: `/categories/proposals?accountId=${accountId}`,
    });
    expect(after.json().proposals).toHaveLength(0);

    await app.close();
  });

  it('validates input and guards ownership and lifecycle', async () => {
    const { app, accountId } = await buildApp();

    expect(
      (await app.inject({ method: 'POST', url: '/categories/proposals/generate', payload: {} }))
        .statusCode,
    ).toBe(400);
    expect((await app.inject({ method: 'GET', url: '/categories/proposals' })).statusCode).toBe(
      400,
    );
    expect(
      (
        await app.inject({
          method: 'POST',
          url: '/categories/proposals/generate',
          payload: { accountId: 'nope' },
        })
      ).statusCode,
    ).toBe(404);
    expect(
      (
        await app.inject({
          method: 'POST',
          url: '/categories/proposals/missing/apply',
          payload: { accountId },
        })
      ).statusCode,
    ).toBe(404);

    // Applying twice is a lifecycle conflict.
    await app.inject({
      method: 'POST',
      url: '/categories/proposals/generate',
      payload: { accountId },
    });
    const pending = (
      await app.inject({ method: 'GET', url: `/categories/proposals?accountId=${accountId}` })
    ).json().proposals as Array<{ id: string }>;
    await app.inject({
      method: 'POST',
      url: `/categories/proposals/${pending[0]!.id}/apply`,
      payload: { accountId },
    });
    const twice = await app.inject({
      method: 'POST',
      url: `/categories/proposals/${pending[0]!.id}/apply`,
      payload: { accountId },
    });
    expect(twice.statusCode).toBe(409);

    await app.close();
  });
});

/**
 * Set up two overlapping active auto categories (a merge pair, the larger surviving) plus one empty
 * active auto category (a retire candidate), so a structural generate run has one of each to make.
 */
function seedStructuralFixture(
  categories: CategoryRepository,
  emails: EmailRepository,
  accountId: string,
) {
  // Same-purpose labels (both map to the invoices/receipts purpose group) so the F1 merge-quality
  // gate accepts the pair; the variable names are kept generic.
  const src = categories.create({
    accountId,
    label: 'Invoices',
    source: 'auto',
    status: 'active',
    canonicalKey: 'invoices',
  });
  const dst = categories.create({
    accountId,
    label: 'Receipts',
    source: 'auto',
    status: 'active',
    canonicalKey: 'receipts',
  });
  const empty = categories.create({
    accountId,
    label: 'Empty',
    source: 'auto',
    status: 'active',
    canonicalKey: 'empty',
  });
  // Near-identical stored centroids so health reports high overlap between src and dst.
  categories.saveCentroid(src.id, MODEL, axis(0), 1);
  categories.saveCentroid(dst.id, MODEL, axis(0), 2);
  // dst is the larger category (2 members vs 1), so it survives as the merge target.
  const members: Array<[string, string]> = [
    [src.id, 'src-a'],
    [dst.id, 'dst-a'],
    [dst.id, 'dst-b'],
  ];
  for (const [categoryId, messageId] of members) {
    emails.upsertBatch([
      { messageId, accountId, folder: 'INBOX', subject: 's', fromAddr: 'a@b.com' },
    ]);
    categories.addAutoAssignments(accountId, [
      {
        messageId,
        accountId,
        categoryId,
        confidence: 0.5,
        assignedBy: 'auto',
        assignedAt: Date.now(),
        method: 'embed',
      },
    ]);
  }
  return { src, dst, empty };
}

describe('structural proposal generation route', () => {
  it('returns 400 for an invalid body and 404 for a missing account', async () => {
    const { app, accountId } = await buildApp();

    const bad = await app.inject({
      method: 'POST',
      url: '/categories/proposals/generate-structural',
      payload: {},
    });
    expect(bad.statusCode).toBe(400);

    const missing = await app.inject({
      method: 'POST',
      url: '/categories/proposals/generate-structural',
      payload: { accountId: 'nope' },
    });
    expect(missing.statusCode).toBe(404);

    await app.close();
  });

  it('creates merge and retire proposals, lists them with correct kind, and does not duplicate on re-run', async () => {
    const { app, accountId, categories, emails } = await buildApp();
    const { src, dst, empty } = seedStructuralFixture(categories, emails, accountId);

    const gen = await app.inject({
      method: 'POST',
      url: '/categories/proposals/generate-structural',
      payload: { accountId },
    });
    expect(gen.statusCode).toBe(200);
    const created = gen.json().created as Array<{
      kind: string;
      categoryId: string;
      sourceCategoryId: string | null;
    }>;
    const merge = created.find((c) => c.kind === 'merge')!;
    const retire = created.find((c) => c.kind === 'retire')!;
    expect(merge).toMatchObject({ categoryId: dst.id, sourceCategoryId: src.id });
    expect(retire).toMatchObject({ categoryId: empty.id, sourceCategoryId: null });

    // The generated structural proposals appear in the review queue with their kind and source.
    const list = await app.inject({
      method: 'GET',
      url: `/categories/proposals?accountId=${accountId}`,
    });
    const pending = list.json().proposals as Array<{
      kind: string;
      categoryId: string;
      sourceCategoryId: string | null;
    }>;
    expect(pending.find((p) => p.kind === 'merge')).toMatchObject({
      categoryId: dst.id,
      sourceCategoryId: src.id,
    });
    expect(pending.find((p) => p.kind === 'retire')).toMatchObject({
      categoryId: empty.id,
      sourceCategoryId: null,
    });
    const structuralCount = pending.filter((p) => p.kind !== 'new_category').length;
    expect(structuralCount).toBe(2);

    // A second run re-proposes nothing (the pending suppression keys already cover both).
    const again = await app.inject({
      method: 'POST',
      url: '/categories/proposals/generate-structural',
      payload: { accountId },
    });
    expect(again.json().created).toHaveLength(0);
    expect(again.json().skippedExisting).toBeGreaterThanOrEqual(2);
    const listAgain = await app.inject({
      method: 'GET',
      url: `/categories/proposals?accountId=${accountId}`,
    });
    expect(
      (listAgain.json().proposals as Array<{ kind: string }>).filter((p) => p.kind !== 'new_category'),
    ).toHaveLength(2);

    await app.close();
  });

  it('skips structural generation for discovery-ineligible personal accounts', async () => {
    const { app, accounts, categories } = await buildApp();
    const personal = accounts.create({ address: 'personal@x.com', kind: 'personal' });
    categories.create({
      accountId: personal.id,
      label: 'Empty Personal',
      source: 'auto',
      status: 'active',
      canonicalKey: 'empty_personal',
    });

    const gen = await app.inject({
      method: 'POST',
      url: '/categories/proposals/generate-structural',
      payload: { accountId: personal.id },
    });
    expect(gen.statusCode).toBe(200);
    expect(gen.json()).toMatchObject({
      runId: '',
      created: [],
      mergeCandidates: 0,
      retireCandidates: 0,
      skippedExisting: 0,
    });

    const list = await app.inject({
      method: 'GET',
      url: `/categories/proposals?accountId=${personal.id}`,
    });
    expect(list.json().proposals).toHaveLength(0);

    await app.close();
  });
});
