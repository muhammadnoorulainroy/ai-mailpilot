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
import { DiscoveryAuditRepository } from '../src/repositories/discovery-audit-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import type { LlmClient } from '../src/llm/client.js';
import type { LlmConfig } from '../src/config/schema.js';
import type { AppContext } from '../src/context.js';
import { ResidualDiscoveryService } from '../src/services/residual-discovery-service.js';
import { DiscoveryProposalService } from '../src/services/discovery-proposal-service.js';
import { DiscoveryProposalOrchestrator } from '../src/services/discovery-proposal-orchestrator.js';
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
    services: { discoveryProposal: orchestrator },
  } as unknown as AppContext;

  const app = Fastify();
  await registerCategoryRoutes(app, ctx);
  await app.ready();
  return { app, accountId: acc.id, categories };
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
