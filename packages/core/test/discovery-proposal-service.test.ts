/**
 * Phase 2b tests: the proposal service wires residual clustering, local-model naming, and the
 * deterministic gate. A fake LLM stands in for the model. Asserts acceptance, gate rejection of bad
 * names, local-by-default routing, and that the run writes nothing (read-only).
 */
import { describe, it, expect } from 'vitest';
import type { Logger } from 'pino';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import type { LlmClient } from '../src/llm/client.js';
import type { LlmConfig } from '../src/config/schema.js';
import { ResidualDiscoveryService } from '../src/services/residual-discovery-service.js';
import { DiscoveryProposalService } from '../src/services/discovery-proposal-service.js';

const MODEL = 'bge-m3';

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

interface Harness {
  captured: { provider?: string };
  makeLlm: (answer: string) => LlmClient;
  accounts: AccountRepository;
  categories: CategoryRepository;
  emails: EmailRepository;
  embeddings: EmbeddingRepository;
  accountId: string;
}

function harness(): Harness {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const categories = new CategoryRepository(db);
  const emails = new EmailRepository(db);
  const embeddings = new EmbeddingRepository(db);
  const acc = accounts.create({ address: 'w@x.com', kind: 'work' });

  // Two well-separated blobs of uncategorized mail: an invoices cluster and a flights cluster.
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

  const captured: { provider?: string } = {};
  const makeLlm = (answer: string): LlmClient =>
    ({
      async chat(opts: { provider?: string }) {
        captured.provider = opts.provider;
        return answer;
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
    }) as unknown as LlmClient;

  return { captured, makeLlm, accounts, categories, emails, embeddings, accountId: acc.id };
}

function service(
  h: Harness,
  answer: string,
  cfg: Partial<LlmConfig> = {},
): DiscoveryProposalService {
  const residual = new ResidualDiscoveryService(h.embeddings, h.categories);
  const getConfig = () => ({ allowCloudDiscovery: false, ...cfg }) as unknown as LlmConfig;
  return new DiscoveryProposalService(
    residual,
    h.emails,
    h.categories,
    h.makeLlm(answer),
    getConfig,
    silentLogger,
  );
}

describe('DiscoveryProposalService', () => {
  it('names residual clusters, accepts sound names, and writes nothing', async () => {
    const h = harness();
    const answer = JSON.stringify({
      clusters: [
        {
          clusterIndex: 0,
          action: 'new_category',
          label: 'Receipts & Invoices',
          description: 'Payment confirmations and invoices for purchases.',
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
    const svc = service(h, answer);

    const result = await svc.propose(h.accountId, MODEL, 'qwen');

    expect(result.clusterCount).toBe(2);
    expect(result.sampledEmails).toBeGreaterThan(0);
    expect(result.accepted.map((c) => c.label).sort()).toEqual([
      'Flight Bookings',
      'Receipts & Invoices',
    ]);
    // Each accepted candidate carries its cluster keyphrases as evidence for the later review UI.
    expect(result.accepted.every((c) => c.evidence.length > 0)).toBe(true);
    // Read-only: no category was created and no run leaves state behind.
    expect(h.categories.listAll(h.accountId)).toHaveLength(0);
    // Local by default: the discovery chat call used the local provider.
    expect(h.captured.provider).toBe('main');
  });

  it('lets the gate reject a vague or abandoned name from the model', async () => {
    const h = harness();
    const answer = JSON.stringify({
      clusters: [
        {
          clusterIndex: 0,
          action: 'new_category',
          label: 'Notifications',
          description: '',
          suggestedKey: '',
        },
        {
          clusterIndex: 1,
          action: 'leave_uncategorized',
          label: '',
          description: '',
          suggestedKey: '',
        },
      ],
    });
    const svc = service(h, answer);

    const result = await svc.propose(h.accountId, MODEL, 'qwen');

    expect(result.accepted).toHaveLength(0);
    const reasons = result.rejected.map((r) => r.reason).sort();
    expect(reasons).toContain('vague_label');
    expect(reasons).toContain('model_left_uncategorized');
  });

  it('returns an empty result when there is no residual to cluster', async () => {
    const h = harness();
    // Mark every email user-assigned to a category so nothing is residual.
    const cat = h.categories.create({ accountId: h.accountId, label: 'Filed', source: 'user' });
    for (const b of ['inv', 'fly']) {
      for (let i = 0; i < 6; i++) {
        h.categories.replaceEmailAssignments(`${b}-${i}`, h.accountId, [
          {
            messageId: `${b}-${i}`,
            accountId: h.accountId,
            categoryId: cat.id,
            confidence: 1,
            assignedBy: 'user',
            assignedAt: 1,
            method: null,
          },
        ]);
      }
    }
    const svc = service(h, '{"clusters":[]}');

    const result = await svc.propose(h.accountId, MODEL, 'qwen');
    expect(result).toMatchObject({ clusterCount: 0, accepted: [], rejected: [] });
  });
});
