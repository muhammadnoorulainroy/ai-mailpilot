/**
 * Regression test for the discovery isolation invariant (Phase 2c): the LLM categorize pass must
 * never offer a suggested (or retired) category as an assignment target. With only a suggested
 * category present, a run has no candidates and does not start, so no mail can be filed into an
 * unapproved category. Before the fix the candidate list came from listForAccount (all statuses).
 */
import { describe, it, expect } from 'vitest';
import type { Logger } from 'pino';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { CategorizeJobRepository } from '../src/repositories/categorize-job-repository.js';
import { LlmCategorizer } from '../src/services/llm-categorizer.js';
import { LlmCategorizeOrchestrator } from '../src/services/llm-categorize-orchestrator.js';
import type { LlmClient } from '../src/llm/client.js';

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

const fakeLlm = {
  async chat() {
    return '{}';
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

describe('LLM categorize candidate isolation', () => {
  it('does not start a run when the only categories are suggested (not active)', () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const emails = new EmailRepository(db);
    const embeddings = new EmbeddingRepository(db);
    const categories = new CategoryRepository(db);
    const jobs = new CategorizeJobRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });

    // Uncategorized mail with embeddings, so there is work to categorize.
    emails.upsertBatch([
      { messageId: 'm1', accountId: acc.id, folder: 'INBOX', subject: 'one' },
      { messageId: 'm2', accountId: acc.id, folder: 'INBOX', subject: 'two' },
    ]);
    embeddings.saveEmbedding({ messageId: 'm1', accountId: acc.id, modelId: 'bge-m3' }, axis(0));
    embeddings.saveEmbedding({ messageId: 'm2', accountId: acc.id, modelId: 'bge-m3' }, axis(1));

    // The only category is a suggested discovery proposal awaiting review.
    categories.create({
      accountId: acc.id,
      label: 'Receipts',
      source: 'auto',
      status: 'suggested',
    });

    const orch = new LlmCategorizeOrchestrator(
      new LlmCategorizer(fakeLlm),
      emails,
      embeddings,
      categories,
      jobs,
      silentLogger,
    );
    const result = orch.start(acc.id, 'qwen', 'bge-m3');

    // No active candidate exists, so the run does not begin and no mail is filed into the suggestion.
    expect(result.started).toBe(false);
    expect(categories.getAssignedMessageIds(acc.id).size).toBe(0);
    db.close();
  });
});
