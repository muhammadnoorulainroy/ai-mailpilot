/**
 * Tests for ChatService covering summary-buffer conversation memory, RAG
 * retrieval and reranking, attachment-aware context, temporal and aggregate
 * ranking heuristics, follow-up anchoring, and failure handling on the answer
 * stream. Uses an in-memory database and a fake LLM client.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import pino from 'pino';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { ConversationRepository, type StoredChatTurn } from '../src/repositories/conversation-repository.js';
import { AttachmentRepository } from '../src/repositories/attachment-repository.js';
import { ChatService, type ChatParams } from '../src/services/chat-service.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import type { LlmClient, ChatCompletionOptions } from '../src/llm/client.js';

const logger = pino({ level: 'silent' });
const params: ChatParams = {
  embeddingModelId: 'bge-m3',
  generationModelId: 'qwen3:8b',
  condenseModelId: 'qwen3:4b',
};

/** Returns a fixed unit embedding pointing along the first axis. */
function unitVector(): number[] {
  const v = new Array<number>(EMBEDDING_DIM).fill(0);
  v[0] = 1;
  return v;
}

/**
 * Builds a stub LlmClient that records embed inputs and chat options and
 * returns chatReply for chat calls, with a fixed reasoning-then-answer stream.
 */
function fakeLlm(chatReply = 'canned'): LlmClient & {
  chatCalls: number;
  embedInputs: string[];
  chatOpts: ChatCompletionOptions[];
  streamOpts: ChatCompletionOptions[];
} {
  const client = {
    chatCalls: 0,
    embedInputs: [] as string[],
    chatOpts: [] as ChatCompletionOptions[],
    streamOpts: [] as ChatCompletionOptions[],
    async health() {
      return { ok: true, models: [] as string[] };
    },
    async embed(text: string) {
      client.embedInputs.push(text);
      return unitVector();
    },
    async embedBatch(texts: string[]) {
      return texts.map(() => unitVector());
    },
    async chat(opts: ChatCompletionOptions) {
      client.chatCalls += 1;
      client.chatOpts.push(opts);
      return chatReply;
    },
    async *chatStream(opts: ChatCompletionOptions) {
      client.streamOpts.push(opts);
      yield 'reasoning';
      yield '</think>';
      yield 'Here is ';
      yield 'the answer.';
    },
  };
  return client;
}

describe('ChatService summary-buffer memory', () => {
  let db: Database;
  let accounts: AccountRepository;
  let emails: EmailRepository;
  let embeddings: EmbeddingRepository;
  let conversations: ConversationRepository;
  let attachments: AttachmentRepository;

  beforeEach(() => {
    db = openDatabase(':memory:');
    accounts = new AccountRepository(db);
    emails = new EmailRepository(db);
    embeddings = new EmbeddingRepository(db);
    conversations = new ConversationRepository(db);
    attachments = new AttachmentRepository(db);
  });
  afterEach(() => db.close());

  /** Appends n alternating user and assistant turns to a conversation. */
  const seed = (convoId: string, n: number): void => {
    const now = Date.now();
    const turns: StoredChatTurn[] = Array.from({ length: n }, (_, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `turn ${i}`,
      at: now + i,
    }));
    conversations.append(convoId, turns);
  };

  /** Runs askStream and concatenates the delta text into the final answer. */
  const drain = async (svc: ChatService, acctId: string, q: string, convoId: string): Promise<string> => {
    let answer = '';
    for await (const e of svc.askStream(acctId, q, convoId, params)) {
      if (e.type === 'delta') answer += e.text;
    }
    return answer;
  };

  it('folds overflow turns into a rolling summary and keeps the recent ones', async () => {
    const acct = accounts.create({ address: 'c@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    seed(convo.id, 10);

    const svc = new ChatService(fakeLlm('Rolling summary of earlier turns.'), embeddings, emails, conversations, attachments, logger);
    const answer = await drain(svc, acct.id, 'write a longer reply', convo.id);

    const after = conversations.get(convo.id)!;
    expect(after.summarizedCount).toBe(4);
    expect(after.summary).toBe('Rolling summary of earlier turns.');
    expect(after.turns).toHaveLength(12);
    expect(answer).toContain('answer');
  });

  it('does not summarize a short conversation', async () => {
    const acct = accounts.create({ address: 'd@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    seed(convo.id, 2);

    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);
    await drain(svc, acct.id, 'write a reply', convo.id);

    const after = conversations.get(convo.id)!;
    expect(after.summarizedCount).toBe(0);
    expect(after.summary).toBe('');
  });

  it('does not fold at the hysteresis boundary when unsummarized equals MAX_HISTORY', async () => {
    const acct = accounts.create({ address: 'b@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    seed(convo.id, 8);

    const svc = new ChatService(fakeLlm('should not be used'), embeddings, emails, conversations, attachments, logger);
    await drain(svc, acct.id, 'write a reply', convo.id);

    const after = conversations.get(convo.id)!;
    expect(after.summarizedCount).toBe(0);
    expect(after.summary).toBe('');
  });

  /** Builds an embedding whose second axis varies, to control similarity order. */
  const evVec = (second: number): Float32Array => {
    const v = new Float32Array(EMBEDDING_DIM);
    v[0] = 1;
    v[1] = second;
    return v;
  };

  /** Seeds three embedded INBOX emails with graded similarity for rank tests. */
  function seedRankable(addr: string): string {
    const acct = accounts.create({ address: addr, kind: 'work' });
    emails.upsertBatch([
      { messageId: 'm0', accountId: acct.id, folder: 'INBOX', subject: 'alpha', body: 'aaa' },
      { messageId: 'm1', accountId: acct.id, folder: 'INBOX', subject: 'beta', body: 'bbb' },
      { messageId: 'm2', accountId: acct.id, folder: 'INBOX', subject: 'gamma', body: 'ccc' },
    ]);
    embeddings.saveEmbedding({ messageId: 'm0', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'm1', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.1));
    embeddings.saveEmbedding({ messageId: 'm2', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.2));
    return acct.id;
  }

  /** Runs askStream with topK 2 and returns the cited source message ids. */
  const sourcesFor = async (svc: ChatService, acctId: string, rerank: boolean): Promise<string[]> => {
    let ids: string[] = [];
    for await (const e of svc.askStream(acctId, 'please tell me everything relevant to this topic here', conversations.create(acctId).id, { ...params, topK: 2, rerank })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    return ids;
  };

  it('reorders the pool by the model order when rerank is on', async () => {
    const acctId = seedRankable('rr@x.y');
    const svc = new ChatService(fakeLlm('2, 0'), embeddings, emails, conversations, attachments, logger);
    expect(await sourcesFor(svc, acctId, true)).toEqual(['m2', 'm0']);
  });

  it('falls back to the fusion order when the reranker output is unusable', async () => {
    const acctId = seedRankable('rr2@x.y');
    const svc = new ChatService(fakeLlm('I cannot rank these'), embeddings, emails, conversations, attachments, logger);
    expect(await sourcesFor(svc, acctId, true)).toEqual(['m0', 'm1']);
  });

  it('feeds only the reranker-judged relevant items, not padded back up to topK', async () => {
    const acctId = seedRankable('rr-prune@x.y');
    const svc = new ChatService(fakeLlm('2'), embeddings, emails, conversations, attachments, logger);
    let ids: string[] = [];
    for await (const e of svc.askStream(acctId, 'please tell me everything relevant to this topic here', conversations.create(acctId).id, { ...params, topK: 10, rerank: true })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids[0]).toBe('m2');
    expect(ids).toHaveLength(3);
  });

  it('does not rerank when rerank is off', async () => {
    const acctId = seedRankable('rr3@x.y');
    const svc = new ChatService(fakeLlm('2, 0'), embeddings, emails, conversations, attachments, logger);
    expect(await sourcesFor(svc, acctId, false)).toEqual(['m0', 'm1']);
  });

  it("pulls a retrieved email's attachment text into context even when the chunk did not match", async () => {
    const acctId = seedRankable('attach-expand@x.y');
    const att = attachments.upsertAttachment({
      messageId: 'm0',
      accountId: acctId,
      filename: 'corrected.pdf',
      partName: '1.2',
    });
    attachments.replaceChunks(att.id, 'm0', acctId, ['Noor Ul Ain Mark: 17 out of 20']);
    attachments.setStatus(att.id, 'extracted', 30);

    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);
    let sources: Array<{ messageId: string; attachmentName?: string }> = [];
    for await (const e of svc.askStream(
      acctId,
      'please tell me everything relevant to this topic here',
      conversations.create(acctId).id,
      { ...params, topK: 3, rerank: false },
    )) {
      if (e.type === 'meta') sources = e.sources;
    }
    expect(sources.some((s) => s.messageId === 'm0' && s.attachmentName === 'corrected.pdf')).toBe(
      true,
    );
  });

  it('surfaces a relevant attachment chunk as a cited source, none when unrelated', async () => {
    const acct = accounts.create({ address: 'att@x.y', kind: 'work' });
    emails.upsertBatch([{ messageId: 'm1', accountId: acct.id, folder: 'INBOX', subject: 'Plan', body: 'see attached' }]);
    const { id } = attachments.upsertAttachment({ messageId: 'm1', accountId: acct.id, filename: 'memo.pdf', partName: '1.2' });
    const rowids = attachments.replaceChunks(id, 'm1', acct.id, ['the project codename is BLUEFOX']);
    attachments.saveChunkEmbedding(rowids[0]!, acct.id, 'bge-m3', unitVector());
    attachments.setStatus(id, 'extracted', 30);

    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let sources: { attachmentName?: string }[] = [];
    for await (const ev of svc.askStream(acct.id, 'what is the project codename', conversations.create(acct.id).id, params)) {
      if (ev.type === 'meta') sources = ev.sources;
    }
    expect(sources.some((s) => s.attachmentName === 'memo.pdf')).toBe(true);
  });

  const localParams: ChatParams = { ...params, thinking: true };
  /** Consumes the full askStream without inspecting events, to drive side effects. */
  const drainAll = async (svc: ChatService, acctId: string, q: string, convoId: string, p = localParams): Promise<void> => {
    for await (const _ of svc.askStream(acctId, q, convoId, p)) {
    }
  };

  /** Seeds an old and a new home insurance contract email, each with dated chunks. */
  const seedAdhContracts = (addr: string): string => {
    const acct = accounts.create({ address: addr, kind: 'work' });
    const day = 86_400_000;
    const nowMs = Date.now();
    emails.upsertBatch([
      { messageId: 'old', accountId: acct.id, folder: 'INBOX', subject: 'ADH 2025', body: 'voir la piece jointe', date: nowMs - 300 * day },
      { messageId: 'new', accountId: acct.id, folder: 'INBOX', subject: 'ADH 2026', body: 'voir la piece jointe', date: nowMs - 5 * day },
    ]);
    embeddings.saveEmbedding({ messageId: 'old', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'new', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.05));
    const a1 = attachments.upsertAttachment({ messageId: 'old', accountId: acct.id, filename: 'Contrat Habitation ADHE-20250808-2731.pdf', partName: '1.2' });
    attachments.replaceChunks(a1.id, 'old', acct.id, [
      'Informations generales sur le contrat et les garanties incluses au dossier.',
      'La presente attestation est valable pour la periode comprise entre le 08/08/2025 et le 31/08/2026.',
    ]);
    attachments.setStatus(a1.id, 'extracted', 200);
    const a2 = attachments.upsertAttachment({ messageId: 'new', accountId: acct.id, filename: 'Contrat Habitation ADHE-20260603-bba7c844.pdf', partName: '1.2' });
    attachments.replaceChunks(a2.id, 'new', acct.id, [
      'Informations generales sur le contrat et les garanties incluses au dossier.',
      'La presente attestation est valable pour la periode comprise entre le 03/06/2026 et le 31/08/2027.',
    ]);
    attachments.setStatus(a2.id, 'extracted', 200);
    return acct.id;
  };

  /** Asks a question and returns the joined message content sent to the generation stream. */
  const genContextFor = async (acctId: string, question: string): Promise<string> => {
    const llm = fakeLlm();
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);
    for await (const e of svc.askStream(acctId, question, conversations.create(acctId).id, { ...params, topK: 8, rerank: false })) {
      void e;
    }
    return (llm.streamOpts.at(-1)?.messages ?? []).map((m) => m.content).join('\n');
  };

  it('surfaces the current contract date chunk for a dates question, not a non-date chunk', async () => {
    const acctId = seedAdhContracts('adh-cur@x.y');
    const ctx = await genContextFor(acctId, 'what are the start and end dates of my current habitation insurance contract?');
    expect(ctx).toContain('03/06/2026');
    expect(ctx).toContain('31/08/2027');
    expect(ctx).not.toContain('08/08/2025');
  });

  it('surfaces the PREVIOUS contract date chunk when the user asks for the previous one', async () => {
    const acctId = seedAdhContracts('adh-prev@x.y');
    const ctx = await genContextFor(acctId, 'what are the dates of my previous habitation insurance contract?');
    expect(ctx).toContain('08/08/2025');
    expect(ctx).not.toContain('03/06/2026');
  });

  it('feeds the FULL date-range chunk, not a truncated one, when the period is split across chunks', async () => {
    const acct = accounts.create({ address: 'adhsplit@x.y', kind: 'work' });
    emails.upsertBatch([{ messageId: 'm', accountId: acct.id, folder: 'INBOX', subject: 'Assurance habitation', body: 'voir la piece jointe', date: Date.now() }]);
    embeddings.saveEmbedding({ messageId: 'm', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    const a = attachments.upsertAttachment({ messageId: 'm', accountId: acct.id, filename: 'Contrat Habitation ADHE-20260603.pdf', partName: '1.2' });
    attachments.replaceChunks(a.id, 'm', acct.id, [
      'La presente attestation est valable pour la periode comprise entre le 01/01/2026 et le 31/08',
      'Le present contrat valable, periode comprise entre les parties, signe le 02/02/2026 et le 31/08',
      'Attestation valable, periode comprise entre la souscription le 04/04/2026 et le 31/08',
      'DUREE DU CONTRAT les garanties sont accordees pour la periode du 03/06/2026 au 31/08/2027 sans tacite',
    ]);
    attachments.setStatus(a.id, 'extracted', 400);
    const ctx = await genContextFor(acct.id, 'what are the start and end dates of my current habitation insurance contract?');
    expect(ctx).toContain('31/08/2027');
  });

  it('retrieves an English-titled document for a French query via bilingual expansion', async () => {
    const acct = accounts.create({ address: 'biling@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'es', accountId: acct.id, folder: 'INBOX', subject: 'Technical Foundations - Evaluation Summary', body: 'MCQ average 18.5. ABS means unjustified absence counted as 0.' },
      { messageId: 'other', accountId: acct.id, folder: 'INBOX', subject: 'random newsletter', body: 'unrelated marketing content here' },
    ]);
    embeddings.saveEmbedding({ messageId: 'other', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'es', accountId: acct.id, modelId: 'bge-m3' }, evVec(1));
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'Dans le resume d evaluation TFSD, que signifie ABS ?', conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids).toContain('es');
  });

  it('does not anchor retrieval to the prior source on a correction follow-up', async () => {
    const acct = accounts.create({ address: 'corr@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    conversations.append(convo.id, [
      { role: 'user', content: 'what is the latest insurance doc?', at: Date.now() },
      { role: 'assistant', content: 'Health insurance info.', at: Date.now(), sources: [{ messageId: 'health', subject: 'Health insurance info', fromAddr: 'cps2@x.y', date: Date.now(), score: 0.9 }] },
    ]);
    const llm = fakeLlm('ADH home insurance contract');
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    await drainAll(svc, acct.id, 'no, i was talking about ADH home insurance', convo.id, params);

    expect(llm.embedInputs.some((t) => /Health insurance/i.test(t))).toBe(false);
  });

  it('appends /no_think and falls back to the original question on a truncated condense reasoning blob', async () => {
    const acct = accounts.create({ address: 'cond@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    conversations.append(convo.id, [
      { role: 'user', content: 'What was my TODO average?', at: Date.now() },
      { role: 'assistant', content: 'It was 20/20.', at: Date.now() },
    ]);
    const llm = fakeLlm('<think>\nOkay, the user is asking about the ABS average, let me find which email');
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    await drainAll(svc, acct.id, 'and the ABS one?', convo.id);

    expect(llm.chatOpts.some((o) => /\/no_think$/.test(o.messages.at(-1)?.content ?? ''))).toBe(true);
    expect(llm.embedInputs).toContain('and the ABS one?');
    expect(llm.embedInputs.some((t) => /Okay, the user/.test(t))).toBe(false);
  });

  it('keeps the prior summary when the fold returns a truncated reasoning blob', async () => {
    const acct = accounts.create({ address: 'sum@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    seed(convo.id, 10);
    conversations.updateSummary(convo.id, 'Good prior summary.', 0);
    const llm = fakeLlm('<think>\nLet me summarize the discussion about deadlines and the viva and');
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    await drainAll(svc, acct.id, 'write a longer reply', convo.id);

    const after = conversations.get(convo.id)!;
    expect(after.summary).toBe('Good prior summary.');
    expect(after.summarizedCount).toBe(0);
  });

  it('does not leave an empty conversation when the answer stream fails', async () => {
    const acct = accounts.create({ address: 'fail@x.y', kind: 'work' });
    const llm = fakeLlm();
    llm.chatStream = async function* () {
      throw new Error('llm timeout');
    };
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    let convoId = '';
    let threw = false;
    try {
      for await (const e of svc.askStream(acct.id, 'q', null, params)) {
        if (e.type === 'meta') convoId = e.conversationId;
      }
    } catch {
      threw = true;
    }
    expect(threw).toBe(true);
    expect(convoId).not.toBe('');
    expect(conversations.get(convoId)).toBeNull();
    expect(conversations.listForAccount(acct.id)).toHaveLength(0);
  });

  it('keeps an existing conversation when a continued turn fails', async () => {
    const acct = accounts.create({ address: 'keep@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    conversations.append(convo.id, [{ role: 'user', content: 'hi', at: Date.now() }]);
    const llm = fakeLlm();
    llm.chatStream = async function* () {
      throw new Error('boom');
    };
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    try {
      for await (const _ of svc.askStream(acct.id, 'q2', convo.id, params)) {
      }
    } catch {
    }
    const after = conversations.get(convo.id);
    expect(after).not.toBeNull();
    expect(after!.turns).toHaveLength(1);
  });

  it('ranks a time+topic query by relevance, not letting newest-but-irrelevant mail crowd it out', async () => {
    const acct = accounts.create({ address: 'tt@x.y', kind: 'work' });
    const day = 86_400_000;
    const nowMs = Date.now();
    emails.upsertBatch([
      { messageId: 'cps2-1', accountId: acct.id, folder: 'INBOX', subject: 'CPS2 defense schedule', body: 'CPS2 defense', date: nowMs - 5 * day },
      { messageId: 'cps2-2', accountId: acct.id, folder: 'INBOX', subject: 'CPS2 defense rooms', body: 'CPS2 defense', date: nowMs - 8 * day },
      { messageId: 'yt-0', accountId: acct.id, folder: 'INBOX', subject: 'YouTube digest', body: 'videos', date: nowMs - 1 * day },
      { messageId: 'yt-1', accountId: acct.id, folder: 'INBOX', subject: 'GitHub notice', body: 'pull request', date: nowMs - 1 * day },
      { messageId: 'yt-2', accountId: acct.id, folder: 'INBOX', subject: 'Sale newsletter', body: 'discount', date: nowMs - 2 * day },
    ]);
    embeddings.saveEmbedding({ messageId: 'cps2-1', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'cps2-2', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.05));
    embeddings.saveEmbedding({ messageId: 'yt-0', accountId: acct.id, modelId: 'bge-m3' }, evVec(2));
    embeddings.saveEmbedding({ messageId: 'yt-1', accountId: acct.id, modelId: 'bge-m3' }, evVec(2.1));
    embeddings.saveEmbedding({ messageId: 'yt-2', accountId: acct.id, modelId: 'bge-m3' }, evVec(2.2));

    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);
    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'Summarize my recent emails about CPS2 defenses', conversations.create(acct.id).id, { ...params, topK: 3, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids.slice(0, 2).sort()).toEqual(['cps2-1', 'cps2-2']);
  });

  it('ranks a pure time query by recency, not by similarity to the filler query', async () => {
    const acct = accounts.create({ address: 'pt@x.y', kind: 'work' });
    const day = 86_400_000;
    const nowMs = Date.now();
    emails.upsertBatch([
      { messageId: 'old', accountId: acct.id, folder: 'INBOX', subject: 'older', body: 'x', date: nowMs - 10 * day },
      { messageId: 'mid', accountId: acct.id, folder: 'INBOX', subject: 'middle', body: 'y', date: nowMs - 5 * day },
      { messageId: 'new', accountId: acct.id, folder: 'INBOX', subject: 'newest', body: 'z', date: nowMs - 1 * day },
    ]);
    embeddings.saveEmbedding({ messageId: 'old', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'mid', accountId: acct.id, modelId: 'bge-m3' }, evVec(1));
    embeddings.saveEmbedding({ messageId: 'new', accountId: acct.id, modelId: 'bge-m3' }, evVec(2));

    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);
    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'show my recent emails', conversations.create(acct.id).id, { ...params, topK: 3, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids).toEqual(['new', 'mid', 'old']);
  });

  it('retrieves a relevant in-window email older than the newest-N, the recall hole fix', async () => {
    const acct = accounts.create({ address: 'recall@x.y', kind: 'work' });
    const day = 86_400_000;
    const nowMs = Date.now();
    const batch = Array.from({ length: 50 }, (_, i) => ({
      messageId: `f${i}`,
      accountId: acct.id,
      folder: 'INBOX',
      subject: 'filler',
      body: 'noise',
      date: nowMs - 1 * day,
    }));
    batch.push({ messageId: 'rel', accountId: acct.id, folder: 'INBOX', subject: 'CPS2 defense', body: 'CPS2 defense', date: nowMs - 20 * day });
    emails.upsertBatch(batch);
    for (let i = 0; i < 50; i++) {
      embeddings.saveEmbedding({ messageId: `f${i}`, accountId: acct.id, modelId: 'bge-m3' }, evVec(2));
    }
    embeddings.saveEmbedding({ messageId: 'rel', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));

    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);
    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'recent emails about CPS2 defenses', conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids).toContain('rel');
  });

  it('answers the user original question; the condensed form is used only for retrieval', async () => {
    const acct = accounts.create({ address: 'orig@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    conversations.append(convo.id, [
      { role: 'user', content: 'In TFSD MCQ Test 2, how much did I score?', at: Date.now() },
      { role: 'assistant', content: '17.', at: Date.now() },
    ]);
    const llm = fakeLlm('What was my average MCQ score in TFSD MCQ Test 2?');
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    await drainAll(svc, acct.id, 'What was my average MCQ?', convo.id);

    expect(llm.embedInputs.some((t) => /Test 2/.test(t))).toBe(true);
    const genMsg = (llm.streamOpts.at(-1)?.messages ?? []).map((m) => m.content).join('\n');
    expect(genMsg).toContain('What was my average MCQ?');
    expect(genMsg).not.toContain('Test 2');
  });

  it('promotes the evaluation-summary email to the front for an aggregate question', async () => {
    const acct = accounts.create({ address: 'agg@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'mcq2', accountId: acct.id, folder: 'INBOX', subject: 'TFSD Lecture 2 MCQ test results', body: 'average 14.32' },
      { messageId: 'summary', accountId: acct.id, folder: 'INBOX', subject: 'TFSD - Evaluation Summary', body: 'Your evaluation results are as follows: average MCQ 18.5, TODO 20' },
    ]);
    embeddings.saveEmbedding({ messageId: 'mcq2', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'summary', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.3));
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'What was my overall average across all MCQ tests?', conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids[0]).toBe('summary');
  });

  it('does not promote a summary email for a specific non-aggregate question', async () => {
    const acct = accounts.create({ address: 'spec@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'mcq2', accountId: acct.id, folder: 'INBOX', subject: 'TFSD Lecture 2 MCQ test results', body: 'your mark 17' },
      { messageId: 'summary', accountId: acct.id, folder: 'INBOX', subject: 'TFSD - Evaluation Summary', body: 'evaluation results as follows: average MCQ 18.5' },
    ]);
    embeddings.saveEmbedding({ messageId: 'mcq2', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'summary', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.3));
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'In TFSD Lecture 2 MCQ test how many marks did I get on it?', conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids[0]).toBe('mcq2');
  });

  it('does not falsely promote an individual email that merely mentions "final grade" in prose', async () => {
    const acct = accounts.create({ address: 'fp@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'close', accountId: acct.id, folder: 'INBOX', subject: 'Lecture 3 result', body: 'your mark 15' },
      { messageId: 'fg', accountId: acct.id, folder: 'INBOX', subject: 'TFSD final exam', body: 'your final grade contribution from this exam is 12' },
    ]);
    embeddings.saveEmbedding({ messageId: 'close', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'fg', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.5));
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'what was my cumulative result across the whole course?', conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids[0]).toBe('close');
  });

  it('rejects a tagless reasoning/prompt leak in the summary fold and keeps the prior summary', async () => {
    const acct = accounts.create({ address: 'leak@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    seed(convo.id, 10);
    conversations.updateSummary(convo.id, 'Good prior summary.', 0);
    const llm = fakeLlm('We are updating the summary with the new exchanges. New exchanges: the user asked about grades.');
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    await drainAll(svc, acct.id, 'write a longer reply', convo.id);

    const after = conversations.get(convo.id)!;
    expect(after.summary).toBe('Good prior summary.');
    expect(after.summarizedCount).toBe(0);
  });

  it('answers from a named filename-targeted document, not a higher-ranked unrelated email', async () => {
    const acct = accounts.create({ address: 'fn@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'doc', accountId: acct.id, folder: 'INBOX', subject: 'Stage a l etranger', body: 'voir la piece jointe' },
      { messageId: 'noise', accountId: acct.id, folder: 'INBOX', subject: 'unrelated', body: 'closest embedding but irrelevant' },
    ]);
    embeddings.saveEmbedding({ messageId: 'noise', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'doc', accountId: acct.id, modelId: 'bge-m3' }, evVec(2));
    const att = attachments.upsertAttachment({ messageId: 'doc', accountId: acct.id, filename: 'StagesEtranger-1.pdf', partName: '1.2' });
    attachments.replaceChunks(att.id, 'doc', acct.id, [
      'Informations generales sur le stage a l etranger et les demarches.',
      "Ordre de signature de la convention de stage: 1 Etudiant, 2 Organisme d'accueil, 3 Responsable de la formation, 4 Directeur.",
    ]);
    attachments.setStatus(att.id, 'extracted', 200);
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let sources: Array<{ messageId: string; attachmentName?: string }> = [];
    for await (const e of svc.askStream(acct.id, "Dans le fichier StagesEtranger-1.pdf, quel est l'ordre exact de signature de la convention ?", conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') sources = e.sources;
    }
    expect(sources[0]?.attachmentName).toBe('StagesEtranger-1.pdf');
    expect(sources[0] && 'attachmentName' in sources[0]).toBe(true);
    expect(sources.some((s) => s.messageId === 'noise')).toBe(false);
  });

  it('anchors a French follow-up retrieval to the previous answer source', async () => {
    const acct = accounts.create({ address: 'anchor@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    conversations.append(convo.id, [
      { role: 'user', content: 'Avant quelle date completer le Dossier Social Etudiant ?', at: Date.now() },
      {
        role: 'assistant',
        content: 'Avant le 31 mai 2026.',
        at: Date.now(),
        sources: [{ messageId: 'crous', subject: 'Communication Ecole pour CROUS 2026-2027', fromAddr: 'ecole@x.y', date: Date.now(), score: 0.9 }],
      },
    ]);
    const llm = fakeLlm('A qui envoyer la notification conditionnelle ?');
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    await drainAll(svc, acct.id, 'Et a qui dois-je envoyer la notification conditionnelle ?', convo.id, params);

    expect(llm.embedInputs.some((t) => /Communication Ecole pour CROUS/.test(t))).toBe(true);
  });

  it('breaks a filename tie by the most recent email across two contract versions', async () => {
    const acct = accounts.create({ address: 'tie@x.y', kind: 'work' });
    const day = 86_400_000;
    const nowMs = Date.now();
    emails.upsertBatch([
      { messageId: 'old', accountId: acct.id, folder: 'INBOX', subject: 'old contract', body: 'see attached', date: nowMs - 300 * day },
      { messageId: 'new', accountId: acct.id, folder: 'INBOX', subject: 'new contract', body: 'see attached', date: nowMs - 5 * day },
    ]);
    const a1 = attachments.upsertAttachment({ messageId: 'old', accountId: acct.id, filename: 'Contrat Habitation ADHE-2025.pdf', partName: '1.2' });
    attachments.replaceChunks(a1.id, 'old', acct.id, ['valable entre 2025 et 2026']);
    attachments.setStatus(a1.id, 'extracted', 30);
    const a2 = attachments.upsertAttachment({ messageId: 'new', accountId: acct.id, filename: 'Contrat Habitation ADHE-2026.pdf', partName: '1.2' });
    attachments.replaceChunks(a2.id, 'new', acct.id, ['valable entre 2026 et 2027']);
    attachments.setStatus(a2.id, 'extracted', 30);
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'dans le document contrat habitation, quelle est la periode de validite ?', conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids.length).toBeGreaterThan(0);
    expect(ids.every((id) => id === 'new')).toBe(true);
  });

  it('feeds only the named attachment chunks, not sibling attachments of the same email', async () => {
    const acct = accounts.create({ address: 'multi@x.y', kind: 'work' });
    emails.upsertBatch([{ messageId: 'm', accountId: acct.id, folder: 'INBOX', subject: 'insurance', body: 'see attached' }]);
    const a1 = attachments.upsertAttachment({ messageId: 'm', accountId: acct.id, filename: 'Attestation de Droits.pdf', partName: '1.2' });
    attachments.replaceChunks(a1.id, 'm', acct.id, ['Je m inscris depuis le site etudiant-etranger.ameli.fr']);
    attachments.setStatus(a1.id, 'extracted', 30);
    const a2 = attachments.upsertAttachment({ messageId: 'm', accountId: acct.id, filename: 'VALIDATION VLS-TS.pdf', partName: '1.3' });
    attachments.replaceChunks(a2.id, 'm', acct.id, ['Validation visa long sejour, sibling content only']);
    attachments.setStatus(a2.id, 'extracted', 30);
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let names: Array<string | undefined> = [];
    for await (const e of svc.askStream(acct.id, "Dans le document Attestation de Droits, sur quel site faut-il s'inscrire ?", conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') names = e.sources.map((s) => s.attachmentName);
    }
    expect(names).toContain('Attestation de Droits.pdf');
    expect(names).not.toContain('VALIDATION VLS-TS.pdf');
  });

  it('deterministically force-includes the prior source document on a follow-up the new search misses', async () => {
    const acct = accounts.create({ address: 'fi@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'crous', accountId: acct.id, folder: 'INBOX', subject: 'CROUS', body: 'Contactez carole.claudinon@mines-stetienne.fr.' },
      { messageId: 'noise', accountId: acct.id, folder: 'INBOX', subject: 'unrelated', body: 'random closest content' },
    ]);
    embeddings.saveEmbedding({ messageId: 'noise', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    const convo = conversations.create(acct.id);
    conversations.append(convo.id, [
      { role: 'user', content: 'avant quelle date ?', at: Date.now() },
      { role: 'assistant', content: 'avant le 31 mai 2026', at: Date.now(), sources: [{ messageId: 'crous', subject: null, fromAddr: 'e@x.y', date: Date.now(), score: 0.9 }] },
    ]);
    const llm = fakeLlm('a qui dois je transmettre le dossier ?');
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'Et a qui dois-je transmettre le dossier ?', convo.id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    expect(ids).toContain('crous');
  });

  it('does not hijack into one file when several distinct documents share a token, so it goes hybrid', async () => {
    const acct = accounts.create({ address: 'amb@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'a', accountId: acct.id, folder: 'INBOX', subject: 'offer', body: 'see attached' },
      { messageId: 'b', accountId: acct.id, folder: 'INBOX', subject: 'convention', body: 'see attached' },
    ]);
    embeddings.saveEmbedding({ messageId: 'a', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'b', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.05));
    const aa = attachments.upsertAttachment({ messageId: 'a', accountId: acct.id, filename: 'internship offer TechCorp.pdf', partName: '1.2' });
    attachments.replaceChunks(aa.id, 'a', acct.id, ['offer details for the internship at TechCorp']);
    attachments.setStatus(aa.id, 'extracted', 30);
    const bb = attachments.upsertAttachment({ messageId: 'b', accountId: acct.id, filename: 'internship report MinesStE.pdf', partName: '1.2' });
    attachments.replaceChunks(bb.id, 'b', acct.id, ['report content for the internship at Mines']);
    attachments.setStatus(bb.id, 'extracted', 30);
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let ids: string[] = [];
    for await (const e of svc.askStream(acct.id, 'que dit le document internship sur les dates exactes ?', conversations.create(acct.id).id, { ...params, topK: 5, rerank: false })) {
      if (e.type === 'meta') ids = e.sources.map((s) => s.messageId);
    }
    const seen = new Set(ids);
    expect(seen.has('a') && seen.has('b')).toBe(true);
  });

  it('accepts a legitimate summary that mentions "previous summary" mid-sentence', async () => {
    const acct = accounts.create({ address: 'sumok@x.y', kind: 'work' });
    const convo = conversations.create(acct.id);
    seed(convo.id, 10);
    const llm = fakeLlm('The user asked Marie about the previous summary report and noted the May 31 deadline.');
    const svc = new ChatService(llm, embeddings, emails, conversations, attachments, logger);

    await drainAll(svc, acct.id, 'write a longer reply', convo.id);

    const after = conversations.get(convo.id)!;
    expect(after.summary).toContain('previous summary report');
    expect(after.summarizedCount).toBe(4);
  });

  it('feeds only the current contract version when an old and a new one compete, temporal dedupe', async () => {
    const acct = accounts.create({ address: 'adh@x.y', kind: 'work' });
    const day = 86_400_000;
    const nowMs = Date.now();
    emails.upsertBatch([
      { messageId: 'old', accountId: acct.id, folder: 'INBOX', subject: 'contract 2025', body: 'see attached', date: nowMs - 300 * day },
      { messageId: 'new', accountId: acct.id, folder: 'INBOX', subject: 'contract 2026', body: 'see attached', date: nowMs - 5 * day },
    ]);
    embeddings.saveEmbedding({ messageId: 'old', accountId: acct.id, modelId: 'bge-m3' }, evVec(0));
    embeddings.saveEmbedding({ messageId: 'new', accountId: acct.id, modelId: 'bge-m3' }, evVec(0.05));
    const a1 = attachments.upsertAttachment({ messageId: 'old', accountId: acct.id, filename: 'Contrat Habitation ADHE-20250808-2731.pdf', partName: '1.2' });
    attachments.replaceChunks(a1.id, 'old', acct.id, ['valable du 08/08/2025 au 31/08/2026']);
    attachments.setStatus(a1.id, 'extracted', 30);
    const a2 = attachments.upsertAttachment({ messageId: 'new', accountId: acct.id, filename: 'Contrat Habitation ADHE-20260603-bba7c844.pdf', partName: '1.2' });
    attachments.replaceChunks(a2.id, 'new', acct.id, ['valable du 03/06/2026 au 31/08/2027']);
    attachments.setStatus(a2.id, 'extracted', 30);
    const svc = new ChatService(fakeLlm(), embeddings, emails, conversations, attachments, logger);

    let names: Array<string | undefined> = [];
    for await (const e of svc.askStream(acct.id, 'what are the dates of my current habitation insurance contract?', conversations.create(acct.id).id, { ...params, topK: 8, rerank: false })) {
      if (e.type === 'meta') names = e.sources.map((s) => s.attachmentName).filter(Boolean);
    }
    expect(names).toContain('Contrat Habitation ADHE-20260603-bba7c844.pdf');
    expect(names).not.toContain('Contrat Habitation ADHE-20250808-2731.pdf');
  });
});
