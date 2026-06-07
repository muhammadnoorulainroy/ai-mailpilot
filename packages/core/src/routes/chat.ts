/**
 * Fastify HTTP routes for chat and conversation management, including streaming
 * answers, retrieval cap resolution per provider, and conversation listing and deletion.
 */
import { Readable } from 'node:stream';
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';
import type { ChatSource, ChatStreamEvent } from '../services/chat-service.js';
import { resolveActiveChatModel } from '../util/chat-model.js';

const ChatBody = z.object({
  accountId: z.string().min(1),
  question: z.string().trim().min(1).max(2000),
  conversationId: z.string().min(1).nullish(),
});

const ListQuery = z.object({ accountId: z.string().min(1) });

const CLOUD_TOP_K = 30;
const CLOUD_SNIPPET = 2000;
const CLOUD_ANSWER_TOKENS = 4000;
const LOCAL_TOP_K_CAP = 10;
const LOCAL_SNIPPET_CAP = 900;

/**
 * Resolve the retrieval caps for a chat call. Cloud keeps the large budget, local clamps a stored
 * value down to a window-safe bound or leaves it unset so the service defaults apply.
 */
export function resolveChatRetrievalCaps(
  cloud: boolean,
  chatTopK: number | null | undefined,
  chatSnippetChars: number | null | undefined,
): { topK?: number; snippetChars?: number } {
  if (cloud) {
    return { topK: chatTopK ?? CLOUD_TOP_K, snippetChars: chatSnippetChars ?? CLOUD_SNIPPET };
  }
  return {
    topK: chatTopK != null ? Math.min(chatTopK, LOCAL_TOP_K_CAP) : undefined,
    snippetChars:
      chatSnippetChars != null ? Math.min(chatSnippetChars, LOCAL_SNIPPET_CAP) : undefined,
  };
}

/**
 * Register the chat and conversation HTTP routes on the Fastify instance.
 */
export async function registerChatRoutes(app: FastifyInstance, ctx: AppContext): Promise<void> {
  /**
   * Build the per-request chat parameters by resolving the active model, then
   * choosing retrieval caps, condense model, and answer budget for cloud or local.
   */
  async function params(): Promise<{
    embeddingModelId: string;
    generationModelId: string;
    condenseModelId: string;
    topK?: number;
    snippetChars?: number;
    rerank?: boolean;
    thinking?: boolean;
    answerTokens?: number;
  }> {
    const llm = ctx.config.llm;
    const active = await resolveActiveChatModel(llm, ctx.llm, ctx.logger);
    const cloud = active.provider === 'cloud';
    const answerModel = active.modelId;
    const caps = resolveChatRetrievalCaps(cloud, llm.chatTopK, llm.chatSnippetChars);
    return {
      embeddingModelId: llm.embeddingModel,
      generationModelId: answerModel,
      condenseModelId: cloud ? answerModel : llm.generationModel,
      topK: caps.topK,
      snippetChars: caps.snippetChars,
      rerank: llm.chatRerank,
      thinking: !cloud,
      answerTokens: cloud ? CLOUD_ANSWER_TOKENS : undefined,
    };
  }

  app.post('/chat/stream', async (req, reply) => {
    const parsed = ChatBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    if (!ctx.repos.accounts.findById(parsed.data.accountId)) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    const events = ctx.services.chat.askStream(
      parsed.data.accountId,
      parsed.data.question,
      parsed.data.conversationId ?? null,
      await params(),
    );

    /**
     * Serialize each stream event as an NDJSON line, emitting an error event if the
     * underlying generator throws so the client still receives a terminal message.
     */
    async function* lines(): AsyncGenerator<string> {
      try {
        for await (const event of events) yield JSON.stringify(event) + '\n';
      } catch (err) {
        ctx.logger.error({ err }, 'chat stream failed');
        const error: ChatStreamEvent = {
          type: 'error',
          message: err instanceof Error ? err.message : String(err),
        };
        yield JSON.stringify(error) + '\n';
      }
    }

    reply.header('Content-Type', 'application/x-ndjson');
    reply.header('Cache-Control', 'no-cache');
    reply.header('X-Accel-Buffering', 'no');
    return reply.send(Readable.from(lines()));
  });

  app.post('/chat', async (req, reply) => {
    const parsed = ChatBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    if (!ctx.repos.accounts.findById(parsed.data.accountId)) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    try {
      let answer = '';
      let conversationId = parsed.data.conversationId ?? '';
      let sources: ChatSource[] = [];
      for await (const event of ctx.services.chat.askStream(
        parsed.data.accountId,
        parsed.data.question,
        parsed.data.conversationId ?? null,
        await params(),
      )) {
        if (event.type === 'meta') {
          conversationId = event.conversationId;
          sources = event.sources;
        } else if (event.type === 'delta') {
          answer += event.text;
        }
      }
      return { conversationId, answer: answer.replace(/<\/?think>/gi, '').trim(), sources };
    } catch (err) {
      ctx.logger.error({ err }, 'chat failed');
      reply.code(500).send({
        error: 'chat failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  });

  app.get('/conversations', async (req, reply) => {
    const parsed = ListQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }
    return { conversations: ctx.repos.conversations.listForAccount(parsed.data.accountId) };
  });

  app.get<{ Params: { id: string } }>('/conversations/:id', async (req, reply) => {
    const q = ListQuery.safeParse(req.query);
    if (!q.success) {
      reply.code(400).send({ error: 'accountId required' });
      return;
    }
    const convo = ctx.repos.conversations.getForAccount(req.params.id, q.data.accountId);
    if (!convo) {
      reply.code(404).send({ error: 'conversation not found' });
      return;
    }
    return convo;
  });

  app.delete<{ Params: { id: string } }>('/conversations/:id', async (req, reply) => {
    const q = ListQuery.safeParse(req.query);
    if (!q.success) {
      reply.code(400).send({ error: 'accountId required' });
      return;
    }
    const deleted = ctx.repos.conversations.deleteForAccount(req.params.id, q.data.accountId);
    if (!deleted) {
      reply.code(404).send({ error: 'conversation not found' });
      return;
    }
    reply.code(204).send();
  });
}
