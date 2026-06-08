/**
 * Core HTTP server entry point. Builds the application context, wires up Fastify
 * with auth, pairing, health checks, and registers all API route groups.
 */
import Fastify from 'fastify';
import cors from '@fastify/cors';
import { CORE_SERVER_PORT, CORE_SERVER_HOST, API_PREFIX } from '@ai-mailpilot/shared';
import { buildContext } from './context.js';
import { EMBEDDING_DIM } from './db/schema.js';
import { registerAccountRoutes } from './routes/accounts.js';
import { registerEmailRoutes } from './routes/emails.js';
import { registerConfigRoutes } from './routes/config.js';
import { registerEmbedRoutes } from './routes/embed.js';
import { registerTriageRoutes } from './routes/triage.js';
import { registerCategoryRoutes } from './routes/categories.js';
import { registerDashboardRoutes } from './routes/dashboard.js';
import { registerChatRoutes } from './routes/chat.js';
import { registerAttachmentRoutes } from './routes/attachments.js';
import { registerEmailAssistantRoutes } from './routes/email-assistant.js';
import { Pairing } from './pairing.js';

const ctx = buildContext();

const pairing = ctx.config.authToken ? new Pairing(ctx.config.authToken, ctx.logger) : null;

/**
 * Probes the configured embedding model once at startup, logging a warning if it
 * is unreachable or an error if it returns a vector of the wrong dimension.
 */
async function preflightEmbeddingModel(): Promise<void> {
  try {
    const vec = await ctx.llm.embed('preflight', ctx.config.llm.embeddingModel);
    if (vec.length !== EMBEDDING_DIM) {
      ctx.logger.error(
        { configured: ctx.config.llm.embeddingModel, expected: EMBEDDING_DIM, actual: vec.length },
        'embedding model returns wrong dimension; storage will reject inserts',
      );
    } else {
      ctx.logger.info(
        { model: ctx.config.llm.embeddingModel, dim: vec.length },
        'embedding model preflight ok',
      );
    }
  } catch (err) {
    ctx.logger.warn({ err }, 'embedding preflight skipped (LLM not reachable)');
  }
}
void preflightEmbeddingModel();

const server = Fastify({
  loggerInstance: ctx.logger,
  bodyLimit: 16 * 1024 * 1024,
});

server.addContentTypeParser('application/json', { parseAs: 'string' }, (_req, body, done) => {
  const text = (body as string).trim();
  if (text === '') {
    done(null, undefined);
    return;
  }
  try {
    done(null, JSON.parse(text));
  } catch (err) {
    (err as { statusCode?: number }).statusCode = 400;
    done(err as Error, undefined);
  }
});

await server.register(cors, { origin: true });

server.addHook('onRequest', async (req, reply) => {
  const path = req.url.split('?')[0];
  if (path === `${API_PREFIX}/health` || path === `${API_PREFIX}/pair`) return;

  const expected = ctx.config.authToken;
  if (!expected) {
    return reply.code(503).send({ error: 'server auth not configured' });
  }
  if (req.headers['authorization'] !== `Bearer ${expected}`) {
    return reply.code(401).send({ error: 'unauthorized' });
  }
});

server.post(`${API_PREFIX}/pair`, async (req, reply) => {
  if (!pairing) return reply.code(503).send({ error: 'server auth not configured' });
  const body = req.body as { code?: unknown } | undefined;
  const code = typeof body?.code === 'string' ? body.code.trim() : '';
  if (!code) return reply.code(400).send({ error: 'pairing code required' });
  const result = pairing.redeem(code);
  if ('error' in result) return reply.code(result.status).send({ error: result.error });
  ctx.logger.info('extension paired successfully');
  return { token: result.token };
});

server.get(`${API_PREFIX}/health`, async () => {
  const llmHealth = await ctx.llm.health();
  return {
    status: 'ok' as const,
    version: '0.1.0',
    llm: {
      connected: llmHealth.ok,
      models: llmHealth.models,
      baseUrl: ctx.config.llm.baseUrl,
    },
    locale: ctx.config.locale,
  };
});

await server.register(
  async (scoped) => {
    await registerAccountRoutes(scoped, ctx);
    await registerEmailRoutes(scoped, ctx);
    await registerConfigRoutes(scoped, ctx);
    await registerEmbedRoutes(scoped, ctx);
    await registerTriageRoutes(scoped, ctx);
    await registerCategoryRoutes(scoped, ctx);
    await registerDashboardRoutes(scoped, ctx);
    await registerChatRoutes(scoped, ctx);
    await registerAttachmentRoutes(scoped, ctx);
    await registerEmailAssistantRoutes(scoped, ctx);
  },
  { prefix: API_PREFIX },
);

try {
  await server.listen({ port: CORE_SERVER_PORT, host: CORE_SERVER_HOST });
} catch (err) {
  ctx.logger.error({ err }, 'failed to start server');
  process.exit(1);
}
