import Fastify from 'fastify';
import cors from '@fastify/cors';
import { CORE_SERVER_PORT, CORE_SERVER_HOST, API_PREFIX } from '@ai-mailpilot/shared';
import { buildContext } from './context.js';

const ctx = buildContext();

const server = Fastify({
  loggerInstance: ctx.logger,
});

await server.register(cors, { origin: true });

server.addHook('onRequest', async (req, reply) => {
  if (req.url === `${API_PREFIX}/health`) return;

  const provided = req.headers['authorization'];
  const expected = ctx.config.authToken ? `Bearer ${ctx.config.authToken}` : null;

  if (expected && provided !== expected) {
    reply.code(401).send({ error: 'unauthorized' });
  }
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

try {
  await server.listen({ port: CORE_SERVER_PORT, host: CORE_SERVER_HOST });
} catch (err) {
  ctx.logger.error({ err }, 'failed to start server');
  process.exit(1);
}
