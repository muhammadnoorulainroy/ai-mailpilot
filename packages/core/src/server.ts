import Fastify from 'fastify';
import cors from '@fastify/cors';
import { CORE_SERVER_PORT, CORE_SERVER_HOST, API_PREFIX } from '@ai-mailpilot/shared';

const server = Fastify({ logger: true });

await server.register(cors, { origin: true });

server.get(`${API_PREFIX}/health`, async () => ({
  status: 'ok',
  version: '0.1.0',
  ollama: { connected: false, models: [] },
  imap: { connected: false },
}));

try {
  await server.listen({ port: CORE_SERVER_PORT, host: CORE_SERVER_HOST });
  console.log(`AI MailPilot Core running at http://${CORE_SERVER_HOST}:${CORE_SERVER_PORT}`);
} catch (err) {
  server.log.error(err);
  process.exit(1);
}
