/**
 * Fastify routes for triggering an account embedding run and polling its progress.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';

const EmbedRunBody = z.object({
  accountId: z.string().min(1),
  modelId: z.string().optional(),
});

/**
 * Registers HTTP routes for starting an account embedding run and polling its progress.
 */
export async function registerEmbedRoutes(app: FastifyInstance, ctx: AppContext): Promise<void> {
  app.post('/embed/run', async (req, reply) => {
    const parsed = EmbedRunBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    const modelId = parsed.data.modelId ?? ctx.config.llm.embeddingModel;
    const result = ctx.services.embedding.start(parsed.data.accountId, modelId);

    const status = result.started
      ? ('started' as const)
      : result.pending === 0
        ? ('up_to_date' as const)
        : ('already_running' as const);

    return { status, pending: result.pending, modelId };
  });

  app.get('/embed/progress', async () => {
    return ctx.services.embedding.getProgress();
  });
}
