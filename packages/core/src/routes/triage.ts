import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { TriageBucket } from '@ai-mailpilot/shared';
import type { AppContext } from '../context.js';

const TriageRunBody = z.object({
  accountId: z.string().min(1),
  modelId: z.string().optional(),
  force: z.boolean().optional(),
});

const SummaryQuery = z.object({
  accountId: z.string().min(1),
});

export async function registerTriageRoutes(app: FastifyInstance, ctx: AppContext): Promise<void> {
  app.post('/triage/run', async (req, reply) => {
    const parsed = TriageRunBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    if (parsed.data.force) {
      ctx.repos.triage.clearForAccount(parsed.data.accountId);
    }

    const modelId = parsed.data.modelId ?? ctx.config.llm.generationModel;
    const result = ctx.services.triage.start(parsed.data.accountId, modelId);

    return {
      status: result.started ? ('started' as const) : ('already_running' as const),
      pending: result.pending,
      modelId,
    };
  });

  app.get('/triage/progress', async () => {
    return ctx.services.triage.getProgress();
  });

  app.get('/triage/summary', async (req, reply) => {
    const parsed = SummaryQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }

    const counts = ctx.repos.triage.countByBucket(parsed.data.accountId);
    const buckets: Record<TriageBucket, number> = {
      urgent: 0,
      summarize: 0,
      spam: 0,
      personal: 0,
    };
    for (const c of counts) buckets[c.bucket] = c.count;

    return { accountId: parsed.data.accountId, buckets };
  });
}
