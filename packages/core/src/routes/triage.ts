/**
 * Fastify routes for email triage, exposing endpoints to run classification,
 * report progress, read bucket summaries, build the priority briefing, and
 * record user resolutions such as dismiss, done, or snooze.
 */
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

const PriorityQuery = z.object({
  accountId: z.string().min(1),
  range: z.enum(['today', 'week', 'all']).default('today'),
  dayStartMs: z.coerce.number().int().nonnegative(),
});

const ResolveBody = z.object({
  accountId: z.string().min(1),
  messageId: z.string().min(1),
  resolution: z.enum(['dismiss', 'done', 'snooze', 'reset']),
  snoozedUntil: z.number().int().positive().optional(),
});

/**
 * Registers the triage HTTP routes for running classification, reading bucket
 * summaries, building the priority briefing, and recording user resolutions.
 */
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

    const cloudOptIn =
      ctx.config.llm.priorityUseChatProvider === true && !!ctx.config.llm.chatBaseUrl;
    const useCloudPriority = cloudOptIn && !!ctx.config.llm.chatApiKey;
    if (cloudOptIn && !useCloudPriority) {
      ctx.logger.warn(
        { accountId: parsed.data.accountId },
        'cloud priority is enabled but no API key is stored; running priority locally',
      );
    }
    const modelId =
      parsed.data.modelId ??
      (useCloudPriority
        ? ctx.config.llm.chatModel || ctx.config.llm.generationModel
        : ctx.config.llm.generationModel);
    const result = ctx.services.triage.start(parsed.data.accountId, modelId, {
      force: parsed.data.force,
      provider: useCloudPriority ? 'chat' : 'main',
    });

    const status = result.started
      ? ('started' as const)
      : result.pending === 0
        ? ('up_to_date' as const)
        : ('already_running' as const);

    return { status, pending: result.pending, modelId };
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

  app.get('/triage/priority', async (req, reply) => {
    const parsed = PriorityQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }
    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    return ctx.services.priority.build(parsed.data.accountId, {
      range: parsed.data.range,
      dayStartMs: parsed.data.dayStartMs,
    });
  });

  app.post('/triage/resolve', async (req, reply) => {
    const parsed = ResolveBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const { accountId, messageId, resolution, snoozedUntil } = parsed.data;
    if (resolution === 'snooze' && !snoozedUntil) {
      reply.code(400).send({ error: 'snoozedUntil is required for a snooze' });
      return;
    }
    const account = ctx.repos.accounts.findById(accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    const now = Date.now();
    const dismissedAt = resolution === 'dismiss' ? now : null;
    const doneAt = resolution === 'done' ? now : null;
    const snoozed = resolution === 'snooze' ? (snoozedUntil ?? null) : null;
    const ok = ctx.repos.triage.setResolution(accountId, messageId, dismissedAt, doneAt, snoozed);
    return { ok };
  });
}
