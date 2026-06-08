/**
 * Fastify routes for the email assistant feature, exposing endpoints to summarize an
 * opened email and to draft a reply while gating any cloud model usage behind confirmation.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';
import { EmailAssistantNotFoundError } from '../services/email-assistant-service.js';
import { resolveActiveChatModel } from '../util/chat-model.js';

const boolish = z
  .union([z.literal('true'), z.literal('false'), z.boolean()])
  .optional()
  .transform((v) => v === true || v === 'true');

const SummaryQuery = z.object({
  accountId: z.string().min(1),
  messageId: z.string().min(1),
  force: boolish,
  confirmCloud: boolish,
});

const DraftBody = z.object({
  accountId: z.string().min(1),
  messageId: z.string().min(1),
  prompt: z.string().trim().max(1000).optional(),
  confirmCloud: z.boolean().optional(),
});

/**
 * Registers the email assistant routes for summarizing an opened email and drafting a reply.
 * Both routes gate cloud usage behind explicit confirmation to avoid leaking email content.
 */
export async function registerEmailAssistantRoutes(
  app: FastifyInstance,
  ctx: AppContext,
): Promise<void> {
  /**
   * Resolves the active chat model, falling back to the local generation model when cloud
   * is configured but no API key is present.
   */
  async function params(): Promise<{ modelId: string; provider: 'local' | 'cloud' }> {
    const resolved = await resolveActiveChatModel(ctx.config.llm, ctx.llm, ctx.logger);
    if (resolved.provider === 'cloud' && !ctx.config.llm.chatApiKey) {
      ctx.logger.warn(
        {},
        'cloud chat is configured without an API key; using local for the email assistant',
      );
      return { modelId: ctx.config.llm.generationModel, provider: 'local' };
    }
    return resolved;
  }

  app.get('/email-assistant/summary', async (req, reply) => {
    const parsed = SummaryQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }
    if (!ctx.repos.accounts.findById(parsed.data.accountId)) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    try {
      const resolved = await params();
      if (resolved.provider === 'cloud' && !parsed.data.confirmCloud) {
        reply.code(412).send({ error: 'cloud_confirmation_required', provider: 'cloud' });
        return;
      }
      const summary = await ctx.services.emailAssistant.summarize(
        parsed.data.accountId,
        parsed.data.messageId,
        resolved,
        parsed.data.force,
      );
      return { summary };
    } catch (err) {
      if (err instanceof EmailAssistantNotFoundError) {
        reply.code(404).send({ error: 'email not found' });
        return;
      }
      ctx.logger.error({ err }, 'email assistant summary failed');
      reply.code(500).send({
        error: 'email assistant summary failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  });

  app.post('/email-assistant/draft', async (req, reply) => {
    const parsed = DraftBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    if (!ctx.repos.accounts.findById(parsed.data.accountId)) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    try {
      const resolved = await params();
      if (resolved.provider === 'cloud' && !parsed.data.confirmCloud) {
        reply.code(412).send({ error: 'cloud_confirmation_required', provider: 'cloud' });
        return;
      }
      return await ctx.services.emailAssistant.draftReply(
        parsed.data.accountId,
        parsed.data.messageId,
        parsed.data.prompt,
        resolved,
      );
    } catch (err) {
      if (err instanceof EmailAssistantNotFoundError) {
        reply.code(404).send({ error: 'email not found' });
        return;
      }
      ctx.logger.error({ err }, 'email assistant draft failed');
      reply.code(500).send({
        error: 'email assistant draft failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  });
}
