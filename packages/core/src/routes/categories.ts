/**
 * Fastify route handlers for category management, topic discovery, and email
 * categorization, including embedding and LLM categorize runs and user corrections.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';
import type { CategoryRow } from '../repositories/category-repository.js';
import type { CategoryDto } from '@ai-mailpilot/shared';
import { discoveryProvider } from '../services/discovery-guard.js';
import { ProposalApplyError } from '../services/discovery-proposal-orchestrator.js';

const DiscoverBody = z.object({
  accountId: z.string().min(1),
  embeddingModelId: z.string().optional(),
  generationModelId: z.string().optional(),
});

const ListQuery = z.object({
  accountId: z.string().min(1),
});

const ProposalActionBody = z.object({
  accountId: z.string().min(1),
});

const RebuildCentroidBody = z.object({
  accountId: z.string().min(1),
  embeddingModelId: z.string().optional(),
  allowAutoFallback: z.boolean().optional(),
});

const CategorizeRunBody = z.object({
  accountId: z.string().min(1),
  embeddingModelId: z.string().optional(),
  force: z.boolean().optional(),
});

/**
 * Maps a category row plus its email count to the client-facing category DTO.
 */
function toDto(row: CategoryRow & { emailCount: number }): CategoryDto {
  return {
    id: row.id,
    accountId: row.accountId,
    label: row.label,
    description: row.description,
    source: row.source,
    emailCount: row.emailCount,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
  };
}

/**
 * Registers the category, topic discovery, and categorization HTTP routes on the Fastify instance.
 */
export async function registerCategoryRoutes(app: FastifyInstance, ctx: AppContext): Promise<void> {
  app.post('/topics/discover', async (req, reply) => {
    const parsed = DiscoverBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    const embeddingModelId = parsed.data.embeddingModelId ?? ctx.config.llm.embeddingModel;
    const generationModelId = parsed.data.generationModelId ?? ctx.config.llm.generationModel;

    try {
      const result = await ctx.services.topicDiscovery.discover(
        parsed.data.accountId,
        embeddingModelId,
        generationModelId,
      );
      return result;
    } catch (err) {
      ctx.logger.error({ err }, 'topic discovery failed');
      reply.code(500).send({
        error: 'topic discovery failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  });

  // Run discovery and persist the review queue: cluster residual mail, name it locally, validate, and
  // store accepted names as suggested categories. Creates nothing active and assigns no mail.
  app.post('/categories/proposals/generate', async (req, reply) => {
    const parsed = DiscoverBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    const embeddingModelId = parsed.data.embeddingModelId ?? ctx.config.llm.embeddingModel;
    const generationModelId = parsed.data.generationModelId ?? ctx.config.llm.generationModel;
    try {
      return await ctx.services.discoveryProposal.generate(
        parsed.data.accountId,
        embeddingModelId,
        generationModelId,
      );
    } catch (err) {
      ctx.logger.error({ err }, 'discovery proposal generation failed');
      reply.code(500).send({
        error: 'discovery proposal generation failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  });

  // The pending review queue for an account.
  app.get('/categories/proposals', async (req, reply) => {
    const parsed = ListQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }
    if (!ctx.repos.accounts.findById(parsed.data.accountId)) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    return { proposals: ctx.services.discoveryProposal.listPending(parsed.data.accountId) };
  });

  // Approve a proposal: promote the suggested category to active, seed its centroid, and assign the
  // still-uncategorized members.
  app.post<{ Params: { id: string } }>('/categories/proposals/:id/apply', async (req, reply) => {
    const parsed = ProposalActionBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    try {
      return ctx.services.discoveryProposal.apply(parsed.data.accountId, req.params.id);
    } catch (err) {
      // A structural block/precondition failure carries its own status (409 conflict or 404 missing).
      if (err instanceof ProposalApplyError) {
        reply.code(err.httpStatus).send({ error: err.message });
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      if (message.includes('not found')) {
        reply.code(404).send({ error: message });
        return;
      }
      if (message.includes('not pending') || message.includes('not suggested')) {
        reply.code(409).send({ error: message });
        return;
      }
      ctx.logger.error({ err, proposalId: req.params.id }, 'apply proposal failed');
      reply.code(500).send({ error: 'failed to apply proposal', message });
    }
  });

  // Dismiss a proposal (soft): retire the suggested category, keep the record so a re-run does not
  // re-propose the same purpose.
  app.post<{ Params: { id: string } }>('/categories/proposals/:id/dismiss', async (req, reply) => {
    const parsed = ProposalActionBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    try {
      return ctx.services.discoveryProposal.dismiss(parsed.data.accountId, req.params.id);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (message.includes('not found')) {
        reply.code(404).send({ error: message });
        return;
      }
      if (message.includes('not pending')) {
        reply.code(409).send({ error: message });
        return;
      }
      ctx.logger.error({ err, proposalId: req.params.id }, 'dismiss proposal failed');
      reply.code(500).send({ error: 'failed to dismiss proposal', message });
    }
  });

  // Recompute a category's centroid from its user-confirmed member embeddings. Centroid-only: it
  // changes no label, status, or assignment. Leaves the centroid unchanged when trusted data is thin.
  app.post<{ Params: { id: string } }>('/categories/:id/rebuild-centroid', async (req, reply) => {
    const parsed = RebuildCentroidBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    const category = ctx.repos.categories.findById(req.params.id);
    if (!category || category.accountId !== parsed.data.accountId) {
      reply.code(404).send({ error: 'category not found for this account' });
      return;
    }
    const embeddingModelId = parsed.data.embeddingModelId ?? ctx.config.llm.embeddingModel;
    return ctx.services.categoryCentroidRebuild.rebuild(
      parsed.data.accountId,
      req.params.id,
      embeddingModelId,
      { allowAutoFallback: parsed.data.allowAutoFallback },
    );
  });

  app.post('/categories/improve/suggest', async (req, reply) => {
    const parsed = DiscoverBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    const embeddingModelId = parsed.data.embeddingModelId ?? ctx.config.llm.embeddingModel;
    // Discovery and improvement share the allowCloudDiscovery privacy switch, not the
    // categorizeUseChatProvider flag used by Refine. Local unless cloud discovery is opted in.
    const provider = discoveryProvider(ctx.config.llm);
    const modelId =
      provider === 'chat'
        ? ctx.config.llm.chatModel || ctx.config.llm.generationModel
        : (parsed.data.generationModelId ?? ctx.config.llm.generationModel);
    try {
      return await ctx.services.categoryImprovement.suggest(
        parsed.data.accountId,
        embeddingModelId,
        modelId,
        provider,
      );
    } catch (err) {
      ctx.logger.error({ err }, 'improve suggest failed');
      reply.code(500).send({
        error: 'improve suggest failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  });

  const ApplyImprovementsBody = z.object({
    accountId: z.string().min(1),
    existingCategoryExpansions: z
      .array(
        z.object({
          categoryId: z.string().min(1),
          messageIds: z.array(z.string().min(1)).max(10000),
        }),
      )
      .max(20)
      .default([]),
    newCategories: z
      .array(
        z.object({
          label: z.string().min(1).max(80),
          description: z.string().max(300),
          messageIds: z.array(z.string().min(1)).max(10000).optional(),
        }),
      )
      .max(20)
      .default([]),
    merges: z
      .array(z.object({ sourceId: z.string().min(1), targetId: z.string().min(1) }))
      .max(20)
      .default([]),
  });

  app.post('/categories/improve/apply', async (req, reply) => {
    const parsed = ApplyImprovementsBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    const embeddingModelId = ctx.config.llm.embeddingModel;
    try {
      return await ctx.services.categoryImprovement.apply(parsed.data.accountId, embeddingModelId, {
        existingCategoryExpansions: parsed.data.existingCategoryExpansions,
        newCategories: parsed.data.newCategories,
        merges: parsed.data.merges,
      });
    } catch (err) {
      ctx.logger.error({ err }, 'improve apply failed');
      reply.code(500).send({
        error: 'improve apply failed',
        message: err instanceof Error ? err.message : String(err),
      });
    }
  });

  app.get('/categories', async (req, reply) => {
    const parsed = ListQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }

    const rows = ctx.repos.categories.listActive(parsed.data.accountId);
    return {
      accountId: parsed.data.accountId,
      categories: rows.map(toDto),
    };
  });

  app.get('/folder-plan', async (req, reply) => {
    const parsed = ListQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }

    const assignments = ctx.repos.categories.getPrimaryCategoryPerEmail(parsed.data.accountId);
    const labelById = new Map(
      ctx.repos.categories.listForAccount(parsed.data.accountId).map((c) => [c.id, c.label]),
    );
    const countById = new Map<string, number>();
    for (const a of assignments) {
      countById.set(a.categoryId, (countById.get(a.categoryId) ?? 0) + 1);
    }
    const categories = [...countById.entries()]
      .map(([id, count]) => ({ id, label: labelById.get(id) ?? 'Unknown', count }))
      .sort((a, b) => b.count - a.count);

    return { accountId: parsed.data.accountId, categories, assignments };
  });

  app.delete<{ Params: { id: string } }>('/categories/:id', async (req, reply) => {
    const deleted = ctx.repos.categories.delete(req.params.id);
    if (!deleted) {
      reply.code(404).send({ error: 'category not found' });
      return;
    }
    reply.code(204).send();
  });

  const PatchBody = z.object({
    label: z.string().trim().min(1).max(120).optional(),
    description: z.string().trim().max(500).nullable().optional(),
  });

  app.patch<{ Params: { id: string } }>('/categories/:id', async (req, reply) => {
    const parsed = PatchBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    if (parsed.data.label === undefined && parsed.data.description === undefined) {
      reply.code(400).send({ error: 'nothing to update' });
      return;
    }

    const existing = ctx.repos.categories.findById(req.params.id);
    if (!existing) {
      reply.code(404).send({ error: 'category not found' });
      return;
    }

    if (parsed.data.label && parsed.data.label !== existing.label) {
      const clash = ctx.repos.categories.findByLabel(existing.accountId, parsed.data.label);
      if (clash && clash.id !== existing.id) {
        reply.code(409).send({ error: 'a category with that label already exists' });
        return;
      }
    }

    const updated = ctx.repos.categories.update(req.params.id, {
      ...(parsed.data.label !== undefined ? { label: parsed.data.label } : {}),
      ...(parsed.data.description !== undefined ? { description: parsed.data.description } : {}),
    });
    if (!updated) {
      reply.code(404).send({ error: 'category not found' });
      return;
    }
    const emailCount = ctx.repos.categories.countEmails(updated.id);
    return toDto({ ...updated, emailCount });
  });

  const MergeBody = z.object({ targetId: z.string().min(1) });

  app.post<{ Params: { id: string } }>('/categories/:id/merge', async (req, reply) => {
    const parsed = MergeBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    if (req.params.id === parsed.data.targetId) {
      reply.code(400).send({ error: 'cannot merge a category into itself' });
      return;
    }

    const source = ctx.repos.categories.findById(req.params.id);
    const target = ctx.repos.categories.findById(parsed.data.targetId);
    if (!source || !target) {
      reply.code(404).send({ error: 'source or target category not found' });
      return;
    }
    if (source.accountId !== target.accountId) {
      reply.code(400).send({ error: 'source and target belong to different accounts' });
      return;
    }

    const result = ctx.repos.categories.mergeInto(req.params.id, parsed.data.targetId);
    return result;
  });

  const EmailsQuery = z.object({
    limit: z.coerce.number().int().positive().max(500).optional(),
    offset: z.coerce.number().int().nonnegative().optional(),
  });

  app.get<{ Params: { id: string } }>('/categories/:id/emails', async (req, reply) => {
    const parsedQuery = EmailsQuery.safeParse(req.query);
    if (!parsedQuery.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsedQuery.error.issues });
      return;
    }
    const category = ctx.repos.categories.findById(req.params.id);
    if (!category) {
      reply.code(404).send({ error: 'category not found' });
      return;
    }
    const limit = parsedQuery.data.limit ?? 200;
    const offset = parsedQuery.data.offset ?? 0;
    const emails = ctx.repos.categories.listEmailsForCategory(req.params.id, limit, offset);
    return { categoryId: req.params.id, emails };
  });

  app.post('/categorize/run', async (req, reply) => {
    const parsed = CategorizeRunBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    const embeddingModelId = parsed.data.embeddingModelId ?? ctx.config.llm.embeddingModel;
    const result = ctx.services.category.start(parsed.data.accountId, embeddingModelId, {
      force: parsed.data.force ?? false,
    });

    const status = result.started
      ? ('started' as const)
      : result.pending === 0
        ? ('up_to_date' as const)
        : ('already_running' as const);

    return { status, pending: result.pending };
  });

  app.get('/categorize/progress', async () => {
    return ctx.services.category.getProgress();
  });

  const LlmCategorizeBody = z.object({
    accountId: z.string().min(1),
    generationModelId: z.string().optional(),
    force: z.boolean().optional(),
    retryUncategorized: z.boolean().optional(),
    messageIds: z.array(z.string().min(1)).max(5000).optional(),
  });

  app.post('/categorize/llm/run', async (req, reply) => {
    const parsed = LlmCategorizeBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    const useCloud = ctx.config.llm.categorizeUseChatProvider && !!ctx.config.llm.chatBaseUrl;
    const provider = useCloud ? ('chat' as const) : ('main' as const);
    const modelId = useCloud
      ? ctx.config.llm.chatModel || ctx.config.llm.generationModel
      : (parsed.data.generationModelId ?? ctx.config.llm.generationModel);
    const result = ctx.services.llmCategorize.start(
      parsed.data.accountId,
      modelId,
      ctx.config.llm.embeddingModel,
      {
        force: parsed.data.force ?? false,
        retryUncategorized: parsed.data.retryUncategorized ?? false,
        messageIds: parsed.data.messageIds,
        provider,
      },
    );

    const status = result.started
      ? ('started' as const)
      : result.pending === 0
        ? ('up_to_date' as const)
        : ('already_running' as const);

    return { status, pending: result.pending };
  });

  app.get('/categorize/llm/progress', async (req) => {
    const accountId = (req.query as { accountId?: string } | undefined)?.accountId;
    return ctx.services.llmCategorize.getProgress(accountId);
  });

  app.post('/categorize/llm/stop', async () => {
    return { stopped: ctx.services.llmCategorize.stop() };
  });

  app.get<{ Params: { messageId: string }; Querystring: { accountId: string } }>(
    '/emails/:messageId/categories',
    async (req, reply) => {
      const accountId = req.query.accountId;
      if (!accountId) {
        reply.code(400).send({ error: 'missing accountId query param' });
        return;
      }

      const assignments = ctx.repos.categories.getEmailCategoriesWithLabels(
        req.params.messageId,
        accountId,
      );

      return {
        messageId: req.params.messageId,
        accountId,
        categories: assignments.map((a) => ({
          categoryId: a.categoryId,
          label: a.label,
          confidence: a.confidence,
          assignedBy: a.assignedBy,
          method: a.method ?? null,
        })),
      };
    },
  );

  const SetCategoriesBody = z.object({
    accountId: z.string().min(1),
    categoryIds: z.array(z.string().min(1)).max(10),
  });

  app.put<{ Params: { messageId: string } }>(
    '/emails/:messageId/categories',
    async (req, reply) => {
      const parsed = SetCategoriesBody.safeParse(req.body);
      if (!parsed.success) {
        reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
        return;
      }
      const { accountId, categoryIds } = parsed.data;

      const account = ctx.repos.accounts.findById(accountId);
      if (!account) {
        reply.code(404).send({ error: 'account not found' });
        return;
      }
      if (!ctx.repos.emails.findById(req.params.messageId, accountId)) {
        reply.code(404).send({ error: 'email not found' });
        return;
      }
      for (const id of categoryIds) {
        const cat = ctx.repos.categories.findById(id);
        if (!cat || cat.accountId !== accountId) {
          reply.code(400).send({ error: `category ${id} not found for this account` });
          return;
        }
      }

      try {
        ctx.services.correction.setUserCategories(
          accountId,
          req.params.messageId,
          categoryIds,
          ctx.config.llm.embeddingModel,
        );
      } catch (err) {
        ctx.logger.error({ err, messageId: req.params.messageId }, 'set user categories failed');
        reply.code(500).send({
          error: 'failed to apply correction',
          message: err instanceof Error ? err.message : String(err),
        });
        return;
      }

      const assignments = ctx.repos.categories.getEmailCategoriesWithLabels(
        req.params.messageId,
        accountId,
      );
      return {
        messageId: req.params.messageId,
        accountId,
        categories: assignments.map((a) => ({
          categoryId: a.categoryId,
          label: a.label,
          confidence: a.confidence,
          assignedBy: a.assignedBy,
          method: a.method ?? null,
        })),
      };
    },
  );
}
