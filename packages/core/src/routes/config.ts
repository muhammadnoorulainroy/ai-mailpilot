/**
 * Fastify routes for reading and updating app configuration, including locale,
 * indexing settings, and LLM model selection with embedding dimension checks.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';
import { LlmConfigSchema, FeaturesConfigSchema } from '../config/schema.js';
import { saveConfig, redactConfig } from '../config/config.js';
import { canonicalizeModelId } from '../util/model-id.js';
import { EMBEDDING_DIM } from '../db/schema.js';

const UpdateConfigBody = z.object({
  locale: z.enum(['en', 'fr']).optional(),
  autoIndex: z.boolean().optional(),
  indexedFolders: z.array(z.string()).optional(),
  llm: LlmConfigSchema.partial().optional(),
  features: FeaturesConfigSchema.partial().optional(),
});

/**
 * Registers GET and PATCH /config routes for reading and updating app configuration.
 */
export async function registerConfigRoutes(app: FastifyInstance, ctx: AppContext): Promise<void> {
  app.get('/config', async () => {
    return redactConfig(ctx.config);
  });

  app.patch('/config', async (req, reply) => {
    const parsed = UpdateConfigBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    const rawLlm = (req.body as { llm?: Record<string, unknown> } | null | undefined)?.llm;
    const llmPatch: Partial<typeof ctx.config.llm> = {};
    if (parsed.data.llm && rawLlm && typeof rawLlm === 'object') {
      for (const key of Object.keys(rawLlm)) {
        if (key in parsed.data.llm) {
          (llmPatch as Record<string, unknown>)[key] = (parsed.data.llm as Record<string, unknown>)[
            key
          ];
        }
      }
    }

    let warning: string | undefined;
    if (llmPatch.embeddingModel) {
      const nextModel = canonicalizeModelId(llmPatch.embeddingModel);
      if (nextModel !== ctx.config.llm.embeddingModel) {
        try {
          const probe = await ctx.llm.embed('preflight', nextModel);
          if (probe.length !== EMBEDDING_DIM) {
            reply.code(400).send({
              error: 'embedding dimension mismatch',
              message: `Model "${nextModel}" produces ${probe.length}-dim vectors; storage requires ${EMBEDDING_DIM}. Change not applied.`,
            });
            return;
          }
        } catch (err) {
          warning = `Could not verify embedding dimension for "${nextModel}" (LLM unreachable). Indexing will fail if it is not ${EMBEDDING_DIM}-dim.`;
          ctx.logger.warn({ err, model: nextModel }, 'embedding dimension probe skipped');
        }
      }
    }

    if (parsed.data.locale) ctx.config.locale = parsed.data.locale;
    if (parsed.data.autoIndex !== undefined) ctx.config.autoIndex = parsed.data.autoIndex;
    if (parsed.data.indexedFolders) ctx.config.indexedFolders = parsed.data.indexedFolders;

    if (Object.keys(llmPatch).length > 0) {
      if (llmPatch.embeddingModel) {
        llmPatch.embeddingModel = canonicalizeModelId(llmPatch.embeddingModel);
      }
      if (llmPatch.generationModel) {
        llmPatch.generationModel = canonicalizeModelId(llmPatch.generationModel);
      }
      if (llmPatch.chatModel) {
        llmPatch.chatModel = canonicalizeModelId(llmPatch.chatModel);
      }
      Object.assign(ctx.config.llm, llmPatch);
    }

    // Merge only the feature keys the caller actually sent, so an absent flag keeps its current value.
    const rawFeatures = (req.body as { features?: Record<string, unknown> } | null | undefined)
      ?.features;
    if (parsed.data.features && rawFeatures && typeof rawFeatures === 'object') {
      for (const key of Object.keys(rawFeatures)) {
        if (Object.prototype.hasOwnProperty.call(parsed.data.features, key)) {
          (ctx.config.features as Record<string, unknown>)[key] = (
            parsed.data.features as Record<string, unknown>
          )[key];
        }
      }
    }

    saveConfig(ctx.config);
    ctx.logger.info('config updated');

    return warning ? { ...redactConfig(ctx.config), warning } : redactConfig(ctx.config);
  });
}
