import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';
import { LlmConfigSchema } from '../config/schema.js';
import { saveConfig } from '../config/config.js';

const UpdateConfigBody = z.object({
  locale: z.enum(['en', 'fr']).optional(),
  autoIndex: z.boolean().optional(),
  indexedFolders: z.array(z.string()).optional(),
  llm: LlmConfigSchema.partial().optional(),
});

export async function registerConfigRoutes(app: FastifyInstance, ctx: AppContext): Promise<void> {
  app.get('/config', async () => {
    const { authToken: _authToken, ...safeConfig } = ctx.config;
    void _authToken;
    return safeConfig;
  });

  app.patch('/config', async (req, reply) => {
    const parsed = UpdateConfigBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    if (parsed.data.locale) ctx.config.locale = parsed.data.locale;
    if (parsed.data.autoIndex !== undefined) ctx.config.autoIndex = parsed.data.autoIndex;
    if (parsed.data.indexedFolders) ctx.config.indexedFolders = parsed.data.indexedFolders;

    if (parsed.data.llm) {
      ctx.config.llm = { ...ctx.config.llm, ...parsed.data.llm };
    }

    saveConfig(ctx.config);
    ctx.logger.info('config updated');

    const { authToken: _authToken, ...safeConfig } = ctx.config;
    void _authToken;
    return safeConfig;
  });
}
