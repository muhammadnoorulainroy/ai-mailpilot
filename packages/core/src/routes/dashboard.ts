/**
 * Fastify route for the account dashboard, validating the request query and
 * delegating to the dashboard service to build the response payload.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';

const DashboardQuery = z.object({
  accountId: z.string().min(1),
  urgentLimit: z.coerce.number().int().positive().max(100).optional(),
  summarizeLimit: z.coerce.number().int().positive().max(100).optional(),
  recentLimit: z.coerce.number().int().positive().max(200).optional(),
  topCategoriesLimit: z.coerce.number().int().positive().max(50).optional(),
});

/**
 * Registers the GET /dashboard route, which validates the query and returns
 * the dashboard payload built for the requested account.
 */
export async function registerDashboardRoutes(
  app: FastifyInstance,
  ctx: AppContext,
): Promise<void> {
  app.get('/dashboard', async (req, reply) => {
    const parsed = DashboardQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }

    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    return ctx.services.dashboard.build(parsed.data.accountId, {
      urgentLimit: parsed.data.urgentLimit,
      summarizeLimit: parsed.data.summarizeLimit,
      recentLimit: parsed.data.recentLimit,
      topCategoriesLimit: parsed.data.topCategoriesLimit,
    });
  });
}
