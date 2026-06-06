/**
 * Fastify routes for ingesting emails from clients, resolving which messages
 * still need fetching, and listing stored emails for an account.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';

const PushEmailItem = z.object({
  messageId: z.string().min(1),
  folder: z.string().min(1),
  subject: z.string().optional(),
  fromAddr: z.string().optional(),
  date: z.number().int().optional(),
  body: z.string().optional(),
  bodyFormat: z.enum(['text', 'html']).optional(),
  hasAttachments: z.boolean().optional(),
  bodyFetched: z.boolean().optional(),
});

const PushEmailsBody = z.object({
  accountId: z.string().min(1),
  emails: z.array(PushEmailItem).min(1).max(500),
});

const SyncStateBody = z.object({
  accountId: z.string().min(1),
  messageIds: z.array(z.string().min(1)).max(5000),
});

const ListEmailsQuery = z.object({
  accountId: z.string().min(1),
  folder: z.string().optional(),
  limit: z.coerce.number().int().positive().max(500).optional(),
  offset: z.coerce.number().int().nonnegative().optional(),
});

/**
 * Registers HTTP routes for pushing emails, querying sync state, and listing emails.
 */
export async function registerEmailRoutes(app: FastifyInstance, ctx: AppContext): Promise<void> {
  app.post('/emails/push', async (req, reply) => {
    const parsed = PushEmailsBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    const items = parsed.data.emails.map((e) => ({
      messageId: e.messageId,
      accountId: parsed.data.accountId,
      folder: e.folder,
      subject: e.subject,
      fromAddr: e.fromAddr,
      date: e.date,
      hasAttachments: e.hasAttachments,
      body: e.body,
      bodyFormat: e.bodyFormat,
      bodyFetched: e.bodyFetched,
    }));

    const inserted = ctx.repos.emails.upsertBatch(items);
    ctx.logger.info({ accountId: parsed.data.accountId, count: inserted }, 'emails pushed');

    return { inserted, total: ctx.repos.emails.count(parsed.data.accountId) };
  });

  app.post('/emails/sync-state', async (req, reply) => {
    const parsed = SyncStateBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }

    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }

    const needFetch = ctx.repos.emails.selectNeedFetch(
      parsed.data.accountId,
      parsed.data.messageIds,
    );
    return { needFetch };
  });

  app.get('/emails', async (req, reply) => {
    const parsed = ListEmailsQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }

    const useFullList = parsed.data.folder !== undefined;
    const emails = useFullList
      ? ctx.repos.emails.list({
          accountId: parsed.data.accountId,
          folder: parsed.data.folder,
          limit: parsed.data.limit,
          offset: parsed.data.offset,
        })
      : ctx.repos.emails.listSummaries({
          accountId: parsed.data.accountId,
          limit: parsed.data.limit,
          offset: parsed.data.offset,
        });

    return {
      emails: emails.map((e) => ({
        messageId: e.messageId,
        folder: e.folder,
        subject: e.subject,
        fromAddr: e.fromAddr,
        date: e.date,
        hasAttachments: e.hasAttachments,
      })),
    };
  });
}
