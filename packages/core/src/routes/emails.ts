/**
 * Fastify routes for ingesting emails from clients, resolving which messages
 * still need fetching, and listing stored emails for an account.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';
import { buildUserLabels } from '../services/user-label-ingest.js';

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
  tags: z
    .array(z.object({ key: z.string().min(1), label: z.string() }))
    .max(200)
    .optional(),
});

const PushEmailsBody = z.object({
  accountId: z.string().min(1),
  emails: z.array(PushEmailItem).min(1).max(500),
});

const SyncStateBody = z.object({
  accountId: z.string().min(1),
  messageIds: z.array(z.string().min(1)).max(5000),
});

const SyncUserLabelsBody = z.object({
  accountId: z.string().min(1),
  items: z
    .array(
      z.object({
        messageId: z.string().min(1),
        folder: z.string().min(1),
        tags: z
          .array(z.object({ key: z.string().min(1), label: z.string() }))
          .max(200)
          .optional(),
      }),
    )
    .min(1)
    .max(1000),
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

    // Mirror the user's own Thunderbird tags and meaningful folder as user-owned labels, replacing
    // each pushed message's set so a removed tag or a moved message is reflected. Only messages in
    // this batch are touched. This never affects AI categories or emails/embeddings.
    const syncedAt = Date.now();
    const labelEntries = parsed.data.emails.map((e) => ({
      messageId: e.messageId,
      labels: buildUserLabels(e.folder, e.tags),
    }));
    ctx.repos.emailUserLabels.replaceForEmails(parsed.data.accountId, labelEntries, syncedAt);

    ctx.logger.info({ accountId: parsed.data.accountId, count: inserted }, 'emails pushed');

    return { inserted, total: ctx.repos.emails.count(parsed.data.accountId) };
  });

  // Body-less label-only sync: capture the user's tags/folder for already-indexed messages without
  // re-fetching bodies. Only touches email_user_labels for messages that already exist; never writes
  // emails or embeddings. Lets an existing mailbox's organization be captured without a full re-sync.
  app.post('/emails/user-labels', async (req, reply) => {
    const parsed = SyncUserLabelsBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const account = ctx.repos.accounts.findById(parsed.data.accountId);
    if (!account) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    const existing = new Set(
      ctx.repos.emails.filterExisting(
        parsed.data.accountId,
        parsed.data.items.map((i) => i.messageId),
      ),
    );
    const entries = parsed.data.items
      .filter((i) => existing.has(i.messageId))
      .map((i) => ({ messageId: i.messageId, labels: buildUserLabels(i.folder, i.tags) }));
    ctx.repos.emailUserLabels.replaceForEmails(parsed.data.accountId, entries, Date.now());
    return { updated: entries.length };
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
