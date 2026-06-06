/**
 * Fastify routes for ingesting message attachments into the RAG store and
 * reporting attachment extraction stats, with base64 size and format guards.
 */
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { AppContext } from '../context.js';

const MAX_DECODED_BYTES = 10 * 1024 * 1024;
const MAX_BASE64_CHARS = Math.ceil(MAX_DECODED_BYTES / 3) * 4 + 1024;
const BASE64_RE = /^[A-Za-z0-9+/]*={0,2}$/;

const AttachmentItem = z.object({
  filename: z.string().min(1),
  contentType: z.string().optional(),
  partName: z.string().min(1),
  size: z.number().int().nonnegative().optional(),
  dataBase64: z
    .string()
    .min(1)
    .max(MAX_BASE64_CHARS)
    .refine((s) => s.length % 4 === 0 && BASE64_RE.test(s), 'invalid base64'),
});

const IngestBody = z.object({
  accountId: z.string().min(1),
  messageId: z.string().min(1),
  attachments: z.array(AttachmentItem).min(1).max(20),
});

const StatsQuery = z.object({ accountId: z.string().min(1) });

/**
 * Registers attachment HTTP routes for ingesting message attachments into the
 * RAG store and reporting extraction stats.
 */
export async function registerAttachmentRoutes(
  app: FastifyInstance,
  ctx: AppContext,
): Promise<void> {
  app.post('/attachments/ingest', async (req, reply) => {
    const parsed = IngestBody.safeParse(req.body);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid body', issues: parsed.error.issues });
      return;
    }
    const { accountId, messageId, attachments } = parsed.data;

    if (!ctx.repos.accounts.findById(accountId)) {
      reply.code(404).send({ error: 'account not found' });
      return;
    }
    if (!ctx.repos.emails.findById(messageId, accountId)) {
      reply.code(409).send({ error: 'email not indexed; push the email before its attachments' });
      return;
    }

    const modelId = ctx.config.llm.embeddingModel;
    const results: Array<{ filename: string; status: string; chunks: number }> = [];
    for (const a of attachments) {
      const bytes = new Uint8Array(Buffer.from(a.dataBase64, 'base64'));
      if (bytes.length > MAX_DECODED_BYTES) {
        results.push({ filename: a.filename, status: 'too_large', chunks: 0 });
        continue;
      }
      const r = await ctx.services.attachment.ingest(
        {
          messageId,
          accountId,
          filename: a.filename,
          contentType: a.contentType,
          partName: a.partName,
          size: a.size,
        },
        bytes,
        modelId,
      );
      results.push({ filename: a.filename, status: r.status, chunks: r.chunks });
    }

    ctx.logger.info({ accountId, messageId, count: attachments.length }, 'attachments ingested');
    return { results };
  });

  app.get('/attachments/stats', async (req, reply) => {
    const parsed = StatsQuery.safeParse(req.query);
    if (!parsed.success) {
      reply.code(400).send({ error: 'invalid query', issues: parsed.error.issues });
      return;
    }
    return { extracted: ctx.repos.attachments.countExtracted(parsed.data.accountId) };
  });
}
