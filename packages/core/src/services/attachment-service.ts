/**
 * Service for ingesting email attachments into the local RAG index by extracting
 * text, chunking it, embedding each chunk, and persisting the results.
 */
import type { Logger } from 'pino';
import type { LlmClient } from '../llm/client.js';
import type {
  AttachmentMeta,
  AttachmentRepository,
  AttachmentStatus,
} from '../repositories/attachment-repository.js';
import { extractAttachmentText } from './attachment-extract.js';
import { chunkText } from '../util/chunk.js';

const MAX_TEXT_CHARS = 200_000;
const MAX_CHUNKS = 200;
const EMBED_BATCH = 16;

/** Outcome of ingesting one attachment: its terminal status and how many chunks were stored. */
export interface IngestResult {
  status: AttachmentStatus;
  chunks: number;
}

/**
 * Ingest one attachment for RAG: extract text, chunk it, embed each chunk, and store.
 * Idempotent, so an already-extracted attachment is skipped on re-sync. Runs fully local.
 */
export class AttachmentService {
  /** Wire up the embedding client, attachment store, and logger. */
  constructor(
    private llm: LlmClient,
    private attachments: AttachmentRepository,
    private logger: Logger,
  ) {}

  /**
   * Ingest one attachment for RAG. Extracts text, chunks and embeds it, then stores the result.
   * Idempotent, so attachments in a terminal status are skipped on re-sync.
   */
  async ingest(
    meta: AttachmentMeta,
    bytes: Uint8Array,
    embeddingModelId: string,
  ): Promise<IngestResult> {
    const { id, priorStatus } = this.attachments.upsertAttachment(meta);
    if (priorStatus === 'extracted' || priorStatus === 'empty' || priorStatus === 'unsupported') {
      return { status: priorStatus, chunks: 0 };
    }

    let extracted;
    try {
      extracted = await extractAttachmentText(bytes, meta.filename, meta.contentType);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.attachments.setStatus(id, 'error', 0, message);
      this.logger.warn({ err, filename: meta.filename }, 'attachment extract failed');
      return { status: 'error', chunks: 0 };
    }

    if (extracted.status !== 'extracted') {
      this.attachments.setStatus(id, extracted.status, 0, null);
      return { status: extracted.status, chunks: 0 };
    }

    const text = extracted.text.slice(0, MAX_TEXT_CHARS);
    const chunks = chunkText(text).slice(0, MAX_CHUNKS);
    if (chunks.length === 0) {
      this.attachments.setStatus(id, 'empty', 0, null);
      return { status: 'empty', chunks: 0 };
    }

    const rowids = this.attachments.replaceChunks(id, meta.messageId, meta.accountId, chunks);
    try {
      for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
        const batch = chunks.slice(i, i + EMBED_BATCH);
        const vectors = await this.llm.embedBatch(batch, embeddingModelId);
        if (vectors.length !== batch.length) {
          throw new Error(
            `attachment embedding count mismatch: requested ${batch.length}, got ${vectors.length}`,
          );
        }
        vectors.forEach((vec, j) => {
          this.attachments.saveChunkEmbedding(
            rowids[i + j]!,
            meta.accountId,
            embeddingModelId,
            vec,
          );
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.attachments.setStatus(id, 'error', text.length, message);
      this.logger.warn({ err, filename: meta.filename }, 'attachment embed failed');
      return { status: 'error', chunks: 0 };
    }

    this.attachments.setStatus(id, 'extracted', text.length, null);
    this.logger.info({ filename: meta.filename, chunks: chunks.length }, 'attachment ingested');
    return { status: 'extracted', chunks: chunks.length };
  }
}
