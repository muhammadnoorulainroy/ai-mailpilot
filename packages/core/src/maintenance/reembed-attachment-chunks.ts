/**
 * Re-embed attachment chunks under the active embedding model, filling in each chunk's document
 * context on the way.
 *
 * Two problems are repaired in one pass. First, chunk vectors are stored per model id and the KNN
 * search filters on it, so after an embedding-model change every chunk still carries the OLD model's
 * vector and the dense arm silently returns nothing: attachment retrieval degrades to keyword-only
 * with no error anywhere. Second, the FTS row carried only the chunk body, so a question naming the
 * document could not reach a chunk whose text never repeats the name; the context header fixes that.
 *
 * The chunk TEXT is already stored, so this needs no original attachment bytes and no re-extraction.
 */
import type { Database } from 'better-sqlite3';
import type { LlmClient } from '../llm/client.js';
import { buildChunkContext } from '../util/text.js';
import { AttachmentRepository } from '../repositories/attachment-repository.js';

const EMBED_BATCH = 16;

/** One chunk needing a vector, with the parent document's identity for its context header. */
interface ChunkRow {
  chunkRowid: number;
  accountId: string;
  text: string;
  filename: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
}

/** Progress callback: how many chunks are embedded so far, out of how many. */
export type ReembedProgress = (done: number, total: number) => void;

/** Outcome of a re-embed pass. */
export interface ReembedResult {
  scanned: number;
  embedded: number;
  failed: number;
  contextsWritten: number;
}

/** Options controlling a re-embed pass. */
export interface ReembedOptions {
  /** Restrict to one account; omit for every account. */
  accountId?: string;
  onProgress?: ReembedProgress;
}

/** Chunks with no vector under `modelId`, newest documents first, with their parent email. */
function chunksMissingModel(db: Database, modelId: string, accountId?: string): ChunkRow[] {
  const where = accountId ? 'AND c.account_id = ?' : '';
  const params: unknown[] = [modelId];
  if (accountId) params.push(accountId);
  return db
    .prepare(
      `SELECT c.rowid AS chunkRowid, c.account_id AS accountId, c.text,
              a.filename, e.subject, e.from_addr AS fromAddr, e.date
         FROM attachment_chunks c
         JOIN attachments a ON a.id = c.attachment_id
         LEFT JOIN emails e ON e.message_id = c.message_id AND e.account_id = c.account_id
        WHERE NOT EXISTS (
                SELECT 1 FROM attachment_chunk_embedding_index ei
                 WHERE ei.chunk_rowid = c.rowid AND ei.model_id = ?
              )
          ${where}
        ORDER BY c.rowid`,
    )
    .all(...params) as ChunkRow[];
}

/**
 * Re-embed every attachment chunk that lacks a vector for `modelId`. The chunk's context header is
 * written to the FTS-indexed column (contextual BM25), but the vector is computed from the BARE
 * chunk text: an ablation showed the metadata header lowers dense top-K when embedded, while helping
 * keyword recall when indexed. Embedding failures are counted and skipped rather than aborting the
 * pass, so one bad batch cannot strand the rest of the corpus.
 */
export async function reembedAttachmentChunks(
  db: Database,
  llm: LlmClient,
  modelId: string,
  opts: ReembedOptions = {},
): Promise<ReembedResult> {
  const rows = chunksMissingModel(db, modelId, opts.accountId);
  const repo = new AttachmentRepository(db);
  const setContext = db.prepare('UPDATE attachment_chunks SET context = ? WHERE rowid = ?');

  const result: ReembedResult = {
    scanned: rows.length,
    embedded: 0,
    failed: 0,
    contextsWritten: 0,
  };

  for (let i = 0; i < rows.length; i += EMBED_BATCH) {
    const batch = rows.slice(i, i + EMBED_BATCH);

    let vectors: number[][];
    try {
      vectors = await llm.embedBatch(
        batch.map((r) => r.text),
        modelId,
        'attachment',
      );
      if (vectors.length !== batch.length) {
        throw new Error(
          `embedding count mismatch: requested ${batch.length}, got ${vectors.length}`,
        );
      }
    } catch {
      result.failed += batch.length;
      opts.onProgress?.(result.embedded + result.failed, rows.length);
      continue;
    }

    const persist = db.transaction(() => {
      batch.forEach((r, j) => {
        const context = buildChunkContext({
          filename: r.filename,
          subject: r.subject,
          fromAddr: r.fromAddr,
          date: r.date,
        });
        if (context) {
          setContext.run(context, r.chunkRowid);
          result.contextsWritten += 1;
        }
        repo.saveChunkEmbedding(r.chunkRowid, r.accountId, modelId, vectors[j]!);
        result.embedded += 1;
      });
    });
    persist();
    opts.onProgress?.(result.embedded + result.failed, rows.length);
  }

  return result;
}
