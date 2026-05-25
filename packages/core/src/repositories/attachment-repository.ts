/**
 * Repository for email attachments and their extracted-text chunks, backing both vector and
 * keyword retrieval over attachment content for chat and citation.
 */
import type { Database, Statement } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import { EMBEDDING_DIM } from '../db/schema.js';
import { vectorToBuffer } from '../util/vector.js';
import { canonicalizeModelId } from '../util/model-id.js';
import { sanitizeFtsQuery } from '../util/text.js';

export type AttachmentStatus = 'pending' | 'extracted' | 'empty' | 'unsupported' | 'error';

export interface AttachmentMeta {
  messageId: string;
  accountId: string;
  filename: string;
  contentType?: string;
  partName: string;
  size?: number;
}

export interface AttachmentRow {
  id: string;
  messageId: string;
  accountId: string;
  filename: string;
  partName: string;
  status: AttachmentStatus;
  charCount: number;
}

/** A retrieved attachment chunk plus the email and file it came from, for citation. */
export interface AttachmentChunkHit {
  chunkRowid: number;
  text: string;
  messageId: string;
  filename: string;
  distance: number;
}

/**
 * Stores attachments and their extracted-text chunks, with a per-chunk vector index
 * and FTS index, mirroring the email embedding and keyword stores. Chunks are keyed by
 * their implicit rowid, shared by both the vec index and attachment_fts.
 */
export class AttachmentRepository {
  private readonly stmts: {
    insertAttachment: Statement<unknown[]>;
    findAttachment: Statement<unknown[]>;
    setStatus: Statement<unknown[]>;
    deleteChunks: Statement<unknown[]>;
    insertChunk: Statement<unknown[]>;
    insertVec: Statement<unknown[]>;
    findEmbRow: Statement<unknown[]>;
    updateVec: Statement<unknown[]>;
    insertEmbIndex: Statement<unknown[]>;
    countAllEmb: Statement<unknown[]>;
    searchKnn: Statement<[Buffer, number, string, string, number]>;
    countExtracted: Statement<unknown[]>;
    listForMessage: Statement<unknown[]>;
  };

  /** Prepare the reusable statements for attachment, chunk, embedding, and FTS access. */
  constructor(private db: Database) {
    this.stmts = {
      insertAttachment: db.prepare(
        `INSERT INTO attachments (id, message_id, account_id, filename, content_type, part_name, size, status, indexed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
         ON CONFLICT (message_id, account_id, part_name) DO UPDATE SET
           filename = excluded.filename,
           content_type = excluded.content_type,
           size = excluded.size,
           indexed_at = excluded.indexed_at`,
      ),
      findAttachment: db.prepare(
        `SELECT id, message_id, account_id, filename, part_name, status, char_count
           FROM attachments WHERE message_id = ? AND account_id = ? AND part_name = ?`,
      ),
      setStatus: db.prepare(
        'UPDATE attachments SET status = ?, error = ?, char_count = ?, indexed_at = ? WHERE id = ?',
      ),
      deleteChunks: db.prepare('DELETE FROM attachment_chunks WHERE attachment_id = ?'),
      insertChunk: db.prepare(
        'INSERT INTO attachment_chunks (attachment_id, message_id, account_id, chunk_index, text, created_at) VALUES (?, ?, ?, ?, ?, ?)',
      ),
      insertVec: db.prepare('INSERT INTO attachment_chunk_embeddings (embedding) VALUES (?)'),
      findEmbRow: db.prepare(
        'SELECT rowid FROM attachment_chunk_embedding_index WHERE chunk_rowid = ? AND model_id = ?',
      ),
      updateVec: db.prepare('UPDATE attachment_chunk_embeddings SET embedding = ? WHERE rowid = ?'),
      insertEmbIndex: db.prepare(
        'INSERT INTO attachment_chunk_embedding_index (rowid, chunk_rowid, account_id, model_id, created_at) VALUES (?, ?, ?, ?, ?)',
      ),
      countAllEmb: db.prepare('SELECT COUNT(*) AS c FROM attachment_chunk_embedding_index'),
      searchKnn: db.prepare<[Buffer, number, string, string, number]>(
        `SELECT ei.chunk_rowid, ev.distance
           FROM attachment_chunk_embeddings ev
           JOIN attachment_chunk_embedding_index ei ON ei.rowid = ev.rowid
          WHERE ev.embedding MATCH ?
            AND k = ?
            AND ei.account_id = ?
            AND ei.model_id = ?
          ORDER BY ev.distance ASC
          LIMIT ?`,
      ),
      countExtracted: db.prepare(
        `SELECT COUNT(*) AS c FROM attachments WHERE account_id = ? AND status = 'extracted'`,
      ),
      listForMessage: db.prepare(
        `SELECT id, message_id, account_id, filename, part_name, status, char_count
           FROM attachments
          WHERE account_id = ? AND message_id = ?
          ORDER BY filename, part_name`,
      ),
    };
  }

  /** Insert or refresh an attachment row, returning its id and prior status for skip logic. */
  upsertAttachment(meta: AttachmentMeta): { id: string; priorStatus: AttachmentStatus | null } {
    const existing = this.stmts.findAttachment.get(
      meta.messageId,
      meta.accountId,
      meta.partName,
    ) as { id: string; status: AttachmentStatus } | undefined;
    const id = existing?.id ?? randomUUID();
    this.stmts.insertAttachment.run(
      id,
      meta.messageId,
      meta.accountId,
      meta.filename,
      meta.contentType ?? null,
      meta.partName,
      meta.size ?? null,
      Date.now(),
    );
    return { id, priorStatus: existing?.status ?? null };
  }

  /** Update an attachment's extraction status, character count, and optional error. */
  setStatus(
    id: string,
    status: AttachmentStatus,
    charCount = 0,
    error: string | null = null,
  ): void {
    this.stmts.setStatus.run(status, error, charCount, Date.now(), id);
  }

  /** Replace an attachment's chunks, dropping stale embeddings and FTS via triggers, and return rowids. */
  replaceChunks(
    attachmentId: string,
    messageId: string,
    accountId: string,
    chunks: string[],
  ): number[] {
    const tx = this.db.transaction(() => {
      this.stmts.deleteChunks.run(attachmentId);
      const now = Date.now();
      const rowids: number[] = [];
      chunks.forEach((text, i) => {
        const r = this.stmts.insertChunk.run(attachmentId, messageId, accountId, i, text, now);
        rowids.push(Number(r.lastInsertRowid));
      });
      return rowids;
    });
    return tx();
  }

  /** Store one chunk's embedding, replacing it if re-embedded under the same model. */
  saveChunkEmbedding(
    chunkRowid: number,
    accountId: string,
    modelId: string,
    vector: ArrayLike<number>,
  ): void {
    if (vector.length !== EMBEDDING_DIM) {
      throw new Error(`attachment chunk embedding dim ${vector.length} != ${EMBEDDING_DIM}`);
    }
    const model = canonicalizeModelId(modelId);
    const buf = vectorToBuffer(vector instanceof Float32Array ? vector : Float32Array.from(vector));
    const tx = this.db.transaction(() => {
      const existing = this.stmts.findEmbRow.get(chunkRowid, model) as
        | { rowid: number }
        | undefined;
      if (existing) {
        this.stmts.updateVec.run(buf, existing.rowid);
        return;
      }
      const ins = this.stmts.insertVec.run(buf);
      this.stmts.insertEmbIndex.run(
        Number(ins.lastInsertRowid),
        chunkRowid,
        accountId,
        model,
        Date.now(),
      );
    });
    tx();
  }

  /** KNN over attachment chunk embeddings; widens the overfetch like the email store. */
  searchChunks(
    accountId: string,
    modelId: string,
    queryVector: ArrayLike<number>,
    k = 8,
  ): Array<{ chunkRowid: number; distance: number }> {
    const model = canonicalizeModelId(modelId);
    const buf = vectorToBuffer(
      queryVector instanceof Float32Array ? queryVector : Float32Array.from(queryVector),
    );
    const total = (this.stmts.countAllEmb.get() as { c: number }).c;
    if (total === 0) return [];
    let overfetch = Math.max(k * 8, 64);
    let rows: Array<{ chunk_rowid: number; distance: number }> = [];
    while (true) {
      rows = this.stmts.searchKnn.all(buf, overfetch, accountId, model, k) as typeof rows;
      if (rows.length >= k || overfetch >= total) break;
      overfetch = Math.min(overfetch * 4, total);
    }
    return rows.map((r) => ({ chunkRowid: r.chunk_rowid, distance: r.distance }));
  }

  /** BM25 keyword search over attachment chunk text. Returns chunk rowids, best first. */
  keywordSearchChunks(accountId: string, query: string, limit = 8): number[] {
    const match = sanitizeFtsQuery(query);
    if (!match) return [];
    const rows = this.db
      .prepare(
        `SELECT c.rowid AS chunk_rowid
           FROM attachment_fts
           JOIN attachment_chunks c ON c.rowid = attachment_fts.rowid
          WHERE attachment_fts MATCH ? AND c.account_id = ?
          ORDER BY attachment_fts.rank
          LIMIT ?`,
      )
      .all(match, accountId, limit) as Array<{ chunk_rowid: number }>;
    return rows.map((r) => r.chunk_rowid);
  }

  /** Load chunk text plus its email and filename for the given chunk rowids. */
  loadChunks(chunkRowids: number[], accountId: string): Map<number, AttachmentChunkHit> {
    const out = new Map<number, AttachmentChunkHit>();
    if (chunkRowids.length === 0) return out;
    const placeholders = chunkRowids.map(() => '?').join(',');
    const rows = this.db
      .prepare(
        `SELECT c.rowid AS chunk_rowid, c.text, c.message_id, a.filename
           FROM attachment_chunks c
           JOIN attachments a ON a.id = c.attachment_id
          WHERE c.account_id = ? AND c.rowid IN (${placeholders})`,
      )
      .all(accountId, ...chunkRowids) as Array<{
      chunk_rowid: number;
      text: string;
      message_id: string;
      filename: string;
    }>;
    for (const r of rows) {
      out.set(r.chunk_rowid, {
        chunkRowid: r.chunk_rowid,
        text: r.text,
        messageId: r.message_id,
        filename: r.filename,
        distance: 0,
      });
    }
    return out;
  }

  /**
   * Load one email's attachment chunks in document order, up to a character budget. Lets the chat
   * pull a retrieved email's document text into context even when no single chunk independently
   * matched the question. Returns [] for an email with no extracted attachments.
   */
  loadChunksForMessage(
    accountId: string,
    messageId: string,
    charBudget: number,
  ): Array<{ text: string; filename: string }> {
    const rows = this.db
      .prepare(
        `SELECT c.text, a.filename
           FROM attachment_chunks c
           JOIN attachments a ON a.id = c.attachment_id
          WHERE c.account_id = ? AND c.message_id = ?
          ORDER BY a.filename, c.chunk_index`,
      )
      .all(accountId, messageId) as Array<{ text: string; filename: string }>;
    const out: Array<{ text: string; filename: string }> = [];
    let used = 0;
    for (const r of rows) {
      if (used >= charBudget) break;
      out.push({ text: r.text, filename: r.filename });
      used += r.text.length;
    }
    return out;
  }

  /** Count attachments whose text has been successfully extracted for an account. */
  countExtracted(accountId: string): number {
    return (this.stmts.countExtracted.get(accountId) as { c: number }).c;
  }

  /** List an email's attachment rows for the given account, ordered by filename and part. */
  listForMessage(accountId: string, messageId: string): AttachmentRow[] {
    const rows = this.stmts.listForMessage.all(accountId, messageId) as Array<{
      id: string;
      message_id: string;
      account_id: string;
      filename: string;
      part_name: string;
      status: AttachmentStatus;
      char_count: number | null;
    }>;
    return rows.map((r) => ({
      id: r.id,
      messageId: r.message_id,
      accountId: r.account_id,
      filename: r.filename,
      partName: r.part_name,
      status: r.status,
      charCount: r.char_count ?? 0,
    }));
  }

  /**
   * List extracted attachments for an account so a chat query that names a file can be matched
   * against the actual filenames, with ties broken by recency. One small indexed scan.
   */
  listExtractedNames(
    accountId: string,
  ): Array<{ attachmentId: string; messageId: string; filename: string; date: number | null }> {
    return this.db
      .prepare(
        `SELECT a.id AS attachmentId, a.message_id AS messageId, a.filename, e.date
           FROM attachments a
           LEFT JOIN emails e ON e.message_id = a.message_id AND e.account_id = a.account_id
          WHERE a.account_id = ? AND a.status = 'extracted'`,
      )
      .all(accountId) as Array<{
      attachmentId: string;
      messageId: string;
      filename: string;
      date: number | null;
    }>;
  }

  /**
   * Load all chunks of one attachment by attachment id, in document order, bounded by maxChunks.
   * Used by filename-targeted retrieval so a multi-attachment email only feeds the chunks of the
   * file the user actually named.
   */
  loadAllChunksForAttachment(
    accountId: string,
    attachmentId: string,
    maxChunks = 400,
  ): Array<{ text: string; filename: string }> {
    return this.db
      .prepare(
        `SELECT c.text, a.filename
           FROM attachment_chunks c
           JOIN attachments a ON a.id = c.attachment_id
          WHERE c.account_id = ? AND c.attachment_id = ?
          ORDER BY c.chunk_index
          LIMIT ?`,
      )
      .all(accountId, attachmentId, maxChunks) as Array<{ text: string; filename: string }>;
  }

  /**
   * Load all chunks of a message's attachments across its files, bounded by maxChunks, so a
   * retrieved email's attachment chunks can be ranked by relevance to the question rather than only
   * the first ones in document order.
   */
  loadAllChunksForMessage(
    accountId: string,
    messageId: string,
    maxChunks = 200,
  ): Array<{ text: string; filename: string }> {
    return this.db
      .prepare(
        `SELECT c.text, a.filename
           FROM attachment_chunks c
           JOIN attachments a ON a.id = c.attachment_id
          WHERE c.account_id = ? AND c.message_id = ?
          ORDER BY a.filename, c.chunk_index
          LIMIT ?`,
      )
      .all(accountId, messageId, maxChunks) as Array<{ text: string; filename: string }>;
  }
}
