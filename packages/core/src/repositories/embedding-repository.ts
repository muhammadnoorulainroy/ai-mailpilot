/**
 * Repository for storing and querying email embeddings backed by a sqlite-vec index,
 * including save, lookup, bulk load, count, and KNN similarity search.
 */
import type { Database, Statement } from 'better-sqlite3';
import { EMBEDDING_DIM } from '../db/schema.js';
import { assertEmbeddingDim, bufferToVector, vectorToBuffer } from '../util/vector.js';
import { canonicalizeModelId } from '../util/model-id.js';

/** Identifies a stored embedding by message, account, and model. */
export interface EmbeddingRef {
  messageId: string;
  accountId: string;
  modelId: string;
}

/** A nearest-neighbour match returned by a similarity search, with its distance. */
export interface SimilarityHit {
  messageId: string;
  accountId: string;
  modelId: string;
  distance: number;
}

/** A single message embedding returned by a bulk load. */
export interface BulkEmbeddingEntry {
  messageId: string;
  vector: Float32Array;
}

/** Thrown when a vector's dimension does not match the configured storage dimension. */
export class EmbeddingDimensionError extends Error {
  /** Builds the error message from the actual vector dimension that was rejected. */
  constructor(actual: number) {
    super(
      `embedding dimension mismatch: storage requires ${EMBEDDING_DIM}, model returned ${actual}. ` +
        `Use a 1024-dim model (e.g. bge-m3).`,
    );
    this.name = 'EmbeddingDimensionError';
  }
}

/** Stores and queries email embeddings backed by a sqlite-vec index. */
export class EmbeddingRepository {
  private readonly stmts: {
    findRowId: Statement<unknown[]>;
    updateVec: Statement<unknown[]>;
    insertVec: Statement<unknown[]>;
    insertIndex: Statement<unknown[]>;
    getEmbedding: Statement<unknown[]>;
    countForModel: Statement<unknown[]>;
    listForAccount: Statement<unknown[]>;
    countAllIndex: Statement<unknown[]>;
    searchKnn: Statement<[Buffer, number, string, string, number]>;
  };

  /** Prepares the reusable statements for the embedding and index tables on the given database. */
  constructor(private db: Database) {
    this.stmts = {
      findRowId: db.prepare(
        'SELECT rowid FROM email_embedding_index WHERE message_id = ? AND account_id = ? AND model_id = ?',
      ),
      updateVec: db.prepare('UPDATE email_embeddings SET embedding = ? WHERE rowid = ?'),
      insertVec: db.prepare('INSERT INTO email_embeddings (embedding) VALUES (?)'),
      insertIndex: db.prepare(
        `INSERT INTO email_embedding_index (rowid, message_id, account_id, model_id, created_at)
         VALUES (?, ?, ?, ?, ?)`,
      ),
      getEmbedding: db.prepare(
        `SELECT ev.embedding FROM email_embeddings ev
           JOIN email_embedding_index ei ON ei.rowid = ev.rowid
          WHERE ei.message_id = ? AND ei.account_id = ? AND ei.model_id = ?`,
      ),
      countForModel: db.prepare(
        'SELECT COUNT(*) AS c FROM email_embedding_index WHERE account_id = ? AND model_id = ?',
      ),
      listForAccount: db.prepare(
        `SELECT ei.message_id, ev.embedding
           FROM email_embeddings ev
           JOIN email_embedding_index ei ON ei.rowid = ev.rowid
          WHERE ei.account_id = ? AND ei.model_id = ?`,
      ),
      countAllIndex: db.prepare('SELECT COUNT(*) AS c FROM email_embedding_index'),
      searchKnn: db.prepare<[Buffer, number, string, string, number]>(
        `SELECT ei.message_id, ei.account_id, ei.model_id, ev.distance
           FROM email_embeddings ev
           JOIN email_embedding_index ei ON ei.rowid = ev.rowid
          WHERE ev.embedding MATCH ?
            AND k = ?
            AND ei.account_id = ?
            AND ei.model_id = ?
          ORDER BY ev.distance ASC
          LIMIT ?`,
      ),
    };
  }

  /** Insert or update the embedding for a message, account, and model. */
  saveEmbedding(ref: EmbeddingRef, vector: ArrayLike<number>): void {
    if (vector.length !== EMBEDDING_DIM) {
      throw new EmbeddingDimensionError(vector.length);
    }
    const modelId = canonicalizeModelId(ref.modelId);
    const f32 = vector instanceof Float32Array ? vector : Float32Array.from(vector);
    const buf = vectorToBuffer(f32);
    const now = Date.now();

    const tx = this.db.transaction(() => {
      const existing = this.stmts.findRowId.get(ref.messageId, ref.accountId, modelId) as
        | { rowid: number }
        | undefined;

      if (existing) {
        this.stmts.updateVec.run(buf, existing.rowid);
        return;
      }

      const ins = this.stmts.insertVec.run(buf);
      const rowid = Number(ins.lastInsertRowid);
      this.stmts.insertIndex.run(rowid, ref.messageId, ref.accountId, modelId, now);
    });

    tx();
  }

  /** Return the stored vector as a Float32Array, or null if not found. */
  getEmbedding(ref: EmbeddingRef): Float32Array | null {
    const modelId = canonicalizeModelId(ref.modelId);
    const row = this.stmts.getEmbedding.get(ref.messageId, ref.accountId, modelId) as
      | { embedding: Buffer }
      | undefined;
    return row ? bufferToVector(row.embedding) : null;
  }

  /** Bulk-load all email embeddings for an account and model in one query. */
  listForAccount(accountId: string, modelId: string): BulkEmbeddingEntry[] {
    const canonical = canonicalizeModelId(modelId);
    const rows = this.stmts.listForAccount.all(accountId, canonical) as Array<{
      message_id: string;
      embedding: Buffer;
    }>;
    return rows.map((r) => ({ messageId: r.message_id, vector: bufferToVector(r.embedding) }));
  }

  /** Count stored embeddings for an account and model. */
  countForModel(accountId: string, modelId: string): number {
    const canonical = canonicalizeModelId(modelId);
    const row = this.stmts.countForModel.get(accountId, canonical) as { c: number } | undefined;
    return row?.c ?? 0;
  }

  /**
   * KNN search via sqlite-vec, returning up to k nearest hits for the account and model.
   * Overfetch widens until enough filtered hits are found, since the vec0 index is global.
   */
  search(
    accountId: string,
    modelId: string,
    queryVector: ArrayLike<number>,
    k = 10,
  ): SimilarityHit[] {
    assertEmbeddingDim(queryVector);
    const canonical = canonicalizeModelId(modelId);
    const f32 = queryVector instanceof Float32Array ? queryVector : Float32Array.from(queryVector);
    const buf = vectorToBuffer(f32);

    const totalIndexed = (this.stmts.countAllIndex.get() as { c: number }).c;

    let overfetch = Math.max(k * 8, 64);
    let rows: Array<{
      message_id: string;
      account_id: string;
      model_id: string;
      distance: number;
    }> = [];

    while (true) {
      rows = this.stmts.searchKnn.all(buf, overfetch, accountId, canonical, k) as typeof rows;
      if (rows.length >= k || overfetch >= totalIndexed) break;
      overfetch = Math.min(overfetch * 4, totalIndexed);
    }

    return rows.map((r) => ({
      messageId: r.message_id,
      accountId: r.account_id,
      modelId: r.model_id,
      distance: r.distance,
    }));
  }
}
