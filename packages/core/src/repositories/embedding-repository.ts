import type { Database } from 'better-sqlite3';

export interface EmbeddingRef {
  messageId: string;
  accountId: string;
  modelId: string;
}

export interface SimilarityHit {
  messageId: string;
  accountId: string;
  modelId: string;
  distance: number;
}

export class EmbeddingRepository {
  constructor(private db: Database) {}

  saveEmbedding(ref: EmbeddingRef, vector: number[]): void {
    const now = Date.now();
    const buf = Buffer.from(new Float32Array(vector).buffer);

    const tx = this.db.transaction(() => {
      const existing = this.db
        .prepare<[string, string, string], { rowid: number }>(
          'SELECT rowid FROM email_embedding_index WHERE message_id = ? AND account_id = ? AND model_id = ?',
        )
        .get(ref.messageId, ref.accountId, ref.modelId);

      if (existing) {
        this.db
          .prepare('UPDATE email_embeddings SET embedding = ? WHERE rowid = ?')
          .run(buf, existing.rowid);
        return;
      }

      const insertVec = this.db
        .prepare('INSERT INTO email_embeddings (embedding) VALUES (?)')
        .run(buf);
      const rowid = Number(insertVec.lastInsertRowid);

      this.db
        .prepare(
          `INSERT INTO email_embedding_index (rowid, message_id, account_id, model_id, created_at)
           VALUES (?, ?, ?, ?, ?)`,
        )
        .run(rowid, ref.messageId, ref.accountId, ref.modelId, now);
    });

    tx();
  }

  getEmbedding(ref: EmbeddingRef): number[] | null {
    const row = this.db
      .prepare<[string, string, string], { embedding: Buffer }>(
        `SELECT ev.embedding FROM email_embeddings ev
           JOIN email_embedding_index ei ON ei.rowid = ev.rowid
          WHERE ei.message_id = ? AND ei.account_id = ? AND ei.model_id = ?`,
      )
      .get(ref.messageId, ref.accountId, ref.modelId);

    if (!row) return null;
    return Array.from(new Float32Array(row.embedding.buffer, row.embedding.byteOffset, row.embedding.byteLength / 4));
  }

  countForModel(accountId: string, modelId: string): number {
    const row = this.db
      .prepare<[string, string], { c: number }>(
        'SELECT COUNT(*) AS c FROM email_embedding_index WHERE account_id = ? AND model_id = ?',
      )
      .get(accountId, modelId);
    return row?.c ?? 0;
  }

  /**
   * KNN search via sqlite-vec. Returns the top K most similar emails to the query vector.
   * Distance is L2; lower = more similar. bge-m3 returns unit vectors, so cosine = 1 - d^2/2.
   *
   * sqlite-vec applies `k` BEFORE the WHERE clause, so we overfetch a multiple of k to
   * ensure enough rows survive the account_id / model_id filters, then trim to k.
   */
  search(
    accountId: string,
    modelId: string,
    queryVector: number[],
    k = 10,
  ): SimilarityHit[] {
    const buf = Buffer.from(new Float32Array(queryVector).buffer);
    const overfetch = Math.max(k * 8, 64);

    const rows = this.db
      .prepare<
        [Buffer, number, string, string],
        { message_id: string; account_id: string; model_id: string; distance: number }
      >(
        `SELECT ei.message_id, ei.account_id, ei.model_id, ev.distance
           FROM email_embeddings ev
           JOIN email_embedding_index ei ON ei.rowid = ev.rowid
          WHERE ev.embedding MATCH ?
            AND k = ?
            AND ei.account_id = ?
            AND ei.model_id = ?
          ORDER BY ev.distance ASC
          LIMIT ?`,
      )
      .all(buf, overfetch, accountId, modelId, k);

    return rows.map((r) => ({
      messageId: r.message_id,
      accountId: r.account_id,
      modelId: r.model_id,
      distance: r.distance,
    }));
  }

  deleteByModel(modelId: string): number {
    const result = this.db.transaction(() => {
      const refs = this.db
        .prepare<[string], { rowid: number }>(
          'SELECT rowid FROM email_embedding_index WHERE model_id = ?',
        )
        .all(modelId);

      const delVec = this.db.prepare('DELETE FROM email_embeddings WHERE rowid = ?');
      for (const r of refs) delVec.run(r.rowid);

      const res = this.db
        .prepare('DELETE FROM email_embedding_index WHERE model_id = ?')
        .run(modelId);
      return res.changes;
    })();

    return result;
  }
}
