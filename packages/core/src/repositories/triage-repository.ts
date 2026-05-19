import type { Database } from 'better-sqlite3';

export type TriageBucket = 'urgent' | 'summarize' | 'spam' | 'personal';

export interface TriageRow {
  messageId: string;
  accountId: string;
  bucket: TriageBucket;
  reasoning: string | null;
  classifiedAt: number;
}

export interface UpsertTriageInput {
  messageId: string;
  accountId: string;
  bucket: TriageBucket;
  reasoning?: string;
}

export interface BucketCount {
  bucket: TriageBucket;
  count: number;
}

export class TriageRepository {
  constructor(private db: Database) {}

  upsert(input: UpsertTriageInput): TriageRow {
    const now = Date.now();
    const row: TriageRow = {
      messageId: input.messageId,
      accountId: input.accountId,
      bucket: input.bucket,
      reasoning: input.reasoning ?? null,
      classifiedAt: now,
    };

    this.db
      .prepare(
        `INSERT INTO triage (message_id, account_id, bucket, reasoning, classified_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT (message_id, account_id) DO UPDATE SET
           bucket = excluded.bucket,
           reasoning = excluded.reasoning,
           classified_at = excluded.classified_at`,
      )
      .run(row.messageId, row.accountId, row.bucket, row.reasoning, row.classifiedAt);

    return row;
  }

  findById(messageId: string, accountId: string): TriageRow | null {
    const row = this.db
      .prepare<
        [string, string],
        {
          message_id: string;
          account_id: string;
          bucket: TriageBucket;
          reasoning: string | null;
          classified_at: number;
        }
      >(
        'SELECT message_id, account_id, bucket, reasoning, classified_at FROM triage WHERE message_id = ? AND account_id = ?',
      )
      .get(messageId, accountId);

    if (!row) return null;
    return {
      messageId: row.message_id,
      accountId: row.account_id,
      bucket: row.bucket,
      reasoning: row.reasoning,
      classifiedAt: row.classified_at,
    };
  }

  clearForAccount(accountId: string): number {
    const result = this.db.prepare('DELETE FROM triage WHERE account_id = ?').run(accountId);
    return result.changes;
  }

  countByBucket(accountId: string): BucketCount[] {
    const rows = this.db
      .prepare<[string], { bucket: TriageBucket; count: number }>(
        'SELECT bucket, COUNT(*) AS count FROM triage WHERE account_id = ? GROUP BY bucket',
      )
      .all(accountId);
    return rows;
  }

  countUnclassified(accountId: string): number {
    const row = this.db
      .prepare<[string], { c: number }>(
        `SELECT COUNT(*) AS c FROM emails e
          LEFT JOIN triage t
            ON t.message_id = e.message_id AND t.account_id = e.account_id
         WHERE e.account_id = ? AND t.message_id IS NULL`,
      )
      .get(accountId);
    return row?.c ?? 0;
  }

  /**
   * Returns emails that have no triage classification yet.
   * Bodies are included if available (helps LLM context).
   */
  findUnclassifiedEmails(
    accountId: string,
    limit = 16,
  ): Array<{
    messageId: string;
    subject: string | null;
    fromAddr: string | null;
    body: string | null;
    bodyFormat: string | null;
  }> {
    return this.db
      .prepare<
        [string, number],
        {
          message_id: string;
          subject: string | null;
          from_addr: string | null;
          body: string | null;
          body_format: string | null;
        }
      >(
        `SELECT e.message_id, e.subject, e.from_addr, e.body, e.body_format
           FROM emails e
           LEFT JOIN triage t
             ON t.message_id = e.message_id AND t.account_id = e.account_id
          WHERE e.account_id = ? AND t.message_id IS NULL
          ORDER BY COALESCE(e.date, e.indexed_at) DESC
          LIMIT ?`,
      )
      .all(accountId, limit)
      .map((r) => ({
        messageId: r.message_id,
        subject: r.subject,
        fromAddr: r.from_addr,
        body: r.body,
        bodyFormat: r.body_format,
      }));
  }
}
