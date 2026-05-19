import type { Database } from 'better-sqlite3';

export type EmailBodyFormat = 'text' | 'html';

export interface EmailRow {
  messageId: string;
  accountId: string;
  folder: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  hasAttachments: boolean;
  body: string | null;
  bodyFormat: EmailBodyFormat;
  indexedAt: number;
}

export interface UpsertEmailInput {
  messageId: string;
  accountId: string;
  folder: string;
  subject?: string;
  fromAddr?: string;
  date?: number;
  hasAttachments?: boolean;
  body?: string;
  bodyFormat?: EmailBodyFormat;
}

export interface ListEmailsOptions {
  accountId: string;
  folder?: string;
  limit?: number;
  offset?: number;
  sinceDate?: number;
}

interface EmailDbRow {
  message_id: string;
  account_id: string;
  folder: string;
  subject: string | null;
  from_addr: string | null;
  date: number | null;
  has_attachments: number;
  body: string | null;
  body_format: string | null;
  indexed_at: number;
}

export class EmailRepository {
  constructor(private db: Database) {}

  upsert(input: UpsertEmailInput): EmailRow {
    const now = Date.now();
    const row: EmailRow = {
      messageId: input.messageId,
      accountId: input.accountId,
      folder: input.folder,
      subject: input.subject ?? null,
      fromAddr: input.fromAddr ?? null,
      date: input.date ?? null,
      hasAttachments: input.hasAttachments ?? false,
      body: input.body ?? null,
      bodyFormat: input.bodyFormat ?? 'text',
      indexedAt: now,
    };

    this.db
      .prepare(
        `INSERT INTO emails (message_id, account_id, folder, subject, from_addr, date, has_attachments, body, body_format, indexed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (message_id, account_id) DO UPDATE SET
           folder = excluded.folder,
           subject = excluded.subject,
           from_addr = excluded.from_addr,
           date = excluded.date,
           has_attachments = excluded.has_attachments,
           body = COALESCE(excluded.body, emails.body),
           body_format = COALESCE(excluded.body_format, emails.body_format)`,
      )
      .run(
        row.messageId,
        row.accountId,
        row.folder,
        row.subject,
        row.fromAddr,
        row.date,
        row.hasAttachments ? 1 : 0,
        row.body,
        row.bodyFormat,
        row.indexedAt,
      );

    return row;
  }

  upsertBatch(inputs: UpsertEmailInput[]): number {
    const tx = this.db.transaction((items: UpsertEmailInput[]) => {
      for (const item of items) {
        this.upsert(item);
      }
    });
    tx(inputs);
    return inputs.length;
  }

  findById(messageId: string, accountId: string): EmailRow | null {
    const row = this.db
      .prepare(this.selectSql('WHERE message_id = ? AND account_id = ?'))
      .get(messageId, accountId) as EmailDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  list(opts: ListEmailsOptions): EmailRow[] {
    const limit = opts.limit ?? 100;
    const offset = opts.offset ?? 0;
    const filters: string[] = ['account_id = ?'];
    const params: (string | number)[] = [opts.accountId];

    if (opts.folder) {
      filters.push('folder = ?');
      params.push(opts.folder);
    }

    if (opts.sinceDate !== undefined) {
      filters.push('date >= ?');
      params.push(opts.sinceDate);
    }

    params.push(limit, offset);

    const rows = this.db
      .prepare(
        `${this.selectSql(`WHERE ${filters.join(' AND ')}`)}
         ORDER BY COALESCE(date, indexed_at) DESC
         LIMIT ? OFFSET ?`,
      )
      .all(...params) as EmailDbRow[];

    return rows.map((r) => this.fromRow(r));
  }

  /**
   * Returns emails that have a body but no embedding for the given model.
   * Used by the embedding orchestrator.
   */
  findUnembedded(accountId: string, modelId: string, limit = 50): EmailRow[] {
    const rows = this.db
      .prepare(
        `SELECT e.message_id, e.account_id, e.folder, e.subject, e.from_addr, e.date,
                e.has_attachments, e.body, e.body_format, e.indexed_at
           FROM emails e
           LEFT JOIN email_embedding_index ei
             ON ei.message_id = e.message_id
            AND ei.account_id = e.account_id
            AND ei.model_id = ?
          WHERE e.account_id = ?
            AND e.body IS NOT NULL
            AND ei.rowid IS NULL
          ORDER BY COALESCE(e.date, e.indexed_at) DESC
          LIMIT ?`,
      )
      .all(modelId, accountId, limit) as EmailDbRow[];

    return rows.map((r) => this.fromRow(r));
  }

  countUnembedded(accountId: string, modelId: string): number {
    const row = this.db
      .prepare(
        `SELECT COUNT(*) AS c FROM emails e
          LEFT JOIN email_embedding_index ei
            ON ei.message_id = e.message_id
           AND ei.account_id = e.account_id
           AND ei.model_id = ?
         WHERE e.account_id = ?
           AND e.body IS NOT NULL
           AND ei.rowid IS NULL`,
      )
      .get(modelId, accountId) as { c: number };
    return row.c;
  }

  count(accountId: string, folder?: string): number {
    const sql = folder
      ? 'SELECT COUNT(*) as c FROM emails WHERE account_id = ? AND folder = ?'
      : 'SELECT COUNT(*) as c FROM emails WHERE account_id = ?';
    const params: string[] = folder ? [accountId, folder] : [accountId];
    const row = this.db.prepare(sql).get(...params) as { c: number };
    return row.c;
  }

  delete(messageId: string, accountId: string): boolean {
    const result = this.db
      .prepare('DELETE FROM emails WHERE message_id = ? AND account_id = ?')
      .run(messageId, accountId);
    return result.changes > 0;
  }

  private selectSql(whereClause: string): string {
    return `SELECT message_id, account_id, folder, subject, from_addr, date,
                   has_attachments, body, body_format, indexed_at
              FROM emails
              ${whereClause}`;
  }

  private fromRow(row: EmailDbRow): EmailRow {
    return {
      messageId: row.message_id,
      accountId: row.account_id,
      folder: row.folder,
      subject: row.subject,
      fromAddr: row.from_addr,
      date: row.date,
      hasAttachments: row.has_attachments === 1,
      body: row.body,
      bodyFormat: (row.body_format as EmailBodyFormat | null) ?? 'text',
      indexedAt: row.indexed_at,
    };
  }
}
