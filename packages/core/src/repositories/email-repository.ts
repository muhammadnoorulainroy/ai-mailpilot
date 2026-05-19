import type { Database } from 'better-sqlite3';

export interface EmailRow {
  messageId: string;
  accountId: string;
  folder: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  hasAttachments: boolean;
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
}

export interface ListEmailsOptions {
  accountId: string;
  folder?: string;
  limit?: number;
  offset?: number;
  sinceDate?: number;
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
      indexedAt: now,
    };

    this.db
      .prepare(
        `INSERT INTO emails (message_id, account_id, folder, subject, from_addr, date, has_attachments, indexed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (message_id, account_id) DO UPDATE SET
           folder = excluded.folder,
           subject = excluded.subject,
           from_addr = excluded.from_addr,
           date = excluded.date,
           has_attachments = excluded.has_attachments`,
      )
      .run(
        row.messageId,
        row.accountId,
        row.folder,
        row.subject,
        row.fromAddr,
        row.date,
        row.hasAttachments ? 1 : 0,
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
      .prepare(
        `SELECT message_id, account_id, folder, subject, from_addr, date, has_attachments, indexed_at
         FROM emails WHERE message_id = ? AND account_id = ?`,
      )
      .get(messageId, accountId) as
      | {
          message_id: string;
          account_id: string;
          folder: string;
          subject: string | null;
          from_addr: string | null;
          date: number | null;
          has_attachments: number;
          indexed_at: number;
        }
      | undefined;

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
        `SELECT message_id, account_id, folder, subject, from_addr, date, has_attachments, indexed_at
         FROM emails
         WHERE ${filters.join(' AND ')}
         ORDER BY COALESCE(date, indexed_at) DESC
         LIMIT ? OFFSET ?`,
      )
      .all(...params) as Array<{
      message_id: string;
      account_id: string;
      folder: string;
      subject: string | null;
      from_addr: string | null;
      date: number | null;
      has_attachments: number;
      indexed_at: number;
    }>;

    return rows.map((r) => this.fromRow(r));
  }

  count(accountId: string, folder?: string): number {
    const sql = folder
      ? 'SELECT COUNT(*) as c FROM emails WHERE account_id = ? AND folder = ?'
      : 'SELECT COUNT(*) as c FROM emails WHERE account_id = ?';
    const params: (string | undefined)[] = folder ? [accountId, folder] : [accountId];
    const row = this.db.prepare(sql).get(...params) as { c: number };
    return row.c;
  }

  delete(messageId: string, accountId: string): boolean {
    const result = this.db
      .prepare('DELETE FROM emails WHERE message_id = ? AND account_id = ?')
      .run(messageId, accountId);
    return result.changes > 0;
  }

  private fromRow(row: {
    message_id: string;
    account_id: string;
    folder: string;
    subject: string | null;
    from_addr: string | null;
    date: number | null;
    has_attachments: number;
    indexed_at: number;
  }): EmailRow {
    return {
      messageId: row.message_id,
      accountId: row.account_id,
      folder: row.folder,
      subject: row.subject,
      fromAddr: row.from_addr,
      date: row.date,
      hasAttachments: row.has_attachments === 1,
      indexedAt: row.indexed_at,
    };
  }
}
