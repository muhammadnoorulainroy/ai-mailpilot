/**
 * Persistence layer for emails and their full-text search index in the local SQLite
 * database, covering upserts, lookups, summary listings, and embedding coverage queries.
 */
import type { Database, Statement } from 'better-sqlite3';
import { canonicalizeModelId } from '../util/model-id.js';
import { sanitizeFtsQuery } from '../util/text.js';

/** Storage format of an email body. */
export type EmailBodyFormat = 'text' | 'html';

/** A stored email including its body. */
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

/** A stored email without its body, for list views and dashboards. */
export interface EmailSummary {
  messageId: string;
  accountId: string;
  folder: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  hasAttachments: boolean;
  indexedAt: number;
}

/** Input for inserting or updating an email. */
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
  bodyFetched?: boolean;
}

/** Filters and paging for listing emails. */
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

interface EmailSummaryDbRow {
  message_id: string;
  account_id: string;
  folder: string;
  subject: string | null;
  from_addr: string | null;
  date: number | null;
  has_attachments: number;
  indexed_at: number;
}

const SELECT_FULL_COLS =
  'message_id, account_id, folder, subject, from_addr, date, has_attachments, body, body_format, indexed_at';
const SELECT_SUMMARY_COLS =
  'message_id, account_id, folder, subject, from_addr, date, has_attachments, indexed_at';

/** Reads and writes emails and their search index in the local database. */
export class EmailRepository {
  private readonly stmts: {
    upsert: Statement<unknown[]>;
    findById: Statement<unknown[]>;
    getBody: Statement<unknown[]>;
    deleteEmbeddings: Statement<unknown[]>;
    delete: Statement<unknown[]>;
    countAll: Statement<unknown[]>;
    countByFolder: Statement<unknown[]>;
    countUnembedded: Statement<unknown[]>;
    listSummariesForAccount: Statement<unknown[]>;
    listSummariesRandomForAccount: Statement<unknown[]>;
    listUncategorizedSummaries: Statement<unknown[]>;
    listUncategorizedSummariesStable: Statement<unknown[]>;
    listSummariesByDomain: Statement<unknown[]>;
    listIdsForAccount: Statement<unknown[]>;
    listSendersForAccount: Statement<unknown[]>;
  };

  /** Prepare and cache the reusable statements used by this repository. */
  constructor(private db: Database) {
    this.stmts = {
      upsert: db.prepare(
        `INSERT INTO emails (message_id, account_id, folder, subject, from_addr, date, has_attachments, body, body_format, body_fetched, indexed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (message_id, account_id) DO UPDATE SET
           folder = excluded.folder,
           subject = excluded.subject,
           from_addr = excluded.from_addr,
           date = excluded.date,
           has_attachments = excluded.has_attachments,
           body = COALESCE(excluded.body, emails.body),
           body_format = COALESCE(excluded.body_format, emails.body_format),
           body_fetched = CASE WHEN excluded.body_fetched = 1 THEN 1 ELSE emails.body_fetched END`,
      ),
      findById: db.prepare(
        `SELECT ${SELECT_FULL_COLS} FROM emails WHERE message_id = ? AND account_id = ?`,
      ),
      getBody: db.prepare('SELECT body FROM emails WHERE message_id = ? AND account_id = ?'),
      deleteEmbeddings: db.prepare(
        'DELETE FROM email_embedding_index WHERE message_id = ? AND account_id = ?',
      ),
      delete: db.prepare('DELETE FROM emails WHERE message_id = ? AND account_id = ?'),
      countAll: db.prepare('SELECT COUNT(*) AS c FROM emails WHERE account_id = ?'),
      countByFolder: db.prepare(
        'SELECT COUNT(*) AS c FROM emails WHERE account_id = ? AND folder = ?',
      ),
      countUnembedded: db.prepare(
        `SELECT COUNT(*) AS c FROM emails e
          LEFT JOIN email_embedding_index ei
            ON ei.message_id = e.message_id
           AND ei.account_id = e.account_id
           AND ei.model_id = ?
         WHERE e.account_id = ?
           AND e.body IS NOT NULL
           AND ei.rowid IS NULL`,
      ),
      listSummariesForAccount: db.prepare(
        `SELECT ${SELECT_SUMMARY_COLS}
           FROM emails WHERE account_id = ?
           ORDER BY COALESCE(date, indexed_at) DESC
           LIMIT ? OFFSET ?`,
      ),
      listSummariesRandomForAccount: db.prepare(
        `SELECT ${SELECT_SUMMARY_COLS}
           FROM emails WHERE account_id = ?
           ORDER BY RANDOM()
           LIMIT ?`,
      ),
      listUncategorizedSummaries: db.prepare(
        `SELECT ${SELECT_SUMMARY_COLS.split(', ')
          .map((c) => `e.${c}`)
          .join(', ')}
           FROM emails e
           LEFT JOIN email_categories ec
             ON ec.message_id = e.message_id AND ec.account_id = e.account_id
          WHERE e.account_id = ? AND ec.message_id IS NULL
          ORDER BY RANDOM()
          LIMIT ?`,
      ),
      listUncategorizedSummariesStable: db.prepare(
        `SELECT ${SELECT_SUMMARY_COLS.split(', ')
          .map((c) => `e.${c}`)
          .join(', ')}
           FROM emails e
           LEFT JOIN email_categories ec
             ON ec.message_id = e.message_id AND ec.account_id = e.account_id
          WHERE e.account_id = ? AND ec.message_id IS NULL
          ORDER BY COALESCE(e.date, e.indexed_at) DESC, e.message_id ASC
          LIMIT ?`,
      ),
      listSummariesByDomain: db.prepare(
        `SELECT ${SELECT_SUMMARY_COLS}
           FROM emails WHERE account_id = ? AND from_addr LIKE ? ESCAPE '\\'
           ORDER BY RANDOM()
           LIMIT ?`,
      ),
      listIdsForAccount: db.prepare(
        `SELECT message_id FROM emails WHERE account_id = ? ORDER BY COALESCE(date, indexed_at) DESC`,
      ),
      listSendersForAccount: db.prepare(
        `SELECT message_id, from_addr FROM emails WHERE account_id = ?`,
      ),
    };
  }

  /** Return the sender of every email for an account without loading bodies. */
  listSenders(accountId: string): Array<{ messageId: string; fromAddr: string | null }> {
    const rows = this.stmts.listSendersForAccount.all(accountId) as Array<{
      message_id: string;
      from_addr: string | null;
    }>;
    return rows.map((r) => ({ messageId: r.message_id, fromAddr: r.from_addr }));
  }

  /**
   * Run a single upsert and drop stale embeddings when the body actually changed, so a
   * later resync re-embeds only what differs.
   */
  private runUpsert(item: UpsertEmailInput, now: number): void {
    const body = item.body ?? null;
    const bodyFetched = (item.bodyFetched ?? true) ? 1 : 0;

    let bodyChanged = false;
    if (body !== null) {
      const prior = this.stmts.getBody.get(item.messageId, item.accountId) as
        | { body: string | null }
        | undefined;
      if (prior && prior.body !== body) bodyChanged = true;
    }

    this.stmts.upsert.run(
      item.messageId,
      item.accountId,
      item.folder,
      item.subject ?? null,
      item.fromAddr ?? null,
      item.date ?? null,
      (item.hasAttachments ?? false) ? 1 : 0,
      body,
      item.bodyFormat ?? 'text',
      bodyFetched,
      now,
    );

    if (bodyChanged) this.stmts.deleteEmbeddings.run(item.messageId, item.accountId);
  }

  /** Insert or update one email and return the resulting row. */
  upsert(input: UpsertEmailInput): EmailRow {
    const now = Date.now();
    this.db.transaction(() => this.runUpsert(input, now))();

    return {
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
  }

  /** Insert or update many emails in a single transaction and return the count. */
  upsertBatch(inputs: UpsertEmailInput[]): number {
    const tx = this.db.transaction((items: UpsertEmailInput[]) => {
      const now = Date.now();
      for (const item of items) this.runUpsert(item, now);
    });
    tx(inputs);
    return inputs.length;
  }

  /**
   * Of the given message ids, return those that still need a body fetched: not present,
   * or present but a prior fetch failed. Already-complete emails are omitted so a resync
   * can skip them.
   */
  selectNeedFetch(accountId: string, messageIds: string[]): string[] {
    if (messageIds.length === 0) return [];
    const synced = new Set<string>();
    const CHUNK = 800;
    for (let i = 0; i < messageIds.length; i += CHUNK) {
      const chunk = messageIds.slice(i, i + CHUNK);
      const placeholders = chunk.map(() => '?').join(',');
      const rows = this.db
        .prepare(
          `SELECT message_id FROM emails
            WHERE account_id = ? AND body_fetched = 1 AND message_id IN (${placeholders})`,
        )
        .all(accountId, ...chunk) as Array<{ message_id: string }>;
      for (const r of rows) synced.add(r.message_id);
    }
    return messageIds.filter((id) => !synced.has(id));
  }

  /**
   * Keyword BM25 search over subject, sender, and body via the FTS5 index, best first.
   * The query is sanitized into a safe FTS5 MATCH expression. An empty query returns [].
   */
  keywordSearch(accountId: string, query: string, limit = 30): string[] {
    const match = sanitizeFtsQuery(query);
    if (!match) return [];
    const rows = this.db
      .prepare(
        `SELECT e.message_id
           FROM email_fts
           JOIN emails e ON e.rowid = email_fts.rowid
          WHERE email_fts MATCH ? AND e.account_id = ?
          ORDER BY email_fts.rank
          LIMIT ?`,
      )
      .all(match, accountId, limit) as Array<{ message_id: string }>;
    return rows.map((r) => r.message_id);
  }

  /**
   * Message ids whose date falls in [from, to], newest first, capped at limit. Backs
   * time-scoped chat retrieval. Emails without a date are excluded.
   */
  listIdsInRange(accountId: string, from: number, to: number, limit = 50): string[] {
    const rows = this.db
      .prepare(
        `SELECT message_id FROM emails
          WHERE account_id = ? AND date IS NOT NULL AND date BETWEEN ? AND ?
          ORDER BY date DESC LIMIT ?`,
      )
      .all(accountId, from, to, limit) as Array<{ message_id: string }>;
    return rows.map((r) => r.message_id);
  }

  /**
   * Of the given message ids, return those whose date falls in [from, to], preserving
   * input order. Unlike listIdsInRange, this filters an already-ranked id list by date so
   * a relevant but older email is not dropped for not being among the newest.
   */
  filterIdsInRange(accountId: string, ids: string[], from: number, to: number): string[] {
    if (ids.length === 0) return [];
    const placeholders = ids.map(() => '?').join(',');
    const rows = this.db
      .prepare(
        `SELECT message_id FROM emails
          WHERE account_id = ? AND date IS NOT NULL AND date BETWEEN ? AND ?
            AND message_id IN (${placeholders})`,
      )
      .all(accountId, from, to, ...ids) as Array<{ message_id: string }>;
    const inRange = new Set(rows.map((r) => r.message_id));
    return ids.filter((id) => inRange.has(id));
  }

  /** Return the full email for a message id and account, or null if absent. */
  findById(messageId: string, accountId: string): EmailRow | null {
    const row = this.stmts.findById.get(messageId, accountId) as EmailDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  /**
   * Return full EmailRow objects including body, with a caller-built WHERE clause.
   * Prefer listSummaries when the body is not needed, it can be far cheaper.
   */
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

    const sql = `SELECT ${SELECT_FULL_COLS} FROM emails WHERE ${filters.join(' AND ')}
                 ORDER BY COALESCE(date, indexed_at) DESC LIMIT ? OFFSET ?`;
    const rows = this.db.prepare(sql).all(...params) as EmailDbRow[];
    return rows.map((r) => this.fromRow(r));
  }

  /**
   * List rows without the body field. Use for list-views and dashboards to avoid loading
   * megabytes of text.
   */
  listSummaries(opts: { accountId: string; limit?: number; offset?: number }): EmailSummary[] {
    const limit = opts.limit ?? 100;
    const offset = opts.offset ?? 0;
    const rows = this.stmts.listSummariesForAccount.all(
      opts.accountId,
      limit,
      offset,
    ) as EmailSummaryDbRow[];
    return rows.map((r) => this.summaryFromRow(r));
  }

  /** Random summaries drawn across the whole account, not just the newest. */
  listSummariesRandom(accountId: string, limit: number): EmailSummary[] {
    const rows = this.stmts.listSummariesRandomForAccount.all(
      accountId,
      limit,
    ) as EmailSummaryDbRow[];
    return rows.map((r) => this.summaryFromRow(r));
  }

  /** Random summaries for emails that have no category assignment yet. */
  listUncategorizedSummaries(accountId: string, limit: number): EmailSummary[] {
    const rows = this.stmts.listUncategorizedSummaries.all(accountId, limit) as EmailSummaryDbRow[];
    return rows.map((r) => this.summaryFromRow(r));
  }

  /** Stable summaries for emails that have no category assignment yet. Used when repeatable QA matters. */
  listUncategorizedSummariesStable(accountId: string, limit: number): EmailSummary[] {
    const rows = this.stmts.listUncategorizedSummariesStable.all(
      accountId,
      limit,
    ) as EmailSummaryDbRow[];
    return rows.map((r) => this.summaryFromRow(r));
  }

  /** Random summaries from a single sender domain. Anchored on '@' so 'mail.com' does not match 'gmail.com'. */
  listSummariesByDomain(accountId: string, domain: string, limit: number): EmailSummary[] {
    const escaped = domain.replace(/[\\%_]/g, '\\$&');
    const rows = this.stmts.listSummariesByDomain.all(
      accountId,
      `%@${escaped}%`,
      limit,
    ) as EmailSummaryDbRow[];
    return rows.map((r) => this.summaryFromRow(r));
  }

  /**
   * Summaries for a specific set of message ids, scoped to the account. Read-only. Chunked to stay
   * within SQLite's bound-parameter limit. Row order is not guaranteed; a caller that needs a stable
   * order should reorder by its own id list.
   */
  summariesByIds(accountId: string, messageIds: string[]): EmailSummary[] {
    const CHUNK = 400;
    const out: EmailSummary[] = [];
    for (let i = 0; i < messageIds.length; i += CHUNK) {
      const chunk = messageIds.slice(i, i + CHUNK);
      if (chunk.length === 0) continue;
      const placeholders = chunk.map(() => '?').join(',');
      const rows = this.db
        .prepare(
          `SELECT ${SELECT_SUMMARY_COLS} FROM emails WHERE account_id = ? AND message_id IN (${placeholders})`,
        )
        .all(accountId, ...chunk) as EmailSummaryDbRow[];
      for (const r of rows) out.push(this.summaryFromRow(r));
    }
    return out;
  }

  /** Stream just the message ids for an account in date-desc order. */
  listIds(accountId: string): string[] {
    const rows = this.stmts.listIdsForAccount.all(accountId) as Array<{ message_id: string }>;
    return rows.map((r) => r.message_id);
  }

  /**
   * Return emails that have a body but no embedding for the given model, skipping ids in
   * skipIds. The caller should chunk skipIds if it approaches SQLite's 999-param limit.
   */
  findUnembedded(
    accountId: string,
    modelId: string,
    limit = 50,
    skipIds: ReadonlySet<string> = new Set(),
  ): EmailRow[] {
    const canonical = canonicalizeModelId(modelId);
    const baseSql = `SELECT e.${SELECT_FULL_COLS.split(', ').join(', e.')}
                       FROM emails e
                       LEFT JOIN email_embedding_index ei
                         ON ei.message_id = e.message_id
                        AND ei.account_id = e.account_id
                        AND ei.model_id = ?
                      WHERE e.account_id = ?
                        AND e.body IS NOT NULL
                        AND ei.rowid IS NULL`;

    if (skipIds.size === 0) {
      const rows = this.db
        .prepare(`${baseSql} ORDER BY COALESCE(e.date, e.indexed_at) DESC LIMIT ?`)
        .all(canonical, accountId, limit) as EmailDbRow[];
      return rows.map((r) => this.fromRow(r));
    }

    const placeholders = Array.from(skipIds, () => '?').join(',');
    const rows = this.db
      .prepare(
        `${baseSql}
           AND e.message_id NOT IN (${placeholders})
           ORDER BY COALESCE(e.date, e.indexed_at) DESC
           LIMIT ?`,
      )
      .all(canonical, accountId, ...skipIds, limit) as EmailDbRow[];
    return rows.map((r) => this.fromRow(r));
  }

  /** Count emails with a body but no embedding for the given model. */
  countUnembedded(accountId: string, modelId: string): number {
    const canonical = canonicalizeModelId(modelId);
    const row = this.stmts.countUnembedded.get(canonical, accountId) as { c: number };
    return row.c;
  }

  /** Count emails for an account, optionally restricted to one folder. */
  count(accountId: string, folder?: string): number {
    const row = folder
      ? (this.stmts.countByFolder.get(accountId, folder) as { c: number })
      : (this.stmts.countAll.get(accountId) as { c: number });
    return row.c;
  }

  /** Delete one email and return whether a row was removed. */
  delete(messageId: string, accountId: string): boolean {
    const result = this.stmts.delete.run(messageId, accountId);
    return result.changes > 0;
  }

  /** Map a raw database row to a full EmailRow with body and typed fields. */
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

  /** Map a raw database row to an EmailSummary, omitting the body. */
  private summaryFromRow(row: EmailSummaryDbRow): EmailSummary {
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
