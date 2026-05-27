/**
 * Persistence layer for email triage classifications and the Today's Focus priority views.
 * Wraps prepared SQL statements that store classifications and query active, sectioned,
 * and carryover priority email lists.
 */
import type { Database, Statement } from 'better-sqlite3';
import type { TriageMetadata } from '@ai-mailpilot/shared';

/** Mutually-exclusive triage classification for an email. */
export type TriageBucket = 'urgent' | 'summarize' | 'spam' | 'personal';

/** One of the mutually-exclusive "Today's Focus" sections. Carryover is computed separately. */
export type PrioritySection = 'needsAction' | 'important' | 'summaries' | 'lowPriority';

/** A persisted triage classification row. */
export interface TriageRow {
  messageId: string;
  accountId: string;
  bucket: TriageBucket;
  reasoning: string | null;
  classifiedAt: number;
}

/** Input for inserting or updating a triage classification. */
export interface UpsertTriageInput {
  messageId: string;
  accountId: string;
  bucket: TriageBucket;
  reasoning?: string;
  metadata?: TriageMetadata;
}

/** Number of emails in a triage bucket. */
export interface BucketCount {
  bucket: TriageBucket;
  count: number;
}

/** Per-section email counts for the priority view. */
export interface PriorityCounts {
  needsAction: number;
  urgent: number;
  important: number;
  summaries: number;
  lowPriority: number;
}

/** An email pending triage classification, with the fields the classifier needs. */
export interface UnclassifiedEmail {
  messageId: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  body: string | null;
  bodyFormat: string | null;
}

/** A classified email joined with its email metadata. */
export interface ClassifiedEmail {
  messageId: string;
  accountId: string;
  bucket: TriageBucket;
  reasoning: string | null;
  classifiedAt: number;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  folder: string;
  hasAttachments: boolean;
}

/** A classified email enriched with its parsed triage metadata, for the priority view. */
export interface PriorityEmail extends ClassifiedEmail {
  actionRequired: boolean;
  needsReply: boolean;
  deadlineAt: number | null;
  importanceScore: number;
  suggestedAction: string | null;
  shortSummary: string | null;
}

const DEFAULT_IMPORTANCE: Record<TriageBucket, number> = {
  urgent: 85,
  personal: 55,
  summarize: 35,
  spam: 5,
};
const IMPORTANT_THRESHOLD = 60;

const ACTIVE = `(t.dismissed_at IS NULL AND t.done_at IS NULL AND (t.snoozed_until IS NULL OR t.snoozed_until <= @now))`;
const EMAIL_DATE = `COALESCE(e.date, t.classified_at)`;
const EMAIL_RECEIVED_DATE = `COALESCE(e.date, e.indexed_at)`;
const IMPORTANCE = `COALESCE(json_extract(t.metadata, '$.importanceScore'), 0)`;
const PRIORITY_RANGE = `${EMAIL_DATE} >= @since AND (@before IS NULL OR ${EMAIL_DATE} < @before)`;
const EMAIL_RANGE = `${EMAIL_RECEIVED_DATE} >= @since AND (@before IS NULL OR ${EMAIL_RECEIVED_DATE} < @before)`;

const SECTION_WHERE: Record<PrioritySection, string> = {
  needsAction: `t.bucket != 'spam' AND t.action_required = 1`,
  important: `t.bucket != 'spam' AND t.action_required = 0 AND (t.bucket IN ('personal','urgent') OR (t.bucket = 'summarize' AND ${IMPORTANCE} >= ${IMPORTANT_THRESHOLD}))`,
  summaries: `t.bucket != 'spam' AND t.action_required = 0 AND t.bucket = 'summarize' AND ${IMPORTANCE} < ${IMPORTANT_THRESHOLD}`,
  lowPriority: `t.bucket = 'spam'`,
};

const PRIORITY_ORDER = `ORDER BY (CASE WHEN t.deadline_at IS NULL THEN 1 ELSE 0 END), t.deadline_at ASC, ${IMPORTANCE} DESC, ${EMAIL_DATE} DESC`;

const SELECT_PRIORITY_COLS = `t.message_id, t.account_id, t.bucket, t.reasoning, t.classified_at, t.metadata,
        e.subject, e.from_addr, e.date, e.folder, e.has_attachments, t.action_required, t.deadline_at`;

interface UnclassifiedDbRow {
  message_id: string;
  subject: string | null;
  from_addr: string | null;
  date: number | null;
  body: string | null;
  body_format: string | null;
}

interface PriorityDbRow {
  message_id: string;
  account_id: string;
  bucket: TriageBucket;
  reasoning: string | null;
  classified_at: number;
  metadata: string | null;
  subject: string | null;
  from_addr: string | null;
  date: number | null;
  folder: string;
  has_attachments: number;
  action_required: number;
  deadline_at: number | null;
}

/** Stores and queries triage classifications and the Today's Focus priority views. */
export class TriageRepository {
  private readonly stmts: {
    upsert: Statement<unknown[]>;
    findById: Statement<unknown[]>;
    clearForAccount: Statement<unknown[]>;
    countByBucket: Statement<unknown[]>;
    countUnclassified: Statement<unknown[]>;
    countUnclassifiedInRange: Statement<unknown[]>;
    countPendingTriage: Statement<unknown[]>;
    countAllEmails: Statement<unknown[]>;
    selectPendingTriage: Statement<unknown[]>;
    selectByBucket: Statement<unknown[]>;
    partitionCounts: Statement<unknown[]>;
    carryover: Statement<unknown[]>;
    setResolution: Statement<unknown[]>;
  };
  private readonly sectionStmts: Record<PrioritySection, Statement<unknown[]>>;

  /** Prepares the fixed and per-section SQL statements against the given database. */
  constructor(private db: Database) {
    this.stmts = {
      upsert: db.prepare(
        `INSERT INTO triage (message_id, account_id, bucket, reasoning, classified_at, metadata, action_required, deadline_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (message_id, account_id) DO UPDATE SET
           bucket = excluded.bucket,
           reasoning = excluded.reasoning,
           classified_at = excluded.classified_at,
           metadata = excluded.metadata,
           action_required = excluded.action_required,
           deadline_at = excluded.deadline_at`,
      ),
      findById: db.prepare(
        'SELECT message_id, account_id, bucket, reasoning, classified_at FROM triage WHERE message_id = ? AND account_id = ?',
      ),
      clearForAccount: db.prepare('DELETE FROM triage WHERE account_id = ?'),
      countByBucket: db.prepare(
        'SELECT bucket, COUNT(*) AS count FROM triage WHERE account_id = ? GROUP BY bucket',
      ),
      countUnclassified: db.prepare(
        `SELECT COUNT(*) AS c FROM emails e
          LEFT JOIN triage t
            ON t.message_id = e.message_id AND t.account_id = e.account_id
         WHERE e.account_id = ? AND t.message_id IS NULL`,
      ),
      countUnclassifiedInRange: db.prepare(
        `SELECT COUNT(*) AS c FROM emails e
          LEFT JOIN triage t
            ON t.message_id = e.message_id AND t.account_id = e.account_id
         WHERE e.account_id = @accountId AND t.message_id IS NULL
           AND ${EMAIL_RANGE}`,
      ),
      countPendingTriage: db.prepare(
        `SELECT COUNT(*) AS c FROM emails e
          LEFT JOIN triage t
            ON t.message_id = e.message_id AND t.account_id = e.account_id
         WHERE e.account_id = ? AND (t.message_id IS NULL OR t.metadata IS NULL)`,
      ),
      countAllEmails: db.prepare('SELECT COUNT(*) AS c FROM emails WHERE account_id = ?'),
      selectPendingTriage: db.prepare(
        `SELECT e.message_id, e.subject, e.from_addr, e.date, e.body, e.body_format
           FROM emails e
           LEFT JOIN triage t
             ON t.message_id = e.message_id AND t.account_id = e.account_id
          WHERE e.account_id = ? AND (t.message_id IS NULL OR t.metadata IS NULL)
          ORDER BY ${EMAIL_RECEIVED_DATE} DESC
          LIMIT ?`,
      ),
      selectByBucket: db.prepare(
        `SELECT t.message_id, t.account_id, t.bucket, t.reasoning, t.classified_at,
                e.subject, e.from_addr, e.date, e.folder, e.has_attachments
           FROM triage t
           JOIN emails e ON e.message_id = t.message_id AND e.account_id = t.account_id
          WHERE t.account_id = ? AND t.bucket = ?
          ORDER BY COALESCE(e.date, t.classified_at) DESC
          LIMIT ?`,
      ),
      partitionCounts: db.prepare(
        `SELECT
           COALESCE(SUM(CASE WHEN ${SECTION_WHERE.needsAction} THEN 1 ELSE 0 END), 0) AS needsAction,
           COALESCE(SUM(CASE WHEN t.bucket = 'urgent' THEN 1 ELSE 0 END), 0) AS urgent,
           COALESCE(SUM(CASE WHEN ${SECTION_WHERE.important} THEN 1 ELSE 0 END), 0) AS important,
           COALESCE(SUM(CASE WHEN ${SECTION_WHERE.summaries} THEN 1 ELSE 0 END), 0) AS summaries,
           COALESCE(SUM(CASE WHEN ${SECTION_WHERE.lowPriority} THEN 1 ELSE 0 END), 0) AS lowPriority
         FROM triage t
         JOIN emails e ON e.message_id = t.message_id AND e.account_id = t.account_id
        WHERE t.account_id = @accountId AND ${ACTIVE} AND ${PRIORITY_RANGE}`,
      ),
      carryover: db.prepare(
        `SELECT ${SELECT_PRIORITY_COLS}
           FROM triage t
           JOIN emails e ON e.message_id = t.message_id AND e.account_id = t.account_id
          WHERE t.account_id = @accountId AND ${ACTIVE}
            AND t.bucket != 'spam' AND (t.action_required = 1 OR t.bucket = 'urgent')
            AND ${EMAIL_DATE} >= @lookbackStart AND ${EMAIL_DATE} < @before
          ${PRIORITY_ORDER}
          LIMIT @limit`,
      ),
      setResolution: db.prepare(
        `UPDATE triage SET dismissed_at = ?, done_at = ?, snoozed_until = ?
          WHERE message_id = ? AND account_id = ?`,
      ),
    };

    this.sectionStmts = {} as Record<PrioritySection, Statement<unknown[]>>;
    for (const section of Object.keys(SECTION_WHERE) as PrioritySection[]) {
      this.sectionStmts[section] = db.prepare(
        `SELECT ${SELECT_PRIORITY_COLS}
          FROM triage t
           JOIN emails e ON e.message_id = t.message_id AND e.account_id = t.account_id
          WHERE t.account_id = @accountId AND ${ACTIVE} AND ${PRIORITY_RANGE}
            AND ${SECTION_WHERE[section]}
          ${PRIORITY_ORDER}
          LIMIT @limit`,
      );
    }
  }

  /** Insert or replace a triage classification, flattening metadata into indexed columns. */
  upsert(input: UpsertTriageInput): TriageRow {
    const now = Date.now();
    const row: TriageRow = {
      messageId: input.messageId,
      accountId: input.accountId,
      bucket: input.bucket,
      reasoning: input.reasoning ?? null,
      classifiedAt: now,
    };
    this.stmts.upsert.run(
      row.messageId,
      row.accountId,
      row.bucket,
      row.reasoning,
      row.classifiedAt,
      input.metadata ? JSON.stringify(input.metadata) : null,
      input.metadata?.actionRequired ? 1 : 0,
      input.metadata?.deadlineAt ?? null,
    );
    return row;
  }

  /** Look up a single classification by message and account, or null if none exists. */
  findById(messageId: string, accountId: string): TriageRow | null {
    const row = this.stmts.findById.get(messageId, accountId) as
      | {
          message_id: string;
          account_id: string;
          bucket: TriageBucket;
          reasoning: string | null;
          classified_at: number;
        }
      | undefined;
    if (!row) return null;
    return {
      messageId: row.message_id,
      accountId: row.account_id,
      bucket: row.bucket,
      reasoning: row.reasoning,
      classifiedAt: row.classified_at,
    };
  }

  /** Delete all classifications for an account and return the number of rows removed. */
  clearForAccount(accountId: string): number {
    return this.stmts.clearForAccount.run(accountId).changes;
  }

  /** Count classified emails grouped by triage bucket for an account. */
  countByBucket(accountId: string): BucketCount[] {
    return this.stmts.countByBucket.all(accountId) as BucketCount[];
  }

  /** Count emails for an account that have no triage row yet. */
  countUnclassified(accountId: string): number {
    const row = this.stmts.countUnclassified.get(accountId) as { c: number } | undefined;
    return row?.c ?? 0;
  }

  /** Unclassified emails in the local-time range. beforeMs null means no upper bound. */
  countUnclassifiedInRange(accountId: string, sinceMs: number, beforeMs: number | null): number {
    if (sinceMs <= 0 && beforeMs === null) return this.countUnclassified(accountId);
    const row = this.stmts.countUnclassifiedInRange.get({
      accountId,
      since: sinceMs,
      before: beforeMs,
    }) as { c: number } | undefined;
    return row?.c ?? 0;
  }

  /**
   * Emails that should be sent through the priority classifier. Incremental runs process both
   * never-triaged emails and legacy rows whose metadata predates Today's Focus. Force runs re-check
   * every email while preserving user resolution columns through upsert.
   */
  countPendingTriage(
    accountId: string,
    force = false,
    skipIds: ReadonlySet<string> = new Set(),
  ): number {
    const base = force
      ? ((this.stmts.countAllEmails.get(accountId) as { c: number } | undefined)?.c ?? 0)
      : ((this.stmts.countPendingTriage.get(accountId) as { c: number } | undefined)?.c ?? 0);
    if (base === 0 || skipIds.size === 0) return base;

    const placeholders = Array.from(skipIds, () => '?').join(',');
    const pendingPredicate = force ? '1=1' : '(t.message_id IS NULL OR t.metadata IS NULL)';
    const row = this.db
      .prepare(
        `SELECT COUNT(*) AS c FROM emails e
          LEFT JOIN triage t
            ON t.message_id = e.message_id AND t.account_id = e.account_id
         WHERE e.account_id = ? AND ${pendingPredicate}
           AND e.message_id NOT IN (${placeholders})`,
      )
      .get(accountId, ...skipIds) as { c: number } | undefined;
    return row?.c ?? 0;
  }

  /** Return classified emails in a bucket joined with email metadata, for the legacy dashboard. */
  listByBucket(accountId: string, bucket: TriageBucket, limit = 20): ClassifiedEmail[] {
    const rows = this.stmts.selectByBucket.all(accountId, bucket, limit) as Array<{
      message_id: string;
      account_id: string;
      bucket: TriageBucket;
      reasoning: string | null;
      classified_at: number;
      subject: string | null;
      from_addr: string | null;
      date: number | null;
      folder: string;
      has_attachments: number;
    }>;

    return rows.map((r) => ({
      messageId: r.message_id,
      accountId: r.account_id,
      bucket: r.bucket,
      reasoning: r.reasoning,
      classifiedAt: r.classified_at,
      subject: r.subject,
      fromAddr: r.from_addr,
      date: r.date,
      folder: r.folder,
      hasAttachments: r.has_attachments === 1,
    }));
  }

  /** Active (non-resolved) per-section counts in the selected local-time range. */
  priorityCounts(
    accountId: string,
    sinceMs: number,
    beforeMs: number | null,
    now: number,
  ): PriorityCounts {
    const r = this.stmts.partitionCounts.get({
      accountId,
      since: sinceMs,
      before: beforeMs,
      now,
    }) as PriorityCounts;
    return {
      needsAction: r.needsAction,
      urgent: r.urgent,
      important: r.important,
      summaries: r.summaries,
      lowPriority: r.lowPriority,
    };
  }

  /** Active emails for one focus section, newest/most-urgent first. */
  listSection(
    accountId: string,
    section: PrioritySection,
    sinceMs: number,
    beforeMs: number | null,
    now: number,
    limit: number,
  ): PriorityEmail[] {
    const rows = this.sectionStmts[section].all({
      accountId,
      since: sinceMs,
      before: beforeMs,
      now,
      limit,
    }) as PriorityDbRow[];
    return rows.map((r) => this.toPriorityEmail(r));
  }

  /**
   * Unresolved actionable/urgent mail from BEFORE the current window, within a recent lookback. Carries
   * forward yesterday's loose ends so they are not lost when the focus moves to today.
   */
  listCarryover(
    accountId: string,
    beforeMs: number,
    lookbackStartMs: number,
    now: number,
    limit: number,
  ): PriorityEmail[] {
    const rows = this.stmts.carryover.all({
      accountId,
      before: beforeMs,
      lookbackStart: lookbackStartMs,
      now,
      limit,
    }) as PriorityDbRow[];
    return rows.map((r) => this.toPriorityEmail(r));
  }

  /** Set the three resolution columns at once. Returns whether a row was updated. */
  setResolution(
    accountId: string,
    messageId: string,
    dismissedAt: number | null,
    doneAt: number | null,
    snoozedUntil: number | null,
  ): boolean {
    return (
      this.stmts.setResolution.run(dismissedAt, doneAt, snoozedUntil, messageId, accountId)
        .changes > 0
    );
  }

  /**
   * Return emails that need priority classification/enrichment, excluding skipIds.
   *
   * A force run reclassifies every email ONCE: it selects rows never classified, missing
   * metadata, or last classified before staleBeforeMs (the moment the run began). persistResult
   * stamps classified_at = now on each upsert, so a reprocessed email drops out of the next page
   * and the loop drains to empty. Without this cursor a force run re-selected the same top rows
   * forever (`1=1`), an unbounded re-classification loop.
   */
  findPendingTriageEmails(
    accountId: string,
    limit = 16,
    skipIds: ReadonlySet<string> = new Set(),
    force = false,
    staleBeforeMs = 0,
  ): UnclassifiedEmail[] {
    if (!force && skipIds.size === 0) {
      const rows = this.stmts.selectPendingTriage.all(accountId, limit) as UnclassifiedDbRow[];
      return rows.map((r) => this.fromRow(r));
    }

    const params: (string | number)[] = [accountId];
    let pendingPredicate: string;
    if (force) {
      pendingPredicate = '(t.message_id IS NULL OR t.metadata IS NULL OR t.classified_at < ?)';
      params.push(staleBeforeMs);
    } else {
      pendingPredicate = '(t.message_id IS NULL OR t.metadata IS NULL)';
    }
    let skipClause = '';
    if (skipIds.size > 0) {
      skipClause = `AND e.message_id NOT IN (${Array.from(skipIds, () => '?').join(',')})`;
      params.push(...skipIds);
    }
    params.push(limit);

    const rows = this.db
      .prepare(
        `SELECT e.message_id, e.subject, e.from_addr, e.date, e.body, e.body_format
           FROM emails e
           LEFT JOIN triage t
             ON t.message_id = e.message_id AND t.account_id = e.account_id
          WHERE e.account_id = ? AND ${pendingPredicate} ${skipClause}
          ORDER BY ${EMAIL_RECEIVED_DATE} DESC
          LIMIT ?`,
      )
      .all(...params) as UnclassifiedDbRow[];
    return rows.map((r) => this.fromRow(r));
  }

  /** Map a raw email row into the classifier input shape. */
  private fromRow(r: UnclassifiedDbRow): UnclassifiedEmail {
    return {
      messageId: r.message_id,
      subject: r.subject,
      fromAddr: r.from_addr,
      date: r.date,
      body: r.body,
      bodyFormat: r.body_format,
    };
  }

  /** Build a priority view email by merging row columns with parsed stored metadata. */
  private toPriorityEmail(r: PriorityDbRow): PriorityEmail {
    const meta = parseStoredMetadata(r.metadata, r.bucket);
    return {
      messageId: r.message_id,
      accountId: r.account_id,
      bucket: r.bucket,
      reasoning: r.reasoning,
      classifiedAt: r.classified_at,
      subject: r.subject,
      fromAddr: r.from_addr,
      date: r.date,
      folder: r.folder,
      hasAttachments: r.has_attachments === 1,
      actionRequired: r.action_required === 1,
      needsReply: meta.needsReply,
      deadlineAt: r.deadline_at,
      importanceScore: meta.importanceScore,
      suggestedAction: meta.suggestedAction,
      shortSummary: meta.shortSummary,
    };
  }
}

/**
 * Parse the stored JSON metadata, falling back to bucket defaults when missing or malformed.
 * Importance defaults to the bucket's baseline so unscored rows still sort sensibly.
 */
function parseStoredMetadata(
  json: string | null,
  bucket: TriageBucket,
): {
  needsReply: boolean;
  importanceScore: number;
  suggestedAction: string | null;
  shortSummary: string | null;
} {
  if (json) {
    try {
      const m = JSON.parse(json) as Partial<TriageMetadata>;
      return {
        needsReply: m.needsReply === true,
        importanceScore:
          typeof m.importanceScore === 'number' ? m.importanceScore : DEFAULT_IMPORTANCE[bucket],
        suggestedAction: typeof m.suggestedAction === 'string' ? m.suggestedAction : null,
        shortSummary: typeof m.shortSummary === 'string' ? m.shortSummary : null,
      };
    } catch {}
  }
  return {
    needsReply: false,
    importanceScore: DEFAULT_IMPORTANCE[bucket],
    suggestedAction: null,
    shortSummary: null,
  };
}
