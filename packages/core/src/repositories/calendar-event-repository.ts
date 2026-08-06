/**
 * Repository for calendar events. Events are captured from mail by the triage pass and stored
 * with a source message link so re-triaging a message replaces its captured event rather than
 * accumulating duplicates. Vector and FTS retrieval methods are added alongside indexing.
 */
import { randomUUID } from 'node:crypto';
import type { Database, Statement } from 'better-sqlite3';
import type { CapturedEvent } from '@ai-mailpilot/shared';
import { sanitizeFtsQuery } from '../util/text.js';

/** A calendar event to persist, already resolved to absolute epoch-ms times. */
export interface CapturedEventInput {
  accountId: string;
  sourceMessageId: string | null;
  title: string;
  startAt: number;
  endAt: number | null;
  allDay: boolean;
  location: string | null;
  source?: 'email' | 'ics' | 'caldav' | 'manual';
}

/** A stored event row. */
export interface EventRow {
  id: string;
  accountId: string;
  sourceMessageId: string | null;
  title: string;
  startAt: number;
  endAt: number | null;
  allDay: boolean;
  location: string | null;
  description: string | null;
  organizer: string | null;
  rrule: string | null;
  status: string;
  source: string;
  createdAt: number;
}

interface RawEventRow {
  id: string;
  account_id: string;
  source_message_id: string | null;
  title: string;
  start_at: number;
  end_at: number | null;
  all_day: number;
  location: string | null;
  description: string | null;
  organizer: string | null;
  rrule: string | null;
  status: string;
  source: string;
  created_at: number;
}

/** Maps a raw database row to an EventRow. */
function toEventRow(r: RawEventRow): EventRow {
  return {
    id: r.id,
    accountId: r.account_id,
    sourceMessageId: r.source_message_id,
    title: r.title,
    startAt: r.start_at,
    endAt: r.end_at,
    allDay: r.all_day === 1,
    location: r.location,
    description: r.description,
    organizer: r.organizer,
    rrule: r.rrule,
    status: r.status,
    source: r.source,
    createdAt: r.created_at,
  };
}

const SELECT_COLUMNS = `id, account_id, source_message_id, title, start_at, end_at, all_day,
  location, organizer, description, rrule, status, source, created_at`;
/** Same columns qualified with the `e` alias, for queries that join the FTS index. */
const SELECT_COLUMNS_E = SELECT_COLUMNS.split(',')
  .map((c) => `e.${c.trim()}`)
  .join(', ');

/** Stores and queries calendar events. */
export class CalendarEventRepository {
  private readonly stmts: {
    deleteByMessage: Statement<unknown[]>;
    insert: Statement<unknown[]>;
    listByMessage: Statement<unknown[]>;
    countForAccount: Statement<unknown[]>;
    listInRange: Statement<unknown[]>;
  };

  /** Prepares the reusable statements for the events table on the given database. */
  constructor(private db: Database) {
    this.stmts = {
      deleteByMessage: db.prepare(
        'DELETE FROM events WHERE account_id = ? AND source_message_id = ? AND source = ?',
      ),
      insert: db.prepare(
        `INSERT INTO events
           (id, account_id, source_message_id, title, start_at, end_at, all_day, location, source, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ),
      listByMessage: db.prepare(
        `SELECT ${SELECT_COLUMNS} FROM events
          WHERE account_id = ? AND source_message_id = ? ORDER BY start_at ASC`,
      ),
      countForAccount: db.prepare('SELECT COUNT(*) AS c FROM events WHERE account_id = ?'),
      listInRange: db.prepare(
        `SELECT ${SELECT_COLUMNS} FROM events
          WHERE account_id = ? AND start_at >= ? AND start_at < ? ORDER BY start_at ASC LIMIT ?`,
      ),
    };
  }

  /**
   * Captures an event extracted from an email, replacing any prior event captured from the same
   * message so re-triaging never accumulates duplicates. Returns the new event id.
   */
  captureFromEmail(input: CapturedEventInput): string {
    const id = randomUUID();
    const source = input.source ?? 'email';
    const tx = this.db.transaction(() => {
      if (input.sourceMessageId !== null) {
        this.stmts.deleteByMessage.run(input.accountId, input.sourceMessageId, source);
      }
      this.stmts.insert.run(
        id,
        input.accountId,
        input.sourceMessageId,
        input.title,
        input.startAt,
        input.endAt,
        input.allDay ? 1 : 0,
        input.location,
        source,
        Date.now(),
      );
    });
    tx();
    return id;
  }

  /** Removes events captured from a message, used when a re-triage no longer finds an event. */
  clearForMessage(
    accountId: string,
    sourceMessageId: string,
    source: 'email' | 'ics' | 'caldav' | 'manual' = 'email',
  ): void {
    this.stmts.deleteByMessage.run(accountId, sourceMessageId, source);
  }

  /** Lists events captured from a specific message, earliest first. */
  listByMessage(accountId: string, sourceMessageId: string): EventRow[] {
    return (this.stmts.listByMessage.all(accountId, sourceMessageId) as RawEventRow[]).map(
      toEventRow,
    );
  }

  /** Counts all events stored for an account. */
  countForAccount(accountId: string): number {
    const row = this.stmts.countForAccount.get(accountId) as { c: number } | undefined;
    return row?.c ?? 0;
  }

  /** Lists events whose start falls within [from, to), earliest first, capped at limit. */
  listInRange(accountId: string, from: number, to: number, limit = 100): EventRow[] {
    return (this.stmts.listInRange.all(accountId, from, to, limit) as RawEventRow[]).map(
      toEventRow,
    );
  }

  /** Returns the earliest captured event per message for the given message ids. */
  mapByMessages(accountId: string, messageIds: string[]): Map<string, CapturedEvent> {
    const out = new Map<string, CapturedEvent>();
    if (messageIds.length === 0) return out;
    const placeholders = messageIds.map(() => '?').join(',');
    const rows = this.db
      .prepare(
        `SELECT ${SELECT_COLUMNS} FROM events
          WHERE account_id = ? AND source_message_id IN (${placeholders})
          ORDER BY start_at ASC`,
      )
      .all(accountId, ...messageIds) as RawEventRow[];
    for (const r of rows) {
      const mid = r.source_message_id;
      if (mid && !out.has(mid)) {
        out.set(mid, {
          title: r.title,
          startAt: r.start_at,
          endAt: r.end_at,
          allDay: r.all_day === 1,
          location: r.location,
        });
      }
    }
    return out;
  }

  /** BM25 keyword search over event title, location, and description. Best match first. */
  keywordSearchEvents(accountId: string, query: string, limit = 6): EventRow[] {
    const match = sanitizeFtsQuery(query);
    if (!match) return [];
    const rows = this.db
      .prepare(
        `SELECT ${SELECT_COLUMNS_E}
           FROM event_fts
           JOIN events e ON e.rowid = event_fts.rowid
          WHERE event_fts MATCH ? AND e.account_id = ?
          ORDER BY event_fts.rank
          LIMIT ?`,
      )
      .all(match, accountId, limit) as RawEventRow[];
    return rows.map(toEventRow);
  }
}
