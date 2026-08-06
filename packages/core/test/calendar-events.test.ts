/**
 * Tests for the calendar events store: capturing an event from a message, replacing it on
 * re-capture, clearing it, range listing, and cascade removal when the source email is deleted.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { CalendarEventRepository } from '../src/repositories/calendar-event-repository.js';

describe('CalendarEventRepository', () => {
  let db: Database;
  let accounts: AccountRepository;
  let emails: EmailRepository;
  let events: CalendarEventRepository;

  beforeEach(() => {
    db = openDatabase(':memory:');
    accounts = new AccountRepository(db);
    emails = new EmailRepository(db);
    events = new CalendarEventRepository(db);
  });
  afterEach(() => db.close());

  /** Seeds an account with one email and returns the account id. */
  function seed(): string {
    const acct = accounts.create({ address: 'a@x.y', kind: 'work' });
    emails.upsertBatch([
      { messageId: 'm1', accountId: acct.id, folder: 'INBOX', subject: 'Defense' },
    ]);
    return acct.id;
  }

  it('captures an event from a message and reads it back', () => {
    const accountId = seed();
    const start = new Date(2026, 2, 12, 14, 0, 0, 0).getTime();
    const end = new Date(2026, 2, 12, 15, 30, 0, 0).getTime();
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm1',
      title: 'Soutenance de stage',
      startAt: start,
      endAt: end,
      allDay: false,
      location: 'Salle B12',
    });

    const rows = events.listByMessage(accountId, 'm1');
    expect(rows).toHaveLength(1);
    expect(rows[0]!.title).toBe('Soutenance de stage');
    expect(rows[0]!.startAt).toBe(start);
    expect(rows[0]!.endAt).toBe(end);
    expect(rows[0]!.allDay).toBe(false);
    expect(rows[0]!.location).toBe('Salle B12');
    expect(rows[0]!.source).toBe('email');
    expect(events.countForAccount(accountId)).toBe(1);
  });

  it('replaces the prior event when the same message is re-captured', () => {
    const accountId = seed();
    const base = new Date(2026, 2, 12, 14, 0, 0, 0).getTime();
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm1',
      title: 'Old title',
      startAt: base,
      endAt: null,
      allDay: false,
      location: null,
    });
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm1',
      title: 'New title',
      startAt: base + 3_600_000,
      endAt: null,
      allDay: false,
      location: 'Room 2',
    });

    const rows = events.listByMessage(accountId, 'm1');
    expect(rows).toHaveLength(1);
    expect(rows[0]!.title).toBe('New title');
    expect(events.countForAccount(accountId)).toBe(1);
  });

  it('clears a captured event for a message', () => {
    const accountId = seed();
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm1',
      title: 'Call',
      startAt: Date.now(),
      endAt: null,
      allDay: false,
      location: null,
    });
    events.clearForMessage(accountId, 'm1');
    expect(events.countForAccount(accountId)).toBe(0);
  });

  it('lists only events whose start falls within the range', () => {
    const accountId = seed();
    emails.upsertBatch([{ messageId: 'm2', accountId, folder: 'INBOX', subject: 'Later' }]);
    const day = 86_400_000;
    const now = Date.now();
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm1',
      title: 'Inside',
      startAt: now + day,
      endAt: null,
      allDay: false,
      location: null,
    });
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm2',
      title: 'Outside',
      startAt: now + 10 * day,
      endAt: null,
      allDay: false,
      location: null,
    });

    const inRange = events.listInRange(accountId, now, now + 2 * day);
    expect(inRange.map((e) => e.title)).toEqual(['Inside']);
  });

  it('finds an event by keyword over title and location via FTS', () => {
    const accountId = seed();
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm1',
      title: 'Soutenance de stage',
      startAt: Date.now(),
      endAt: null,
      allDay: false,
      location: 'Salle B12',
    });
    expect(events.keywordSearchEvents(accountId, 'soutenance', 6).map((e) => e.title)).toContain(
      'Soutenance de stage',
    );
    expect(events.keywordSearchEvents(accountId, 'B12', 6)).toHaveLength(1);
    expect(events.keywordSearchEvents(accountId, 'invoice', 6)).toHaveLength(0);
  });

  it('maps the earliest event per message for a set of messages', () => {
    const accountId = seed();
    emails.upsertBatch([{ messageId: 'm2', accountId, folder: 'INBOX', subject: 'X' }]);
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm1',
      title: 'A',
      startAt: 1000,
      endAt: null,
      allDay: false,
      location: null,
    });
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm2',
      title: 'B',
      startAt: 2000,
      endAt: null,
      allDay: false,
      location: 'Room',
    });

    const map = events.mapByMessages(accountId, ['m1', 'm2', 'mX']);
    expect(map.get('m1')!.title).toBe('A');
    expect(map.get('m2')!.location).toBe('Room');
    expect(map.has('mX')).toBe(false);
  });

  it('cascades event removal when the source email is deleted', () => {
    const accountId = seed();
    events.captureFromEmail({
      accountId,
      sourceMessageId: 'm1',
      title: 'Meeting',
      startAt: Date.now(),
      endAt: null,
      allDay: false,
      location: null,
    });
    expect(events.countForAccount(accountId)).toBe(1);
    emails.delete('m1', accountId);
    expect(events.countForAccount(accountId)).toBe(0);
  });
});
