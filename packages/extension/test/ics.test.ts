/**
 * Tests for the .ics builder: a timed event with escaped text, an all-day event with DATE values,
 * and the filename slug.
 */
import { describe, it, expect } from 'vitest';
import { buildIcs, slugify } from '../src/ui/dashboard/ics.js';

describe('buildIcs', () => {
  it('builds a timed VEVENT with summary, start, end, and location', () => {
    const start = new Date(2026, 2, 12, 14, 0, 0).getTime();
    const end = new Date(2026, 2, 12, 15, 30, 0).getTime();
    const ics = buildIcs(
      {
        title: 'Soutenance, de stage',
        startAt: start,
        endAt: end,
        allDay: false,
        location: 'Salle B12',
      },
      'uid-1',
    );
    expect(ics).toContain('BEGIN:VCALENDAR');
    expect(ics).toContain('BEGIN:VEVENT');
    expect(ics).toContain('UID:uid-1');
    expect(ics).toContain('SUMMARY:Soutenance\\, de stage');
    expect(ics).toContain('DTSTART:20260312T140000');
    expect(ics).toContain('DTEND:20260312T153000');
    expect(ics).toContain('LOCATION:Salle B12');
    expect(ics).toContain('END:VCALENDAR');
    expect(ics.includes('\r\n')).toBe(true);
  });

  it('builds an all-day event with DATE values and no location line', () => {
    const start = new Date(2026, 2, 12, 0, 0, 0).getTime();
    const ics = buildIcs(
      { title: 'Holiday', startAt: start, endAt: null, allDay: true, location: null },
      'uid-2',
    );
    expect(ics).toContain('DTSTART;VALUE=DATE:20260312');
    expect(ics).toContain('DTEND;VALUE=DATE:20260313');
    expect(ics).not.toContain('LOCATION:');
  });

  it('slugify produces a filename-safe slug', () => {
    expect(slugify('Soutenance de stage!')).toBe('soutenance-de-stage');
    expect(slugify('')).toBe('event');
  });
});
