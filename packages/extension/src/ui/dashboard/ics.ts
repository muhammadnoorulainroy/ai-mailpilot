/**
 * Builds a single-event iCalendar (.ics) document from a captured event, so a triaged mail's
 * meeting can be added to any calendar. Event times are floating local wall-clock; DTSTAMP is UTC.
 */
import type { CapturedEvent } from '@ai-mailpilot/shared';

/** Zero-pads a number to two digits. */
function pad(n: number): string {
  return String(n).padStart(2, '0');
}

/** Floating local-time value YYYYMMDDTHHMMSS (no zone), for DTSTART/DTEND. */
function icsLocal(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}T${pad(d.getHours())}${pad(d.getMinutes())}00`;
}

/** UTC value YYYYMMDDTHHMMSSZ, for DTSTAMP. */
function icsUtc(ms: number): string {
  const d = new Date(ms);
  return `${d.getUTCFullYear()}${pad(d.getUTCMonth() + 1)}${pad(d.getUTCDate())}T${pad(d.getUTCHours())}${pad(d.getUTCMinutes())}${pad(d.getUTCSeconds())}Z`;
}

/** Date-only value YYYYMMDD, for all-day events. */
function icsDate(ms: number): string {
  const d = new Date(ms);
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}`;
}

/** Escapes a text value per RFC 5545 (backslash, comma, semicolon, newline). */
function escapeText(s: string): string {
  return s
    .replace(/\\/g, '\\\\')
    .replace(/;/g, '\\;')
    .replace(/,/g, '\\,')
    .replace(/\r?\n/g, '\\n');
}

/** Turns a title into a short filename-safe slug. */
export function slugify(title: string): string {
  const slug = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 40);
  return slug || 'event';
}

/** Builds a VCALENDAR document with one VEVENT for the captured event. */
export function buildIcs(event: CapturedEvent, uid: string): string {
  const lines = [
    'BEGIN:VCALENDAR',
    'VERSION:2.0',
    'PRODID:-//ai-mailpilot//calendar//EN',
    'CALSCALE:GREGORIAN',
    'BEGIN:VEVENT',
    `UID:${uid}`,
    `DTSTAMP:${icsUtc(Date.now())}`,
    `SUMMARY:${escapeText(event.title)}`,
  ];
  if (event.allDay) {
    lines.push(`DTSTART;VALUE=DATE:${icsDate(event.startAt)}`);
    lines.push(`DTEND;VALUE=DATE:${icsDate(event.startAt + 86_400_000)}`);
  } else {
    lines.push(`DTSTART:${icsLocal(event.startAt)}`);
    if (event.endAt) lines.push(`DTEND:${icsLocal(event.endAt)}`);
  }
  if (event.location) lines.push(`LOCATION:${escapeText(event.location)}`);
  lines.push('END:VEVENT', 'END:VCALENDAR');
  return lines.join('\r\n');
}
