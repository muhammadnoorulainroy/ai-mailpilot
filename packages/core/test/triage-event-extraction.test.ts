/**
 * Tests that the triage pass extracts a structured calendar event from the LLM response,
 * resolving its date and clock into absolute local-time timestamps, and returns no event
 * when the model reports none or gives an unusable date.
 */
import { describe, it, expect } from 'vitest';
import type { Logger } from 'pino';
import { TriageService } from '../src/services/triage-service.js';
import type { LlmClient } from '../src/llm/client.js';
import type { EmailToTriage } from '../src/services/triage-service.js';

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

/** Builds a TriageService whose LLM returns a fixed raw JSON string. */
function serviceReturning(raw: string): TriageService {
  const llm = { chat: async () => raw } as unknown as LlmClient;
  return new TriageService(llm, silentLogger);
}

const email: EmailToTriage = {
  messageId: 'm1',
  subject: 'Soutenance',
  fromAddr: 'scolarite@mines.fr',
  date: Date.now(),
  body: 'Votre soutenance est fixee.',
  bodyFormat: 'text',
};

/** Formats a Date as YYYY-MM-DD in local time. */
function localDate(d: Date): string {
  const p = (n: number): string => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

describe('triage event extraction', () => {
  it('resolves a dated event into absolute local-time timestamps', async () => {
    const future = new Date();
    future.setDate(future.getDate() + 30);
    const date = localDate(future);
    const raw = JSON.stringify({
      bucket: 'urgent',
      actionRequired: true,
      needsReply: false,
      deadlineHours: null,
      importanceScore: 88,
      suggestedAction: 'Prepare the defense',
      shortSummary: 'Defense scheduled.',
      reasoning: 'Scheduled defense.',
      event: {
        title: 'Soutenance de stage',
        date,
        start: '14:00',
        end: '15:30',
        allDay: false,
        location: 'Salle B12',
      },
    });

    const result = await serviceReturning(raw).classify(email);
    expect(result.event).not.toBeNull();
    const ev = result.event!;
    expect(ev.title).toBe('Soutenance de stage');
    expect(ev.allDay).toBe(false);
    expect(ev.location).toBe('Salle B12');

    const s = new Date(ev.startAt);
    expect(localDate(s)).toBe(date);
    expect(s.getHours()).toBe(14);
    expect(s.getMinutes()).toBe(0);
    const e = new Date(ev.endAt!);
    expect(e.getHours()).toBe(15);
    expect(e.getMinutes()).toBe(30);
    expect(ev.endAt!).toBeGreaterThan(ev.startAt);
  });

  it('treats an event with no start time as all-day', async () => {
    const future = new Date();
    future.setDate(future.getDate() + 10);
    const date = localDate(future);
    const raw = JSON.stringify({
      bucket: 'summarize',
      event: {
        title: 'Public holiday',
        date,
        start: null,
        end: null,
        allDay: true,
        location: null,
      },
    });

    const ev = (await serviceReturning(raw).classify(email)).event!;
    expect(ev.allDay).toBe(true);
    expect(ev.endAt).toBeNull();
    expect(new Date(ev.startAt).getHours()).toBe(0);
  });

  it('returns no event when the model reports none', async () => {
    const raw = JSON.stringify({ bucket: 'summarize', event: null });
    expect((await serviceReturning(raw).classify(email)).event).toBeNull();
  });

  it('rejects an event with an unusable date', async () => {
    const raw = JSON.stringify({
      bucket: 'summarize',
      event: { title: 'Vague meeting', date: 'next week', start: '10:00' },
    });
    expect((await serviceReturning(raw).classify(email)).event).toBeNull();
  });

  it('rejects an event dated implausibly far in the future', async () => {
    const raw = JSON.stringify({
      bucket: 'summarize',
      event: { title: 'Distant', date: '2999-01-01', start: '10:00' },
    });
    expect((await serviceReturning(raw).classify(email)).event).toBeNull();
  });
});
