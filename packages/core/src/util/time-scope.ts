/**
 * Parses natural-language time expressions in English and French from chat queries into absolute
 * date ranges, and helps separate the time part of a query from its semantic topic.
 */

/**
 * An absolute date range derived from a time expression in a chat query, with the matched
 * text so callers can strip it from the semantic part.
 */
export interface TimeScope {
  from: number;
  to: number;
  label: string;
  matched: string;
}

const MONTHS: Record<string, number> = {
  january: 0,
  february: 1,
  march: 2,
  april: 3,
  may: 4,
  june: 5,
  july: 6,
  august: 7,
  september: 8,
  october: 9,
  november: 10,
  december: 11,
  jan: 0,
  feb: 1,
  mar: 2,
  apr: 3,
  jun: 5,
  jul: 6,
  aug: 7,
  sep: 8,
  sept: 8,
  oct: 9,
  nov: 10,
  dec: 11,
  janvier: 0,
  février: 1,
  fevrier: 1,
  mars: 2,
  avril: 3,
  mai: 4,
  juin: 5,
  juillet: 6,
  août: 7,
  aout: 7,
  septembre: 8,
  octobre: 9,
  novembre: 10,
  décembre: 11,
  decembre: 11,
};
const MONTH_ALT = Object.keys(MONTHS)
  .sort((a, b) => b.length - a.length)
  .join('|');

const RECENT_DAYS = 30;

/** Local-time timestamp for midnight at the start of the given date. */
function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate(), 0, 0, 0, 0).getTime();
}
/** Local-time timestamp for the last millisecond of the given date. */
function endOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate(), 23, 59, 59, 999).getTime();
}
/** Returns the date shifted by the given number of days, positive or negative. */
function dayOffset(d: Date, days: number): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate() + days);
}
/** Full local-time range covering the given calendar month. */
function monthRange(year: number, monthIndex: number): { from: number; to: number } {
  return {
    from: new Date(year, monthIndex, 1, 0, 0, 0, 0).getTime(),
    to: new Date(year, monthIndex + 1, 0, 23, 59, 59, 999).getTime(),
  };
}
/** Full local-time range covering the given calendar year. */
function yearRange(year: number): { from: number; to: number } {
  return {
    from: new Date(year, 0, 1, 0, 0, 0, 0).getTime(),
    to: new Date(year, 11, 31, 23, 59, 59, 999).getTime(),
  };
}
/**
 * Monday-to-Sunday range for the week containing `now`, shifted by `offsetWeeks` (0 this week,
 * -1 last week, +1 next week).
 */
function weekRange(now: Date, offsetWeeks: number): { from: number; to: number } {
  const dow = (now.getDay() + 6) % 7;
  const monday = dayOffset(now, -dow + offsetWeeks * 7);
  return { from: startOfDay(monday), to: endOfDay(dayOffset(monday, 6)) };
}
/** Range for the next occurrence of a weekday (0 Sunday to 6 Saturday), today included. */
function nextWeekdayRange(now: Date, targetDow: number): { from: number; to: number } {
  const delta = (targetDow - now.getDay() + 7) % 7;
  const d = dayOffset(now, delta);
  return { from: startOfDay(d), to: endOfDay(d) };
}
/** Range for the upcoming Saturday and Sunday, today included when it is the weekend. */
function weekendRange(now: Date): { from: number; to: number } {
  const toSat = (6 - now.getDay() + 7) % 7;
  const sat = dayOffset(now, toSat);
  return { from: startOfDay(sat), to: endOfDay(dayOffset(sat, 1)) };
}

/** Weekday names in English and French mapped to JS day-of-week (0 Sunday to 6 Saturday). */
const WEEKDAYS: Record<string, number> = {
  sunday: 0,
  monday: 1,
  tuesday: 2,
  wednesday: 3,
  thursday: 4,
  friday: 5,
  saturday: 6,
  dimanche: 0,
  lundi: 1,
  mardi: 2,
  mercredi: 3,
  jeudi: 4,
  vendredi: 5,
  samedi: 6,
};
const WEEKDAY_ALT = Object.keys(WEEKDAYS)
  .sort((a, b) => b.length - a.length)
  .join('|');

/**
 * Detect a time expression in a chat query and return its absolute date range, or null when
 * none is found. Ranges are computed in local time, and `now` is passed in to stay deterministic.
 */
export function parseTimeScope(query: string, now: number): TimeScope | null {
  const N = new Date(now);

  let m =
    /\b(?:in\s+the\s+)?(?:last|past|previous)\s+(\d{1,3})\s+(day|week|month|year)s?\b/i.exec(
      query,
    ) ??
    /\b(?:les\s+)?(\d{1,3})\s+(?:derni(?:ers?|ères?)\s+)?(jours?|semaines?|mois|ans?|années?)\b/i.exec(
      query,
    );
  if (m) {
    const n = parseInt(m[1]!, 10);
    const unit = m[2]!.toLowerCase();
    const from = new Date(N);
    if (/day|jour/.test(unit)) from.setDate(from.getDate() - n);
    else if (/week|semaine/.test(unit)) from.setDate(from.getDate() - 7 * n);
    else if (/month|mois/.test(unit)) from.setMonth(from.getMonth() - n);
    else from.setFullYear(from.getFullYear() - n);
    return { from: startOfDay(from), to: endOfDay(N), label: `last ${n} ${unit}`, matched: m[0]! };
  }

  m = new RegExp(`\\b(${MONTH_ALT})\\s+((?:19|20)\\d{2})\\b`, 'i').exec(query);
  if (m) {
    const r = monthRange(parseInt(m[2]!, 10), MONTHS[m[1]!.toLowerCase()]!);
    return { ...r, label: `${m[1]} ${m[2]}`, matched: m[0]! };
  }

  const named: Array<[RegExp, () => { from: number; to: number }, string]> = [
    [/\baujourd'?hui\b|\btoday\b/i, () => ({ from: startOfDay(N), to: endOfDay(N) }), 'today'],
    [
      /\btomorrow\b|\bdemain\b/i,
      () => {
        const d = dayOffset(N, 1);
        return { from: startOfDay(d), to: endOfDay(d) };
      },
      'tomorrow',
    ],
    [
      /\byesterday\b|\bhier\b/i,
      () => {
        const d = dayOffset(N, -1);
        return { from: startOfDay(d), to: endOfDay(d) };
      },
      'yesterday',
    ],
    [/\bthis\s+weekend\b|\bce\s+week-?end\b/i, () => weekendRange(N), 'this weekend'],
    [/\bnext\s+week\b|\b(?:la\s+)?semaine\s+prochaine\b/i, () => weekRange(N, 1), 'next week'],
    [/\bthis\s+week\b|\bcette\s+semaine\b/i, () => weekRange(N, 0), 'this week'],
    [
      /\blast\s+week\b|\b(?:la\s+)?semaine\s+(?:derni(?:è|e)re|pass(?:é|e)e)\b/i,
      () => weekRange(N, -1),
      'last week',
    ],
    [
      /\bnext\s+month\b|\b(?:le\s+)?mois\s+prochain\b/i,
      () => monthRange(N.getFullYear(), N.getMonth() + 1),
      'next month',
    ],
    [
      /\bthis\s+month\b|\bce\s+mois(?:-ci)?\b/i,
      () => monthRange(N.getFullYear(), N.getMonth()),
      'this month',
    ],
    [
      /\blast\s+month\b|\b(?:le\s+)?mois\s+(?:dernier|pass(?:é|e))\b/i,
      () => monthRange(N.getFullYear(), N.getMonth() - 1),
      'last month',
    ],
    [
      /\bnext\s+year\b|\b(?:l'?)?ann(?:é|e)e\s+prochaine\b/i,
      () => yearRange(N.getFullYear() + 1),
      'next year',
    ],
    [/\bthis\s+year\b|\bcette\s+ann(?:é|e)e\b/i, () => yearRange(N.getFullYear()), 'this year'],
    [
      /\blast\s+year\b|\b(?:l'?)?ann(?:é|e)e\s+(?:derni(?:è|e)re|pass(?:é|e)e)\b/i,
      () => yearRange(N.getFullYear() - 1),
      'last year',
    ],
  ];
  for (const [re, range, label] of named) {
    const hit = re.exec(query);
    if (hit) return { ...range(), label, matched: hit[0]! };
  }

  // A weekday name ("Friday", "vendredi"), optionally prefixed with on/this/next, resolves to the
  // next occurrence of that day. Placed after the fixed phrases so "next week" is not shadowed.
  m = new RegExp(`\\b(?:on|this|next|ce|le)?\\s*(${WEEKDAY_ALT})\\b`, 'i').exec(query);
  if (m) {
    const dow = WEEKDAYS[m[1]!.toLowerCase()]!;
    return { ...nextWeekdayRange(N, dow), label: m[1]!.toLowerCase(), matched: m[0]! };
  }

  // A superlative recency qualifier ("most recent", "latest", "le plus récent") asks for the newest
  // matching item ("my most recent contract" = the newest contract), a preference version dedup
  // handles, not for mail received in a recent window; leave it to normal retrieval so the topic is
  // not hard-filtered to the last 30 days.
  const superlativeRecent = /\bmost\s+recent\b|\blatest\b|\bplus\s+r(?:é|e)cent(?:e)?\b/i.test(
    query,
  );
  m = superlativeRecent
    ? null
    : /\brecent(?:ly)?\b|\blately\b|\br(?:é|e)cemment\b|\br(?:é|e)cents?\b|\b(?:ces\s+)?derniers\s+jours\b/i.exec(
        query,
      );
  if (m) {
    return {
      from: startOfDay(dayOffset(N, -RECENT_DAYS)),
      to: endOfDay(N),
      label: 'recent',
      matched: m[0]!,
    };
  }

  m = /\b(?:in|en|during|durant|of|from|de)\s+((?:19|20)\d{2})\b/i.exec(query);
  if (m) {
    return { ...yearRange(parseInt(m[1]!, 10)), label: m[1]!, matched: m[0]! };
  }

  return null;
}

/**
 * Remove the matched time expression from a query, leaving the semantic part for retrieval,
 * or the original query when nothing meaningful remains.
 */
export function stripTimeScope(query: string, scope: TimeScope): string {
  const stripped = query.replace(scope.matched, ' ').replace(/\s+/g, ' ').trim();
  return stripped.length >= 2 ? stripped : query;
}

const FILLER = new Set([
  'summarize',
  'summary',
  'show',
  'list',
  'find',
  'get',
  'give',
  'tell',
  'read',
  'see',
  'me',
  'my',
  'mine',
  'our',
  'i',
  'the',
  'a',
  'an',
  'of',
  'about',
  'on',
  'in',
  'for',
  'to',
  'from',
  'any',
  'all',
  'some',
  'please',
  'regarding',
  're',
  'what',
  'whats',
  'which',
  'who',
  'email',
  'emails',
  'mail',
  'mails',
  'message',
  'messages',
  'inbox',
  'resume',
  'resumer',
  'montre',
  'montrer',
  'liste',
  'trouve',
  'donne',
  'quoi',
  'quel',
  'quelle',
  'mes',
  'mon',
  'ma',
  'les',
  'des',
  'de',
  'du',
  'le',
  'la',
  'un',
  'une',
  'sur',
  'au',
  'sujet',
  'courriel',
  'courriels',
  'pour',
  'dans',
]);

/**
 * True if a time-stripped query still names a topic, meaning a token beyond the generic command
 * and filler words. Drives whether a time scope ranks by recency for a pure time query or only
 * filters to the window for a time plus topic query.
 */
export function hasTopicTerms(strippedQuery: string): boolean {
  const tokens = strippedQuery.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];
  return tokens.some((t) => t.length >= 2 && !FILLER.has(t));
}
