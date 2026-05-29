/**
 * Read-only aggregation service that assembles the "Today's Focus" priority view
 * from triage data, grouping emails into sections with counts and carryover.
 */
import type { PriorityRange, PriorityResponse, PriorityEmailDto } from '@ai-mailpilot/shared';
import type { PriorityEmail, TriageRepository } from '../repositories/triage-repository.js';

const DAY_MS = 24 * 60 * 60 * 1000;
const SECTION_LIMIT = 50;
const CARRYOVER_LIMIT = 50;
const CARRYOVER_LOOKBACK_DAYS = 14;

/**
 * Inputs for building the priority view. dayStartMs is the start of today in the
 * caller's local timezone as epoch ms, since boundaries are computed in local time.
 */
export interface PriorityOptions {
  range: PriorityRange;
  dayStartMs: number;
  now?: number;
}

/**
 * Builds the "Today's Focus" priority view as a pure read aggregation over the
 * triage repository. Range boundaries come from the caller in local time, and
 * carryover only applies to the bounded ranges (today and week).
 */
export class PriorityService {
  /** Creates the service over the triage repository. */
  constructor(private triage: TriageRepository) {}

  /**
   * Aggregates the priority sections, counts, and carryover for an account
   * within the requested range and returns the assembled response.
   */
  build(accountId: string, opts: PriorityOptions): PriorityResponse {
    const now = opts.now ?? Date.now();
    const dayStart = opts.dayStartMs;
    const sinceMs =
      opts.range === 'today' ? dayStart : opts.range === 'week' ? dayStart - 6 * DAY_MS : 0;
    const beforeMs = opts.range === 'all' ? null : dayStart + DAY_MS;

    const counts = this.triage.priorityCounts(accountId, sinceMs, beforeMs, now);
    const unclassified = this.triage.countUnclassifiedInRange(accountId, sinceMs, beforeMs);

    const needsAction = this.triage.listSection(
      accountId,
      'needsAction',
      sinceMs,
      beforeMs,
      now,
      SECTION_LIMIT,
    );
    const important = this.triage.listSection(
      accountId,
      'important',
      sinceMs,
      beforeMs,
      now,
      SECTION_LIMIT,
    );
    const summaries = this.triage.listSection(
      accountId,
      'summaries',
      sinceMs,
      beforeMs,
      now,
      SECTION_LIMIT,
    );
    const lowPriority = this.triage.listSection(
      accountId,
      'lowPriority',
      sinceMs,
      beforeMs,
      now,
      SECTION_LIMIT,
    );

    const carryover =
      sinceMs > 0
        ? this.triage.listCarryover(
            accountId,
            sinceMs,
            Math.max(0, sinceMs - CARRYOVER_LOOKBACK_DAYS * DAY_MS),
            now,
            CARRYOVER_LIMIT,
          )
        : [];

    return {
      accountId,
      range: opts.range,
      generatedAt: now,
      counts: {
        needsAction: counts.needsAction,
        urgent: counts.urgent,
        important: counts.important,
        summaries: counts.summaries,
        lowPriority: counts.lowPriority,
        unclassified,
      },
      needsAction: needsAction.map(toDto),
      important: important.map(toDto),
      summaries: summaries.map(toDto),
      carryover: carryover.map(toDto),
      lowPriority: lowPriority.map(toDto),
    };
  }
}

/** Maps an internal priority email row to the wire-facing DTO shape. */
function toDto(e: PriorityEmail): PriorityEmailDto {
  return {
    messageId: e.messageId,
    folder: e.folder,
    subject: e.subject,
    fromAddr: e.fromAddr,
    date: e.date,
    hasAttachments: e.hasAttachments,
    bucket: e.bucket,
    reasoning: e.reasoning,
    classifiedAt: e.classifiedAt,
    actionRequired: e.actionRequired,
    needsReply: e.needsReply,
    deadlineAt: e.deadlineAt,
    importanceScore: e.importanceScore,
    suggestedAction: e.suggestedAction,
    shortSummary: e.shortSummary,
  };
}
