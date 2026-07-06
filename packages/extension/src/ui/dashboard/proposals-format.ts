/**
 * Pure copy helpers for the suggested-categories review queue (Phase 2d). Kept apart from the DOM so
 * the user-facing wording and its edge cases (zero, singular, large counts) can be unit-tested.
 */
import type { ProposalKindDto } from '@ai-mailpilot/shared';

/** A short tag naming the proposal kind, so a merge/retire/split never reads as an add-category card. */
export function proposalKindTag(kind: ProposalKindDto): string {
  switch (kind) {
    case 'merge':
      return 'Merge';
    case 'retire':
      return 'Retire';
    case 'split':
      return 'Split';
    default:
      return 'New category';
  }
}

/** The primary action button label for a proposal card. Never "Add" for a structural change. */
export function proposalActionLabel(kind: ProposalKindDto): string {
  switch (kind) {
    case 'merge':
      return 'Merge';
    case 'retire':
      return 'Retire';
    case 'split':
      return 'Split';
    default:
      return 'Add';
  }
}

/**
 * Whether the primary action is safe to offer from this queue. A split hides which child categories
 * it would create (that detail is not exposed to the review view), so it is never applyable here.
 */
export function proposalIsApplyable(kind: ProposalKindDto): boolean {
  return kind !== 'split';
}

/** The status line after applying a proposal, phrased for its kind (never "Added" for merge/retire/split). */
export function proposalApplySuccessCopy(
  kind: ProposalKindDto,
  label: string,
  assigned: number,
): string {
  const emails = `${assigned} email${assigned === 1 ? '' : 's'}`;
  switch (kind) {
    case 'merge':
      return assigned > 0
        ? `Merged into "${label}" and moved ${emails}.`
        : `Merged into "${label}".`;
    case 'retire':
      return `Retired "${label}".`;
    case 'split':
      return assigned > 0 ? `Split "${label}" and moved ${emails}.` : `Split "${label}".`;
    default:
      return assigned > 0 ? `Added "${label}" and filed ${emails}.` : `Added "${label}".`;
  }
}

/** The estimated-size label for a proposal, capped so a large cluster reads as "~100+". */
export function proposalCountLabel(count: number): string {
  const n = Math.max(0, Math.floor(count));
  if (n >= 100) return '~100+ emails';
  return `~${n} email${n === 1 ? '' : 's'}`;
}

/**
 * The review-queue summary line for a given number of pending proposals. Kind-neutral: the queue can
 * mix new-category, merge, and retire suggestions, so the header never says "Add the ones you want".
 */
export function proposalsSummary(count: number): string {
  if (count <= 0) {
    return 'No suggestions right now. Use "Find new categories" or "Find cleanup suggestions" to look for changes worth reviewing.';
  }
  const noun = count === 1 ? 'suggestion' : 'suggestions';
  return `${count} ${noun} to review. Each one is described on its card; nothing changes until you approve it.`;
}

/** The "Suggested categories" toolbar button label, showing the pending count when there is one. */
export function proposalsBadgeLabel(count: number): string {
  return count > 0 ? `Suggested categories (${count})` : 'Suggested categories';
}
