/**
 * Pure copy helpers for the suggested-categories review queue (Phase 2d). Kept apart from the DOM so
 * the user-facing wording and its edge cases (zero, singular, large counts) can be unit-tested.
 */

/** The estimated-size label for a proposal, capped so a large cluster reads as "~100+". */
export function proposalCountLabel(count: number): string {
  const n = Math.max(0, Math.floor(count));
  if (n >= 100) return '~100+ emails';
  return `~${n} email${n === 1 ? '' : 's'}`;
}

/** The review-queue summary line for a given number of pending proposals. */
export function proposalsSummary(count: number): string {
  if (count <= 0) {
    return 'No suggestions right now. Click "Find new categories" to look for recurring topics in emails that fit none of your current categories.';
  }
  const noun = count === 1 ? 'suggested category' : 'suggested categories';
  return `${count} ${noun} from emails that fit none of your current ones. Add the ones you want; nothing changes until you do.`;
}

/** The "Suggested categories" toolbar button label, showing the pending count when there is one. */
export function proposalsBadgeLabel(count: number): string {
  return count > 0 ? `Suggested categories (${count})` : 'Suggested categories';
}
