/**
 * Phase 2d tests: the review-queue copy helpers. These guard the user-facing wording and its edge
 * cases (zero, singular, plural, large counts) so the queue never shows misleading text.
 */
import { describe, it, expect } from 'vitest';
import {
  proposalCountLabel,
  proposalsSummary,
  proposalsBadgeLabel,
  proposalKindTag,
  proposalActionLabel,
  proposalIsApplyable,
  proposalDisabledReason,
  proposalAffectedLine,
  proposalUserImpactNote,
  proposalApplySuccessCopy,
} from '../src/ui/dashboard/proposals-format.js';

describe('proposalCountLabel', () => {
  it('pluralizes and caps the estimated email count', () => {
    expect(proposalCountLabel(0)).toBe('~0 emails');
    expect(proposalCountLabel(1)).toBe('~1 email');
    expect(proposalCountLabel(7)).toBe('~7 emails');
    expect(proposalCountLabel(100)).toBe('~100+ emails');
    expect(proposalCountLabel(9999)).toBe('~100+ emails');
  });

  it('never shows a negative or fractional count', () => {
    expect(proposalCountLabel(-5)).toBe('~0 emails');
    expect(proposalCountLabel(3.9)).toBe('~3 emails');
  });
});

describe('proposalsSummary', () => {
  it('gives an actionable empty state pointing at both generators', () => {
    const empty = proposalsSummary(0);
    expect(empty).toContain('Find new categories');
    expect(empty).toContain('Find cleanup suggestions');
    expect(empty).not.toContain('undefined');
  });

  it('agrees in number with the count and stays kind-neutral (no add-category framing)', () => {
    expect(proposalsSummary(1)).toContain('1 suggestion ');
    expect(proposalsSummary(3)).toContain('3 suggestions ');
    expect(proposalsSummary(2)).toContain('nothing changes until you approve it');
    // The queue can hold merge/retire cards, so the header must not tell the user to "Add" them.
    expect(proposalsSummary(2)).not.toContain('Add the ones you want');
  });
});

describe('proposalsBadgeLabel', () => {
  it('shows the count only when there are pending proposals', () => {
    expect(proposalsBadgeLabel(0)).toBe('Suggested categories');
    expect(proposalsBadgeLabel(4)).toBe('Suggested categories (4)');
  });
});

describe('proposalKindTag', () => {
  it('names each kind distinctly so a structural card never reads as an add-category card', () => {
    expect(proposalKindTag('new_category')).toBe('New category');
    expect(proposalKindTag('merge')).toBe('Merge');
    expect(proposalKindTag('retire')).toBe('Retire');
    expect(proposalKindTag('split')).toBe('Split');
  });
});

describe('proposalActionLabel', () => {
  it('labels the primary action per kind and never "Add" for a structural change', () => {
    expect(proposalActionLabel('new_category')).toBe('Add');
    expect(proposalActionLabel('merge')).toBe('Merge');
    expect(proposalActionLabel('retire')).toBe('Retire');
    expect(proposalActionLabel('split')).toBe('Split');
  });
});

describe('proposalIsApplyable', () => {
  it('offers apply for every non-split kind regardless of child count', () => {
    expect(proposalIsApplyable('new_category')).toBe(true);
    expect(proposalIsApplyable('merge')).toBe(true);
    expect(proposalIsApplyable('retire')).toBe(true);
  });

  it('offers apply for a split only when it carries at least two reviewable children', () => {
    expect(proposalIsApplyable('split', 2)).toBe(true);
    expect(proposalIsApplyable('split', 3)).toBe(true);
    expect(proposalIsApplyable('split', 1)).toBe(false);
    expect(proposalIsApplyable('split', 0)).toBe(false);
    // Default child count is zero, so a split with no children detail is not applyable.
    expect(proposalIsApplyable('split')).toBe(false);
  });
});

describe('proposalDisabledReason', () => {
  it('explains only the unsafe case (a split without reviewable children)', () => {
    expect(proposalDisabledReason('split', 0)).toContain('cannot be applied');
    expect(proposalDisabledReason('split', 1)).toContain('cannot be applied');
    expect(proposalDisabledReason('split', 2)).toBeNull();
    expect(proposalDisabledReason('merge')).toBeNull();
    expect(proposalDisabledReason('retire')).toBeNull();
    expect(proposalDisabledReason('new_category')).toBeNull();
  });
});

describe('proposalAffectedLine', () => {
  it('describes the affected mail per structural kind and stays silent for new_category', () => {
    expect(proposalAffectedLine('retire', 4)).toBe('4 emails currently assigned');
    expect(proposalAffectedLine('retire', 1)).toBe('1 email currently assigned');
    expect(proposalAffectedLine('merge', 9)).toContain('moves up to 9 emails');
    expect(proposalAffectedLine('new_category', 5)).toBeNull();
  });

  it('scopes a split to auto-assigned mail only, since user rows do not move', () => {
    expect(proposalAffectedLine('split', 12)).toBe(
      'up to 12 auto-assigned emails may move into child categories',
    );
    expect(proposalAffectedLine('split', 1)).toBe(
      'up to 1 auto-assigned email may move into child categories',
    );
    // It must not claim every affected email is reassigned.
    expect(proposalAffectedLine('split', 12)).not.toContain('to reassign');
  });
});

describe('proposalUserImpactNote', () => {
  it('warns only when user-confirmed assignments are involved (merge/retire)', () => {
    expect(proposalUserImpactNote('merge', 0)).toBeNull();
    expect(proposalUserImpactNote('merge', 1)).toBe('1 user-confirmed assignment affected');
    expect(proposalUserImpactNote('retire', 3)).toBe('3 user-confirmed assignments affected');
  });

  it('tells the user their confirmed mail stays on the source for a split, never "affected"', () => {
    expect(proposalUserImpactNote('split', 0)).toBeNull();
    expect(proposalUserImpactNote('split', 1)).toBe(
      '1 user-confirmed email will stay on the source category',
    );
    expect(proposalUserImpactNote('split', 3)).toBe(
      '3 user-confirmed emails will stay on the source category',
    );
    expect(proposalUserImpactNote('split', 3)).not.toContain('affected');
  });
});

describe('proposalApplySuccessCopy', () => {
  it('phrases new_category success as adding and filing', () => {
    expect(proposalApplySuccessCopy('new_category', 'Receipts', 3)).toBe(
      'Added "Receipts" and filed 3 emails.',
    );
    expect(proposalApplySuccessCopy('new_category', 'Receipts', 1)).toBe(
      'Added "Receipts" and filed 1 email.',
    );
    expect(proposalApplySuccessCopy('new_category', 'Receipts', 0)).toBe('Added "Receipts".');
  });

  it('phrases merge success as merging into the surviving category, never "Added"', () => {
    const copy = proposalApplySuccessCopy('merge', 'Bills', 5);
    expect(copy).toBe('Merged into "Bills" and moved 5 emails.');
    expect(copy).not.toContain('Added');
    expect(proposalApplySuccessCopy('merge', 'Bills', 0)).toBe('Merged into "Bills".');
  });

  it('phrases retire success as retiring, never "Added"', () => {
    const copy = proposalApplySuccessCopy('retire', 'Old Newsletters', 0);
    expect(copy).toBe('Retired "Old Newsletters".');
    expect(copy).not.toContain('Added');
  });

  it('phrases split success as splitting, never "Added"', () => {
    const copy = proposalApplySuccessCopy('split', 'Big Bucket', 4);
    expect(copy).toBe('Split "Big Bucket" and moved 4 emails.');
    expect(copy).not.toContain('Added');
  });
});
