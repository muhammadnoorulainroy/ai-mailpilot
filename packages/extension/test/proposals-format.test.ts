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
  it('offers apply for every kind except split, whose child detail is not exposed here', () => {
    expect(proposalIsApplyable('new_category')).toBe(true);
    expect(proposalIsApplyable('merge')).toBe(true);
    expect(proposalIsApplyable('retire')).toBe(true);
    expect(proposalIsApplyable('split')).toBe(false);
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
