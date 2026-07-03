/**
 * Phase 2d tests: the review-queue copy helpers. These guard the user-facing wording and its edge
 * cases (zero, singular, plural, large counts) so the queue never shows misleading text.
 */
import { describe, it, expect } from 'vitest';
import {
  proposalCountLabel,
  proposalsSummary,
  proposalsBadgeLabel,
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
  it('gives an actionable empty state', () => {
    const empty = proposalsSummary(0);
    expect(empty).toContain('Find new categories');
    expect(empty).not.toContain('undefined');
  });

  it('agrees in number with the count', () => {
    expect(proposalsSummary(1)).toContain('1 suggested category ');
    expect(proposalsSummary(3)).toContain('3 suggested categories ');
    expect(proposalsSummary(2)).toContain('nothing changes until you do');
  });
});

describe('proposalsBadgeLabel', () => {
  it('shows the count only when there are pending proposals', () => {
    expect(proposalsBadgeLabel(0)).toBe('Suggested categories');
    expect(proposalsBadgeLabel(4)).toBe('Suggested categories (4)');
  });
});
