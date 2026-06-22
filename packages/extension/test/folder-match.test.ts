/**
 * Tests for pathHasExcludedSegment, verifying that folder paths are matched
 * by exact segment against the exclude list rather than substring, including
 * case-insensitive matches and empty list handling.
 */
import { describe, it, expect } from 'vitest';
import { pathHasExcludedSegment } from '../src/settings/sync-prefs.js';

const DEFAULTS = ['Trash', 'Junk', 'Drafts', 'Spam', 'Outbox'];

describe('pathHasExcludedSegment (M12)', () => {
  it('excludes folders whose segment exactly matches (case-insensitive)', () => {
    expect(pathHasExcludedSegment('/Trash', DEFAULTS)).toBe(true);
    expect(pathHasExcludedSegment('/[Gmail]/Spam', DEFAULTS)).toBe(true);
    expect(pathHasExcludedSegment('/junk', DEFAULTS)).toBe(true);
    expect(pathHasExcludedSegment('/Archive/Drafts/2024', DEFAULTS)).toBe(true);
  });

  it('does NOT exclude folders that merely contain an excluded word', () => {
    expect(pathHasExcludedSegment('/Work/Draft Contracts', DEFAULTS)).toBe(false);
    expect(pathHasExcludedSegment('/Receipts/Junk Mail Receipts', DEFAULTS)).toBe(false);
    expect(pathHasExcludedSegment('/Clients/Spamhaus Project', DEFAULTS)).toBe(false);
    expect(pathHasExcludedSegment('/INBOX', DEFAULTS)).toBe(false);
  });

  it('returns false for empty exclude lists', () => {
    expect(pathHasExcludedSegment('/Trash', [])).toBe(false);
    expect(pathHasExcludedSegment('/Trash', ['', '   '])).toBe(false);
  });
});
