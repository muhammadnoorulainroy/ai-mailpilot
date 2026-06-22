/**
 * Tests for mergeMailpilotTags, verifying that user and built-in tags are
 * preserved while stale MailPilot tags are replaced by the desired set and
 * duplicates are removed.
 */
import { describe, it, expect } from 'vitest';
import { mergeMailpilotTags } from '../src/thunderbird/tags.js';

describe('mergeMailpilotTags (never strip the user own tags)', () => {
  it('keeps user tags and adds the desired MailPilot tags', () => {
    expect(mergeMailpilotTags(['important', 'mailpilot_old'], ['mailpilot_work']).sort()).toEqual([
      'important',
      'mailpilot_work',
    ]);
  });

  it('drops a stale MailPilot tag not in the desired set, but never a user/built-in tag', () => {
    expect(mergeMailpilotTags(['$label1', 'mailpilot_a', 'mailpilot_b'], ['mailpilot_a']).sort()).toEqual([
      '$label1',
      'mailpilot_a',
    ]);
  });

  it('dedups and handles a message with no existing tags', () => {
    expect(mergeMailpilotTags([], ['mailpilot_x', 'mailpilot_x'])).toEqual(['mailpilot_x']);
  });
});
