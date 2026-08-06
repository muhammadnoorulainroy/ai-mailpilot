/**
 * Part B (extension side): the sync tag resolver maps a message's Thunderbird tag keys to the user's
 * own tags with visible labels, dropping MailPilot-managed tags so they are never mistaken for user
 * organization.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { loadUserTagResolver } from '../src/thunderbird/mailbox.js';

afterEach(() => vi.unstubAllGlobals());

describe('loadUserTagResolver', () => {
  it('resolves user tag keys to visible labels and drops MailPilot tags', async () => {
    vi.stubGlobal('browser', {
      messages: {
        tags: {
          list: async () => [
            { key: '$work', tag: 'Work', color: '#000', ordinal: '' },
            { key: '$uni', tag: 'University', color: '#000', ordinal: '' },
            { key: 'mailpilot_banking', tag: 'Banking Transactions', color: '#000', ordinal: '' },
          ],
        },
      },
    });
    const resolve = await loadUserTagResolver();
    expect(resolve(['$work', 'mailpilot_banking', '$uni'])).toEqual([
      { key: '$work', label: 'Work' },
      { key: '$uni', label: 'University' },
    ]);
  });

  it('falls back to the key as label for an unknown tag and handles no tags', async () => {
    vi.stubGlobal('browser', { messages: { tags: { list: async () => [] } } });
    const resolve = await loadUserTagResolver();
    expect(resolve(['$ghost'])).toEqual([{ key: '$ghost', label: '$ghost' }]);
    expect(resolve([])).toEqual([]);
  });
});
