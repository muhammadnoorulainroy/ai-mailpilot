/**
 * Tests for MailboxSnapshot.fetchMessages, covering targeted new-mail indexing,
 * deduplication, skipping headers without a Message-ID, and retry handling when
 * a body fetch fails.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { MailboxSnapshot } from '../src/thunderbird/mailbox.js';

/** Builds a minimal TbMessageHeader fixture for the given id and Message-ID. */
const header = (id: number, messageId: string): TbMessageHeader =>
  ({
    id,
    headerMessageId: messageId,
    subject: `Subj ${id}`,
    author: `a${id}@x.y`,
    date: new Date('2026-06-22T10:00:00Z'),
    recipients: [],
    read: false,
    flagged: false,
    folder: { accountId: 'acc1', path: 'INBOX', name: 'Inbox' },
  }) as TbMessageHeader;

describe('MailboxSnapshot.fetchMessages (targeted new-mail indexing)', () => {
  let getFull: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    getFull = vi.fn(async (id: number) => ({ contentType: 'text/plain', body: `Body ${id}`, headers: {} }));
    vi.stubGlobal('browser', { messages: { getFull } });
  });
  afterEach(() => vi.unstubAllGlobals());

  it('fetches each unique message once, skipping duplicates and headers with no Message-ID', async () => {
    const snap = new MailboxSnapshot([]);
    const headers = [
      header(1, 'm1'),
      header(1, 'm1'),
      header(2, 'm2'),
      header(3, ''),
    ];

    const { items, attachmentMsgs } = await snap.fetchMessages('INBOX', headers);

    expect(items.map((i) => i.messageId)).toEqual(['m1', 'm2']);
    expect(getFull).toHaveBeenCalledTimes(2);
    expect(attachmentMsgs).toEqual([]);
    expect(items[0]).toMatchObject({
      messageId: 'm1',
      folder: 'INBOX',
      subject: 'Subj 1',
      fromAddr: 'a1@x.y',
      bodyFetched: true,
      bodyFormat: 'text',
    });
    expect(items[0]!.body).toContain('Body 1');
  });

  it('records a message whose body fetch fails as not-fetched, so a later sync retries it', async () => {
    getFull.mockRejectedValueOnce(new Error('IMAP transient error'));
    const snap = new MailboxSnapshot([]);

    const { items } = await snap.fetchMessages('INBOX', [header(5, 'm5')]);

    expect(items).toHaveLength(1);
    expect(items[0]).toMatchObject({ messageId: 'm5', bodyFetched: false });
    expect(items[0]!.body).toBeUndefined();
  });
});
