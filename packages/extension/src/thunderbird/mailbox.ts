/**
 * Reads accounts, folders, and messages from Thunderbird and turns them into the push items Core
 * indexes, plus helpers for fetching attachments and mapping Message-IDs back to Thunderbird ids.
 */
import type { PushEmailItem, PushAttachmentItem, AccountKind } from '@ai-mailpilot/shared';
import { pathHasExcludedSegment } from '../settings/sync-prefs.js';

const MAX_BODY_CHARS = 24_000;
const BATCH_BYTE_BUDGET = 4 * 1024 * 1024;

/** Truncate a body to max chars without splitting a surrogate pair at the boundary. */
function capBody(body: string | undefined, max: number): string | undefined {
  if (!body || body.length <= max) return body;
  let end = max;
  const last = body.charCodeAt(end - 1);
  if (last >= 0xd800 && last <= 0xdbff) end -= 1;
  return body.slice(0, end);
}

const ATTACHMENT_EXT = /\.(pdf|docx|txt|text|csv|tsv|md|markdown|log|json|xml|html?|xhtml)$/i;
const ATTACHMENT_TYPE = /pdf|wordprocessingml|^text\/|application\/(json|xml|csv|xhtml)/i;
const MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024;

/** Whether an attachment is a supported text-like type under the size cap, so worth extracting. */
function isIndexableAttachment(
  name: string,
  contentType: string | undefined,
  size: number | undefined,
): boolean {
  if (typeof size === 'number' && size > MAX_ATTACHMENT_BYTES) return false;
  return ATTACHMENT_EXT.test(name ?? '') || ATTACHMENT_TYPE.test(contentType ?? '');
}

/** Base64-encode bytes in chunks to avoid blowing the argument limit on large buffers. */
function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

interface TbAttachmentPart {
  name: string;
  contentType: string;
  partName: string;
  size: number;
}

/** Narrow browser.messages to the attachment methods, which are missing from the typings. */
function tbMessages(): {
  listAttachments(messageId: number): Promise<TbAttachmentPart[]>;
  getAttachmentFile(messageId: number, partName: string): Promise<Blob>;
} {
  return browser.messages as unknown as {
    listAttachments(messageId: number): Promise<TbAttachmentPart[]>;
    getAttachmentFile(messageId: number, partName: string): Promise<Blob>;
  };
}

/**
 * Fetch a message's indexable attachments as base64 for Core to extract and embed. Only
 * supported types under the size cap are returned, and errors are swallowed so a bad attachment
 * never breaks a sync.
 */
export async function fetchAttachmentFiles(messageTbId: number): Promise<PushAttachmentItem[]> {
  let parts: TbAttachmentPart[];
  try {
    parts = await tbMessages().listAttachments(messageTbId);
  } catch {
    return [];
  }
  const out: PushAttachmentItem[] = [];
  for (const p of parts) {
    if (!isIndexableAttachment(p.name, p.contentType, p.size)) continue;
    try {
      const file = await tbMessages().getAttachmentFile(messageTbId, p.partName);
      const buf = new Uint8Array(await file.arrayBuffer());
      if (buf.length === 0 || buf.length > MAX_ATTACHMENT_BYTES) continue;
      out.push({
        filename: p.name,
        contentType: p.contentType,
        partName: p.partName,
        size: buf.length,
        dataBase64: bytesToBase64(buf),
      });
    } catch {}
  }
  return out;
}

/** A mail account exposed by Thunderbird, narrowed to the fields sync needs. */
export interface MailboxAccount {
  tbId: string;
  name: string;
  type: string;
  address: string;
  kind: AccountKind;
}

/** A folder within a mail account, narrowed to the fields sync needs. */
export interface MailboxFolder {
  path: string;
  name: string;
  type?: string;
}

/**
 * A synced message that has attachments. Holds the Thunderbird id used to fetch the files and
 * the Message-ID that Core stores the email under.
 */
export interface AttachmentMsg {
  tbId: number;
  messageId: string;
}

/**
 * Snapshot of Thunderbird's account tree captured once per sync, so the account list is read
 * once rather than calling browser.accounts.list repeatedly.
 */
export class MailboxSnapshot {
  /** Wraps a snapshot of the given Thunderbird accounts. */
  constructor(private readonly accounts: TbAccount[]) {}

  /** Load the current account tree from Thunderbird. */
  static async load(): Promise<MailboxSnapshot> {
    const accounts = await browser.accounts.list(true);
    return new MailboxSnapshot(accounts);
  }

  /** List the IMAP and POP3 accounts, with a guessed kind for each address. */
  listAccounts(): MailboxAccount[] {
    return this.accounts
      .filter((a) => a.type === 'imap' || a.type === 'pop3')
      .map((a) => {
        const identity = a.identities?.[0];
        const address = identity?.email ?? a.name;
        return {
          tbId: a.id,
          name: a.name,
          type: a.type,
          address,
          kind: guessKind(address),
        };
      });
  }

  /** List every folder under an account, flattened across the folder tree. */
  listFolders(accountTbId: string): MailboxFolder[] {
    const account = this.accounts.find((a) => a.id === accountTbId);
    if (!account?.folders) return [];

    const result: MailboxFolder[] = [];
    walkFolders(account.folders, (f) => {
      result.push({ path: f.path, name: f.name, type: f.type });
    });
    return result;
  }

  /**
   * Fetch bodies for a specific set of message headers and build the push items, so an arriving
   * message can be indexed without re-listing the whole inbox. Dedupes by numeric id since a
   * debounced burst can deliver the same message in overlapping events.
   */
  async fetchMessages(
    folderPath: string,
    headers: TbMessageHeader[],
  ): Promise<{ items: PushEmailItem[]; attachmentMsgs: AttachmentMsg[] }> {
    const items: PushEmailItem[] = [];
    const attachmentMsgs: AttachmentMsg[] = [];
    const seen = new Set<number>();
    for (const header of headers) {
      if (!header.headerMessageId || seen.has(header.id)) continue;
      seen.add(header.id);
      const { body, format, hasAttachments, fetchOk } = await fetchBody(header.id);
      items.push({
        messageId: header.headerMessageId,
        folder: folderPath,
        subject: header.subject,
        fromAddr: header.author,
        date: header.date.getTime(),
        hasAttachments,
        body: capBody(body, MAX_BODY_CHARS),
        bodyFormat: format,
        bodyFetched: fetchOk,
      });
      if (hasAttachments && fetchOk) {
        attachmentMsgs.push({ tbId: header.id, messageId: header.headerMessageId });
      }
    }
    return { items, attachmentMsgs };
  }

  /**
   * Build a headerMessageId to Thunderbird numeric id map across every subscribed folder, so the
   * tag-apply pipeline can translate Core's RFC-822 message IDs into the ids messages.update
   * accepts. Folders whose path has a segment in skipFolderNames are skipped, using the same
   * segment-equality rule as sync.
   */
  async buildMessageIdMap(
    accountTbId: string,
    skipFolderNames: string[] = [],
  ): Promise<Map<string, number>> {
    const account = this.accounts.find((a) => a.id === accountTbId);
    if (!account?.folders) return new Map();

    const map = new Map<string, number>();

    const folders: TbFolder[] = [];
    walkFolders(account.folders, (f) => {
      if (pathHasExcludedSegment(f.path, skipFolderNames)) return;
      folders.push(f);
    });
    folders.sort((a, b) => folderRank(a) - folderRank(b));

    for (const folder of folders) {
      let page = await browser.messages.list(folder);
      while (true) {
        for (const header of page.messages) {
          if (header.headerMessageId && !map.has(header.headerMessageId)) {
            map.set(header.headerMessageId, header.id);
          }
        }
        if (!page.id) break;
        page = await browser.messages.continueList(page.id);
        if (page.messages.length === 0) break;
      }
    }

    return map;
  }

  /**
   * Stream every message in a folder, building PushEmailItems and invoking onBatch for each chunk
   * of up to batchSize, so a whole mailbox can be synced without loading it all at once.
   *
   * opts.selectIds, if given, is called once per page with that page's message ids and returns the
   * subset whose body should be fetched. Omitted ids are skipped without a getFull call, which is
   * how resync avoids re-fetching already-synced emails. Omit it to fetch every message.
   *
   * opts.onProgress is called once per page with the number of messages examined, so a caller can
   * drive a progress bar even when most messages are skipped.
   *
   * Returns counts of fetched bodies pushed, skipped with no Message-ID, and upToDate that had a
   * Message-ID but were skipped by selectIds.
   */
  async forEachEmailBatch(
    accountTbId: string,
    folderPath: string,
    batchSize: number,
    onBatch: (items: PushEmailItem[]) => Promise<void>,
    opts: {
      selectIds?: (messageIds: string[]) => Promise<Set<string>>;
      onProgress?: (examinedDelta: number) => void;
    } = {},
  ): Promise<{
    fetched: number;
    skipped: number;
    upToDate: number;
    attachmentMsgs: AttachmentMsg[];
  }> {
    const empty = { fetched: 0, skipped: 0, upToDate: 0, attachmentMsgs: [] as AttachmentMsg[] };
    const account = this.accounts.find((a) => a.id === accountTbId);
    if (!account?.folders) return empty;

    const folder = findFolder(account.folders, folderPath);
    if (!folder) return empty;

    let fetched = 0;
    let skipped = 0;
    let upToDate = 0;
    const attachmentMsgs: AttachmentMsg[] = [];
    let batch: PushEmailItem[] = [];
    let batchBytes = 0;
    let page = await browser.messages.list(folder);

    while (true) {
      const withId = page.messages.filter((h) => h.headerMessageId);
      skipped += page.messages.length - withId.length;

      const toFetch = opts.selectIds
        ? await opts.selectIds(withId.map((h) => h.headerMessageId))
        : null;

      for (const header of withId) {
        if (toFetch && !toFetch.has(header.headerMessageId)) {
          upToDate += 1;
          continue;
        }

        const { body, format, hasAttachments, fetchOk } = await fetchBody(header.id);
        const cappedBody = capBody(body, MAX_BODY_CHARS);
        batch.push({
          messageId: header.headerMessageId,
          folder: folderPath,
          subject: header.subject,
          fromAddr: header.author,
          date: header.date.getTime(),
          hasAttachments,
          body: cappedBody,
          bodyFormat: format,
          bodyFetched: fetchOk,
        });
        if (hasAttachments && fetchOk) {
          attachmentMsgs.push({ tbId: header.id, messageId: header.headerMessageId });
        }
        fetched += 1;
        batchBytes +=
          (cappedBody?.length ?? 0) +
          (header.subject?.length ?? 0) +
          (header.author?.length ?? 0) +
          200;

        if (batch.length >= batchSize || batchBytes >= BATCH_BYTE_BUDGET) {
          await onBatch(batch);
          batch = [];
          batchBytes = 0;
        }
      }

      opts.onProgress?.(page.messages.length);

      if (!page.id) break;
      page = await browser.messages.continueList(page.id);
      if (page.messages.length === 0) break;
    }

    if (batch.length > 0) await onBatch(batch);
    return { fetched, skipped, upToDate, attachmentMsgs };
  }
}

/** Rank inbox folders ahead of others so they are scanned first when building the id map. */
function folderRank(f: TbFolder): number {
  return f.type === 'inbox' || f.name.toLowerCase() === 'inbox' ? 0 : 1;
}

/** Visit every folder in a tree, recursing into subfolders. */
function walkFolders(folders: TbFolder[], visit: (f: TbFolder) => void): void {
  for (const f of folders) {
    visit(f);
    if (f.subFolders) walkFolders(f.subFolders, visit);
  }
}

/** Find a folder by exact path anywhere in the tree, or null if absent. */
function findFolder(folders: TbFolder[], path: string): TbFolder | null {
  for (const f of folders) {
    if (f.path === path) return f;
    if (f.subFolders) {
      const found = findFolder(f.subFolders, path);
      if (found) return found;
    }
  }
  return null;
}

/** Guess an account kind from its address domain, used to weight categorization. */
function guessKind(address: string): AccountKind {
  const domain = address.split('@')[1]?.toLowerCase() ?? '';
  if (/\b(edu|ac\.|mines-|univ-|cnrs|inria)\b/.test(domain)) return 'institutional';
  if (/\b(gmail|yahoo|hotmail|outlook|protonmail|icloud)\b/.test(domain)) return 'personal';
  return 'work';
}

/**
 * Fetch a message's full part tree and pull out the best body, preferring plain text over HTML.
 * fetchOk is false only when the getFull call itself failed, distinguishing a fetch error from a
 * message that genuinely has no body.
 */
async function fetchBody(messageId: number): Promise<{
  body: string | undefined;
  format: 'text' | 'html' | undefined;
  hasAttachments: boolean;
  fetchOk: boolean;
}> {
  try {
    const full = await browser.messages.getFull(messageId);
    const hasAttachments = detectAttachments(full);
    const plain = extractBody(full, 'text/plain');
    if (plain) return { body: plain, format: 'text', hasAttachments, fetchOk: true };

    const html = extractBody(full, 'text/html');
    if (html) return { body: html, format: 'html', hasAttachments, fetchOk: true };

    return { body: undefined, format: undefined, hasAttachments, fetchOk: true };
  } catch {
    return { body: undefined, format: undefined, hasAttachments: false, fetchOk: false };
  }
}

/**
 * Whether a message part tree contains a real downloadable attachment, excluding inline content
 * embedded in HTML. A part counts only when it is explicitly Content-Disposition: attachment, or
 * has a filename and is neither a body part nor an inline-referenced part.
 */
export function detectAttachments(part: TbMessagePart): boolean {
  const headers = part.headers ?? {};
  const cd = (headers['content-disposition']?.[0] ?? '').toLowerCase();
  const ct = (headers['content-type']?.[0] ?? '').toLowerCase();
  if (cd.includes('attachment')) return true;

  const type = part.contentType?.toLowerCase() ?? '';
  const isBody = type.startsWith('text/') || type.startsWith('multipart/') || type === '';
  const isInline =
    cd.includes('inline') || !!headers['content-id'] || !!headers['content-location'];
  const hasFilename =
    (!!part.name && part.name.trim().length > 0) ||
    /filename\*?=/.test(cd) ||
    /\bname\*?=/.test(ct);
  if (hasFilename && !isBody && !isInline) return true;

  if (part.parts) {
    for (const sub of part.parts) {
      if (detectAttachments(sub)) return true;
    }
  }
  return false;
}

/** Recursively find the first non-empty body part matching the given MIME type. */
function extractBody(part: TbMessagePart, mimeType: string): string | undefined {
  if (part.contentType === mimeType && typeof part.body === 'string' && part.body.length > 0) {
    return part.body;
  }
  if (part.parts) {
    for (const sub of part.parts) {
      const found = extractBody(sub, mimeType);
      if (found) return found;
    }
  }
  return undefined;
}
