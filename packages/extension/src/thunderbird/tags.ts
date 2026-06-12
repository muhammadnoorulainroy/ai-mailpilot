/**
 * Syncs MailPilot category assignments into Thunderbird message tags, creating and renaming
 * per-category tags, applying them to matching messages, and clearing tags that have gone stale.
 */
import type { CategoryDto } from '@ai-mailpilot/shared';
import { coreClient } from '../api-client/core-client.js';
import { MailboxSnapshot } from './mailbox.js';
import { loadSyncPrefs } from '../settings/sync-prefs.js';

const KEY_PREFIX = 'mailpilot_';

const PAGE_SIZE = 500;

const PALETTE: string[] = [
  '#C8553D',
  '#15803D',
  '#B45309',
  '#7A3A08',
  '#5A564E',
  '#8B341F',
  '#A8412C',
  '#0D5A2B',
];

/** Derive a Thunderbird-safe tag key from a category id, namespaced with the MailPilot prefix. */
function tagKeyFor(categoryId: string): string {
  return (KEY_PREFIX + categoryId).toLowerCase().replace(/[^a-z0-9_-]/g, '_');
}

/**
 * Merge MailPilot's category tags into a message's tags without dropping the user's own tags.
 * Thunderbird's messages.update replaces the entire tag set, so keep every non-MailPilot tag and
 * set the MailPilot tags to exactly desired.
 */
export function mergeMailpilotTags(existing: string[], desired: Iterable<string>): string[] {
  const userTags = existing.filter((k) => !k.startsWith(KEY_PREFIX));
  return [...new Set([...userTags, ...desired])];
}

/** Pick a stable color for a category by cycling through the fixed palette. */
function colorFor(index: number): string {
  return PALETTE[index % PALETTE.length] as string;
}

/**
 * Ensure one Thunderbird tag per category. Creates missing tags, renames tags whose label
 * changed, and returns a categoryId to tagKey map for setting tags on messages.
 */
async function ensureCategoryTags(categories: CategoryDto[]): Promise<Map<string, string>> {
  const existing = await browser.messages.tags.list();
  const byKey = new Map(existing.map((t) => [t.key, t]));

  const result = new Map<string, string>();
  for (let i = 0; i < categories.length; i++) {
    const cat = categories[i];
    if (!cat) continue;

    const key = tagKeyFor(cat.id);
    const color = colorFor(i);
    const tag = byKey.get(key);

    if (!tag) {
      await browser.messages.tags.create(key, cat.label, color);
    } else if (tag.tag !== cat.label) {
      await browser.messages.tags.update(key, { tag: cat.label });
    }
    result.set(cat.id, key);
  }

  return result;
}

/** Counts summarizing the outcome of applying category tags to an account's messages. */
export interface ApplyTagsResult {
  taggedMessages: number;
  missingMessages: number;
  tagsCreated: number;
  staleTagsCleared: number;
}

/**
 * Load the account's category assignments from Core, ensure tags exist in Thunderbird, then set
 * tags on every message that has a match. Returns counts for a status line.
 */
export async function applyTagsForAccount(
  coreAccountId: string,
  accountTbId: string,
): Promise<ApplyTagsResult> {
  const prefs = await loadSyncPrefs();

  const list = await coreClient.listCategories(coreAccountId);
  if (list.categories.length === 0) {
    return { taggedMessages: 0, missingMessages: 0, tagsCreated: 0, staleTagsCleared: 0 };
  }

  const existingBefore = await browser.messages.tags.list();
  const beforeKeys = new Set(existingBefore.map((t) => t.key));

  const tagKeyByCategory = await ensureCategoryTags(list.categories);
  const tagsCreated = [...tagKeyByCategory.values()].filter((k) => !beforeKeys.has(k)).length;

  const snapshot = await MailboxSnapshot.load();
  const idMap = await snapshot.buildMessageIdMap(accountTbId, prefs.excludedFolderPaths);

  const tagsByMessageId = new Map<string, Set<string>>();
  for (const cat of list.categories) {
    const tagKey = tagKeyByCategory.get(cat.id);
    if (!tagKey) continue;
    for (let offset = 0; ; offset += PAGE_SIZE) {
      const page = await coreClient.listEmailsInCategory(cat.id, PAGE_SIZE, offset);
      for (const email of page.emails) {
        const set = tagsByMessageId.get(email.messageId) ?? new Set();
        set.add(tagKey);
        tagsByMessageId.set(email.messageId, set);
      }
      if (page.emails.length < PAGE_SIZE) break;
    }
  }

  let tagged = 0;
  let missing = 0;
  for (const [headerMessageId, tagSet] of tagsByMessageId) {
    const tbId = idMap.get(headerMessageId);
    if (tbId === undefined) {
      missing += 1;
      continue;
    }
    try {
      const current = await browser.messages.get(tbId);
      const merged = mergeMailpilotTags(current.tags ?? [], tagSet);
      await browser.messages.update(tbId, { tags: merged });
      tagged += 1;
    } catch (err) {
      console.warn('[MailPilot] failed to tag message', tbId, err);
      missing += 1;
    }
  }

  const staleTagsCleared = await clearStaleTags(
    accountTbId,
    beforeKeys,
    tagKeyByCategory,
    tagsByMessageId,
  );

  return { taggedMessages: tagged, missingMessages: missing, tagsCreated, staleTagsCleared };
}

/**
 * Strip MailPilot tags from this account's messages that have dropped out of every category, so
 * re-categorization never leaves a stale category tag behind. Only messages no longer in the
 * desired set are touched, user tags are kept.
 */
async function clearStaleTags(
  accountTbId: string,
  beforeKeys: Set<string>,
  tagKeyByCategory: Map<string, string>,
  desired: Map<string, Set<string>>,
): Promise<number> {
  const keys = [...new Set([...beforeKeys, ...tagKeyByCategory.values()])].filter((k) =>
    k.startsWith(KEY_PREFIX),
  );
  if (keys.length === 0) return 0;

  const tagFilter = Object.fromEntries(keys.map((k) => [k, true]));
  let cleared = 0;
  let page = await browser.messages.query({
    accountId: accountTbId,
    tags: { mode: 'any', tags: tagFilter },
  });
  while (true) {
    for (const msg of page.messages) {
      if (msg.folder.accountId !== accountTbId) continue;
      if (desired.has(msg.headerMessageId)) continue;
      const cleaned = mergeMailpilotTags(msg.tags, []);
      if (cleaned.length !== msg.tags.length) {
        try {
          await browser.messages.update(msg.id, { tags: cleaned });
          cleared += 1;
        } catch (err) {
          console.warn('[MailPilot] failed to clear stale tags', msg.id, err);
        }
      }
    }
    if (!page.id) break;
    page = await browser.messages.continueList(page.id);
    if (page.messages.length === 0) break;
  }
  return cleared;
}
