/**
 * Syncs MailPilot category assignments into Thunderbird message tags, creating and renaming
 * per-category tags, applying them to matching messages, and clearing tags that have gone stale.
 */
import type { CategoryDto } from '@ai-mailpilot/shared';
import { coreClient } from '../api-client/core-client.js';
import { MailboxSnapshot } from './mailbox.js';
import { loadSyncPrefs } from '../settings/sync-prefs.js';

const KEY_PREFIX = 'mailpilot_';

/**
 * Prefix for a MailPilot tag's visible label when the user already owns a tag with the same name.
 * Keeps the user's tag untouched while still creating a distinct MailPilot tag Thunderbird accepts.
 */
const DISAMBIGUATION_PREFIX = 'AI: ';

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

/**
 * Derive a Thunderbird-safe MailPilot tag key from a category's canonical key, falling back to its
 * id only when the canonical key is missing. Keying on the canonical key (stable across id churn)
 * means a category reset/re-discovery that mints new ids reuses the same tag instead of colliding.
 */
export function tagKeyFor(cat: { id: string; canonicalKey?: string }): string {
  const base = cat.canonicalKey && cat.canonicalKey.length > 0 ? cat.canonicalKey : cat.id;
  return (KEY_PREFIX + base).toLowerCase().replace(/[^a-z0-9_-]/g, '_');
}

/** Whether a Thunderbird tag is MailPilot-owned (safe to create, rename, adopt, or clean). */
function isMailpilotTag(key: string): boolean {
  return key.startsWith(KEY_PREFIX);
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
 * Ensure one Thunderbird tag per category and return a categoryId to tagKey map for setting tags on
 * messages. Collision-safe and non-destructive to user tags:
 * - Exact key present: reuse it, renaming only the visible label when the category label changed.
 * - Key absent but a MailPilot tag already shows this label (e.g. left over from an old category id
 *   after a reset): adopt that tag's key instead of creating a duplicate label.
 * - A non-MailPilot user tag already shows this label: never touch it; create the MailPilot tag under
 *   a disambiguated visible label ("AI: <label>") so Thunderbird accepts it.
 * Never updates, renames, or deletes a non-MailPilot tag.
 */
export async function ensureCategoryTags(categories: CategoryDto[]): Promise<Map<string, string>> {
  const existing = await browser.messages.tags.list();
  const byKey = new Map(existing.map((t) => [t.key, t]));
  const mailpilotByLabel = new Map<string, string>();
  const userLabels = new Set<string>();
  for (const t of existing) {
    if (isMailpilotTag(t.key)) mailpilotByLabel.set(t.tag.toLowerCase(), t.key);
    else userLabels.add(t.tag.toLowerCase());
  }
  // A visible label the user owns must not be taken by a create/rename; use a disambiguated one.
  const safeLabel = (label: string): string =>
    userLabels.has(label.toLowerCase()) ? DISAMBIGUATION_PREFIX + label : label;

  const result = new Map<string, string>();
  const usedKeys = new Set<string>();
  for (let i = 0; i < categories.length; i++) {
    const cat = categories[i];
    if (!cat) continue;

    const key = tagKeyFor(cat);
    const color = colorFor(i);
    const existingTag = byKey.get(key);

    if (existingTag) {
      const desired = safeLabel(cat.label);
      if (existingTag.tag !== desired) {
        await browser.messages.tags.update(key, { tag: desired });
      }
      result.set(cat.id, key);
      usedKeys.add(key);
      continue;
    }

    // Adopt a leftover MailPilot tag that already shows this label (old id-keyed tag after a reset),
    // so we never ask Thunderbird to create a second tag with a label it already has.
    const adoptKey = mailpilotByLabel.get(cat.label.toLowerCase());
    if (adoptKey && !usedKeys.has(adoptKey)) {
      result.set(cat.id, adoptKey);
      usedKeys.add(adoptKey);
      continue;
    }

    await browser.messages.tags.create(key, safeLabel(cat.label), color);
    result.set(cat.id, key);
    usedKeys.add(key);
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
