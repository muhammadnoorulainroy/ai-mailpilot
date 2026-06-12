/**
 * Files categorized emails into Thunderbird folders by their primary category and can restore
 * the messages MailPilot itself moved back into the inbox, tracking moves in local storage.
 */
import { coreClient } from '../api-client/core-client.js';
import { MailboxSnapshot } from './mailbox.js';
import { loadSyncPrefs } from '../settings/sync-prefs.js';

const MOVE_BATCH = 100;

const MOVED_STORE_PREFIX = 'mailpilot_moved_';

/**
 * Load the set of header message ids MailPilot has moved for the given Thunderbird account.
 */
async function loadMovedIds(tbAccountId: string): Promise<Set<string>> {
  const key = MOVED_STORE_PREFIX + tbAccountId;
  const stored = (await browser.storage.local.get(key)) as Record<string, unknown>;
  const arr = stored[key];
  return new Set(Array.isArray(arr) ? (arr as string[]) : []);
}

/**
 * Persist the set of moved header message ids for the given Thunderbird account.
 */
async function saveMovedIds(tbAccountId: string, ids: Set<string>): Promise<void> {
  await browser.storage.local.set({ [MOVED_STORE_PREFIX + tbAccountId]: [...ids] });
}

/**
 * Progress update emitted while organizing or restoring folders.
 */
export interface OrganizeProgress {
  phase: 'folders' | 'moving' | 'done';
  moved: number;
  total: number;
}

/**
 * Outcome of an organize run, counting moved messages, ones whose Thunderbird id could not be
 * resolved, and how many category folders were used.
 */
export interface OrganizeResult {
  moved: number;
  missing: number;
  foldersUsed: number;
}

/**
 * Normalize a category or parent name into a safe folder name, stripping path separators and
 * collapsing whitespace, and falling back to a default when the result is empty.
 */
function sanitizeFolderName(name: string): string {
  const clean = name
    .replace(/[/\\]+/g, '-')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 100);
  return clean.length > 0 ? clean : 'Unnamed';
}

/**
 * Return the named subfolder of the parent, creating it if it does not already exist.
 */
async function findOrCreateChild(parent: TbAccount | TbFolder, name: string): Promise<TbFolder> {
  const subs = await browser.folders.getSubFolders(parent, false);
  const existing = subs.find((f) => f.name === name);
  return existing ?? browser.folders.create(parent, name);
}

/**
 * Recursively search a folder tree for the first folder matching the predicate.
 */
function findFolderBy(
  folders: TbFolder[] | undefined,
  pred: (f: TbFolder) => boolean,
): TbFolder | null {
  if (!folders) return null;
  for (const f of folders) {
    if (pred(f)) return f;
    const nested = findFolderBy(f.subFolders, pred);
    if (nested) return nested;
  }
  return null;
}

/**
 * Locate the inbox folder, preferring the typed inbox and falling back to a folder named inbox.
 */
function findInbox(folders: TbFolder[] | undefined): TbFolder | null {
  return (
    findFolderBy(folders, (f) => f.type === 'inbox') ??
    findFolderBy(folders, (f) => f.name.toLowerCase() === 'inbox')
  );
}

/**
 * Page through a folder and collect the Thunderbird ids and header message ids of messages
 * present in the tracked set, so only MailPilot-moved messages are returned.
 */
async function listTrackedIds(
  folder: TbFolder,
  tracked: Set<string>,
): Promise<{ tbIds: number[]; messageIds: string[] }> {
  const tbIds: number[] = [];
  const messageIds: string[] = [];
  let page = await browser.messages.list(folder);
  while (true) {
    for (const m of page.messages) {
      if (tracked.has(m.headerMessageId)) {
        tbIds.push(m.id);
        messageIds.push(m.headerMessageId);
      }
    }
    if (!page.id) break;
    page = await browser.messages.continueList(page.id);
    if (page.messages.length === 0) break;
  }
  return { tbIds, messageIds };
}

/**
 * Move message ids to the destination folder in fixed-size batches, invoking onMoved after each
 * batch so callers can track progress.
 */
async function moveInBatches(
  ids: number[],
  destination: TbFolder,
  onMoved: (movedIds: number[]) => void,
): Promise<void> {
  for (let i = 0; i < ids.length; i += MOVE_BATCH) {
    const batch = ids.slice(i, i + MOVE_BATCH);
    await browser.messages.move(batch, destination);
    onMoved(batch);
  }
}

/**
 * File each categorized email into a Thunderbird folder named after its primary category,
 * under a single parent folder. Idempotent: re-running re-files emails to their current
 * primary category.
 */
export async function organizeIntoFolders(
  coreAccountId: string,
  tbAccountId: string,
  parentName: string,
  onProgress: (p: OrganizeProgress) => void,
): Promise<OrganizeResult> {
  const plan = await coreClient.folderPlan(coreAccountId);
  if (plan.assignments.length === 0) return { moved: 0, missing: 0, foldersUsed: 0 };

  const account = await browser.accounts.get(tbAccountId);
  if (!account) throw new Error('Thunderbird account not found');

  onProgress({ phase: 'folders', moved: 0, total: plan.assignments.length });

  const parentFolder = await findOrCreateChild(account, sanitizeFolderName(parentName));
  const folderByCategory = new Map<string, TbFolder>();
  for (const cat of plan.categories) {
    folderByCategory.set(
      cat.id,
      await findOrCreateChild(parentFolder, sanitizeFolderName(cat.label)),
    );
  }

  const prefs = await loadSyncPrefs();
  const snapshot = await MailboxSnapshot.load();
  const idMap = await snapshot.buildMessageIdMap(tbAccountId, prefs.excludedFolderPaths);

  const idsByCategory = new Map<string, number[]>();
  const tbIdToMessageId = new Map<number, string>();
  let missing = 0;
  for (const a of plan.assignments) {
    const tbId = idMap.get(a.messageId);
    if (tbId === undefined) {
      missing += 1;
      continue;
    }
    const list = idsByCategory.get(a.categoryId) ?? [];
    list.push(tbId);
    idsByCategory.set(a.categoryId, list);
    tbIdToMessageId.set(tbId, a.messageId);
  }

  const total = [...idsByCategory.values()].reduce((n, l) => n + l.length, 0);
  let moved = 0;
  onProgress({ phase: 'moving', moved, total });

  const tracked = await loadMovedIds(tbAccountId);
  try {
    for (const [categoryId, ids] of idsByCategory) {
      const folder = folderByCategory.get(categoryId);
      if (!folder) continue;
      await moveInBatches(ids, folder, (movedIds) => {
        moved += movedIds.length;
        for (const tbId of movedIds) {
          const messageId = tbIdToMessageId.get(tbId);
          if (messageId) tracked.add(messageId);
        }
        onProgress({ phase: 'moving', moved, total });
      });
    }
  } finally {
    await saveMovedIds(tbAccountId, tracked);
  }

  onProgress({ phase: 'done', moved, total });
  return { moved, missing, foldersUsed: folderByCategory.size };
}

/**
 * Undo organizeIntoFolders: move only the messages MailPilot itself filed out of the category
 * subfolders back into the inbox. Messages the user moved into those folders are left
 * untouched. Messages organized before tracking existed are restored only after re-running
 * Organize re-records them.
 */
export async function restoreFromFolders(
  tbAccountId: string,
  parentName: string,
  onProgress: (p: OrganizeProgress) => void,
): Promise<{ movedBack: number }> {
  const account = await browser.accounts.get(tbAccountId);
  if (!account) throw new Error('Thunderbird account not found');

  const inbox = findInbox(account.folders);
  if (!inbox) throw new Error('Inbox not found');

  const tracked = await loadMovedIds(tbAccountId);
  if (tracked.size === 0) return { movedBack: 0 };

  const topSubs = await browser.folders.getSubFolders(account, false);
  const parent = topSubs.find((f) => f.name === sanitizeFolderName(parentName));
  if (!parent) return { movedBack: 0 };

  const catFolders = await browser.folders.getSubFolders(parent, false);
  const batches: number[][] = [];
  const tbIdToMessageId = new Map<number, string>();
  for (const folder of catFolders) {
    const { tbIds, messageIds } = await listTrackedIds(folder, tracked);
    if (tbIds.length > 0) {
      batches.push(tbIds);
      tbIds.forEach((id, i) => tbIdToMessageId.set(id, messageIds[i]!));
    }
  }

  const total = batches.reduce((n, l) => n + l.length, 0);
  let movedBack = 0;
  onProgress({ phase: 'moving', moved: 0, total });
  try {
    for (const ids of batches) {
      await moveInBatches(ids, inbox, (movedIds) => {
        movedBack += movedIds.length;
        for (const tbId of movedIds) {
          const messageId = tbIdToMessageId.get(tbId);
          if (messageId) tracked.delete(messageId);
        }
        onProgress({ phase: 'moving', moved: movedBack, total });
      });
    }
  } finally {
    await saveMovedIds(tbAccountId, tracked);
  }

  onProgress({ phase: 'done', moved: movedBack, total });
  return { movedBack };
}
