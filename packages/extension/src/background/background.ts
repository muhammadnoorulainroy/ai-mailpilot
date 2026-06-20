/**
 * Thunderbird extension background script. Drives mailbox sync to the Core
 * server, embedding and categorization, the category context menu, auto-index
 * of new mail, assistant summary prefetch, and debug helpers.
 */
import { coreClient } from '../api-client/core-client.js';
import {
  MailboxSnapshot,
  fetchAttachmentFiles,
  type MailboxAccount,
  type AttachmentMsg,
} from '../thunderbird/mailbox.js';
import {
  loadSyncPrefs,
  saveSyncPrefs,
  shouldSyncAccount,
  isFolderExcluded,
  type SyncPrefs,
} from '../settings/sync-prefs.js';

const PUSH_BATCH = 500;
const PROGRESS_POLL_MS = 2000;
const SYNC_DIFF_CHUNK = 1000;

interface SyncState {
  phase: 'idle' | 'syncing' | 'embedding' | 'done' | 'error';
  processed: number;
  total: number;
  pushed: number;
  upToDate: number;
  embedProcessed: number;
  embedTotal: number;
  error?: string;
}

/** Builds a zeroed sync state with the given starting phase. */
function freshSyncState(phase: SyncState['phase']): SyncState {
  return {
    phase,
    processed: 0,
    total: 0,
    pushed: 0,
    upToDate: 0,
    embedProcessed: 0,
    embedTotal: 0,
  };
}

let syncState: SyncState = freshSyncState('idle');
let syncInProgress = false;

/** Runs a full sync, guarding against overlapping runs. Force re-fetches all messages. */
async function sync(force = false): Promise<void> {
  if (syncInProgress) {
    console.log('[MailPilot] sync already in progress; ignoring trigger');
    return;
  }
  syncInProgress = true;
  try {
    await runSyncImpl(force);
  } finally {
    syncInProgress = false;
  }
}

/**
 * Core sync loop. Pushes each eligible account's inbox to Core in batches,
 * ingests attachments, then embeds the synced accounts. Skips already-synced
 * messages unless force is set.
 */
async function runSyncImpl(force = false): Promise<void> {
  console.log(`[MailPilot] sync started${force ? ' (force re-fetch)' : ''}`);
  syncState = freshSyncState('syncing');

  const prefs = await loadSyncPrefs();
  const snapshot = await MailboxSnapshot.load();
  const tbAccounts = snapshot.listAccounts();
  if (tbAccounts.length === 0) {
    console.warn('[MailPilot] no IMAP/POP3 accounts found in Thunderbird');
    syncState.phase = 'done';
    return;
  }

  const syncedAccountIds: string[] = [];

  for (const tb of tbAccounts) {
    const decision = shouldSyncAccount(tb.address, tb.kind, prefs);
    if (!decision.sync) {
      console.log(`[MailPilot] SKIP ${tb.address} (${tb.kind}): ${decision.reason}`);
      continue;
    }
    console.log(`[MailPilot] SYNC ${tb.address} (${tb.kind}): ${decision.reason}`);

    try {
      const created = await coreClient.createAccount({
        address: tb.address,
        displayName: tb.name,
        kind: tb.kind,
      });

      const folders = snapshot.listFolders(tb.tbId);
      console.log(`[MailPilot] ${folders.length} folders for ${tb.address}`);

      const inbox =
        folders.find((f) => f.type === 'inbox') ??
        folders.find((f) => f.name.toLowerCase() === 'inbox');

      if (!inbox) {
        console.warn(`[MailPilot] no inbox found for ${tb.address}`);
        continue;
      }

      if (isFolderExcluded(inbox.path, prefs)) {
        console.log(`[MailPilot] inbox ${inbox.path} matches excluded pattern, skipping`);
        continue;
      }

      try {
        const info = await browser.folders.getFolderInfo({
          accountId: tb.tbId,
          path: inbox.path,
          name: inbox.name,
        });
        syncState.total += info.totalMessageCount ?? 0;
      } catch {}

      console.log(
        `[MailPilot] syncing messages from "${inbox.path}" (streamed${force ? ', force re-fetch' : ', skipping already-synced'})...`,
      );

      const coreAccountId = created.account.id;
      const selectIds = force
        ? undefined
        : async (ids: string[]): Promise<Set<string>> => {
            const need = new Set<string>();
            for (let i = 0; i < ids.length; i += SYNC_DIFF_CHUNK) {
              const chunk = ids.slice(i, i + SYNC_DIFF_CHUNK);
              const res = await coreClient.syncState({
                accountId: coreAccountId,
                messageIds: chunk,
              });
              for (const id of res.needFetch) need.add(id);
            }
            return need;
          };

      let total = 0;
      const { fetched, skipped, upToDate, attachmentMsgs } = await snapshot.forEachEmailBatch(
        tb.tbId,
        inbox.path,
        PUSH_BATCH,
        async (batch) => {
          const result = await coreClient.pushEmails({ accountId: coreAccountId, emails: batch });
          total = result.total;
          console.log(
            `[MailPilot] pushed batch of ${batch.length} (new ${result.inserted}, total in Core ${result.total})`,
          );
        },
        {
          selectIds,
          onProgress: (n) => {
            syncState.processed += n;
          },
        },
      );
      syncState.pushed += fetched;
      syncState.upToDate += upToDate;

      await ingestAttachments(coreAccountId, attachmentMsgs);
      console.log(
        `[MailPilot] ${tb.address}: ${fetched} fetched, ${upToDate} already up to date, ${skipped} skipped (no Message-ID), ${total} total in Core`,
      );

      syncedAccountIds.push(coreAccountId);
    } catch (err) {
      console.error(`[MailPilot] sync failed for ${tb.address}:`, err);
    }
  }

  console.log('[MailPilot] sync complete');

  syncState.phase = 'embedding';
  for (const accountId of syncedAccountIds) {
    await embed(accountId);
  }
  syncState.phase = 'done';
}

/** Fetches and ingests attachment files for the given messages, skipping failures. */
async function ingestAttachments(accountId: string, msgs: AttachmentMsg[]): Promise<void> {
  if (msgs.length === 0) return;
  let indexed = 0;
  for (const m of msgs) {
    let files;
    try {
      files = await fetchAttachmentFiles(m.tbId);
    } catch (err) {
      console.warn(`[MailPilot] attachment fetch failed for ${m.messageId}:`, err);
      continue;
    }
    for (const file of files) {
      try {
        await coreClient.ingestAttachments({
          accountId,
          messageId: m.messageId,
          attachments: [file],
        });
        indexed += 1;
      } catch (err) {
        console.warn(
          `[MailPilot] attachment ingest failed for ${m.messageId} (${file.filename}):`,
          err,
        );
      }
    }
  }
  if (indexed > 0)
    console.log(`[MailPilot] ingested ${indexed} attachment(s) across ${msgs.length} message(s)`);
}

/** Starts embedding for an account and polls until it finishes, unless already up to date. */
async function embed(accountId: string): Promise<void> {
  try {
    const start = await coreClient.runEmbed({ accountId });
    console.log(
      `[MailPilot] embed: ${start.status}, ${start.pending} pending with model ${start.modelId}`,
    );

    if (start.status === 'up_to_date') return;
    await pollEmbedProgress();
  } catch (err) {
    console.error('[MailPilot] embed failed:', err);
  }
}

/** Polls Core embed progress into syncState until it reaches a terminal status. */
async function pollEmbedProgress(): Promise<void> {
  while (true) {
    await new Promise((r) => setTimeout(r, PROGRESS_POLL_MS));
    const progress = await coreClient.embedProgress();
    syncState.embedProcessed = progress.processed;
    syncState.embedTotal = progress.total;
    console.log(
      `[MailPilot] embed progress: ${progress.processed}/${progress.total} (${progress.failed} failed) - ${progress.status}`,
    );
    if (
      progress.status === 'completed' ||
      progress.status === 'completed_with_failures' ||
      progress.status === 'error' ||
      progress.status === 'idle'
    ) {
      return;
    }
  }
}

const CATEGORY_PARENT_ID = 'mailpilot-add-parent';
const CATEGORY_CHILD_PREFIX = 'mailpilot-cat:';
const CATEGORY_NONE_ID = `${CATEGORY_CHILD_PREFIX}__none`;
let categoryChildIds: string[] = [];
let rebuildingMenu = false;
let pendingSelection: TbMessageList | undefined;
let hasPendingRebuild = false;

/** Creates the parent "Add to category" context menu on the message list. */
async function setupMenus(): Promise<void> {
  try {
    await browser.menus.create({
      id: CATEGORY_PARENT_ID,
      title: 'Add to AI MailPilot category',
      contexts: ['message_list'],
    });
  } catch (err) {
    console.warn('[MailPilot] menu setup failed:', err);
  }
}

/** Maps a Thunderbird account id to its matching Core account id by address, or null. */
function resolveCoreAccountId(
  tbAccountId: string,
  tbAccounts: MailboxAccount[],
  coreAccounts: { id: string; address: string }[],
): string | null {
  const tb = tbAccounts.find((a) => a.tbId === tbAccountId);
  if (!tb) return null;
  const core = coreAccounts.find((a) => a.address.toLowerCase() === tb.address.toLowerCase());
  return core?.id ?? null;
}

/** Coalesces concurrent rebuild requests so only one menu rebuild runs at a time. */
async function rebuildCategoryMenu(selected: TbMessageList | undefined): Promise<void> {
  pendingSelection = selected;
  hasPendingRebuild = true;
  if (rebuildingMenu) return;

  rebuildingMenu = true;
  try {
    while (hasPendingRebuild) {
      hasPendingRebuild = false;
      await rebuildCategoryMenuInner(pendingSelection);
    }
  } finally {
    rebuildingMenu = false;
  }
}

/**
 * Rebuilds the category submenu for the selected message, listing the account's
 * categories or a disabled note when none apply.
 */
async function rebuildCategoryMenuInner(selected: TbMessageList | undefined): Promise<void> {
  for (const id of categoryChildIds) {
    try {
      await browser.menus.remove(id);
    } catch {}
  }
  categoryChildIds = [];

  const tbAccountId = selected?.messages?.[0]?.folder?.accountId;
  let cats: { id: string; label: string }[] = [];
  let note = '';

  if (!tbAccountId) {
    note = 'Select a message first';
  } else {
    try {
      const snapshot = await MailboxSnapshot.load();
      const coreAccounts = (await coreClient.listAccounts()).accounts;
      const coreAccountId = resolveCoreAccountId(
        tbAccountId,
        snapshot.listAccounts(),
        coreAccounts,
      );
      if (!coreAccountId) {
        note = 'Sync this account first';
      } else {
        cats = (await coreClient.listCategories(coreAccountId)).categories.map((c) => ({
          id: c.id,
          label: c.label,
        }));
        if (cats.length === 0) note = 'No categories yet (run Organize)';
      }
    } catch {
      note = 'Core server not reachable';
    }
  }

  if (cats.length > 0) {
    for (const cat of cats) {
      const id = CATEGORY_CHILD_PREFIX + cat.id;
      await browser.menus.create({
        id,
        parentId: CATEGORY_PARENT_ID,
        title: cat.label,
        contexts: ['message_list'],
      });
      categoryChildIds.push(id);
    }
  } else {
    await browser.menus.create({
      id: CATEGORY_NONE_ID,
      parentId: CATEGORY_PARENT_ID,
      title: note,
      enabled: false,
      contexts: ['message_list'],
    });
    categoryChildIds.push(CATEGORY_NONE_ID);
  }

  await browser.menus.refresh();
}

browser.menus.onShown.addListener((info) => {
  if (!info.contexts?.includes('message_list')) return;
  if (!info.menuIds?.includes(CATEGORY_PARENT_ID)) return;
  void rebuildCategoryMenu(info.selectedMessages);
});

browser.menus.onClicked.addListener((info) => {
  const id = String(info.menuItemId);
  if (!id.startsWith(CATEGORY_CHILD_PREFIX) || id === CATEGORY_NONE_ID) return;
  const categoryId = id.slice(CATEGORY_CHILD_PREFIX.length);
  const messages = info.selectedMessages?.messages ?? [];
  if (messages.length === 0) return;
  void applyCategoryToMessages(categoryId, messages);
});

/** Adds the chosen category to each selected message, merging with existing categories. */
async function applyCategoryToMessages(
  categoryId: string,
  messages: TbMessageHeader[],
): Promise<void> {
  const snapshot = await MailboxSnapshot.load();
  const tbAccounts = snapshot.listAccounts();
  const coreAccounts = (await coreClient.listAccounts()).accounts;

  let applied = 0;
  for (const msg of messages) {
    const tbAccountId = msg.folder?.accountId;
    if (!tbAccountId || !msg.headerMessageId) continue;
    const coreAccountId = resolveCoreAccountId(tbAccountId, tbAccounts, coreAccounts);
    if (!coreAccountId) continue;
    try {
      const current = await coreClient.getEmailCategories(msg.headerMessageId, coreAccountId);
      const ids = new Set(current.categories.map((c) => c.categoryId));
      ids.add(categoryId);
      await coreClient.setEmailCategories(msg.headerMessageId, coreAccountId, [...ids]);
      applied += 1;
    } catch (err) {
      console.error('[MailPilot] categorize from menu failed:', msg.headerMessageId, err);
    }
  }
  console.log(`[MailPilot] added category to ${applied}/${messages.length} message(s)`);
}

const ASSISTANT_PREFETCH_MAX_PENDING = 20;

interface AssistantSummaryQueueItem {
  key: string;
  accountId: string;
  messageId: string;
  subject: string;
}

const pendingAssistantSummaries = new Map<string, AssistantSummaryQueueItem>();
let assistantSummaryPrefetchRunning = false;

/** Enqueues a message for summary prefetch, deduping and evicting the oldest when full. */
function queueAssistantSummaryPrefetch(item: Omit<AssistantSummaryQueueItem, 'key'>): void {
  const key = `${item.accountId}\u0000${item.messageId}`;
  if (pendingAssistantSummaries.has(key)) return;

  if (pendingAssistantSummaries.size >= ASSISTANT_PREFETCH_MAX_PENDING) {
    const oldest = pendingAssistantSummaries.keys().next().value as string | undefined;
    if (oldest) pendingAssistantSummaries.delete(oldest);
  }

  pendingAssistantSummaries.set(key, { ...item, key });
  void processAssistantSummaryPrefetchQueue();
}

/** Queues summary prefetch for each currently displayed message with a known Core account. */
async function prefetchDisplayedAssistantSummaries(displayed: TbMessageList): Promise<void> {
  const messages = displayed.messages ?? [];
  if (messages.length === 0) return;

  let snapshot: MailboxSnapshot;
  let coreAccounts: { id: string; address: string }[];
  try {
    snapshot = await MailboxSnapshot.load();
    coreAccounts = (await coreClient.listAccounts()).accounts;
  } catch (err) {
    console.debug('[MailPilot] assistant summary prefetch skipped; account lookup failed:', err);
    return;
  }

  const tbAccounts = snapshot.listAccounts();
  for (const msg of messages) {
    const tbAccountId = msg.folder?.accountId;
    const messageId = msg.headerMessageId;
    if (!tbAccountId || !messageId) continue;

    const accountId = resolveCoreAccountId(tbAccountId, tbAccounts, coreAccounts);
    if (!accountId) continue;

    queueAssistantSummaryPrefetch({
      accountId,
      messageId,
      subject: msg.subject || '(no subject)',
    });
  }
}

/** True when the error is a Core 404, the expected miss for an unsynced message. */
function isExpectedAssistantPrefetchMiss(err: unknown): boolean {
  return err instanceof Error && /\bCore API 404\b/.test(err.message);
}

/** Drains the prefetch queue one item at a time, requesting each summary from Core. */
async function processAssistantSummaryPrefetchQueue(): Promise<void> {
  if (assistantSummaryPrefetchRunning) return;
  assistantSummaryPrefetchRunning = true;

  try {
    while (pendingAssistantSummaries.size > 0) {
      const next = pendingAssistantSummaries.entries().next().value as
        | [string, AssistantSummaryQueueItem]
        | undefined;
      if (!next) return;

      const [key, item] = next;
      pendingAssistantSummaries.delete(key);

      try {
        const res = await coreClient.emailAssistantSummary({
          accountId: item.accountId,
          messageId: item.messageId,
        });
        if (!res.summary.cached) {
          console.log(`[MailPilot] prefetched assistant summary: ${item.subject}`);
        }
      } catch (err) {
        if (isExpectedAssistantPrefetchMiss(err)) {
          console.debug(
            `[MailPilot] assistant summary prefetch skipped for unsynced message: ${item.subject}`,
          );
        } else {
          console.warn('[MailPilot] assistant summary prefetch failed:', err);
        }
      }
    }
  } finally {
    assistantSummaryPrefetchRunning = false;
  }
}

browser.browserAction.onClicked.addListener(async () => {
  try {
    await browser.tabs.create({ url: browser.runtime.getURL('dashboard/dashboard.html') });
  } catch (err) {
    console.error('[MailPilot] failed to open dashboard:', err);
  }
});

const AUTO_SYNC_DEBOUNCE_MS = 8000;
let autoSyncTimer: ReturnType<typeof setTimeout> | null = null;
const pendingNewMail = new Map<string, { folder: TbFolder; headers: TbMessageHeader[] }>();

/** Accumulates new-mail headers per folder for the next auto-sync run. */
function addPending(folder: TbFolder, headers: TbMessageHeader[]): void {
  const key = `${folder.accountId}\u0000${folder.path}`;
  const entry = pendingNewMail.get(key);
  if (entry) entry.headers.push(...headers);
  else pendingNewMail.set(key, { folder, headers: [...headers] });
}

/** Records newly received mail and schedules a debounced auto-sync. */
function queueNewMail(folder: TbFolder, headers: TbMessageHeader[]): void {
  addPending(folder, headers);
  scheduleAutoSync();
}

/** (Re)arms the debounce timer that triggers auto-sync after new mail settles. */
function scheduleAutoSync(): void {
  if (autoSyncTimer) clearTimeout(autoSyncTimer);
  autoSyncTimer = setTimeout(() => {
    autoSyncTimer = null;
    void runAutoSync();
  }, AUTO_SYNC_DEBOUNCE_MS);
}

const AUTO_SYNC_RETRY_BASE_MS = 15_000;
const AUTO_SYNC_RETRY_MAX_MS = 5 * 60_000;
let autoSyncRetries = 0;

/** Schedules an auto-sync retry with exponential backoff capped at the max delay. */
function scheduleAutoSyncRetry(): void {
  const delay = Math.min(AUTO_SYNC_RETRY_BASE_MS * 2 ** autoSyncRetries, AUTO_SYNC_RETRY_MAX_MS);
  autoSyncRetries += 1;
  if (autoSyncTimer) clearTimeout(autoSyncTimer);
  autoSyncTimer = setTimeout(() => {
    autoSyncTimer = null;
    void runAutoSync();
  }, delay);
  console.log(`[MailPilot] auto-sync retry in ${Math.round(delay / 1000)}s`);
}

/**
 * Indexes pending new mail when auto-index is enabled, then triggers
 * categorization. Requeues and retries on failure.
 */
async function runAutoSync(): Promise<void> {
  if (pendingNewMail.size === 0) return;
  let enabled = false;
  try {
    enabled = (await coreClient.getConfig()).autoIndex === true;
  } catch {
    scheduleAutoSyncRetry();
    return;
  }
  if (!enabled) {
    pendingNewMail.clear();
    autoSyncRetries = 0;
    return;
  }
  if (syncInProgress) {
    scheduleAutoSync();
    return;
  }

  const toProcess = new Map(pendingNewMail);
  pendingNewMail.clear();
  syncInProgress = true;
  let anyFailed = false;
  const categorizeByAccount = new Map<string, string[]>();
  try {
    const snapshot = await MailboxSnapshot.load();
    const prefs = await loadSyncPrefs();
    for (const { folder, headers } of toProcess.values()) {
      try {
        const indexed = await indexNewMessages(snapshot, prefs, folder, headers);
        if (indexed) {
          const ids = categorizeByAccount.get(indexed.accountId) ?? [];
          ids.push(...indexed.messageIds);
          categorizeByAccount.set(indexed.accountId, ids);
        }
      } catch (err) {
        console.error(`[MailPilot] auto-index failed for ${folder.path}; will retry:`, err);
        addPending(folder, headers);
        anyFailed = true;
      }
    }
  } finally {
    syncInProgress = false;
  }

  for (const [accountId, messageIds] of categorizeByAccount) {
    await triggerCategorize(accountId, messageIds);
  }

  if (anyFailed) scheduleAutoSyncRetry();
  else autoSyncRetries = 0;
}

const pendingCategorize = new Map<string, Set<string>>();
let categorizeRetryTimer: ReturnType<typeof setTimeout> | null = null;
let categorizeRetryRounds = 0;
const CATEGORIZE_RETRY_MS = 30_000;
const MAX_CATEGORIZE_RETRY_ROUNDS = 5;

/** Queues message ids for a later categorize retry and arms the retry timer. */
function queueCategorizeRetry(accountId: string, messageIds: string[]): void {
  const set = pendingCategorize.get(accountId) ?? new Set<string>();
  for (const id of messageIds) set.add(id);
  pendingCategorize.set(accountId, set);
  if (!categorizeRetryTimer) {
    categorizeRetryTimer = setTimeout(() => {
      categorizeRetryTimer = null;
      void flushPendingCategorize();
    }, CATEGORIZE_RETRY_MS);
  }
}

/** Asks Core to categorize the messages, queueing a retry if it is busy or fails. */
async function triggerCategorize(accountId: string, messageIds: string[]): Promise<void> {
  if (messageIds.length === 0) return;
  try {
    const res = await coreClient.runLlmCategorize({ accountId, messageIds });
    if (res.status === 'already_running') queueCategorizeRetry(accountId, messageIds);
  } catch (err) {
    console.warn('[MailPilot] auto-categorize trigger failed; will retry:', err);
    queueCategorizeRetry(accountId, messageIds);
  }
}

/** Retries pending categorize requests, giving up after the max retry rounds. */
async function flushPendingCategorize(): Promise<void> {
  if (pendingCategorize.size === 0) return;
  const batch = new Map(pendingCategorize);
  pendingCategorize.clear();
  for (const [accountId, ids] of batch) {
    await triggerCategorize(accountId, [...ids]);
  }
  if (pendingCategorize.size > 0) {
    categorizeRetryRounds += 1;
    if (categorizeRetryRounds >= MAX_CATEGORIZE_RETRY_ROUNDS) {
      console.warn('[MailPilot] auto-categorize gave up after retries; run Refine to catch up');
      pendingCategorize.clear();
      categorizeRetryRounds = 0;
    }
  } else {
    categorizeRetryRounds = 0;
  }
}

/**
 * Pushes, ingests, and embeds new messages for an eligible folder. Returns the
 * Core account id and indexed message ids, or null when the folder is skipped.
 */
async function indexNewMessages(
  snapshot: MailboxSnapshot,
  prefs: SyncPrefs,
  folder: TbFolder,
  headers: TbMessageHeader[],
): Promise<{ accountId: string; messageIds: string[] } | null> {
  const account = snapshot.listAccounts().find((a) => a.tbId === folder.accountId);
  if (!account) return null;
  if (!shouldSyncAccount(account.address, account.kind, prefs).sync) return null;
  if (isFolderExcluded(folder.path, prefs)) return null;

  const { items, attachmentMsgs } = await snapshot.fetchMessages(folder.path, headers);
  if (items.length === 0) return null;

  const created = await coreClient.createAccount({
    address: account.address,
    displayName: account.name,
    kind: account.kind,
  });
  const coreAccountId = created.account.id;
  for (let i = 0; i < items.length; i += PUSH_BATCH) {
    await coreClient.pushEmails({
      accountId: coreAccountId,
      emails: items.slice(i, i + PUSH_BATCH),
    });
  }
  await ingestAttachments(coreAccountId, attachmentMsgs);
  await embed(coreAccountId);
  console.log(`[MailPilot] auto-indexed ${items.length} new message(s) from ${folder.path}`);
  return { accountId: coreAccountId, messageIds: items.map((it) => it.messageId) };
}

browser.runtime.onMessage.addListener((message): Promise<unknown> | undefined => {
  const type = (message as { type?: unknown } | null)?.type;

  if (type === 'mailpilot:sync-progress') {
    return Promise.resolve({ ...syncState });
  }

  if (type === 'mailpilot:sync') {
    const force = (message as { force?: unknown } | null)?.force === true;
    return sync(force)
      .then(() => ({ ok: true }))
      .catch((err) => {
        console.error('[MailPilot] sync from message failed:', err);
        syncState.phase = 'error';
        syncState.error = err instanceof Error ? err.message : String(err);
        return { ok: false, error: syncState.error };
      });
  }
  return undefined;
});

/**
 * Bootstraps the background script. Loads the token, sets up menus, checks Core
 * health, registers mail and display listeners, and exposes debug helpers on globalThis.
 */
async function init(): Promise<void> {
  console.log('[MailPilot] starting...');
  await coreClient.loadToken();
  await setupMenus();

  try {
    const health = await coreClient.health();
    console.log('[MailPilot] Core connected:', health);
  } catch (err) {
    console.warn('[MailPilot] Core not reachable. Start the Core Server first.', err);
  }

  try {
    browser.messages.onNewMailReceived.addListener((folder, messages) =>
      queueNewMail(folder, messages.messages),
    );
    console.log('[MailPilot] watching for new mail (auto-index)');
  } catch (err) {
    console.warn('[MailPilot] could not register new-mail listener:', err);
  }

  try {
    browser.messageDisplay.onMessagesDisplayed.addListener((_tab, displayedMessages) => {
      void prefetchDisplayedAssistantSummaries(displayedMessages);
    });
    console.log('[MailPilot] watching displayed messages (assistant summary prefetch)');
  } catch (err) {
    console.warn('[MailPilot] could not register displayed-message listener:', err);
  }

  const debug = globalThis as unknown as {
    mailpilotSync: typeof sync;
    mailpilotEmbed: typeof embed;
    mailpilotSetToken: (token: string) => Promise<void>;
    mailpilotShowPrefs: () => Promise<void>;
    mailpilotEnableAccount: (address: string) => Promise<void>;
    mailpilotDisableAccount: (address: string) => Promise<void>;
  };
  debug.mailpilotSync = sync;
  debug.mailpilotEmbed = embed;
  debug.mailpilotSetToken = async (token: string) => {
    await coreClient.setToken(token);
    console.log('[MailPilot] token stored and loaded in-memory');
  };
  debug.mailpilotShowPrefs = async () => {
    console.log('[MailPilot] sync prefs:', await loadSyncPrefs());
  };
  debug.mailpilotEnableAccount = async (address: string) => {
    const prefs = await loadSyncPrefs();
    if (!prefs.enabledAddresses.includes(address)) prefs.enabledAddresses.push(address);
    prefs.excludedAddresses = prefs.excludedAddresses.filter((a) => a !== address);
    await saveSyncPrefs(prefs);
    console.log(`[MailPilot] enabled sync for ${address}`);
  };
  debug.mailpilotDisableAccount = async (address: string) => {
    const prefs = await loadSyncPrefs();
    if (!prefs.excludedAddresses.includes(address)) prefs.excludedAddresses.push(address);
    prefs.enabledAddresses = prefs.enabledAddresses.filter((a) => a !== address);
    await saveSyncPrefs(prefs);
    console.log(`[MailPilot] disabled sync for ${address}`);
  };

  console.log(
    '[MailPilot] debug helpers: mailpilotSync(), mailpilotEmbed(id), mailpilotSetToken(token), ' +
      'mailpilotShowPrefs(), mailpilotEnableAccount(addr), mailpilotDisableAccount(addr)',
  );
}

init();
