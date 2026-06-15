/**
 * Dashboard UI controller. Wires up the inbox dashboard view, including stats,
 * category management, the priority focus list, chat, and Thunderbird folder
 * and tag organization against the local Core service.
 */
import { coreClient } from '../../api-client/core-client.js';
import { renderMarkdown } from '../shared/markdown.js';
import type {
  AccountDto,
  CategoryEmailDto,
  ChatSourceDto,
  ConversationDto,
  ConversationSummaryDto,
  DashboardCategorySummaryDto,
  DashboardResponse,
  FolderPlanResponse,
  ImproveSuggestionsResponse,
  PriorityRange,
  PriorityResponse,
  PriorityEmailDto,
  TriageResolution,
} from '@ai-mailpilot/shared';
import { MailboxSnapshot } from '../../thunderbird/mailbox.js';
import { applyTagsForAccount } from '../../thunderbird/tags.js';
import {
  organizeIntoFolders,
  restoreFromFolders,
  type OrganizeProgress,
} from '../../thunderbird/folders.js';
import { loadSyncPrefs } from '../../settings/sync-prefs.js';

/** Looks up an element by id and throws if it is missing, narrowing to the given element type. */
const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`element #${id} not found`);
  return el as T;
};

interface State {
  accounts: AccountDto[];
  currentAccountId: string | null;
  dashboard: DashboardResponse | null;
  folderPlan: FolderPlanResponse | null;
  priorityRange: PriorityRange;
  priority: PriorityResponse | null;
}

const state: State = {
  accounts: [],
  currentAccountId: null,
  dashboard: null,
  folderPlan: null,
  priorityRange: 'today',
  priority: null,
};

const POLL_MS = 1500;

/** Entry point. Loads the auth token and accounts, renders the dashboard, and starts chat for the first account. */
async function init(): Promise<void> {
  setStatus('Loading...');
  attachHandlers();

  try {
    await coreClient.loadToken();
    if (!coreClient.hasToken()) {
      showEmpty('Pair the extension first', 'Open settings and paste your auth token to begin.');
      return;
    }

    await refreshChatProvider();
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') void refreshChatProvider();
    });

    await loadAccounts();
    if (state.accounts.length === 0) {
      showEmpty(
        'No accounts indexed yet',
        'Open settings, enable an account, then click "Sync inbox now".',
      );
      return;
    }

    state.currentAccountId = state.accounts[0]?.id ?? null;
    populateAccountSelect();
    await refreshDashboard();
    await surfaceInterruptedRun();
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }

  void loadChatForAccount(state.currentAccountId);
}

/** Fetches the list of indexed accounts from Core into state. */
async function loadAccounts(): Promise<void> {
  const res = await coreClient.listAccounts();
  state.accounts = res.accounts;
}

/** Fills the account selector and hides it when only one account exists. */
function populateAccountSelect(): void {
  const sel = $<HTMLSelectElement>('account-select');
  sel.innerHTML = '';
  for (const acc of state.accounts) {
    const opt = document.createElement('option');
    opt.value = acc.id;
    opt.textContent = acc.displayName ? `${acc.displayName} (${acc.address})` : acc.address;
    if (acc.id === state.currentAccountId) opt.selected = true;
    sel.appendChild(opt);
  }
  sel.hidden = state.accounts.length < 2;
}

/** Shows the empty-state panel with the given title and description, hiding the main content. */
function showEmpty(title: string, description: string): void {
  $('empty-title').textContent = title;
  $('empty-description').textContent = description;
  $('empty-state').hidden = false;
  $('content').hidden = true;
  setStatus('');
}

/** Hides the empty-state panel and reveals the main content. */
function showContent(): void {
  $('empty-state').hidden = true;
  $('content').hidden = false;
}

/** Fetches and renders the dashboard for the current account, ignoring results if the account changed mid-request. */
async function refreshDashboard(): Promise<void> {
  const account = state.currentAccountId;
  if (!account) return;
  setStatus('Refreshing...');

  try {
    const data = await coreClient.dashboard(account);
    if (account !== state.currentAccountId) return;
    state.dashboard = data;
    showContent();
    renderStats();
    renderCategories();
    renderRecent();
    setStatus(`Updated ${formatTime(data.generatedAt)}`);
    void loadFolderPlan(account);
  } catch (err) {
    if (account !== state.currentAccountId) return;
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

/** Loads the folder organization plan for an account and renders its preview tree. */
async function loadFolderPlan(account: string): Promise<void> {
  try {
    const plan = await coreClient.folderPlan(account);
    if (account !== state.currentAccountId) return;
    state.folderPlan = plan;
    renderFolderTree();
  } catch {}
}

/** Renders the planned folder hierarchy under the chosen parent folder name, with per-category counts. */
function renderFolderTree(): void {
  const container = $('folder-tree');
  const plan = state.folderPlan;
  if (!plan || plan.categories.length === 0) {
    container.innerHTML =
      '<div class="empty-row">Organize your inbox into categories first, then preview here.</div>';
    return;
  }

  const parentName = $<HTMLInputElement>('folder-parent-input').value.trim() || 'AI MailPilot';
  container.innerHTML = '';

  const root = document.createElement('div');
  root.className = 'folder-tree-root';
  root.textContent = parentName;
  container.appendChild(root);

  const children = document.createElement('div');
  children.className = 'folder-tree-children';
  for (const cat of plan.categories) {
    const row = document.createElement('div');
    row.className = 'folder-tree-row';

    const name = document.createElement('span');
    name.className = 'folder-tree-name';
    name.textContent = cat.label;

    const count = document.createElement('span');
    count.className = 'folder-tree-count';
    count.textContent = String(cat.count);

    row.append(name, count);
    children.appendChild(row);
  }
  container.appendChild(children);
}

/** Renders the summary stat tiles (totals, uncategorized, category count, last update). */
function renderStats(): void {
  const d = state.dashboard;
  if (!d) return;
  $('stat-total').textContent = d.emails.total.toLocaleString();
  $('stat-unclassified').textContent = d.emails.uncategorized.toLocaleString();
  $('stat-categories').textContent = d.categoryCount.toLocaleString();
  $('stat-time').textContent = formatTime(d.generatedAt);
}

/** Returns the epoch millis for midnight at the start of today in the local timezone. */
function localDayStartMs(): number {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  return d.getTime();
}

const RANGE_SUBTITLE: Record<PriorityRange, string> = {
  today: 'What needs your attention today.',
  week: 'Action and updates from the last 7 days.',
  all: 'Everything triaged so far.',
};

const BUCKET_LABEL: Record<PriorityEmailDto['bucket'], string> = {
  urgent: 'Urgent',
  summarize: 'Summary',
  personal: 'Personal',
  spam: 'Spam',
};

/** Loads the priority focus list for the current account and selected range, then renders it. */
async function loadPriority(): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  try {
    const data = await coreClient.priority({
      accountId,
      range: state.priorityRange,
      dayStartMs: localDayStartMs(),
    });
    if (accountId !== state.currentAccountId) return;
    state.priority = data;
    renderPriority();
  } catch (err) {
    if (accountId !== state.currentAccountId) return;
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

/** Renders the priority view metrics, the unclassified hint, and each priority bucket list. */
function renderPriority(): void {
  const p = state.priority;
  if (!p) return;

  $('focus-sub').textContent = RANGE_SUBTITLE[p.range];
  $('metric-needs-action').textContent = p.counts.needsAction.toLocaleString();
  $('metric-urgent').textContent = p.counts.urgent.toLocaleString();
  $('metric-important').textContent = p.counts.important.toLocaleString();
  $('metric-low').textContent = p.counts.lowPriority.toLocaleString();

  const unclassified = $('priority-unclassified');
  if (p.counts.unclassified > 0) {
    unclassified.hidden = false;
    const scope = p.range === 'all' ? '' : ' in this range';
    const n = p.counts.unclassified.toLocaleString();
    unclassified.textContent = `${n} email${p.counts.unclassified === 1 ? '' : 's'}${scope} not classified yet. Run the priority pass to triage them.`;
  } else {
    unclassified.hidden = true;
  }

  const needsEmpty =
    p.range === 'today' ? 'No urgent email today.' : 'Nothing needs action in this range.';
  renderPriorityList('needs-action-list', p.needsAction, needsEmpty);
  renderPriorityList('important-list', p.important, 'No important updates.');
  renderPriorityList('summaries-list', p.summaries, 'No summaries or digests.');
  renderPriorityList('low-list', p.lowPriority, 'Nothing low priority.');

  const carrySection = $('section-carryover');
  if (p.carryover.length > 0) {
    carrySection.hidden = false;
    renderPriorityList('carryover-list', p.carryover, 'No carryover.');
  } else {
    carrySection.hidden = true;
  }

  $('low-count').textContent = p.counts.lowPriority > 0 ? `(${p.counts.lowPriority})` : '';
}

/** Renders a list of priority emails into a container, showing emptyText when there are none. */
function renderPriorityList(
  containerId: string,
  emails: PriorityEmailDto[],
  emptyText: string,
): void {
  const container = $(containerId);
  container.innerHTML = '';
  if (emails.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-row';
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }
  for (const email of emails) container.appendChild(priorityRow(email));
}

/** Builds a single priority email row with its subject, bucket tag, summary, and resolve actions. */
function priorityRow(email: PriorityEmailDto): HTMLElement {
  const row = document.createElement('div');
  row.className = 'email-row priority-row';

  const meta = document.createElement('div');
  meta.className = 'email-meta';

  const head = document.createElement('div');
  head.className = 'priority-head';
  const subject = document.createElement('span');
  subject.className = 'email-subject';
  subject.textContent = email.subject ?? '(no subject)';
  head.appendChild(subject);
  const tag = document.createElement('span');
  tag.className = `priority-tag tag-${email.bucket}`;
  tag.textContent = BUCKET_LABEL[email.bucket];
  head.appendChild(tag);
  if (email.deadlineAt) {
    const dl = document.createElement('span');
    dl.className = 'priority-deadline';
    dl.textContent = `due ${formatTime(email.deadlineAt)}`;
    head.appendChild(dl);
  }
  meta.appendChild(head);

  const from = document.createElement('div');
  from.className = 'email-from';
  from.textContent = email.fromAddr ?? '(unknown sender)';
  meta.appendChild(from);

  const summary = email.shortSummary ?? email.reasoning;
  if (summary) {
    const s = document.createElement('div');
    s.className = 'email-reasoning';
    s.textContent = summary;
    meta.appendChild(s);
  }
  if (email.suggestedAction) {
    const a = document.createElement('div');
    a.className = 'priority-action';
    a.textContent = email.suggestedAction;
    meta.appendChild(a);
  }

  const side = document.createElement('div');
  side.className = 'priority-side';
  const date = document.createElement('div');
  date.className = 'email-date';
  date.textContent = email.date ? formatTime(email.date) : '';
  side.appendChild(date);

  const actions = document.createElement('div');
  actions.className = 'priority-resolve';
  actions.appendChild(resolveButton('Done', 'done', email.messageId));
  actions.appendChild(resolveButton('Snooze', 'snooze', email.messageId));
  actions.appendChild(resolveButton('Dismiss', 'dismiss', email.messageId));
  side.appendChild(actions);

  row.appendChild(meta);
  row.appendChild(side);
  return row;
}

/** Builds a button that resolves a triaged email to the given resolution when clicked. */
function resolveButton(
  label: string,
  resolution: TriageResolution,
  messageId: string,
): HTMLButtonElement {
  const btn = document.createElement('button');
  btn.className = 'btn btn-ghost btn-xs';
  btn.type = 'button';
  btn.textContent = label;
  btn.addEventListener('click', () => void resolvePriority(messageId, resolution));
  return btn;
}

/** Applies a triage resolution to an email, reloads the priority view, and offers an undo. Snooze hides it until tomorrow. */
async function resolvePriority(messageId: string, resolution: TriageResolution): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  try {
    const snoozedUntil =
      resolution === 'snooze' ? localDayStartMs() + 24 * 60 * 60 * 1000 : undefined;
    await coreClient.resolveTriage({ accountId, messageId, resolution, snoozedUntil });
    await loadPriority();
    showResolutionUndo(accountId, messageId, resolution);
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  }
}

/** Maps a triage resolution to its past-tense status label for the undo prompt. */
function resolutionPastTense(resolution: TriageResolution): string {
  switch (resolution) {
    case 'done':
      return 'Marked done';
    case 'snooze':
      return 'Snoozed until tomorrow';
    case 'dismiss':
      return 'Dismissed';
    case 'reset':
      return 'Restored';
  }
}

/** Renders the category cards with assignment counts and a relative bar, each opening an edit modal. */
function renderCategories(): void {
  const d = state.dashboard;
  if (!d) return;

  const summary = $('categories-summary');
  const container = $('categories-list');
  container.innerHTML = '';

  if (d.categories.length === 0) {
    summary.hidden = true;
    const empty = document.createElement('div');
    empty.className = 'empty-row';
    empty.textContent =
      'No categories yet. Click "Organize inbox" to discover topics and assign emails.';
    container.appendChild(empty);
    return;
  }

  const assigned = d.emails.total - d.emails.uncategorized;
  const totalLabels = d.categories.reduce((sum, c) => sum + c.emailCount, 0);
  summary.hidden = false;
  summary.textContent =
    totalLabels > assigned
      ? `${assigned} of ${d.emails.total} emails assigned to ${d.categoryCount} categories (${totalLabels} labels total; some emails belong to more than one).`
      : `${assigned} of ${d.emails.total} emails assigned to ${d.categoryCount} categories.`;

  const maxCount = Math.max(...d.categories.map((c) => c.emailCount), 1);

  for (const cat of d.categories) {
    const card = document.createElement('div');
    card.className = 'category-card';
    card.setAttribute('role', 'button');
    card.setAttribute('tabindex', '0');
    card.title = 'Open to rename, merge, delete, or re-file emails';
    card.addEventListener('click', () => openCategoryModal(cat));
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openCategoryModal(cat);
      }
    });

    const top = document.createElement('div');
    top.className = 'category-card-top';

    const label = document.createElement('div');
    label.className = 'category-label';
    label.textContent = cat.label;

    const count = document.createElement('span');
    count.className = 'category-count';
    count.textContent = String(cat.emailCount);
    top.append(label, count);
    card.appendChild(top);

    const desc = document.createElement('div');
    desc.className = 'category-description';
    desc.textContent = cat.description ?? '';
    card.appendChild(desc);

    const bar = document.createElement('div');
    bar.className = 'category-bar';
    const fill = document.createElement('div');
    fill.className = 'category-bar-fill';
    fill.style.width = `${Math.max(3, (cat.emailCount / maxCount) * 100)}%`;
    bar.appendChild(fill);
    card.appendChild(bar);

    container.appendChild(card);
  }
}

/** Wires tab buttons to toggle the active tab and panel, lazily loading the priority view when selected. */
function setupTabs(): void {
  const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>('.tab'));
  const panels = Array.from(document.querySelectorAll<HTMLElement>('.tab-panel'));
  for (const tab of tabs) {
    tab.addEventListener('click', () => {
      const name = tab.dataset.tab;
      if (!name) return;
      for (const t of tabs) {
        const active = t === tab;
        t.classList.toggle('tab-active', active);
        t.setAttribute('aria-selected', active ? 'true' : 'false');
      }
      for (const panel of panels) {
        const active = panel.id === `panel-${name}`;
        panel.hidden = !active;
        panel.classList.toggle('tab-panel-active', active);
      }
      if (name === 'priority') void loadPriority();
    });
  }
}

let currentConversationId: string | null = null;
let chatBusy = false;
let chatAbort: AbortController | null = null;

let chatCloudProvider: string | null = null;
let categorizeCloudProvider: string | null = null;
let priorityCloudProvider: string | null = null;

/** Derives a human-readable cloud provider label from a base URL, or null when no URL is set. */
function providerLabelFor(url: string | null | undefined): string | null {
  if (!url) return null;
  if (url.includes('openai.com')) return 'OpenAI';
  try {
    return new URL(url).hostname;
  } catch {
    return 'a cloud provider';
  }
}

/** Builds the chat placeholder text, noting whether questions stay local or go to a cloud provider. */
function chatEmptyText(): string {
  const base =
    'Ask anything about your inbox. It searches your emails and answers from what it ' +
    'finds, citing the messages it used. ';
  return chatCloudProvider
    ? base +
        `Cloud chat is on: your question and the retrieved emails are sent to ${chatCloudProvider}.`
    : base + 'Everything stays on your machine.';
}

/** Reads the current LLM config to cache which chat, categorize, and priority calls use a cloud provider, then updates the privacy badge. */
async function refreshChatProvider(): Promise<void> {
  if (!coreClient.hasToken()) return;
  try {
    const cfg = await coreClient.getConfig();
    chatCloudProvider = providerLabelFor(cfg.llm.chatBaseUrl);
    categorizeCloudProvider = cfg.llm.categorizeUseChatProvider
      ? providerLabelFor(cfg.llm.chatBaseUrl)
      : null;
    priorityCloudProvider = cfg.llm.priorityUseChatProvider
      ? providerLabelFor(cfg.llm.chatBaseUrl)
      : null;
  } catch {
    chatCloudProvider = null;
    categorizeCloudProvider = null;
    priorityCloudProvider = null;
  }
  renderChatPrivacy();
}

/** Returns the cloud provider label if the priority pass is actually configured to use one with a key set, otherwise null. */
async function resolvePriorityCloudProvider(): Promise<string | null> {
  if (!coreClient.hasToken()) return null;
  try {
    const cfg = await coreClient.getConfig();
    const active =
      cfg.llm.priorityUseChatProvider === true &&
      !!cfg.llm.chatBaseUrl &&
      cfg.llm.chatApiKeySet === true;
    return active ? providerLabelFor(cfg.llm.chatBaseUrl) : null;
  } catch {
    return 'a cloud provider';
  }
}

/** Updates the chat privacy badge to reflect whether chat runs locally or against a cloud provider. */
function renderChatPrivacy(): void {
  const badge = document.getElementById('chat-privacy');
  if (!badge) return;
  if (chatCloudProvider) {
    badge.textContent = `Cloud: ${chatCloudProvider}`;
    badge.className = 'chat-privacy cloud';
    badge.title = `Your question and the retrieved emails are sent to ${chatCloudProvider}. Embeddings stay local.`;
  } else {
    badge.textContent = 'Local';
    badge.className = 'chat-privacy local';
    badge.title = 'Chat runs on your machine; nothing is sent to a cloud provider.';
  }
}

/** Builds the localStorage key that remembers the last open conversation for an account. */
function convoStorageKey(accountId: string): string {
  return `mailpilot_convo_${accountId}`;
}

/** Wires the chat form, input, stop, and new-chat controls, including submit on Enter and textarea auto-grow. */
function setupChat(): void {
  const form = $<HTMLFormElement>('chat-form');
  const input = $<HTMLTextAreaElement>('chat-input');
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    void sendChat();
  });
  $<HTMLButtonElement>('chat-stop').addEventListener('click', () => chatAbort?.abort());
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      void sendChat();
    }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
  });
  $<HTMLButtonElement>('chat-new').addEventListener('click', () => newChat());
}

/** Clears the chat transcript and shows the empty placeholder message. */
function showChatEmpty(): void {
  const container = $('chat-messages');
  container.innerHTML = '';
  const empty = document.createElement('div');
  empty.className = 'chat-empty';
  empty.textContent = chatEmptyText();
  container.appendChild(empty);
}

/** Starts a fresh conversation, clearing the remembered id, transcript, and input. */
function newChat(): void {
  currentConversationId = null;
  if (state.currentAccountId) localStorage.removeItem(convoStorageKey(state.currentAccountId));
  showChatEmpty();
  $<HTMLTextAreaElement>('chat-input').value = '';
  void refreshConversationList();
}

/** Renders a saved conversation's turns, rendering markdown and source lists for assistant replies. */
function renderConversationTurns(turns: ConversationDto['turns']): void {
  const container = $('chat-messages');
  container.innerHTML = '';
  if (turns.length === 0) {
    showChatEmpty();
    return;
  }
  for (const turn of turns) {
    const el = appendChatMessage(turn.role, turn.content);
    if (turn.role === 'assistant') renderAnswerMarkdown(el);
    if (turn.role === 'assistant' && turn.sources?.length) appendSources(turn.sources);
  }
}

/** Renders the saved conversation sidebar, each item selectable and deletable, highlighting the current one. */
function renderConversationList(items: ConversationSummaryDto[]): void {
  const list = $('chat-conversations');
  list.innerHTML = '';
  if (items.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'chat-convo-empty';
    empty.textContent = 'No conversations yet.';
    list.appendChild(empty);
    return;
  }
  for (const c of items) {
    const item = document.createElement('div');
    item.className = `chat-convo-item${c.id === currentConversationId ? ' active' : ''}`;
    item.addEventListener('click', () => void selectConversation(c.id));

    const body = document.createElement('div');
    body.className = 'chat-convo-body';
    const title = document.createElement('div');
    title.className = 'chat-convo-title';
    title.textContent = c.preview || 'New conversation';
    title.title = c.preview;
    const time = document.createElement('div');
    time.className = 'chat-convo-time';
    time.textContent = formatTime(c.updatedAt);
    body.append(title, time);

    const del = document.createElement('button');
    del.className = 'chat-convo-del';
    del.type = 'button';
    del.textContent = '×';
    del.title = 'Delete conversation';
    del.addEventListener('click', (e) => {
      e.stopPropagation();
      void deleteConversation(c.id, item);
    });

    item.append(body, del);
    list.appendChild(item);
  }
}

/** Fetches and renders the conversation list for the current account, falling back to empty on error. */
async function refreshConversationList(): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) {
    renderConversationList([]);
    return;
  }
  try {
    const res = await coreClient.listConversations(accountId);
    renderConversationList(res.conversations);
  } catch {
    renderConversationList([]);
  }
}

/** Opens a saved conversation, loading its turns and remembering it for the account. No-op while chat is busy. */
async function selectConversation(id: string): Promise<void> {
  if (chatBusy || id === currentConversationId) return;
  const accountId = state.currentAccountId;
  if (!accountId) return;
  try {
    const convo = await coreClient.getConversation(id, accountId);
    currentConversationId = convo.id;
    if (state.currentAccountId) {
      localStorage.setItem(convoStorageKey(state.currentAccountId), convo.id);
    }
    renderConversationTurns(convo.turns);
    void refreshConversationList();
  } catch {
    void refreshConversationList();
  }
}

/** Deletes a conversation after confirmation, removing its row and clearing the view if it was open. */
async function deleteConversation(id: string, row?: HTMLElement): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  const ok = await confirmDialog(
    'Delete conversation',
    'This permanently removes this chat and its history. This cannot be undone.',
    'Delete',
  );
  if (!ok) return;

  try {
    await coreClient.deleteConversation(id, accountId);
  } catch (err) {
    setStatus(`Could not delete conversation: ${err instanceof Error ? err.message : String(err)}`);
    return;
  }

  row?.remove();
  if (id === currentConversationId) {
    currentConversationId = null;
    if (state.currentAccountId) localStorage.removeItem(convoStorageKey(state.currentAccountId));
    showChatEmpty();
  }
  void refreshConversationList();
}

/** Restores the last open conversation for an account if one was saved, then refreshes the conversation list. */
async function loadChatForAccount(accountId: string | null): Promise<void> {
  currentConversationId = null;
  showChatEmpty();
  if (accountId) {
    const id = localStorage.getItem(convoStorageKey(accountId));
    if (id) {
      try {
        const convo = await coreClient.getConversation(id, accountId);
        if (convo.turns.length > 0) {
          renderConversationTurns(convo.turns);
          currentConversationId = convo.id;
        }
      } catch {
        localStorage.removeItem(convoStorageKey(accountId));
      }
    }
  }
  await refreshConversationList();
}

/**
 * Creates a pending assistant message and returns controls to stream into it.
 * The returned handles append thinking text, switch to the final answer, and
 * promote streamed thinking into the answer body.
 */
function createAssistantTurn(): {
  answer: HTMLElement;
  think: (text: string) => void;
  answerStart: () => void;
  promote: () => void;
} {
  const empty = document.querySelector('.chat-empty');
  if (empty) empty.remove();
  const container = $('chat-messages');

  const answer = document.createElement('div');
  answer.className = 'chat-msg chat-msg-assistant chat-msg-pending';
  answer.textContent = 'Thinking…';
  container.appendChild(answer);
  scrollChat();

  let details: HTMLDetailsElement | null = null;
  let body: HTMLElement | null = null;
  let startedAt = 0;

  /** Lazily creates the collapsible thinking panel on first use and returns its body element. */
  function ensure(): HTMLElement {
    if (body) return body;
    startedAt = Date.now();
    details = document.createElement('details');
    details.className = 'chat-thinking';
    details.open = true;
    const summary = document.createElement('summary');
    summary.textContent = 'Thinking…';
    body = document.createElement('div');
    body.className = 'chat-thinking-body';
    details.append(summary, body);
    container.insertBefore(details, answer);
    answer.textContent = '';
    answer.classList.remove('chat-msg-pending');
    return body;
  }

  return {
    answer,
    think(text: string) {
      const b = ensure();
      b.textContent = `${b.textContent ?? ''}${text}`;
      b.scrollTop = b.scrollHeight;
      scrollChat();
    },
    answerStart() {
      if (details) {
        details.open = false;
        const secs = Math.max(1, Math.round((Date.now() - startedAt) / 1000));
        const summary = details.querySelector('summary');
        if (summary) summary.textContent = `Thought for ${secs}s`;
      }
    },
    promote() {
      if (body) {
        answer.textContent = body.textContent ?? '';
        details?.remove();
        details = null;
        body = null;
      }
    },
  };
}

/** Sends the typed question, streams the assistant reply with thinking and sources, and handles abort and error states. */
async function sendChat(): Promise<void> {
  if (chatBusy) return;
  const input = $<HTMLTextAreaElement>('chat-input');
  const question = input.value.trim();
  if (!question) return;

  const accountId = state.currentAccountId;
  if (!accountId) {
    appendChatMessage('assistant', 'Select an account first.');
    return;
  }

  chatBusy = true;
  chatAbort = new AbortController();
  $<HTMLButtonElement>('chat-send').disabled = true;
  $<HTMLButtonElement>('chat-send').hidden = true;
  $<HTMLButtonElement>('chat-stop').hidden = false;
  appendChatMessage('user', question);
  input.value = '';
  input.style.height = 'auto';

  const turn = createAssistantTurn();
  let answerStarted = false;
  let errored = false;
  let pendingSources: ChatSourceDto[] = [];
  /** True while the account this turn was started for is still the active account. */
  const stillCurrent = (): boolean => state.currentAccountId === accountId;

  /** Switches the turn from thinking to answer mode the first time answer content arrives. */
  const beginAnswer = (): void => {
    if (!answerStarted) {
      turn.answerStart();
      turn.answer.textContent = '';
      turn.answer.classList.remove('chat-msg-pending');
      answerStarted = true;
    }
  };
  /** Marks the turn as failed, appending an interruption note or replacing the answer with an error. */
  const fail = (message: string): void => {
    errored = true;
    if (answerStarted && (turn.answer.textContent ?? '') !== '') {
      turn.answer.textContent = `${turn.answer.textContent}\n\n[Interrupted: ${message}]`;
    } else {
      beginAnswer();
      turn.answer.textContent = `Sorry, that failed: ${message}`;
    }
  };

  try {
    await coreClient.chatStream(
      { accountId, question, conversationId: currentConversationId ?? undefined },
      {
        onMeta: (conversationId, sources) => {
          if (!stillCurrent()) return;
          currentConversationId = conversationId;
          localStorage.setItem(convoStorageKey(accountId), conversationId);
          pendingSources = sources;
        },
        onThink: (text) => {
          if (stillCurrent()) turn.think(text);
        },
        onDelta: (text) => {
          if (!stillCurrent()) return;
          beginAnswer();
          turn.answer.textContent = `${turn.answer.textContent ?? ''}${text}`;
          scrollChat();
        },
        onPromote: () => {
          if (!stillCurrent()) return;
          turn.answerStart();
          turn.promote();
          answerStarted = true;
        },
        onError: (message) => {
          if (stillCurrent()) fail(message);
        },
      },
      chatAbort.signal,
    );
    if (!stillCurrent()) return;
    if (!errored && (turn.answer.textContent ?? '').trim() === '') {
      turn.answer.classList.remove('chat-msg-pending');
      turn.answer.textContent = '(no answer)';
    } else if (!answerStarted && !errored) {
      turn.answer.classList.remove('chat-msg-pending');
    }
    if (answerStarted && !errored) renderAnswerMarkdown(turn.answer);
    if (answerStarted && !errored && pendingSources.length > 0) appendSources(pendingSources);
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      if (stillCurrent() && answerStarted) {
        turn.answer.classList.remove('chat-msg-pending');
        renderAnswerMarkdown(turn.answer);
      } else if (stillCurrent()) {
        turn.answer.classList.remove('chat-msg-pending');
        turn.answer.textContent = '(stopped)';
      }
    } else if (stillCurrent()) {
      fail(err instanceof Error ? err.message : String(err));
    }
  } finally {
    chatBusy = false;
    chatAbort = null;
    $<HTMLButtonElement>('chat-stop').hidden = true;
    $<HTMLButtonElement>('chat-send').hidden = false;
    $<HTMLButtonElement>('chat-send').disabled = false;
    input.focus();
    if (stillCurrent()) void refreshConversationList();
  }
}

/** Scrolls the chat transcript to the bottom. */
function scrollChat(): void {
  const c = $('chat-messages');
  c.scrollTop = c.scrollHeight;
}

/** Replaces an element's plain text with rendered markdown when it has non-empty content. */
function renderAnswerMarkdown(el: HTMLElement): void {
  const raw = el.textContent ?? '';
  if (raw.trim()) el.innerHTML = renderMarkdown(raw);
}

/** Appends a chat message bubble for the given role and returns it. The pending flag styles it as in-progress. */
function appendChatMessage(role: 'user' | 'assistant', text: string, pending = false): HTMLElement {
  const empty = document.querySelector('.chat-empty');
  if (empty) empty.remove();

  const container = $('chat-messages');
  const msg = document.createElement('div');
  msg.className = `chat-msg chat-msg-${role}${pending ? ' chat-msg-pending' : ''}`;
  msg.textContent = text;
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
  return msg;
}

/** Appends a collapsible sources panel, deduplicating by message and attachment, listing each cited email. */