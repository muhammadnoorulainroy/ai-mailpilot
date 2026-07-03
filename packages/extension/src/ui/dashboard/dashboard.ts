/**
 * Dashboard UI controller. Wires up the inbox dashboard view, including stats,
 * category management, the priority focus list, chat, and Thunderbird folder
 * and tag organization against the local Core service.
 */
import { coreClient } from '../../api-client/core-client.js';
import { renderMarkdown } from '../shared/markdown.js';
import { proposalCountLabel, proposalsSummary, proposalsBadgeLabel } from './proposals-format.js';
import type {
  AccountDto,
  AssignmentMethodDto,
  CategoryEmailDto,
  ProposalDto,
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
    void refreshProposalsBadge();
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
function appendSources(sources: ChatSourceDto[]): void {
  const container = $('chat-messages');
  const details = document.createElement('details');
  details.className = 'chat-sources';

  const seen = new Set<string>();
  const unique = sources.filter((s) => {
    const key = `${s.messageId}|${s.attachmentName ?? ''}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  const summary = document.createElement('summary');
  summary.textContent = `Sources (${unique.length})`;
  details.appendChild(summary);

  unique.forEach((s, i) => {
    const row = document.createElement('div');
    row.className = 'chat-source';

    const num = document.createElement('span');
    num.className = 'chat-source-num';
    num.textContent = `[${i + 1}]`;

    const subject = document.createElement('span');
    subject.textContent = s.subject ?? '(no subject)';

    const meta = document.createElement('span');
    meta.className = 'chat-source-from';
    meta.textContent = ` ${s.fromAddr ?? 'unknown'}${s.date ? ` \u00b7 ${formatTime(s.date)}` : ''}`;

    row.append(num, subject, meta);

    if (s.attachmentName) {
      const att = document.createElement('span');
      att.className = 'chat-source-attachment';
      att.textContent = `\u{1F4CE} ${s.attachmentName}`;
      row.appendChild(att);
    }

    details.appendChild(row);
  });

  container.appendChild(details);
  container.scrollTop = container.scrollHeight;
}

/** Renders the recent emails list, or an empty hint when nothing is indexed. */
function renderRecent(): void {
  const d = state.dashboard;
  if (!d) return;
  const container = $('recent-list');
  container.innerHTML = '';
  if (d.recent.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-row';
    empty.textContent = 'No emails indexed yet.';
    container.appendChild(empty);
    return;
  }
  for (const e of d.recent) {
    container.appendChild(simpleEmailRow(e.subject, e.fromAddr, e.date));
  }
}

/** Builds a basic email row showing subject, sender, and date, with fallbacks for missing fields. */
function simpleEmailRow(
  subject: string | null,
  fromAddr: string | null,
  date: number | null,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'email-row';
  const meta = document.createElement('div');
  meta.className = 'email-meta';
  const subj = document.createElement('div');
  subj.className = 'email-subject';
  subj.textContent = subject ?? '(no subject)';
  meta.appendChild(subj);
  const from = document.createElement('div');
  from.className = 'email-from';
  from.textContent = fromAddr ?? '(unknown sender)';
  meta.appendChild(from);
  const dateEl = document.createElement('div');
  dateEl.className = 'email-date';
  dateEl.textContent = date ? formatTime(date) : '';
  row.appendChild(meta);
  row.appendChild(dateEl);
  return row;
}

/** Runs the priority triage pass, confirming first when a cloud provider is used, and polls progress to completion. */
async function runTriage(): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  priorityCloudProvider = await resolvePriorityCloudProvider();
  if (priorityCloudProvider) {
    const ok = await confirmDialog(
      'Run Priority with cloud AI?',
      `This priority pass will send email subjects and snippets to ${priorityCloudProvider}. Embeddings stay local. Continue?`,
      'Run priority pass',
    );
    if (!ok) return;
  }
  const btn = $<HTMLButtonElement>('btn-run-triage');
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Starting...';

  let finalLabel = original;

  try {
    const result = await coreClient.runTriage({ accountId });
    if (result.status === 'up_to_date') {
      setStatus('Priority already up to date.');
      finalLabel = 'Already up to date';
    } else if (result.status === 'already_running') {
      setStatus('Another run is already in progress. Try again when it finishes.');
      finalLabel = 'Busy';
    } else {
      setStatus(
        priorityCloudProvider
          ? `Priority pass running on ${result.pending} emails with ${priorityCloudProvider}...`
          : `Priority pass running on ${result.pending} emails locally...`,
      );
      const outcome = await pollProgress('triage', accountId);
      finalLabel = reportOutcome(outcome, 'Priority pass complete.');
    }
    await Promise.all([refreshDashboard(), loadPriority()]);
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
    finalLabel = 'Failed';
  } finally {
    btn.textContent = finalLabel;
    window.setTimeout(() => {
      btn.textContent = original;
      btn.disabled = false;
    }, 2000);
  }
}

/** Enables or disables all category action buttons while a run is in progress. */
function setCategoryActionsDisabled(disabled: boolean): void {
  $<HTMLButtonElement>('btn-organize').disabled = disabled;
  $<HTMLButtonElement>('btn-rediscover').disabled = disabled;
  $<HTMLButtonElement>('btn-refine-ai').disabled = disabled;
  $<HTMLButtonElement>('btn-retry-uncategorized').disabled = disabled;
  $<HTMLButtonElement>('btn-improve').disabled = disabled;
  $<HTMLButtonElement>('btn-proposals').disabled = disabled;
}

/** Returns whether the user opted to recategorize every email rather than only new ones. */
function wantsForce(): boolean {
  return $<HTMLInputElement>('chk-recategorize-all').checked;
}

/**
 * Runs the fast categorization pass, optionally rediscovering topics first.
 * Confirms a full recategorize, polls progress, then refreshes and applies tags.
 */
async function organizeInbox(rediscover: boolean): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  const force = wantsForce();
  if (force) {
    const ok = await confirmDialog(
      'Recategorize everything',
      'This re-files every email with the fast pass, replacing the current auto-categorization. Your manual corrections are kept. Continue?',
      'Recategorize',
    );
    if (!ok) return;
  }

  const btn = $<HTMLButtonElement>(rediscover ? 'btn-rediscover' : 'btn-organize');
  const original = btn.textContent;
  setCategoryActionsDisabled(true);
  btn.textContent = rediscover ? 'Re-discovering...' : 'Organizing...';

  let finalLabel = original;

  try {
    const hasCategories = (state.dashboard?.categories.length ?? 0) > 0;
    if (rediscover || !hasCategories) {
      setStatus('Discovering topics from your inbox...');
      const discovery = await coreClient.discoverTopics({ accountId });
      if (discovery.status === 'insufficient_categories') {
        setStatus(
          hasCategories
            ? 'Could not find enough clear categories, so your existing categories were kept. Use "Improve categories" to grow them from uncategorized mail, or try a stronger model in Settings.'
            : 'Could not build clear categories from your inbox yet. Try Re-discover again, switch to a stronger model in Settings, or add categories manually.',
        );
        finalLabel = hasCategories ? 'Kept existing' : 'No categories';
        return;
      }
      setStatus(
        `Discovered ${discovery.topicsCreated} topics from ${discovery.emailsSampled} sampled emails. Assigning...`,
      );
    } else {
      setStatus(force ? 'Recategorizing all emails...' : 'Categorizing new emails...');
    }

    const result = await coreClient.runCategorize({ accountId, force });
    if (result.status === 'up_to_date') {
      setStatus('Inbox already organized.');
      finalLabel = 'Already up to date';
    } else if (result.status === 'already_running') {
      setStatus('Another run is already in progress. Try again when it finishes.');
      finalLabel = 'Busy';
    } else {
      setStatus(`Categorizing ${result.pending} emails...`);
      const outcome = await pollProgress('categorize', accountId);
      finalLabel = reportOutcome(outcome, 'Inbox organized.');
    }
    await refreshDashboard();
    await applyTagsIfEnabled();
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
    finalLabel = 'Failed';
  } finally {
    btn.textContent = finalLabel;
    window.setTimeout(() => {
      btn.textContent = original;
      setCategoryActionsDisabled(false);
    }, 2000);
  }
}

/**
 * Runs the slower LLM categorization pass after a confirmation describing scope, cloud use, and local cost.
 * With opts.retry it only re-reads previously uncategorized mail.
 */
async function refineWithAi(opts: { retry?: boolean } = {}): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  if ((state.dashboard?.categories.length ?? 0) === 0) {
    setStatus('No categories yet. Click "Organize inbox" or "Re-discover" first.');
    return;
  }

  const retry = opts.retry ?? false;
  const force = retry ? false : wantsForce();
  const total = state.dashboard?.emails.total ?? 0;
  const scope = force
    ? `It re-files all ${total.toLocaleString()} emails (your manual corrections are kept).`
    : `It only upgrades mail the fast pass categorized plus anything uncategorized - existing AI results and your corrections are left alone.`;
  const cloudNote = categorizeCloudProvider
    ? ` Cloud categorization is on: email subjects and snippets are sent to ${categorizeCloudProvider}.`
    : '';
  const LARGE_LOCAL_RUN = 5000;
  const localSlowNote =
    !categorizeCloudProvider && total > LARGE_LOCAL_RUN
      ? ` Heads up: on an inbox this size the local model can take a long time, often hours. Turning on cloud categorization in Settings is much faster. You can stop the run at any time and the progress is saved.`
      : '';
  const ok = await confirmDialog(
    retry ? 'Retry uncategorized' : 'Refine with AI',
    (retry
      ? 'The AI re-reads the emails it previously left uncategorized and files any that now fit a category. Useful after you add or rename categories. Runs in the background. Start now?'
      : `The AI reads your email content and files each one into the categories that actually fit. It groups near-identical emails so it makes far fewer model calls. ${scope} Runs in the background; you can keep working and refresh later. Start now?`) +
      cloudNote +
      localSlowNote,
    'Start',
  );
  if (!ok) return;

  const btn = $<HTMLButtonElement>('btn-refine-ai');
  const stopBtn = $<HTMLButtonElement>('btn-stop-ai');
  const original = btn.textContent;
  setCategoryActionsDisabled(true);
  btn.textContent = 'Refining...';

  let finalLabel = original;
  try {
    const result = await coreClient.runLlmCategorize({
      accountId,
      force,
      retryUncategorized: retry,
    });
    if (result.status === 'up_to_date') {
      setStatus('Nothing to categorize.');
      finalLabel = 'Up to date';
    } else if (result.status === 'already_running') {
      setStatus('A categorization run is already in progress.');
      finalLabel = 'Busy';
    } else {
      setStatus(`AI is reading ${result.pending.toLocaleString()} emails...`);
      stopBtn.hidden = false;
      const outcome = await pollProgress('llmCategorize', accountId);
      finalLabel = reportOutcome(outcome, 'AI categorization complete.');
    }
    await refreshDashboard();
    await applyTagsIfEnabled();
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
    finalLabel = 'Failed';
  } finally {
    stopBtn.hidden = true;
    btn.textContent = finalLabel;
    window.setTimeout(() => {
      btn.textContent = original;
      setCategoryActionsDisabled(false);
    }, 2000);
  }
}

/** Requests that the running LLM categorization stop after the current batch. */
async function stopAiRun(): Promise<void> {
  const stopBtn = $<HTMLButtonElement>('btn-stop-ai');
  stopBtn.disabled = true;
  try {
    await coreClient.stopLlmCategorize();
    setStatus('Stopping after the current batch...');
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  } finally {
    window.setTimeout(() => {
      stopBtn.disabled = false;
    }, 1500);
  }
}

/** True when a run status string represents a finished or stopped state. */
const isTerminal = (s: string): boolean =>
  s === 'completed' ||
  s === 'completed_with_failures' ||
  s === 'error' ||
  s === 'idle' ||
  s === 'stopped' ||
  s === 'interrupted';

const MAX_POLLS = 1200;

interface RunOutcome {
  status: string;
  failed: number;
  error?: string;
}

/** Sets a status message for a finished run and returns the short button label for its outcome. */
function reportOutcome(outcome: RunOutcome | null, successMsg: string): string {
  if (!outcome) return 'In background';
  if (outcome.status === 'error') {
    setStatus(
      outcome.error ??
        'The run ended with an error; some emails may be unprocessed. Check the Core log.',
    );
    return 'Failed';
  }
  if (outcome.status === 'stopped') {
    setStatus('Stopped. Partial results were saved.');
    return 'Stopped';
  }
  if (outcome.status === 'interrupted') {
    setStatus(
      'The previous run was interrupted before finishing. Click Refine to continue where it left off.',
    );
    return 'Interrupted';
  }
  if (outcome.status === 'completed_with_failures') {
    setStatus(
      `${successMsg} ${outcome.failed} email(s) could not be processed and will be retried.`,
    );
    return 'Done, some failed';
  }
  setStatus(successMsg);
  return 'Done';
}

/** On load, surfaces a status hint if a previous Refine run was interrupted so the user can resume it. */
async function surfaceInterruptedRun(): Promise<void> {
  if (!state.currentAccountId) return;
  try {
    const p = await coreClient.llmCategorizeProgress(state.currentAccountId);
    if (p.status !== 'interrupted') return;
    if (p.accountId !== null && p.accountId !== state.currentAccountId) return;
    setStatus(
      `A previous Refine run was interrupted at ${p.processed.toLocaleString()}/${p.total.toLocaleString()} emails. ` +
        'Click Refine to continue where it left off.',
    );
  } catch {}
}

/**
 * Polls a background run of the given kind until it reaches a terminal state,
 * updating the status line each tick. Returns the outcome, or null if it is
 * still running after the poll cap.
 */
async function pollProgress(
  kind: 'triage' | 'categorize' | 'llmCategorize',
  accountId: string,
): Promise<RunOutcome | null> {
  const maxPolls = kind === 'llmCategorize' ? MAX_POLLS * 20 : MAX_POLLS;
  for (let i = 0; i < maxPolls; i++) {
    await new Promise((r) => setTimeout(r, POLL_MS));
    if (kind === 'triage') {
      const p = await coreClient.triageProgress();
      if (p.accountId !== null && p.accountId !== accountId) continue;
      setStatus(`Triage ${p.processed}/${p.total} (${p.failed} failed)`);
      if (isTerminal(p.status)) return { status: p.status, failed: p.failed };
    } else if (kind === 'categorize') {
      const p = await coreClient.categorizeProgress();
      if (p.accountId !== null && p.accountId !== accountId) continue;
      setStatus(`Categorize ${p.processed}/${p.total} (${p.assigned} assigned)`);
      if (isTerminal(p.status))
        return { status: p.status, failed: (p as { failed?: number }).failed ?? 0 };
    } else {
      const p = await coreClient.llmCategorizeProgress(accountId);
      if (p.accountId !== null && p.accountId !== accountId) continue;
      if (p.phase === 'preparing' || p.phase === 'clustering') {
        setStatus('Preparing groups from your inbox... this can take a moment on a large inbox.');
      } else {
        const groups =
          p.clusters > 0
            ? ` - ${p.clustersProcessed.toLocaleString()}/${p.clusters.toLocaleString()} groups, ${p.llmCalls.toLocaleString()} AI calls`
            : '';
        setStatus(
          `AI categorizing${groups}: ${p.processed.toLocaleString()}/${p.total.toLocaleString()} emails ` +
            `(${p.assigned.toLocaleString()} labels, ${p.failed} failed)`,
        );
      }
      if (isTerminal(p.status)) return { status: p.status, failed: p.failed, error: p.error };
    }
  }
  setStatus('Still running in the background; refresh to see the latest.');
  return null;
}

/** When tag sync is enabled, applies category tags to the matching Thunderbird account and reports the result. */
async function applyTagsIfEnabled(): Promise<void> {
  const prefs = await loadSyncPrefs();
  if (!prefs.applyTags) return;
  if (!state.currentAccountId) return;

  const coreAccount = state.accounts.find((a) => a.id === state.currentAccountId);
  if (!coreAccount) return;

  const snapshot = await MailboxSnapshot.load();
  const tbAccount = snapshot
    .listAccounts()
    .find((a) => a.address.toLowerCase() === coreAccount.address.toLowerCase());
  if (!tbAccount) {
    setStatus('Tag apply skipped: matching Thunderbird account not found.');
    return;
  }

  setStatus('Applying tags in Thunderbird...');
  try {
    const result = await applyTagsForAccount(state.currentAccountId, tbAccount.tbId);
    const created = result.tagsCreated > 0 ? `, ${result.tagsCreated} new tag(s)` : '';
    const missed = result.missingMessages > 0 ? `, ${result.missingMessages} not found` : '';
    const cleared = result.staleTagsCleared > 0 ? `, ${result.staleTagsCleared} stale cleared` : '';
    setStatus(
      `Tagged ${result.taggedMessages} message(s) in Thunderbird${created}${cleared}${missed}.`,
    );
  } catch (err) {
    setStatus(`Tag apply failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/** Finds the Thunderbird account id whose address matches the current Core account, or null. */
async function resolveTbAccountId(): Promise<string | null> {
  const coreAccount = state.accounts.find((a) => a.id === state.currentAccountId);
  if (!coreAccount) return null;
  const snapshot = await MailboxSnapshot.load();
  const tb = snapshot
    .listAccounts()
    .find((a) => a.address.toLowerCase() === coreAccount.address.toLowerCase());
  return tb?.tbId ?? null;
}

/** Sets the status text in the folders panel. */
function setFoldersStatus(msg: string): void {
  $('folders-status').textContent = msg;
}

/** Updates the folder operation progress bar and label from a progress event. */
function showFolderProgress(p: OrganizeProgress): void {
  $('folder-progress').hidden = false;
  const pct = p.total > 0 ? Math.round((p.moved / p.total) * 100) : p.phase === 'done' ? 100 : 0;
  $('folder-progress-fill').style.width = `${pct}%`;
  $('folder-progress-label').textContent =
    p.phase === 'folders'
      ? 'Creating folders...'
      : p.phase === 'done'
        ? 'Done'
        : `Moving ${p.moved}/${p.total}...`;
}

/** Enables or disables the organize and restore folder buttons during an operation. */
function setFolderButtonsDisabled(disabled: boolean): void {
  $<HTMLButtonElement>('btn-organize-folders').disabled = disabled;
  $<HTMLButtonElement>('btn-restore-folders').disabled = disabled;
}

/** Returns the trimmed parent folder name from the input, defaulting to "AI MailPilot". */
function folderParentName(): string {
  return $<HTMLInputElement>('folder-parent-input').value.trim() || 'AI MailPilot';
}

/** Moves emails into Thunderbird folders by primary category after confirmation, reporting progress and results. */
async function organizeFolders(): Promise<void> {
  const account = state.currentAccountId;
  const plan = state.folderPlan;
  if (!account || !plan) return;
  if (plan.assignments.length === 0) {
    setFoldersStatus('Nothing to organize yet. Categorize your inbox first.');
    return;
  }
  const tbAccountId = await resolveTbAccountId();
  if (!tbAccountId) {
    setFoldersStatus('Matching Thunderbird account not found.');
    return;
  }
  const parentName = folderParentName();

  const ok = await confirmDialog(
    'Organize into folders',
    `Move ${plan.assignments.length} email(s) into ${plan.categories.length} folder(s) under "${parentName}"? Each email moves to its primary category. You can move everything back to the Inbox anytime.`,
    'Organize',
  );
  if (!ok) return;

  setFolderButtonsDisabled(true);
  setFoldersStatus('');
  try {
    const result = await organizeIntoFolders(account, tbAccountId, parentName, showFolderProgress);
    const missed = result.missing > 0 ? `, ${result.missing} not found in Thunderbird` : '';
    setFoldersStatus(
      `Moved ${result.moved} email(s) into ${result.foldersUsed} folder(s)${missed}.`,
    );
  } catch (err) {
    setFoldersStatus(err instanceof Error ? err.message : String(err));
  } finally {
    setFolderButtonsDisabled(false);
    window.setTimeout(() => ($('folder-progress').hidden = true), 1500);
  }
}

/** Moves all emails out of the AI MailPilot folders back into the Inbox after confirmation, then refreshes. */
async function restoreFolders(): Promise<void> {
  const tbAccountId = await resolveTbAccountId();
  if (!tbAccountId) {
    setFoldersStatus('Matching Thunderbird account not found.');
    return;
  }
  const parentName = folderParentName();

  const ok = await confirmDialog(
    'Move everything back to Inbox',
    `Move all emails from the "${parentName}" folders back into the Inbox?`,
    'Move back',
  );
  if (!ok) return;

  setFolderButtonsDisabled(true);
  setFoldersStatus('');
  try {
    const result = await restoreFromFolders(tbAccountId, parentName, showFolderProgress);
    setFoldersStatus(`Moved ${result.movedBack} email(s) back to the Inbox.`);
    await refreshDashboard();
  } catch (err) {
    setFoldersStatus(err instanceof Error ? err.message : String(err));
  } finally {
    setFolderButtonsDisabled(false);
    window.setTimeout(() => ($('folder-progress').hidden = true), 1500);
  }
}

let improveSuggestions: ImproveSuggestionsResponse | null = null;

/** Asks Core for category improvement suggestions and opens the review modal, or reports when there are none. */
async function improveCategories(): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  setCategoryActionsDisabled(true);
  setStatus('Looking for categories to add or merge...');
  try {
    const s = await coreClient.improveSuggest({ accountId });
    improveSuggestions = s;
    if (s.warning) {
      setStatus(s.warning);
      return;
    }
    if (
      s.existingCategoryExpansions.length === 0 &&
      s.newCategories.length === 0 &&
      s.merges.length === 0
    ) {
      setStatus(
        s.diagnostics?.recommendation ??
          (s.uncategorizedCount > 0
            ? `No clear improvements found for ${s.uncategorizedCount.toLocaleString()} uncategorized emails.`
            : 'Your inbox is well categorized; nothing to improve.'),
      );
      return;
    }
    renderImproveModal(s);
    $('improve-modal').hidden = false;
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  } finally {
    setCategoryActionsDisabled(false);
  }
}

/** Renders the improve modal sections for new categories, existing-category expansions, and merges. */
function renderImproveModal(s: ImproveSuggestionsResponse): void {
  $('improve-summary').textContent =
    `${s.uncategorizedCount.toLocaleString()} emails fit no current category. ` +
    'Pick the changes to apply; nothing happens until you click Apply.';
  const body = $('improve-body');
  body.innerHTML = '';

  if (s.newCategories.length > 0) {
    const rows = s.newCategories.map((c, i) => {
      const count =
        c.estimatedCount >= 100
          ? ' (~100+ similar)'
          : c.estimatedCount > 0
            ? ` (~${c.estimatedCount} similar)`
            : '';
      const example = c.sampleSubjects.length > 0 ? ` Example: "${c.sampleSubjects[0]}"` : '';
      return improveRow(`new-${i}`, `${c.label}${count}`, `${c.description}${example}`);
    });
    body.appendChild(improveSection('New categories', rows));
  }
  if (s.existingCategoryExpansions.length > 0) {
    const rows = s.existingCategoryExpansions.map((e, i) => {
      const count =
        e.estimatedCount >= 100
          ? ' (~100+ similar)'
          : e.estimatedCount > 0
            ? ` (~${e.estimatedCount} similar)`
            : '';
      const example = e.sampleSubjects.length > 0 ? ` Example: "${e.sampleSubjects[0]}"` : '';
      return improveRow(`expand-${i}`, `${e.categoryLabel}${count}`, `${e.reason}${example}`);
    });
    body.prepend(improveSection('File into existing categories', rows));
  }
  if (s.merges.length > 0) {
    const rows = s.merges.map((m, i) =>
      improveRow(`merge-${i}`, `${m.sourceLabel} -> ${m.targetLabel}`, m.reason),
    );
    body.appendChild(improveSection('Merge overlapping categories', rows));
  }
}

/** Wraps a titled group of improvement rows into a section element. */
function improveSection(title: string, rows: HTMLElement[]): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'improve-section';
  const head = document.createElement('div');
  head.className = 'section-title';
  head.textContent = title;
  wrap.appendChild(head);
  for (const r of rows) wrap.appendChild(r);
  return wrap;
}

/** Builds a checked checkbox row for one improvement suggestion, keyed so apply can find selected ones. */
function improveRow(key: string, title: string, detail: string): HTMLElement {
  const label = document.createElement('label');
  label.className = 'improve-row';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = true;
  cb.dataset.key = key;
  const text = document.createElement('div');
  const t = document.createElement('div');
  t.className = 'improve-row-title';
  t.textContent = title;
  const d = document.createElement('div');
  d.className = 'improve-row-detail';
  d.textContent = detail;
  text.append(t, d);
  label.append(cb, text);
  return label;
}

/** Closes the improve modal and clears the cached suggestions. */
function closeImproveModal(): void {
  $('improve-modal').hidden = true;
  improveSuggestions = null;
}

/** Applies the checked improvement suggestions (new categories, expansions, merges) and refreshes the dashboard. */
async function applyImprovements(): Promise<void> {
  const accountId = state.currentAccountId;
  const s = improveSuggestions;
  if (!accountId || !s) {
    closeImproveModal();
    return;
  }
  const checked = new Set(
    Array.from($('improve-body').querySelectorAll<HTMLInputElement>('input[type=checkbox]'))
      .filter((c) => c.checked)
      .map((c) => c.dataset.key ?? ''),
  );
  const newCategories = s.newCategories
    .filter((_, i) => checked.has(`new-${i}`))
    .map((c) => ({ label: c.label, description: c.description, messageIds: c.messageIds }));
  const existingCategoryExpansions = s.existingCategoryExpansions
    .filter((_, i) => checked.has(`expand-${i}`))
    .map((e) => ({ categoryId: e.categoryId, messageIds: e.messageIds }));
  const merges = s.merges
    .filter((_, i) => checked.has(`merge-${i}`))
    .map((m) => ({ sourceId: m.sourceId, targetId: m.targetId }));

  closeImproveModal();
  if (existingCategoryExpansions.length === 0 && newCategories.length === 0 && merges.length === 0)
    return;

  setCategoryActionsDisabled(true);
  setStatus('Applying category changes...');
  try {
    const r = await coreClient.improveApply({
      accountId,
      existingCategoryExpansions,
      newCategories,
      merges,
    });
    setStatus(
      `Filed ${r.expanded.toLocaleString()} email(s), added ${r.created} categor${
        r.created === 1 ? 'y' : 'ies'
      }, merged ${r.merged}. Run Refine to continue improving coverage.`,
    );
    await refreshDashboard();
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
  } finally {
    setCategoryActionsDisabled(false);
  }
}

/** Opens the suggested-categories review modal and loads the pending queue. */
async function openProposalsModal(): Promise<void> {
  $('proposals-modal').hidden = false;
  await loadProposals();
}

/** Closes the suggested-categories modal. */
function closeProposalsModal(): void {
  $('proposals-modal').hidden = true;
}

/** Fetches the pending proposals for the current account, renders them, and refreshes the button count. */
async function loadProposals(): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  $('proposals-summary').textContent = 'Loading suggestions...';
  try {
    const res = await coreClient.listProposals(accountId);
    if (accountId !== state.currentAccountId) return;
    renderProposals(res.proposals);
    setProposalsBadge(res.proposals.length);
  } catch (err) {
    $('proposals-summary').textContent = err instanceof Error ? err.message : String(err);
  }
}

/** Renders the review queue: one card per proposal with Add and Ignore actions. */
function renderProposals(list: ProposalDto[]): void {
  $('proposals-summary').textContent = proposalsSummary(list.length);
  const body = $('proposals-body');
  body.innerHTML = '';
  for (const p of list) body.appendChild(proposalCard(p));
}

/** Builds one proposal card: label, estimated size, evidence keywords, and Add/Ignore buttons. */
function proposalCard(p: ProposalDto): HTMLElement {
  const card = document.createElement('div');
  card.className = 'proposal-card';

  const top = document.createElement('div');
  top.className = 'proposal-top';
  const label = document.createElement('div');
  label.className = 'proposal-label';
  label.textContent = p.label;
  const count = document.createElement('span');
  count.className = 'proposal-count';
  count.textContent = proposalCountLabel(p.proposedCount);
  top.append(label, count);
  card.appendChild(top);

  if (p.description) {
    const desc = document.createElement('div');
    desc.className = 'proposal-description';
    desc.textContent = p.description;
    card.appendChild(desc);
  }

  if (p.evidence.length > 0) {
    const chips = document.createElement('div');
    chips.className = 'proposal-evidence';
    for (const term of p.evidence.slice(0, 6)) {
      const chip = document.createElement('span');
      chip.className = 'proposal-chip';
      chip.textContent = term;
      chips.appendChild(chip);
    }
    card.appendChild(chips);
  }

  const actions = document.createElement('div');
  actions.className = 'proposal-actions';
  const ignore = document.createElement('button');
  ignore.className = 'btn btn-ghost btn-sm';
  ignore.type = 'button';
  ignore.textContent = 'Ignore';
  const add = document.createElement('button');
  add.className = 'btn btn-primary btn-sm';
  add.type = 'button';
  add.textContent = 'Add';
  // Disable both buttons on this card while its request is in flight, so a fast double-click cannot
  // send a duplicate apply/dismiss. On success the queue refresh replaces the card; on failure the
  // action re-enables them.
  const setBusy = (busy: boolean): void => {
    add.disabled = busy;
    ignore.disabled = busy;
  };
  ignore.addEventListener('click', () => void dismissProposal(p, setBusy));
  add.addEventListener('click', () => void applyProposal(p, setBusy));
  actions.append(ignore, add);
  card.appendChild(actions);
  return card;
}

/** Runs discovery to find new proposals, then reloads the queue. */
async function generateProposals(): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  const btn = $<HTMLButtonElement>('proposals-generate');
  btn.disabled = true;
  $('proposals-summary').textContent = 'Looking for new categories in your emails...';
  try {
    const res = await coreClient.generateProposals({ accountId });
    if (accountId !== state.currentAccountId) return;
    setStatus(
      res.created.length > 0
        ? `Found ${res.created.length} new suggestion${res.created.length === 1 ? '' : 's'}.`
        : 'No new categories found.',
    );
    await loadProposals();
  } catch (err) {
    $('proposals-summary').textContent = err instanceof Error ? err.message : String(err);
  } finally {
    btn.disabled = false;
  }
}

/** Approves a proposal, files its emails, and refreshes the queue and dashboard. */
async function applyProposal(p: ProposalDto, setBusy: (busy: boolean) => void): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  setBusy(true);
  try {
    const res = await coreClient.applyProposal(p.id, accountId);
    setStatus(
      res.assigned > 0
        ? `Added "${res.label}" and filed ${res.assigned} email${res.assigned === 1 ? '' : 's'}.`
        : `Added "${res.label}".`,
    );
    await loadProposals();
    await refreshDashboard();
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
    setBusy(false);
  }
}

/** Dismisses a proposal so it leaves the queue and is not suggested again. */
async function dismissProposal(p: ProposalDto, setBusy: (busy: boolean) => void): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  setBusy(true);
  try {
    await coreClient.dismissProposal(p.id, accountId);
    setStatus(`Ignored "${p.label}".`);
    await loadProposals();
  } catch (err) {
    setStatus(err instanceof Error ? err.message : String(err));
    setBusy(false);
  }
}

/** Sets the toolbar button label to reflect the pending proposal count. */
function setProposalsBadge(count: number): void {
  $<HTMLButtonElement>('btn-proposals').textContent = proposalsBadgeLabel(count);
}

/** Refreshes the pending-proposal count on the toolbar button, non-fatally. */
async function refreshProposalsBadge(): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId) return;
  try {
    const res = await coreClient.listProposals(accountId);
    if (accountId !== state.currentAccountId) return;
    setProposalsBadge(res.proposals.length);
  } catch {
    // Non-fatal: leave the button label unchanged if the count cannot be fetched.
  }
}

/** Shows the modal confirmation dialog and resolves true on confirm, false on cancel, backdrop click, or Escape. */
function confirmDialog(title: string, message: string, okLabel = 'Confirm'): Promise<boolean> {
  return new Promise((resolve) => {
    const backdrop = $('confirm-modal');
    const okBtn = $<HTMLButtonElement>('confirm-ok');
    const cancelBtn = $<HTMLButtonElement>('confirm-cancel');

    $('confirm-title').textContent = title;
    $('confirm-message').textContent = message;
    okBtn.textContent = okLabel;
    backdrop.hidden = false;

    /** Hides the dialog, detaches its listeners, and resolves the promise with the result. */
    const finish = (result: boolean): void => {
      backdrop.hidden = true;
      okBtn.removeEventListener('click', onOk);
      cancelBtn.removeEventListener('click', onCancel);
      backdrop.removeEventListener('click', onBackdrop);
      document.removeEventListener('keydown', onKey);
      resolve(result);
    };
    /** Confirms the dialog. */
    const onOk = (): void => finish(true);
    /** Cancels the dialog. */
    const onCancel = (): void => finish(false);
    /** Cancels when the click lands on the backdrop rather than the dialog body. */
    const onBackdrop = (e: Event): void => {
      if (e.target === backdrop) finish(false);
    };
    /** Cancels the dialog on the Escape key. */
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') finish(false);
    };

    okBtn.addEventListener('click', onOk);
    cancelBtn.addEventListener('click', onCancel);
    backdrop.addEventListener('click', onBackdrop);
    document.addEventListener('keydown', onKey);
  });
}

interface ModalCategory {
  id: string;
  label: string;
  description: string | null;
  emailCount: number;
}

let modalCategory: ModalCategory | null = null;

/** Opens the category edit modal for a category, populating its fields, merge targets, and email list. */
function openCategoryModal(cat: DashboardCategorySummaryDto): void {
  modalCategory = {
    id: cat.id,
    label: cat.label,
    description: cat.description,
    emailCount: cat.emailCount,
  };

  $<HTMLInputElement>('cat-label-input').value = cat.label;
  $<HTMLTextAreaElement>('cat-description-input').value = cat.description ?? '';
  $('cat-count').textContent = cat.emailCount === 1 ? '1 email' : `${cat.emailCount} emails`;
  $('cat-source').textContent = '';
  setSaveStatus('');

  const sel = $<HTMLSelectElement>('cat-merge-select');
  sel.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = 'Merge into...';
  sel.appendChild(placeholder);
  for (const other of state.dashboard?.categories ?? []) {
    if (other.id === cat.id) continue;
    const opt = document.createElement('option');
    opt.value = other.id;
    opt.textContent = other.label;
    sel.appendChild(opt);
  }
  $<HTMLButtonElement>('cat-merge').disabled = true;

  $('category-modal').hidden = false;
  void loadCategoryEmails(cat.id);
}

/** Closes the category edit modal and clears its active category. */
function closeCategoryModal(): void {
  $('category-modal').hidden = true;
  modalCategory = null;
}

/** Loads and renders the emails in a category into the modal, ignoring results if the modal moved on. */
async function loadCategoryEmails(categoryId: string): Promise<void> {
  const list = $('cat-email-list');
  list.innerHTML = '<div class="empty-row">Loading...</div>';
  try {
    const result = await coreClient.listEmailsInCategory(categoryId, 200);
    if (modalCategory?.id !== categoryId) return;
    renderCategoryEmails(list, result.emails);
  } catch (err) {
    if (modalCategory?.id !== categoryId) return;
    list.innerHTML = '';
    const row = document.createElement('div');
    row.className = 'empty-row';
    row.textContent = err instanceof Error ? err.message : String(err);
    list.appendChild(row);
  }
}

/** Renders the category modal's email rows, each with an inline category membership editor. */
function renderCategoryEmails(container: HTMLElement, emails: CategoryEmailDto[]): void {
  container.innerHTML = '';

  if (emails.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-row';
    empty.textContent = 'No emails in this category.';
    container.appendChild(empty);
    return;
  }

  for (const email of emails) {
    const row = document.createElement('div');
    row.className = 'email-row';

    const meta = document.createElement('div');
    meta.className = 'email-meta';

    const subject = document.createElement('div');
    subject.className = 'email-subject';
    subject.textContent = email.subject ?? '(no subject)';
    meta.appendChild(subject);

    const from = document.createElement('div');
    from.className = 'email-from';
    from.textContent = email.fromAddr ?? '(unknown sender)';
    meta.appendChild(from);

    meta.appendChild(buildCategoryEditor(email));

    const date = document.createElement('div');
    date.className = 'email-date';
    date.textContent = email.date ? formatTime(email.date) : '';

    row.appendChild(meta);
    row.appendChild(date);
    container.appendChild(row);
  }
}

type ChipProvenance = {
  assignedBy: 'user' | 'auto';
  method?: AssignmentMethodDto | null;
  confidence: number;
};

/** Builds a small provenance badge showing how a category was assigned (you, fast, or AI), or null. */
function chipProvenance(cat: ChipProvenance): HTMLElement | null {
  let text: string;
  let kind: string;
  if (cat.assignedBy === 'user') {
    text = 'You';
    kind = 'cp-user';
  } else if (cat.method === 'embed') {
    text = `Fast ${Math.round(cat.confidence * 100)}%`;
    kind = 'cp-fast';
  } else if (cat.method === 'gate' || cat.method === 'proposal') {
    text = 'Fast';
    kind = 'cp-fast';
  } else if (cat.method === 'llm') {
    text = 'AI';
    kind = 'cp-ai';
  } else {
    return null;
  }
  const el = document.createElement('span');
  el.className = `cat-chip-prov ${kind}`;
  el.textContent = text;
  return el;
}

/** Builds the per-email category editor with removable chips and an add-category dropdown. */
function buildCategoryEditor(email: CategoryEmailDto): HTMLElement {
  const wrap = document.createElement('div');
  wrap.className = 'cat-editor';

  for (const cat of email.categories) {
    const chip = document.createElement('span');
    chip.className = 'cat-chip';

    const label = document.createElement('span');
    label.textContent = cat.label;
    chip.appendChild(label);

    const prov = chipProvenance(cat);
    if (prov) chip.appendChild(prov);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'cat-chip-x';
    remove.textContent = '×';
    remove.title = `Remove from ${cat.label}`;
    remove.addEventListener('click', () => {
      const next = email.categories.filter((c) => c.id !== cat.id).map((c) => c.id);
      void applyMembership(email.messageId, next);
    });
    chip.appendChild(remove);
    wrap.appendChild(chip);
  }

  const present = new Set(email.categories.map((c) => c.id));
  const add = document.createElement('select');
  add.className = 'select select-compact cat-add';
  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = '+ Add';
  add.appendChild(placeholder);
  for (const cat of state.dashboard?.categories ?? []) {
    if (present.has(cat.id)) continue;
    const opt = document.createElement('option');
    opt.value = cat.id;
    opt.textContent = cat.label;
    add.appendChild(opt);
  }
  add.addEventListener('change', () => {
    if (!add.value) return;
    void applyMembership(email.messageId, [...email.categories.map((c) => c.id), add.value]);
  });
  wrap.appendChild(add);

  return wrap;
}

/** Saves the new category membership for an email, then reloads the modal list and dashboard. */
async function applyMembership(messageId: string, categoryIds: string[]): Promise<void> {
  const accountId = state.currentAccountId;
  if (!accountId || !modalCategory) return;
  const fromCategoryId = modalCategory.id;
  setSaveStatus('Saving...');
  try {
    await coreClient.setEmailCategories(messageId, accountId, categoryIds);
    setSaveStatus('Saved. The categories will learn from this.', 'success');
    if (modalCategory?.id === fromCategoryId) await loadCategoryEmails(fromCategoryId);
    await refreshDashboard();
  } catch (err) {
    setSaveStatus(err instanceof Error ? err.message : String(err), 'error');
  }
}

/** Saves edited category label and description when changed, validating that the label is not empty. */
async function saveCategoryEdits(): Promise<void> {
  if (!modalCategory) return;
  const label = $<HTMLInputElement>('cat-label-input').value.trim();
  const description = $<HTMLTextAreaElement>('cat-description-input').value.trim();

  if (!label) {
    setSaveStatus('Label cannot be empty.', 'error');
    return;
  }

  const labelChanged = label !== modalCategory.label;
  const descChanged = description !== (modalCategory.description ?? '');
  if (!labelChanged && !descChanged) {
    setSaveStatus('No changes.', 'muted');
    return;
  }

  setSaveStatus('Saving...');
  try {
    await coreClient.updateCategory(modalCategory.id, {
      ...(labelChanged ? { label } : {}),
      ...(descChanged ? { description: description || null } : {}),
    });
    modalCategory.label = label;
    modalCategory.description = description || null;
    setSaveStatus('Saved.', 'success');
    await refreshDashboard();
  } catch (err) {
    setSaveStatus(err instanceof Error ? err.message : String(err), 'error');
  }
}

/** Merges the modal's category into a selected target after confirmation, reassigning emails and closing the modal. */
async function mergeCurrentCategory(): Promise<void> {
  if (!modalCategory) return;
  const targetId = $<HTMLSelectElement>('cat-merge-select').value;
  if (!targetId) return;

  const targetLabel = state.dashboard?.categories.find((c) => c.id === targetId)?.label ?? 'target';

  const confirmed = window.confirm(
    `Merge "${modalCategory.label}" into "${targetLabel}"? This will reassign every email and delete the source category.`,
  );
  if (!confirmed) return;

  setSaveStatus('Merging...');
  try {
    const result = await coreClient.mergeCategory(modalCategory.id, { targetId });
    setSaveStatus(`Merged ${result.reassigned} email(s).`, 'success');
    await refreshDashboard();
    closeCategoryModal();
  } catch (err) {
    setSaveStatus(err instanceof Error ? err.message : String(err), 'error');
  }
}

/** Deletes the modal's category after confirmation, leaving its emails unassigned, then closes the modal. */
async function deleteCurrentCategory(): Promise<void> {
  if (!modalCategory) return;
  const confirmed = window.confirm(
    `Delete category "${modalCategory.label}"? Its emails will become unassigned (you can re-organize to redistribute them).`,
  );
  if (!confirmed) return;

  setSaveStatus('Deleting...');
  try {
    await coreClient.deleteCategory(modalCategory.id);
    await refreshDashboard();
    closeCategoryModal();
  } catch (err) {
    setSaveStatus(err instanceof Error ? err.message : String(err), 'error');
  }
}

/** Sets the category modal save-status message and colors it by tone. */
function setSaveStatus(msg: string, tone: 'muted' | 'success' | 'error' = 'muted'): void {
  const el = $('cat-save-status');
  el.textContent = msg;
  el.style.color =
    tone === 'error'
      ? 'var(--urgent-text)'
      : tone === 'success'
        ? 'var(--success-text)'
        : 'var(--text-tertiary)';
}

/** Attaches all top-level event handlers for toolbar buttons, account selection, range tabs, and modals. */
function attachHandlers(): void {
  setupTabs();
  setupChat();

  $<HTMLButtonElement>('btn-refresh').addEventListener('click', () => {
    void refreshDashboard();
  });

  $<HTMLButtonElement>('btn-settings').addEventListener('click', () => {
    void browser.runtime.openOptionsPage();
  });

  $<HTMLButtonElement>('btn-empty-settings').addEventListener('click', () => {
    void browser.runtime.openOptionsPage();
  });

  $<HTMLSelectElement>('account-select').addEventListener('change', (e) => {
    const sel = e.target as HTMLSelectElement;
    state.currentAccountId = sel.value;
    state.priority = null;
    void loadChatForAccount(sel.value);
    void refreshDashboard();
    if (!$('panel-priority').hidden) void loadPriority();
  });

  for (const seg of Array.from(
    document.querySelectorAll<HTMLButtonElement>('#priority-range .seg'),
  )) {
    seg.addEventListener('click', () => {
      const range = seg.dataset.range as PriorityRange | undefined;
      if (!range || range === state.priorityRange) return;
      state.priorityRange = range;
      for (const s of Array.from(
        document.querySelectorAll<HTMLButtonElement>('#priority-range .seg'),
      )) {
        s.classList.toggle('seg-active', s === seg);
      }
      void loadPriority();
    });
  }

  $<HTMLButtonElement>('btn-run-triage').addEventListener('click', () => {
    void runTriage();
  });
  $<HTMLButtonElement>('btn-organize').addEventListener('click', () => {
    void organizeInbox(false);
  });
  $<HTMLButtonElement>('btn-refine-ai').addEventListener('click', () => {
    void refineWithAi();
  });
  $<HTMLButtonElement>('btn-retry-uncategorized').addEventListener('click', () => {
    void refineWithAi({ retry: true });
  });
  $<HTMLButtonElement>('btn-stop-ai').addEventListener('click', () => {
    void stopAiRun();
  });
  $<HTMLButtonElement>('btn-rediscover').addEventListener('click', () => {
    void organizeInbox(true);
  });
  $<HTMLButtonElement>('btn-improve').addEventListener('click', () => {
    void improveCategories();
  });
  $<HTMLButtonElement>('improve-cancel').addEventListener('click', closeImproveModal);
  $<HTMLButtonElement>('improve-apply').addEventListener('click', () => {
    void applyImprovements();
  });
  $('improve-modal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeImproveModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('improve-modal').hidden) closeImproveModal();
  });

  $<HTMLButtonElement>('btn-proposals').addEventListener('click', () => {
    void openProposalsModal();
  });
  $<HTMLButtonElement>('proposals-generate').addEventListener('click', () => {
    void generateProposals();
  });
  $<HTMLButtonElement>('proposals-close').addEventListener('click', closeProposalsModal);
  $('proposals-modal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeProposalsModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('proposals-modal').hidden) closeProposalsModal();
  });

  $<HTMLButtonElement>('btn-organize-folders').addEventListener('click', () => {
    void organizeFolders();
  });
  $<HTMLButtonElement>('btn-restore-folders').addEventListener('click', () => {
    void restoreFolders();
  });
  $<HTMLInputElement>('folder-parent-input').addEventListener('input', renderFolderTree);

  $<HTMLButtonElement>('cat-close').addEventListener('click', closeCategoryModal);
  $('category-modal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeCategoryModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('category-modal').hidden) closeCategoryModal();
  });

  $<HTMLButtonElement>('cat-save').addEventListener('click', () => {
    void saveCategoryEdits();
  });
  $<HTMLSelectElement>('cat-merge-select').addEventListener('change', (e) => {
    const target = (e.target as HTMLSelectElement).value;
    $<HTMLButtonElement>('cat-merge').disabled = !target;
  });
  $<HTMLButtonElement>('cat-merge').addEventListener('click', () => {
    void mergeCurrentCategory();
  });
  $<HTMLButtonElement>('cat-delete').addEventListener('click', () => {
    void deleteCurrentCategory();
  });
}

let statusUndoTimer: number | null = null;

/** Sets the status line text and cancels any pending undo timer. */
function setStatus(msg: string): void {
  if (statusUndoTimer !== null) {
    window.clearTimeout(statusUndoTimer);
    statusUndoTimer = null;
  }
  $('status-line').textContent = msg;
}

/** Shows a status message with an Undo button that resets the triage resolution, auto-clearing after ten seconds. */
function showResolutionUndo(
  accountId: string,
  messageId: string,
  resolution: TriageResolution,
): void {
  if (statusUndoTimer !== null) window.clearTimeout(statusUndoTimer);
  const el = $('status-line');
  el.textContent = `${resolutionPastTense(resolution)}. `;

  const undo = document.createElement('button');
  undo.type = 'button';
  undo.className = 'status-undo';
  undo.textContent = 'Undo';
  undo.addEventListener('click', async () => {
    undo.disabled = true;
    try {
      await coreClient.resolveTriage({ accountId, messageId, resolution: 'reset' });
      if (accountId === state.currentAccountId) await loadPriority();
      setStatus('Restored to the focus view.');
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err));
    }
  });
  el.appendChild(undo);

  statusUndoTimer = window.setTimeout(() => {
    statusUndoTimer = null;
    if (el.contains(undo)) el.textContent = '';
  }, 10_000);
}

/** Formats an epoch-millis timestamp as a short local date and time, or "--" when missing. */
function formatTime(ms: number): string {
  if (!ms) return '--';
  const d = new Date(ms);
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

init().catch((err) => {
  console.error('[MailPilot dashboard] init failed:', err);
});
