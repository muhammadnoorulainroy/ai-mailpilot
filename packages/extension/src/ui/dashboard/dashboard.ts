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