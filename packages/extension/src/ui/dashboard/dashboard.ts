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