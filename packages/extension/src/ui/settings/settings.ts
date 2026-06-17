/**
 * Settings page controller for the extension. Wires up the UI for Core connection
 * and pairing, per-account sync preferences, model selection, chat provider
 * configuration, and triggering a mailbox sync with live progress.
 */
import { coreClient } from '../../api-client/core-client.js';
import { MailboxSnapshot, type MailboxAccount } from '../../thunderbird/mailbox.js';
import {
  loadSyncPrefs,
  saveSyncPrefs,
  shouldSyncAccount,
  type SyncPrefs,
} from '../../settings/sync-prefs.js';

/** Looks up a required element by id and throws when it is missing. */
const $ = <T extends HTMLElement = HTMLElement>(id: string): T => {
  const el = document.getElementById(id);
  if (!el) throw new Error(`element #${id} not found`);
  return el as T;
};

interface State {
  prefs: SyncPrefs;
  accounts: MailboxAccount[];
  availableModels: string[];
  savedChatBaseUrl: string | null;
  savedChatModel: string | null;
}

const state: State = {
  prefs: { enabledAddresses: [], excludedAddresses: [], excludedFolderPaths: [], applyTags: true },
  accounts: [],
  availableModels: [],
  savedChatBaseUrl: null,
  savedChatModel: null,
};

const EMBEDDING_PATTERNS = /bge|nomic-embed|mxbai-embed|arctic-embed|minilm|embed/i;
/** Returns true when the model id looks like an embedding model rather than a chat model. */
const isEmbeddingModel = (id: string): boolean => EMBEDDING_PATTERNS.test(id);

/** Strips the trailing :latest tag so model ids compare equal regardless of tag. */
const canonicalModelId = (id: string): string => id.replace(/:latest$/, '');

/** Attaches handlers and loads connection, auth, accounts, prefs, and config to render the page. */
async function init(): Promise<void> {
  attachHandlers();

  try {
    await Promise.all([refreshConnection(), refreshAuth(), loadAccounts(), loadPrefs()]);
    renderAccounts();
    renderFolders();
    renderApplyTagsToggle();
    await loadConfigIfAuthed();
  } catch (err) {
    setStatus(
      'models-status',
      `Could not load settings: ${err instanceof Error ? err.message : String(err)}`,
      'error',
    );
  }
}

/** Syncs the apply-tags toggle to the current preference value. */
function renderApplyTagsToggle(): void {
  $<HTMLInputElement>('toggle-apply-tags').checked = state.prefs.applyTags;
}

/** Pings Core health, updates the connection and LLM badges, and caches available models. */
async function refreshConnection(): Promise<void> {
  setBadge('health-dot', 'health-label', 'health-badge', 'neutral', 'Checking');

  try {
    const h = await coreClient.health();
    setBadge('health-dot', 'health-label', 'health-badge', 'success', 'Connected');

    $('llm-row').hidden = false;
    $('llm-url').textContent = h.llm.baseUrl;
    $('llm-models').textContent =
      h.llm.models.length > 0 ? `${h.llm.models.length} models available` : 'no models discovered';

    if (h.llm.connected) {
      setBadge('llm-dot', 'llm-label', 'llm-badge', 'success', 'ready');
    } else {
      setBadge('llm-dot', 'llm-label', 'llm-badge', 'warning', 'unreachable');
    }

    state.availableModels = h.llm.models;
    $<HTMLSpanElement>('version').textContent = `v${h.version}`;
  } catch {
    setBadge('health-dot', 'health-label', 'health-badge', 'urgent', 'Unreachable');
    state.availableModels = [];
  }
}

/** Loads the stored token and updates the auth badge to reflect pairing state. */
async function refreshAuth(): Promise<void> {
  await coreClient.loadToken();
  if (coreClient.hasToken()) {
    setBadge('auth-dot', 'auth-label', 'auth-badge', 'success', 'Paired');
  } else {
    setBadge('auth-dot', 'auth-label', 'auth-badge', 'warning', 'No token');
  }
}

/** Loads the mailbox account list into state, falling back to empty on failure. */
async function loadAccounts(): Promise<void> {
  try {
    const snapshot = await MailboxSnapshot.load();
    state.accounts = snapshot.listAccounts();
  } catch {
    state.accounts = [];
  }
}

/** Loads persisted sync preferences into state. */
async function loadPrefs(): Promise<void> {
  state.prefs = await loadSyncPrefs();
}

/** Renders one toggle row per account, persisting enabled and excluded lists on change. */
function renderAccounts(): void {
  const list = $('accounts-list');
  list.innerHTML = '';

  if (state.accounts.length === 0) {
    list.innerHTML =
      '<div class="empty-row">No IMAP or POP3 accounts configured in Thunderbird.</div>';
    return;
  }

  for (const account of state.accounts) {
    const decision = shouldSyncAccount(account.address, account.kind, state.prefs);
    const row = document.createElement('div');
    row.className = 'account-row';
    row.innerHTML = `
      <span class="dot ${decision.sync ? 'dot-success' : 'dot-neutral'}"></span>
      <div class="account-info">
        <div class="account-name"></div>
        <div class="account-meta">
          <span class="account-address"></span>
          <span class="account-kind"></span>
        </div>
      </div>
      <label class="toggle">
        <input type="checkbox" />
        <span class="toggle-track"><span class="toggle-thumb"></span></span>
      </label>
    `;

    (row.querySelector('.account-name') as HTMLElement).textContent = account.name;
    (row.querySelector('.account-address') as HTMLElement).textContent = account.address;
    (row.querySelector('.account-kind') as HTMLElement).textContent = account.kind;

    const checkbox = row.querySelector('input[type="checkbox"]') as HTMLInputElement;
    checkbox.checked = decision.sync;
    checkbox.addEventListener('change', async () => {
      if (checkbox.checked) {
        if (!state.prefs.enabledAddresses.includes(account.address)) {
          state.prefs.enabledAddresses.push(account.address);
        }
        state.prefs.excludedAddresses = state.prefs.excludedAddresses.filter(
          (a) => a !== account.address,
        );
      } else {
        if (!state.prefs.excludedAddresses.includes(account.address)) {
          state.prefs.excludedAddresses.push(account.address);
        }
        state.prefs.enabledAddresses = state.prefs.enabledAddresses.filter(
          (a) => a !== account.address,
        );
      }
      await saveSyncPrefs(state.prefs);
      renderAccounts();
    });

    list.appendChild(row);
  }
}

/** Renders the excluded folder list with a remove button per entry. */
function renderFolders(): void {
  const list = $('folders-list');
  list.innerHTML = '';

  if (state.prefs.excludedFolderPaths.length === 0) {
    list.innerHTML = '<div class="empty-row">No folders are excluded.</div>';
    return;
  }

  for (const path of state.prefs.excludedFolderPaths) {
    const row = document.createElement('div');
    row.className = 'folder-row';
    row.innerHTML = `
      <span class="folder-name"></span>
      <button class="btn btn-ghost btn-sm" type="button">Remove</button>
    `;
    (row.querySelector('.folder-name') as HTMLElement).textContent = path;
    const removeBtn = row.querySelector('button') as HTMLButtonElement;
    removeBtn.addEventListener('click', async () => {
      state.prefs.excludedFolderPaths = state.prefs.excludedFolderPaths.filter((p) => p !== path);
      await saveSyncPrefs(state.prefs);
      renderFolders();
    });
    list.appendChild(row);
  }
}

/** Fetches Core config when paired and populates every model and chat provider field. */