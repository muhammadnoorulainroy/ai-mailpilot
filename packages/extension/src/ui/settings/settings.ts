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
async function loadConfigIfAuthed(): Promise<void> {
  if (!coreClient.hasToken()) {
    populateModelSelects(null, null, null);
    return;
  }
  try {
    const cfg = await coreClient.getConfig();
    $('endpoint-url').textContent = cfg.llm.baseUrl;
    const chatBaseUrl = cfg.llm.chatBaseUrl ?? '';
    state.savedChatBaseUrl = cfg.llm.chatBaseUrl ?? null;
    state.savedChatModel = cfg.llm.chatModel ?? null;
    const provider =
      chatBaseUrl === '' ? 'local' : chatBaseUrl.includes('openai.com') ? 'openai' : 'custom';
    populateModelSelects(
      cfg.llm.embeddingModel,
      cfg.llm.generationModel,
      localChatModelForSelect(cfg.llm.chatModel, cfg.llm.generationModel),
    );
    $<HTMLInputElement>('chat-topk').value = cfg.llm.chatTopK?.toString() ?? '';
    $<HTMLInputElement>('chat-snippet').value = cfg.llm.chatSnippetChars?.toString() ?? '';
    $<HTMLInputElement>('chat-rerank').checked = cfg.llm.chatRerank ?? true;

    $<HTMLSelectElement>('chat-provider').value = provider;
    $<HTMLInputElement>('chat-base-url').value = chatBaseUrl;
    const modelField = $<HTMLInputElement>('chat-model-text');
    modelField.value = provider === 'local' ? '' : (cfg.llm.chatModel ?? '');
    if (
      provider === 'openai' &&
      (!modelField.value.trim() || modelField.value.trim() === cfg.llm.generationModel)
    ) {
      modelField.value = 'gpt-4o-mini';
    }
    $<HTMLInputElement>('chat-categorize').checked = cfg.llm.categorizeUseChatProvider ?? false;
    $<HTMLInputElement>('chat-priority').checked = cfg.llm.priorityUseChatProvider ?? false;
    $('chat-key-status').textContent = cfg.llm.chatApiKeySet
      ? 'A key is configured. Leave blank to keep it.'
      : 'No key stored yet.';
    applyChatProvider(provider);

    $<HTMLInputElement>('toggle-auto-index').checked = cfg.autoIndex;
  } catch (err) {
    setStatus('models-status', err instanceof Error ? err.message : String(err), 'error');
    populateModelSelects(null, null, null);
  }
}

/**
 * Picks which local model to preselect for chat, using the configured chat model when
 * it is available locally and otherwise falling back to the generation model.
 */
function localChatModelForSelect(
  configuredChat: string | undefined,
  generationModel: string,
): string {
  const generationModels = state.availableModels.filter((m) => !isEmbeddingModel(m));
  if (
    configuredChat &&
    generationModels.some((model) => canonicalModelId(model) === canonicalModelId(configuredChat))
  ) {
    return configuredChat;
  }
  return generationModel;
}

/** Shows the local model field or the cloud fields depending on the selected provider. */
function applyChatProvider(provider: string): void {
  const isLocal = provider === 'local';
  $('chat-local-field').hidden = !isLocal;
  $('chat-cloud-fields').hidden = isLocal;
}

/** Fills the embedding, generation, and chat selects from the cached available models. */
function populateModelSelects(
  currentEmbedding: string | null,
  currentGeneration: string | null,
  currentChat: string | null,
): void {
  const embed = $<HTMLSelectElement>('embedding-select');
  const gen = $<HTMLSelectElement>('generation-select');
  const chat = $<HTMLSelectElement>('chat-select');

  const embeddingModels = state.availableModels.filter(isEmbeddingModel).sort();
  const generationModels = state.availableModels.filter((m) => !isEmbeddingModel(m)).sort();

  fillSelect(embed, embeddingModels, currentEmbedding, 'No embedding models found');
  fillSelect(gen, generationModels, currentGeneration, 'No generation models found');
  fillSelect(chat, generationModels, currentChat, 'No chat models found');
}

/**
 * Rebuilds a select from available options, selecting the current value and adding a
 * "not pulled" option when the configured model is not present locally.
 */
function fillSelect(
  select: HTMLSelectElement,
  available: string[],
  current: string | null,
  emptyLabel: string,
): void {
  select.innerHTML = '';

  if (available.length === 0 && !current) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = emptyLabel;
    opt.disabled = true;
    opt.selected = true;
    select.appendChild(opt);
    select.disabled = true;
    return;
  }

  const currentCanonical = current ? canonicalModelId(current) : null;
  const matchInAvailable = currentCanonical
    ? (available.find((m) => canonicalModelId(m) === currentCanonical) ?? null)
    : null;

  if (current && !matchInAvailable) {
    const opt = document.createElement('option');
    opt.value = current;
    opt.textContent = `${current} (not pulled)`;
    opt.selected = true;
    select.appendChild(opt);
  }

  for (const id of available) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = id;
    if (id === matchInAvailable) opt.selected = true;
    select.appendChild(opt);
  }

  select.disabled = false;
}

/** Binds all button, toggle, and input event listeners for the settings page. */