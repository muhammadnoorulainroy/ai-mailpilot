/**
 * Settings page controller for the extension. Wires up the UI for Core connection
 * and pairing, per-account sync preferences, model selection, chat provider
 * configuration, and triggering a mailbox sync with live progress.
 */
import { coreClient } from '../../api-client/core-client.js';
import { MailboxSnapshot, type MailboxAccount } from '../../thunderbird/mailbox.js';
import type { AccountDto } from '@ai-mailpilot/shared';
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
  coreAccounts: AccountDto[];
  availableModels: string[];
  savedChatBaseUrl: string | null;
  savedChatModel: string | null;
}

const state: State = {
  prefs: { enabledAddresses: [], excludedAddresses: [], excludedFolderPaths: [], applyTags: true },
  accounts: [],
  coreAccounts: [],
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
    await loadCoreAccounts();
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

/** Loads Core's account records so settings can show per-account AI discovery consent. */
async function loadCoreAccounts(): Promise<void> {
  if (!coreClient.hasToken()) {
    state.coreAccounts = [];
    return;
  }
  try {
    state.coreAccounts = (await coreClient.listAccounts()).accounts;
  } catch {
    state.coreAccounts = [];
  }
}

/** Account ids with an in-flight AI discovery PATCH, used to block duplicate concurrent saves. */
const discoveryPending = new Set<string>();

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
    const coreAccount =
      state.coreAccounts.find((a) => a.address.toLowerCase() === account.address.toLowerCase()) ??
      null;
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
        <div class="account-discovery-hint"></div>
      </div>
      <div class="account-controls">
        <label class="account-toggle">
          <span>Sync</span>
          <span class="toggle">
            <input class="sync-toggle" type="checkbox" />
            <span class="toggle-track"><span class="toggle-thumb"></span></span>
          </span>
        </label>
        <label class="account-toggle">
          <span>AI discovery</span>
          <span class="toggle">
            <input class="discovery-toggle" type="checkbox" />
            <span class="toggle-track"><span class="toggle-thumb"></span></span>
          </span>
        </label>
      </div>
    `;

    (row.querySelector('.account-name') as HTMLElement).textContent = account.name;
    (row.querySelector('.account-address') as HTMLElement).textContent = account.address;
    (row.querySelector('.account-kind') as HTMLElement).textContent = account.kind;
    const hint = row.querySelector('.account-discovery-hint') as HTMLElement;
    if (!coreAccount) {
      hint.textContent = 'AI discovery setting appears after this account is synced.';
    } else if (coreAccount.discoveryEnabled) {
      hint.textContent =
        'AI discovery can inspect this account to suggest categories and cleanup. Discovery stays local unless cloud discovery is explicitly enabled.';
    } else {
      hint.textContent =
        'AI discovery is off. Topic discovery, category proposals, and cleanup suggestions skip this account.';
    }

    const checkbox = row.querySelector('.sync-toggle') as HTMLInputElement;
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

    const discovery = row.querySelector('.discovery-toggle') as HTMLInputElement;
    discovery.checked = coreAccount?.discoveryEnabled ?? false;
    discovery.disabled = !coreAccount || discoveryPending.has(coreAccount.id);
    discovery.addEventListener('change', async () => {
      if (!coreAccount || discoveryPending.has(coreAccount.id)) {
        discovery.checked = coreAccount?.discoveryEnabled ?? false;
        return;
      }
      const next = discovery.checked;
      if (next) {
        const ok = window.confirm(
          `Allow AI discovery for ${account.address}? This lets AI MailPilot inspect this account's subjects, snippets, and embeddings to discover categories and cleanup suggestions. Discovery runs locally unless you explicitly enable cloud discovery in Core.`,
        );
        if (!ok) {
          discovery.checked = false;
          return;
        }
      }
      discoveryPending.add(coreAccount.id);
      discovery.disabled = true;
      try {
        const updated = await coreClient.updateAccountDiscovery(coreAccount.id, {
          discoveryEnabled: next,
        });
        state.coreAccounts = state.coreAccounts.map((a) =>
          a.id === updated.account.id ? updated.account : a,
        );
      } catch (err) {
        setStatus(
          'accounts-status',
          `Could not save AI discovery setting: ${err instanceof Error ? err.message : String(err)}`,
          'error',
        );
      } finally {
        discoveryPending.delete(coreAccount.id);
        renderAccounts();
      }
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
    $<HTMLInputElement>('toggle-multi-prototype').checked =
      cfg.features?.multiPrototypeCategories ?? false;
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
function attachHandlers(): void {
  $<HTMLButtonElement>('btn-open-dashboard').addEventListener('click', () => {
    void browser.tabs.create({ url: browser.runtime.getURL('dashboard/dashboard.html') });
  });

  $<HTMLInputElement>('toggle-apply-tags').addEventListener('change', async (e) => {
    const checked = (e.target as HTMLInputElement).checked;
    state.prefs.applyTags = checked;
    await saveSyncPrefs(state.prefs);
  });

  $<HTMLInputElement>('toggle-auto-index').addEventListener('change', async (e) => {
    const input = e.target as HTMLInputElement;
    try {
      await coreClient.updateConfig({ autoIndex: input.checked });
    } catch (err) {
      input.checked = !input.checked;
      setStatus(
        'models-status',
        `Could not save auto-index: ${err instanceof Error ? err.message : String(err)}`,
        'error',
      );
    }
  });

  $<HTMLInputElement>('toggle-multi-prototype').addEventListener('change', async (e) => {
    const input = e.target as HTMLInputElement;
    try {
      await coreClient.updateConfig({ features: { multiPrototypeCategories: input.checked } });
    } catch (err) {
      input.checked = !input.checked;
      setStatus(
        'models-status',
        `Could not save the experimental toggle: ${err instanceof Error ? err.message : String(err)}`,
        'error',
      );
    }
  });

  $<HTMLButtonElement>('btn-test').addEventListener('click', async () => {
    await refreshConnection();
    if (coreClient.hasToken()) await loadConfigIfAuthed();
  });

  $<HTMLButtonElement>('btn-pair').addEventListener('click', async () => {
    const input = $<HTMLInputElement>('pair-code-input');
    const status = $('pair-status');
    const code = input.value.trim();
    if (!code) {
      status.textContent = 'Enter the 6-digit code from the Core console.';
      return;
    }
    status.textContent = 'Pairing...';
    try {
      await coreClient.pair(code);
      input.value = '';
      status.textContent = 'Paired.';
      setBadge('auth-dot', 'auth-label', 'auth-badge', 'success', 'Paired');
      await loadConfigIfAuthed();
    } catch (err) {
      status.textContent = `Pairing failed: ${err instanceof Error ? err.message : String(err)}`;
    }
  });

  $<HTMLButtonElement>('btn-save-token').addEventListener('click', async () => {
    const input = $<HTMLInputElement>('token-input');
    const token = input.value.trim();
    if (!token) {
      setBadge('auth-dot', 'auth-label', 'auth-badge', 'warning', 'No token');
      return;
    }
    await coreClient.setToken(token);
    input.value = '';
    setBadge('auth-dot', 'auth-label', 'auth-badge', 'success', 'Paired');
    await loadConfigIfAuthed();
  });

  $<HTMLSelectElement>('chat-provider').addEventListener('change', (e) => {
    const select = e.target as HTMLSelectElement;
    const provider = select.value;
    if (provider !== 'local') {
      const ok = window.confirm(
        'Cloud chat sends your question and the retrieved email snippets to the cloud provider ' +
          'you configure. Embeddings still run locally. Continue?',
      );
      if (!ok) {
        select.value = 'local';
        applyChatProvider('local');
        return;
      }
    }
    applyChatProvider(provider);
    if (provider === 'openai') {
      const url = $<HTMLInputElement>('chat-base-url');
      if (!url.value.trim()) url.value = 'https://api.openai.com/v1';
      const model = $<HTMLInputElement>('chat-model-text');
      if (!model.value.trim()) model.value = 'gpt-4o-mini';
    }
  });

  $<HTMLButtonElement>('btn-save-models').addEventListener('click', async () => {
    const embeddingModel = $<HTMLSelectElement>('embedding-select').value;
    const generationModel = $<HTMLSelectElement>('generation-select').value;
    if (!embeddingModel || !generationModel) {
      setStatus('models-status', 'Embedding and generation models are required', 'error');
      return;
    }

    const topK = parseTuning($<HTMLInputElement>('chat-topk').value, 1, 50);
    const snippet = parseTuning($<HTMLInputElement>('chat-snippet').value, 200, 8000);
    if (topK === 'invalid' || snippet === 'invalid') {
      setStatus(
        'models-status',
        'Emails per answer (1-50) and chars per email (200-8000) must be valid numbers',
        'error',
      );
      return;
    }

    const llmPatch: NonNullable<Parameters<typeof coreClient.updateConfig>[0]['llm']> = {
      embeddingModel,
      generationModel,
      chatTopK: topK,
      chatSnippetChars: snippet,
      chatRerank: $<HTMLInputElement>('chat-rerank').checked,
    };

    const provider = $<HTMLSelectElement>('chat-provider').value;
    if (provider === 'local') {
      llmPatch.chatBaseUrl = null;
      llmPatch.chatModel = $<HTMLSelectElement>('chat-select').value || generationModel;
      llmPatch.categorizeUseChatProvider = false;
      llmPatch.priorityUseChatProvider = false;
    } else {
      const baseUrl =
        $<HTMLInputElement>('chat-base-url').value.trim() || (state.savedChatBaseUrl ?? '');
      const typedModel = $<HTMLInputElement>('chat-model-text').value.trim();
      const savedCloudModel =
        state.savedChatModel && state.savedChatModel !== generationModel
          ? state.savedChatModel
          : '';
      const model = typedModel || savedCloudModel || (provider === 'openai' ? 'gpt-4o-mini' : '');
      if (!baseUrl || !model) {
        setStatus('models-status', 'Cloud chat needs an API base URL and a model name', 'error');
        return;
      }
      llmPatch.chatBaseUrl = baseUrl;
      llmPatch.chatModel = model;
      llmPatch.categorizeUseChatProvider = $<HTMLInputElement>('chat-categorize').checked;
      llmPatch.priorityUseChatProvider = $<HTMLInputElement>('chat-priority').checked;
      const key = $<HTMLInputElement>('chat-api-key').value.trim();
      if (key) llmPatch.chatApiKey = key;
    }

    setStatus('models-status', 'Saving...', 'info');
    try {
      await coreClient.updateConfig({ llm: llmPatch });
      $<HTMLInputElement>('chat-api-key').value = '';
      setStatus('models-status', 'Saved', 'success');
    } catch (err) {
      setStatus('models-status', err instanceof Error ? err.message : String(err), 'error');
    }
  });

  $<HTMLButtonElement>('btn-add-folder').addEventListener('click', async () => {
    const input = $<HTMLInputElement>('folder-input');
    const path = input.value.trim();
    if (!path) return;
    if (!state.prefs.excludedFolderPaths.includes(path)) {
      state.prefs.excludedFolderPaths.push(path);
      await saveSyncPrefs(state.prefs);
    }
    input.value = '';
    renderFolders();
  });

  $<HTMLInputElement>('folder-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      $<HTMLButtonElement>('btn-add-folder').click();
    }
  });

  $<HTMLButtonElement>('btn-sync').addEventListener('click', async () => {
    const btn = $<HTMLButtonElement>('btn-sync');
    const force = $<HTMLInputElement>('toggle-force-sync').checked;
    btn.disabled = true;
    btn.textContent = 'Syncing...';

    renderSyncBanner(emptyProgress('syncing'));
    const poll = window.setInterval(() => {
      void browser.runtime
        .sendMessage({ type: 'mailpilot:sync-progress' })
        .then((p) => {
          const sp = p as SyncProgress | undefined;
          if (sp?.phase) renderSyncBanner(sp);
        })
        .catch(() => {});
    }, 700);

    try {
      await browser.runtime.sendMessage({ type: 'mailpilot:sync', force });
      const final = (await browser.runtime.sendMessage({
        type: 'mailpilot:sync-progress',
      })) as SyncProgress | undefined;
      renderSyncBanner(final?.phase ? final : emptyProgress('done'));
      await loadCoreAccounts();
      renderAccounts();
      btn.textContent = 'Sync complete';
    } catch (err) {
      renderSyncBanner({
        ...emptyProgress('error'),
        error: err instanceof Error ? err.message : String(err),
      });
      btn.textContent = 'Sync failed';
      console.error('sync trigger failed:', err);
    } finally {
      window.clearInterval(poll);
      window.setTimeout(() => {
        btn.textContent = 'Sync inbox now';
        btn.disabled = false;
        $('sync-banner').hidden = true;
      }, 3500);
    }
  });
}

interface SyncProgress {
  phase: 'idle' | 'syncing' | 'embedding' | 'done' | 'error';
  processed: number;
  total: number;
  pushed: number;
  upToDate: number;
  embedProcessed: number;
  embedTotal: number;
  error?: string;
}

/** Builds a zeroed progress object for the given phase. */
function emptyProgress(phase: SyncProgress['phase']): SyncProgress {
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

/** Updates the sync progress banner label, bar, and count for the current phase. */
function renderSyncBanner(p: SyncProgress): void {
  $('sync-banner').hidden = false;
  const fill = $<HTMLElement>('sync-banner-fill');
  const label = $('sync-banner-label');
  const count = $('sync-banner-count');

  /** Renders a filled progress bar and a done over total count. */
  const determinate = (done: number, total: number): void => {
    fill.classList.remove('indeterminate');
    fill.style.width = `${Math.min(100, Math.round((done / total) * 100))}%`;
    count.textContent = `${done.toLocaleString()} / ${total.toLocaleString()}`;
  };
  /** Renders an animated bar with optional text when no total is known. */
  const indeterminate = (text: string): void => {
    fill.classList.add('indeterminate');
    fill.style.width = '';
    count.textContent = text;
  };

  if (p.phase === 'embedding') {
    label.textContent = 'Embedding emails';
    if (p.embedTotal > 0) determinate(p.embedProcessed, p.embedTotal);
    else indeterminate('');
  } else if (p.phase === 'done') {
    label.textContent = 'Sync complete';
    fill.classList.remove('indeterminate');
    fill.style.width = '100%';
    const parts: string[] = [];
    if (p.pushed > 0) parts.push(`${p.pushed.toLocaleString()} synced`);
    if (p.upToDate > 0) parts.push(`${p.upToDate.toLocaleString()} already up to date`);
    count.textContent = parts.join(', ');
  } else if (p.phase === 'error') {
    label.textContent = `Sync failed: ${p.error ?? 'unknown error'}`;
    fill.classList.remove('indeterminate');
  } else {
    label.textContent = 'Syncing emails from Thunderbird';
    if (p.total > 0) determinate(p.processed, p.total);
    else indeterminate(p.processed > 0 ? `${p.processed.toLocaleString()} emails` : '');
  }
}

type Tone = 'success' | 'warning' | 'urgent' | 'neutral' | 'info' | 'error';

/** Sets the dot, badge, and label classes and text for a status indicator to match the tone. */
function setBadge(
  dotId: string,
  labelId: string,
  badgeId: string,
  tone: Tone,
  label: string,
): void {
  const dot = $(dotId);
  const labelEl = $(labelId);
  const badge = $(badgeId);

  dot.className = `dot dot-${tone === 'info' || tone === 'error' ? 'neutral' : tone}`;

  if (tone === 'urgent' || tone === 'error') {
    badge.className = 'badge badge-urgent';
    dot.className = 'dot dot-urgent';
  } else if (tone === 'warning') {
    badge.className = 'badge badge-warning';
    dot.className = 'dot dot-warning';
  } else if (tone === 'success') {
    badge.className = 'badge badge-success';
    dot.className = 'dot dot-success';
  } else {
    badge.className = 'badge badge-neutral';
    dot.className = 'dot dot-neutral';
  }

  labelEl.textContent = label;
}

/** Writes a status message into an element and colors it according to the tone. */
function setStatus(id: string, message: string, tone: Tone): void {
  const el = $(id);
  el.textContent = message;
  el.className = 'hint';
  if (tone === 'error' || tone === 'urgent') el.style.color = 'var(--urgent-text)';
  else if (tone === 'success') el.style.color = 'var(--success-text)';
  else if (tone === 'warning') el.style.color = 'var(--warning-text)';
  else el.style.color = 'var(--text-secondary)';
}

/**
 * Parses a tuning field, returning null for blank input, 'invalid' when out of the
 * inclusive min to max integer range, or the parsed number.
 */
function parseTuning(raw: string, min: number, max: number): number | null | 'invalid' {
  const t = raw.trim();
  if (t === '') return null;
  const n = Number(t);
  if (!Number.isInteger(n) || n < min || n > max) return 'invalid';
  return n;
}

init().catch((err) => {
  console.error('[MailPilot settings] init failed:', err);
});
