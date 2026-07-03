/**
 * HTTP client for the Core API used by the extension. Manages auth token storage and
 * pairing, then exposes typed methods for every Core endpoint including chat streaming.
 */
import { CORE_SERVER_URL, API_PREFIX } from '@ai-mailpilot/shared';
import type {
  HealthResponse,
  AccountListResponse,
  AccountResponse,
  CreateAccountRequest,
  PushEmailsRequest,
  PushEmailsResponse,
  IngestAttachmentsRequest,
  IngestAttachmentsResponse,
  SyncStateRequest,
  SyncStateResponse,
  EmbedRunRequest,
  EmbedRunResponse,
  EmbedProgressResponse,
  DashboardResponse,
  TriageRunRequest,
  TriageRunResponse,
  TriageProgressResponse,
  PriorityRequest,
  PriorityResponse,
  TriageResolveRequest,
  TriageResolveResponse,
  CategorizeRunRequest,
  CategorizeRunResponse,
  CategorizeProgressResponse,
  LlmCategorizeRunRequest,
  LlmCategorizeRunResponse,
  LlmCategorizeProgressResponse,
  ChatRequest,
  ChatResponse,
  ChatStreamEvent,
  ChatSourceDto,
  EmailAssistantSummaryResponse,
  EmailAssistantDraftRequest,
  EmailAssistantDraftResponse,
  ConversationDto,
  ConversationSummaryDto,
  DiscoverTopicsRequest,
  DiscoverTopicsResponse,
  ImproveCategoriesRequest,
  ImproveSuggestionsResponse,
  ApplyImprovementsRequest,
  ApplyImprovementsResponse,
  CategoryDto,
  CategoryListResponse,
  CategoryUpdateRequest,
  CategoryMergeRequest,
  CategoryMergeResponse,
  CategoryEmailListResponse,
  EmailCategoriesResponse,
  FolderPlanResponse,
  GenerateProposalsRequest,
  GenerateProposalsResponse,
  ProposalListResponse,
  ApplyProposalResponse,
  DismissProposalResponse,
} from '@ai-mailpilot/shared';

const TOKEN_KEY = 'core_auth_token';

/**
 * Server configuration returned by the Core API, including LLM provider settings.
 * chatApiKey is write-only, sent on update and never returned, while chatApiKeySet
 * reports whether a key is stored.
 */
export interface ServerConfig {
  version: number;
  locale: 'en' | 'fr';
  autoIndex: boolean;
  indexedFolders: string[];
  llm: {
    baseUrl: string;
    apiKey?: string;
    embeddingModel: string;
    generationModel: string;
    chatModel?: string;
    embeddingDimensions: number;
    chatTopK?: number | null;
    chatSnippetChars?: number | null;
    chatRerank?: boolean;
    chatBaseUrl?: string | null;
    chatApiKey?: string | null;
    chatApiKeySet?: boolean;
    categorizeUseChatProvider?: boolean;
    priorityUseChatProvider?: boolean;
    allowCloudDiscovery?: boolean;
  };
}

/** Partial update payload for the server configuration. */
export interface UpdateConfigRequest {
  locale?: 'en' | 'fr';
  autoIndex?: boolean;
  indexedFolders?: string[];
  llm?: Partial<ServerConfig['llm']>;
}

/** HTTP client for the Core API, handling auth token storage, pairing, and all endpoints. */
export class CoreClient {
  private baseUrl: string;
  private authToken: string | null = null;

  /** Creates a client targeting the given Core base URL. */
  constructor(baseUrl: string = `${CORE_SERVER_URL}${API_PREFIX}`) {
    this.baseUrl = baseUrl;
  }

  /** Whether a non-empty auth token is currently held in memory. */
  hasToken(): boolean {
    return this.authToken !== null && this.authToken.length > 0;
  }

  /** Load the auth token from extension storage into memory. */
  async loadToken(): Promise<void> {
    const stored = (await browser.storage.local.get(TOKEN_KEY)) as Record<string, unknown>;
    if (typeof stored[TOKEN_KEY] === 'string') {
      this.authToken = stored[TOKEN_KEY];
    }
  }

  /** Store the auth token in memory and extension storage. */
  async setToken(token: string): Promise<void> {
    this.authToken = token;
    await browser.storage.local.set({ [TOKEN_KEY]: token });
  }

  /** Clear the stored auth token from memory and extension storage. */
  async clearToken(): Promise<void> {
    this.authToken = null;
    await browser.storage.local.set({ [TOKEN_KEY]: '' });
  }

  /**
   * Exchange a one-time pairing code for the auth token and store it.
   * Hits the unauthenticated /pair endpoint, so it sends no Authorization header.
   */
  async pair(code: string): Promise<void> {
    const res = await fetch(`${this.baseUrl}/pair`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code }),
    });
    if (!res.ok) {
      let message = `pairing failed (${res.status})`;
      try {
        const data = (await res.json()) as { error?: string };
        if (data.error) message = data.error;
      } catch {}
      throw new Error(message);
    }
    const data = (await res.json()) as { token?: string };
    if (!data.token) throw new Error('pairing response did not include a token');
    await this.setToken(data.token);
  }

  /**
   * Core request helper. Attaches the auth token, sets the JSON content type when a body
   * is present, and throws on non-2xx responses with the 401 case surfaced for re-pairing.
   */
  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    if (!this.authToken) await this.loadToken();

    const headers: Record<string, string> = {};
    if (options?.body != null) headers['Content-Type'] = 'application/json';
    if (this.authToken) headers['Authorization'] = `Bearer ${this.authToken}`;

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: { ...headers, ...(options?.headers as Record<string, string> | undefined) },
    });

    if (response.status === 401) {
      await this.loadToken();
      throw new Error(`Core API 401 ${path}: token rejected. Update the token in Settings.`);
    }

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Core API ${response.status} ${path}: ${error}`);
    }

    if (response.status === 204) return undefined as T;
    return response.json() as Promise<T>;
  }

  /** Server health check. The only unauthenticated endpoint. */
  health(): Promise<HealthResponse> {
    return this.request('/health');
  }

  /** Fetch the current server configuration. */
  getConfig(): Promise<ServerConfig> {
    return this.request('/config');
  }

  /** Apply a partial update to the server configuration. */
  updateConfig(patch: UpdateConfigRequest): Promise<ServerConfig> {
    return this.request('/config', { method: 'PATCH', body: JSON.stringify(patch) });
  }

  /** List all known accounts. */
  listAccounts(): Promise<AccountListResponse> {
    return this.request('/accounts');
  }

  /** Create a new account. */
  createAccount(req: CreateAccountRequest): Promise<AccountResponse> {
    return this.request('/accounts', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Fetch a single account by id. */
  getAccount(id: string): Promise<AccountResponse> {
    return this.request(`/accounts/${id}`);
  }

  /** Push a batch of emails to the server. */
  pushEmails(req: PushEmailsRequest): Promise<PushEmailsResponse> {
    return this.request('/emails/push', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Ingest a message's attachments for RAG. Core extracts and embeds them locally. */
  ingestAttachments(req: IngestAttachmentsRequest): Promise<IngestAttachmentsResponse> {
    return this.request('/attachments/ingest', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Report which of the given message ids still need a body fetched, so resync can skip complete emails. */
  syncState(req: SyncStateRequest): Promise<SyncStateResponse> {
    return this.request('/emails/sync-state', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Start an embedding run. */
  runEmbed(req: EmbedRunRequest): Promise<EmbedRunResponse> {
    return this.request('/embed/run', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Fetch progress of the current embedding run. */
  embedProgress(): Promise<EmbedProgressResponse> {
    return this.request('/embed/progress');
  }

  /** Start a triage run. */
  runTriage(req: TriageRunRequest): Promise<TriageRunResponse> {
    return this.request('/triage/run', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Fetch progress of the current triage run. */
  triageProgress(): Promise<TriageProgressResponse> {
    return this.request('/triage/progress');
  }

  /** Fetch the priority briefing for an account over a date range. */
  priority(req: PriorityRequest): Promise<PriorityResponse> {
    const params = new URLSearchParams({
      accountId: req.accountId,
      range: req.range,
      dayStartMs: String(req.dayStartMs),
    });
    return this.request(`/triage/priority?${params.toString()}`);
  }

  /** Mark a triage item as resolved. */
  resolveTriage(req: TriageResolveRequest): Promise<TriageResolveResponse> {
    return this.request('/triage/resolve', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Discover candidate topics for category suggestions. */
  discoverTopics(req: DiscoverTopicsRequest): Promise<DiscoverTopicsResponse> {
    return this.request('/topics/discover', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Request suggested category improvements. */
  improveSuggest(req: ImproveCategoriesRequest): Promise<ImproveSuggestionsResponse> {
    return this.request('/categories/improve/suggest', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  /** Apply selected category improvements. */
  improveApply(req: ApplyImprovementsRequest): Promise<ApplyImprovementsResponse> {
    return this.request('/categories/improve/apply', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Run discovery and persist new category proposals for review. */
  generateProposals(req: GenerateProposalsRequest): Promise<GenerateProposalsResponse> {
    return this.request('/categories/proposals/generate', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  }

  /** List the pending category proposals awaiting review for an account. */
  listProposals(accountId: string): Promise<ProposalListResponse> {
    const qs = new URLSearchParams({ accountId });
    return this.request(`/categories/proposals?${qs.toString()}`);
  }

  /** Approve a proposal: promote it to an active category and file its still-uncategorized emails. */
  applyProposal(proposalId: string, accountId: string): Promise<ApplyProposalResponse> {
    return this.request(`/categories/proposals/${encodeURIComponent(proposalId)}/apply`, {
      method: 'POST',
      body: JSON.stringify({ accountId }),
    });
  }

  /** Dismiss a proposal so it leaves the queue and is not proposed again. */
  dismissProposal(proposalId: string, accountId: string): Promise<DismissProposalResponse> {
    return this.request(`/categories/proposals/${encodeURIComponent(proposalId)}/dismiss`, {
      method: 'POST',
      body: JSON.stringify({ accountId }),
    });
  }

  /** Start an embedding-based categorization run. */
  runCategorize(req: CategorizeRunRequest): Promise<CategorizeRunResponse> {
    return this.request('/categorize/run', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Fetch progress of the current categorization run. */
  categorizeProgress(): Promise<CategorizeProgressResponse> {
    return this.request('/categorize/progress');
  }

  /** Start an LLM-based categorization run. */
  runLlmCategorize(req: LlmCategorizeRunRequest): Promise<LlmCategorizeRunResponse> {
    return this.request('/categorize/llm/run', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Fetch progress of the current LLM categorization run for an optional account. */
  llmCategorizeProgress(accountId?: string): Promise<LlmCategorizeProgressResponse> {
    const q = accountId ? `?accountId=${encodeURIComponent(accountId)}` : '';
    return this.request(`/categorize/llm/progress${q}`);
  }

  /** Request that the running LLM categorization stop. */
  stopLlmCategorize(): Promise<{ stopped: boolean }> {
    return this.request('/categorize/llm/stop', { method: 'POST', body: '{}' });
  }

  /** Fetch or regenerate an AI summary for a single message. */
  emailAssistantSummary(req: {
    accountId: string;
    messageId: string;
    force?: boolean;
    confirmCloud?: boolean;
  }): Promise<EmailAssistantSummaryResponse> {
    const params = new URLSearchParams({
      accountId: req.accountId,
      messageId: req.messageId,
    });
    if (req.force) params.set('force', 'true');
    if (req.confirmCloud) params.set('confirmCloud', 'true');
    return this.request(`/email-assistant/summary?${params.toString()}`);
  }

  /** Generate an AI reply draft for a message. */
  emailAssistantDraft(req: EmailAssistantDraftRequest): Promise<EmailAssistantDraftResponse> {
    return this.request('/email-assistant/draft', { method: 'POST', body: JSON.stringify(req) });
  }

  /** Send a chat request and await the full answer. */
  chat(req: ChatRequest): Promise<ChatResponse> {
    return this.request('/chat', { method: 'POST', body: JSON.stringify(req) });
  }

  /**
   * Stream a chat answer token by token. Reads the NDJSON event stream and dispatches
   * meta, delta, and error events. Resolves when the stream ends.
   */
  async chatStream(
    req: ChatRequest,
    handlers: {
      onMeta?: (conversationId: string, sources: ChatSourceDto[]) => void;
      onThink?: (text: string) => void;
      onDelta?: (text: string) => void;
      onPromote?: () => void;
      onError?: (message: string) => void;
    },
    signal?: AbortSignal,
  ): Promise<void> {
    if (!this.authToken) await this.loadToken();
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (this.authToken) headers['Authorization'] = `Bearer ${this.authToken}`;

    const res = await fetch(`${this.baseUrl}/chat/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(req),
      signal,
    });
    if (res.status === 401) {
      await this.loadToken();
      throw new Error('Core API 401 /chat/stream: token rejected. Update the token in Settings.');
    }
    if (!res.ok || !res.body) {
      const text = res.body ? await res.text() : '';
      throw new Error(`Core API ${res.status} /chat/stream: ${text}`);
    }

    /** Parse one NDJSON line into a stream event and route it to the matching handler. */
    const dispatch = (line: string): void => {
      if (!line) return;
      let event: ChatStreamEvent;
      try {
        event = JSON.parse(line) as ChatStreamEvent;
      } catch {
        return;
      }
      if (event.type === 'meta') handlers.onMeta?.(event.conversationId, event.sources);
      else if (event.type === 'think') handlers.onThink?.(event.text);
      else if (event.type === 'delta') handlers.onDelta?.(event.text);
      else if (event.type === 'promote') handlers.onPromote?.();
      else if (event.type === 'error') handlers.onError?.(event.message);
    };

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    try {
      while (true) {
        const { done, value } = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        let nl: number;
        while ((nl = buffer.indexOf('\n')) !== -1) {
          dispatch(buffer.slice(0, nl).trim());
          buffer = buffer.slice(nl + 1);
        }
        if (done) {
          dispatch(buffer.trim());
          break;
        }
      }
    } finally {
      reader.cancel().catch(() => {});
    }
  }

  /** List conversation summaries for an account. */
  listConversations(accountId: string): Promise<{ conversations: ConversationSummaryDto[] }> {
    return this.request(`/conversations?accountId=${encodeURIComponent(accountId)}`);
  }

  /** Fetch a full conversation by id. */
  getConversation(id: string, accountId: string): Promise<ConversationDto> {
    return this.request(
      `/conversations/${encodeURIComponent(id)}?accountId=${encodeURIComponent(accountId)}`,
    );
  }

  /** Delete a conversation by id. */
  deleteConversation(id: string, accountId: string): Promise<void> {
    return this.request(
      `/conversations/${encodeURIComponent(id)}?accountId=${encodeURIComponent(accountId)}`,
      { method: 'DELETE' },
    );
  }

  /** List categories for an account. */
  listCategories(accountId: string): Promise<CategoryListResponse> {
    const qs = new URLSearchParams({ accountId });
    return this.request(`/categories?${qs.toString()}`);
  }

  /** Fetch the proposed folder plan for an account. */
  folderPlan(accountId: string): Promise<FolderPlanResponse> {
    const qs = new URLSearchParams({ accountId });
    return this.request(`/folder-plan?${qs.toString()}`);
  }

  /** Apply a partial update to a category. */
  updateCategory(id: string, patch: CategoryUpdateRequest): Promise<CategoryDto> {
    return this.request(`/categories/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
  }

  /** Delete a category by id. */
  deleteCategory(id: string): Promise<void> {
    return this.request(`/categories/${encodeURIComponent(id)}`, { method: 'DELETE' });
  }

  /** Merge one category into another. */
  mergeCategory(sourceId: string, body: CategoryMergeRequest): Promise<CategoryMergeResponse> {
    return this.request(`/categories/${encodeURIComponent(sourceId)}/merge`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  /** List emails assigned to a category, paginated. */
  listEmailsInCategory(id: string, limit = 200, offset = 0): Promise<CategoryEmailListResponse> {
    const qs = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    return this.request(`/categories/${encodeURIComponent(id)}/emails?${qs.toString()}`);
  }

  /** Fetch the categories assigned to a single email. */
  getEmailCategories(messageId: string, accountId: string): Promise<EmailCategoriesResponse> {
    const qs = new URLSearchParams({ accountId });
    return this.request(`/emails/${encodeURIComponent(messageId)}/categories?${qs.toString()}`);
  }

  /** Replace the set of categories assigned to a single email. */
  setEmailCategories(
    messageId: string,
    accountId: string,
    categoryIds: string[],
  ): Promise<EmailCategoriesResponse> {
    return this.request(`/emails/${encodeURIComponent(messageId)}/categories`, {
      method: 'PUT',
      body: JSON.stringify({ accountId, categoryIds }),
    });
  }

  /** Fetch the dashboard summary for an account. */
  dashboard(accountId: string): Promise<DashboardResponse> {
    const qs = new URLSearchParams({ accountId });
    return this.request(`/dashboard?${qs.toString()}`);
  }
}

export const coreClient = new CoreClient();
