/**
 * Shared request and response type definitions for the Core HTTP API, covering
 * accounts, mailbox sync, embedding, triage, categorization, chat, and dashboards.
 */

/** Health check response with LLM connectivity and locale. */
export interface HealthResponse {
  status: 'ok' | 'error';
  version: string;
  llm: {
    connected: boolean;
    models: string[];
    baseUrl: string;
  };
  locale: string;
}

/** Classification of an email account. */
export type AccountKind = 'personal' | 'work' | 'institutional';

/** A configured email account. */
export interface AccountDto {
  id: string;
  address: string;
  displayName: string | null;
  kind: AccountKind;
  createdAt: number;
}

/** Request body to create an account. */
export interface CreateAccountRequest {
  address: string;
  displayName?: string;
  kind: AccountKind;
}

/** Response listing all accounts. */
export interface AccountListResponse {
  accounts: AccountDto[];
}

/** Response wrapping a single account. */
export interface AccountResponse {
  account: AccountDto;
}

/** Metadata for a mailbox folder. */
export interface FolderInfo {
  name: string;
  path: string;
  totalMessages: number;
  unreadMessages: number;
}

/** Response listing mailbox folders. */
export interface FolderListResponse {
  folders: FolderInfo[];
}

/** A single email pushed from the extension to Core. */
export interface PushEmailItem {
  messageId: string;
  folder: string;
  subject?: string;
  fromAddr?: string;
  date?: number;
  body?: string;
  bodyFormat?: 'text' | 'html';
  hasAttachments?: boolean;
  bodyFetched?: boolean;
}

/** Request to push a batch of emails for an account. */
export interface PushEmailsRequest {
  accountId: string;
  emails: PushEmailItem[];
}

/** Result of pushing emails, with inserted and total counts. */
export interface PushEmailsResponse {
  inserted: number;
  total: number;
}

/** Request to reconcile the mailbox state by message ids. */
export interface SyncStateRequest {
  accountId: string;
  messageIds: string[];
}

/** Subset of message ids that still need a body fetched. */
export interface SyncStateResponse {
  needFetch: string[];
}

/** Request to start an embedding run for an account. */
export interface EmbedRunRequest {
  accountId: string;
  modelId?: string;
}

/** Result of starting an embedding run. */
export interface EmbedRunResponse {
  status: 'started' | 'already_running' | 'up_to_date';
  pending: number;
  modelId: string;
}

/** Progress of an in-flight or completed embedding run. */
export interface EmbedProgressResponse {
  status: 'idle' | 'running' | 'completed' | 'completed_with_failures' | 'error';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  failed: number;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

/** Triage classification bucket for an email. */
export type TriageBucket = 'urgent' | 'summarize' | 'spam' | 'personal';

/** Request to start a triage run for an account. */
export interface TriageRunRequest {
  accountId: string;
  modelId?: string;
  force?: boolean;
}

/** Result of starting a triage run. */
export interface TriageRunResponse {
  status: 'started' | 'already_running' | 'up_to_date';
  pending: number;
  modelId: string;
}

/** Progress of an in-flight or completed triage run. */
export interface TriageProgressResponse {
  status: 'idle' | 'running' | 'completed' | 'completed_with_failures' | 'error';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  failed: number;
  buckets: Record<TriageBucket, number>;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

/** Per-account triage bucket counts. */
export interface TriageSummaryResponse {
  accountId: string;
  buckets: Record<TriageBucket, number>;
}

/**
 * Richer per-email triage metadata. Optional everywhere since old rows predate it and a
 * model may omit fields, so the parser fills safe defaults.
 */
export interface TriageMetadata {
  actionRequired: boolean;
  needsReply: boolean;
  deadlineAt: number | null;
  importanceScore: number;
  suggestedAction?: string | null;
  shortSummary?: string | null;
  confidence?: number | null;
}

/** Time range for the priority "Today's Focus" view. */
export type PriorityRange = 'today' | 'week' | 'all';

/** Request for the priority view, scoped to a range and the user's local day start. */
export interface PriorityRequest {
  accountId: string;
  range: PriorityRange;
  dayStartMs: number;
}

/** An email as rendered in the priority view, with triage metadata. */
export interface PriorityEmailDto {
  messageId: string;
  folder: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  hasAttachments: boolean;
  bucket: TriageBucket;
  reasoning: string | null;
  classifiedAt: number;
  actionRequired: boolean;
  needsReply: boolean;
  deadlineAt: number | null;
  importanceScore: number;
  suggestedAction: string | null;
  shortSummary: string | null;
}

/** The priority view payload with counts and rendered sections. */
export interface PriorityResponse {
  accountId: string;
  range: PriorityRange;
  generatedAt: number;
  counts: {
    needsAction: number;
    urgent: number;
    important: number;
    summaries: number;
    lowPriority: number;
    unclassified: number;
  };
  needsAction: PriorityEmailDto[];
  important: PriorityEmailDto[];
  summaries: PriorityEmailDto[];
  carryover: PriorityEmailDto[];
  lowPriority: PriorityEmailDto[];
}

/** Action taken on a triaged email in the focus view. */
export type TriageResolution = 'dismiss' | 'done' | 'snooze' | 'reset';

/** Request to resolve a triaged email. snoozedUntil is required for 'snooze'. */
export interface TriageResolveRequest {
  accountId: string;
  messageId: string;
  resolution: TriageResolution;
  snoozedUntil?: number;
}

/** Result of resolving a triaged email. */
export interface TriageResolveResponse {
  ok: boolean;
}

/** Origin of a category. */
export type CategorySource = 'auto' | 'user' | 'imported';

/** A category with its label, description, and email count. */
export interface CategoryDto {
  id: string;
  accountId: string;
  label: string;
  description: string | null;
  source: CategorySource;
  emailCount: number;
  createdAt: number;
  updatedAt: number;
}

/** Response listing categories for an account. */
export interface CategoryListResponse {
  accountId: string;
  categories: CategoryDto[];
}

/** Request to update a category's label or description. */
export interface CategoryUpdateRequest {
  label?: string;
  description?: string | null;
}

/** Request to merge a category into the given target. */
export interface CategoryMergeRequest {
  targetId: string;
}

/** Result of a merge, with the number of reassigned emails. */
export interface CategoryMergeResponse {
  reassigned: number;
}

/** An email as listed within a category, with per-category provenance. */
export interface CategoryEmailDto {
  messageId: string;
  accountId: string;
  folder: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  hasAttachments: boolean;
  confidence: number;
  assignedBy: 'user' | 'auto';
  method?: 'embed' | 'llm' | 'gate' | null;
  categories: Array<{
    id: string;
    label: string;
    confidence: number;
    assignedBy: 'user' | 'auto';
    method?: 'embed' | 'llm' | 'gate' | null;
  }>;
}

/** Response listing the emails within a category. */
export interface CategoryEmailListResponse {
  categoryId: string;
  emails: CategoryEmailDto[];
}

/** Request to discover topics and seed categories for an account. */
export interface DiscoverTopicsRequest {
  accountId: string;
  embeddingModelId?: string;
  generationModelId?: string;
}

/** Result of a topic discovery run. */
export interface DiscoverTopicsResponse {
  status: 'ok' | 'insufficient_categories';
  topicsCreated: number;
  emailsSampled: number;
  centroidsComputed: number;
}

/** Request to generate suggested taxonomy improvements for an account. */
export interface ImproveCategoriesRequest {
  accountId: string;
}

/** A proposed new category derived from uncategorized mail. */
export interface SuggestedCategory {
  label: string;
  description: string;
  estimatedCount: number;
  sampleSubjects: string[];
  messageIds?: string[];
}

/** A proposed expansion of an existing category with more emails. */
export interface SuggestedCategoryExpansion {
  categoryId: string;
  categoryLabel: string;
  estimatedCount: number;
  sampleSubjects: string[];
  sampleSenders: string[];
  reason: string;
  messageIds: string[];
}

/** A proposed merge of one category into another. */
export interface SuggestedMerge {
  sourceId: string;
  sourceLabel: string;
  targetId: string;
  targetLabel: string;
  reason: string;
}

/** Suggested taxonomy changes for the user to review and approve. */
export interface ImproveSuggestionsResponse {
  uncategorizedCount: number;
  sampledCount: number;
  existingCategoryExpansions: SuggestedCategoryExpansion[];
  newCategories: SuggestedCategory[];
  merges: SuggestedMerge[];
  leaveUncategorized?: {
    estimatedCount: number;
    reason: string;
    sampleSubjects: string[];
  };
  warning?: string;
  diagnostics?: {
    existingCategoriesLikelyCoverBacklog: boolean;
    recommendation: string;
  };
}

/** Request to apply the approved taxonomy changes. */
export interface ApplyImprovementsRequest {
  accountId: string;
  existingCategoryExpansions?: Array<{ categoryId: string; messageIds: string[] }>;
  newCategories: Array<{ label: string; description: string; messageIds?: string[] }>;
  merges: Array<{ sourceId: string; targetId: string }>;
}

/** Counts of changes applied during an improvement run. */
export interface ApplyImprovementsResponse {
  expanded: number;
  created: number;
  merged: number;
}

/** Request to start an embedding-based categorization run. */
export interface CategorizeRunRequest {
  accountId: string;
  embeddingModelId?: string;
  force?: boolean;
}

/** Result of starting a categorization run. */
export interface CategorizeRunResponse {
  status: 'started' | 'already_running' | 'up_to_date';
  pending: number;
}

/** Progress of an in-flight or completed categorization run. */
export interface CategorizeProgressResponse {
  status: 'idle' | 'running' | 'completed' | 'error';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  uncategorized: number;
  assigned: number;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

/** Request to start an LLM-based categorization run. */
export interface LlmCategorizeRunRequest {
  accountId: string;
  generationModelId?: string;
  force?: boolean;
  retryUncategorized?: boolean;
  messageIds?: string[];
}

/** Result of starting an LLM categorization run. */
export interface LlmCategorizeRunResponse {
  status: 'started' | 'already_running' | 'up_to_date';
  pending: number;
}

/** Progress of an in-flight or completed LLM categorization run. */
export interface LlmCategorizeProgressResponse {
  status:
    | 'idle'
    | 'running'
    | 'completed'
    | 'completed_with_failures'
    | 'error'
    | 'stopped'
    | 'interrupted';
  phase?: 'preparing' | 'clustering' | 'categorizing';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  assigned: number;
  uncategorized: number;
  failed: number;
  clusters: number;
  clustersProcessed: number;
  gatedClusters: number;
  llmCalls: number;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

/** A category assignment on an email, with how it was decided. */
export interface EmailCategoryAssignment {
  categoryId: string;
  label: string;
  confidence: number;
  assignedBy: 'user' | 'auto';
  method?: 'embed' | 'llm' | 'gate' | null;
}

/** Response listing the category assignments for an email. */
export interface EmailCategoriesResponse {
  messageId: string;
  accountId: string;
  categories: EmailCategoryAssignment[];
}

/** Request to set the categories on an email. */
export interface SetEmailCategoriesRequest {
  accountId: string;
  categoryIds: string[];
}

/** A category in a folder organization plan. */
export interface FolderPlanCategory {
  id: string;
  label: string;
  count: number;
}

/** A planned assignment of an email to a category folder. */
export interface FolderPlanAssignment {
  messageId: string;
  categoryId: string;
}

/** A folder organization plan with categories and assignments. */
export interface FolderPlanResponse {
  accountId: string;
  categories: FolderPlanCategory[];
  assignments: FolderPlanAssignment[];
}

/** An email as shown on the dashboard. */
export interface DashboardEmailDto {
  messageId: string;
  folder: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  hasAttachments: boolean;
}

/** A triaged dashboard email with its classification reasoning. */
export interface DashboardClassifiedEmailDto extends DashboardEmailDto {
  reasoning: string | null;
  classifiedAt: number;
}

/** A category summary as shown on the dashboard. */
export interface DashboardCategorySummaryDto {
  id: string;
  label: string;
  description: string | null;
  emailCount: number;
}

/** The dashboard payload with email counts, triage buckets, and categories. */
export interface DashboardResponse {
  accountId: string;
  generatedAt: number;
  emails: {
    total: number;
    unclassified: number;
    uncategorized: number;
  };
  triage: {
    buckets: Record<TriageBucket, number>;
    urgent: DashboardClassifiedEmailDto[];
    summarize: DashboardClassifiedEmailDto[];
  };
  recent: DashboardEmailDto[];
  categoryCount: number;
  categories: DashboardCategorySummaryDto[];
}

/** A single turn in a chat conversation. */
export interface ChatTurnDto {
  role: 'user' | 'assistant';
  content: string;
}

/** Request to ask a question of the inbox, optionally continuing a conversation. */
export interface ChatRequest {
  accountId: string;
  question: string;
  conversationId?: string;
}

/** An NDJSON stream event from the chat stream endpoint, one per line. */
export type ChatStreamEvent =
  | { type: 'meta'; conversationId: string; sources: ChatSourceDto[] }
  | { type: 'think'; text: string }
  | { type: 'delta'; text: string }
  | { type: 'promote' }
  | { type: 'done' }
  | { type: 'error'; message: string };

/** A short summary of a conversation for listing. */
export interface ConversationSummaryDto {
  id: string;
  updatedAt: number;
  preview: string;
}

/** A single turn within a stored conversation. */
export interface ConversationTurnDto {
  role: 'user' | 'assistant';
  content: string;
  at: number;
  sources?: ChatSourceDto[];
}

/** A full stored conversation with all its turns. */
export interface ConversationDto {
  id: string;
  accountId: string;
  turns: ConversationTurnDto[];
  updatedAt: number;
}

/** An email cited as a source for a chat answer. */
export interface ChatSourceDto {
  messageId: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  score: number;
  attachmentName?: string;
}

/** An attachment pushed from the extension for ingestion. */
export interface PushAttachmentItem {
  filename: string;
  contentType?: string;
  partName: string;
  size?: number;
  dataBase64: string;
}

/** Request to ingest attachments for an email. */
export interface IngestAttachmentsRequest {
  accountId: string;
  messageId: string;
  attachments: PushAttachmentItem[];
}

/** Result of ingesting attachments, per file. */
export interface IngestAttachmentsResponse {
  results: Array<{ filename: string; status: string; chunks: number }>;
}

/** Count of attachments with extracted text. */
export interface AttachmentStatsResponse {
  extracted: number;
}

/** A chat answer with the emails retrieved as context. */
export interface ChatResponse {
  answer: string;
  sources: ChatSourceDto[];
}
