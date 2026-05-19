import type { EmbeddingProgress } from './embedding.js';
import type { SemanticCategory, CategorizationResult, ClusteringResult, ClusteringParams } from './category.js';
import type { EmailSummary } from './email.js';

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

export type AccountKind = 'personal' | 'work' | 'institutional';

export interface AccountDto {
  id: string;
  address: string;
  displayName: string | null;
  kind: AccountKind;
  createdAt: number;
}

export interface CreateAccountRequest {
  address: string;
  displayName?: string;
  kind: AccountKind;
}

export interface AccountListResponse {
  accounts: AccountDto[];
}

export interface AccountResponse {
  account: AccountDto;
}

export interface FolderListResponse {
  folders: FolderInfo[];
}

export interface FolderInfo {
  name: string;
  path: string;
  totalMessages: number;
  unreadMessages: number;
}

export interface PushEmailItem {
  messageId: string;
  folder: string;
  subject?: string;
  fromAddr?: string;
  date?: number;
  body?: string;
  bodyFormat?: 'text' | 'html';
  hasAttachments?: boolean;
}

export interface PushEmailsRequest {
  accountId: string;
  emails: PushEmailItem[];
}

export interface PushEmailsResponse {
  inserted: number;
  total: number;
}

export interface EmbedRunRequest {
  accountId: string;
  modelId?: string;
}

export interface EmbedRunResponse {
  status: 'started' | 'already_running';
  pending: number;
  modelId: string;
}

export interface EmbedProgressResponse {
  status: 'idle' | 'running' | 'completed' | 'error';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  failed: number;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

export interface IndexFolderRequest {
  folder: string;
}

export interface IndexFolderResponse {
  status: 'started' | 'already_running';
  folder: string;
}

export interface EmbeddingProgressResponse {
  progress: EmbeddingProgress;
}

export interface ClusterRequest {
  params?: Partial<ClusteringParams>;
}

export interface ClusterResponse {
  result: ClusteringResult;
}

export interface CategoryListResponse {
  categories: SemanticCategory[];
}

export interface CategoryUpdateRequest {
  label?: string;
  description?: string;
}

export interface CategoryMergeRequest {
  targetCategoryId: string;
}

export interface CategorizeRequest {
  messageId: string;
}

export interface CategorizeResponse {
  result: CategorizationResult;
}

export interface ConfirmCategorizationRequest {
  messageId: string;
  categoryId: string;
}

export interface SearchRequest {
  query: string;
  limit?: number;
  folders?: string[];
  dateFrom?: string;
  dateTo?: string;
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface SearchResult {
  email: EmailSummary;
  score: number;
}

export interface ChatRequest {
  question: string;
  conversationId?: string;
}

export interface ChatResponse {
  answer: string;
  sources: EmailSummary[];
  conversationId: string;
}
