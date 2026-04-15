import type { EmbeddingProgress } from './embedding.js';
import type { SemanticCategory, CategorizationResult, ClusteringResult, ClusteringParams } from './category.js';
import type { EmailSummary } from './email.js';

export interface HealthResponse {
  status: 'ok' | 'error';
  version: string;
  ollama: { connected: boolean; models: string[] };
  imap: { connected: boolean };
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
