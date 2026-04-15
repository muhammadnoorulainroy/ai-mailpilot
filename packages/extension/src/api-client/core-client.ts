import { CORE_SERVER_URL, API_PREFIX } from '@ai-mailpilot/shared';
import type {
  HealthResponse,
  FolderListResponse,
  IndexFolderRequest,
  IndexFolderResponse,
  EmbeddingProgressResponse,
  ClusterRequest,
  ClusterResponse,
  CategoryListResponse,
  CategorizeRequest,
  CategorizeResponse,
  ConfirmCategorizationRequest,
  SearchRequest,
  SearchResponse,
  ChatRequest,
  ChatResponse,
} from '@ai-mailpilot/shared';

class CoreClient {
  private baseUrl: string;

  constructor(baseUrl: string = `${CORE_SERVER_URL}${API_PREFIX}`) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Core API error (${response.status}): ${error}`);
    }
    return response.json() as Promise<T>;
  }

  async health(): Promise<HealthResponse> {
    return this.request('/health');
  }

  async listFolders(): Promise<FolderListResponse> {
    return this.request('/folders');
  }

  async indexFolder(req: IndexFolderRequest): Promise<IndexFolderResponse> {
    return this.request('/embed/index', { method: 'POST', body: JSON.stringify(req) });
  }

  async embeddingProgress(): Promise<EmbeddingProgressResponse> {
    return this.request('/embed/progress');
  }

  async cluster(req?: ClusterRequest): Promise<ClusterResponse> {
    return this.request('/cluster', { method: 'POST', body: JSON.stringify(req ?? {}) });
  }

  async listCategories(): Promise<CategoryListResponse> {
    return this.request('/categories');
  }

  async deleteCategory(id: string): Promise<void> {
    await this.request(`/categories/${id}`, { method: 'DELETE' });
  }

  async categorize(req: CategorizeRequest): Promise<CategorizeResponse> {
    return this.request('/categorize', { method: 'POST', body: JSON.stringify(req) });
  }

  async confirmCategorization(req: ConfirmCategorizationRequest): Promise<void> {
    await this.request('/categorize/confirm', { method: 'POST', body: JSON.stringify(req) });
  }

  async search(req: SearchRequest): Promise<SearchResponse> {
    return this.request('/search', { method: 'POST', body: JSON.stringify(req) });
  }

  async chat(req: ChatRequest): Promise<ChatResponse> {
    return this.request('/chat', { method: 'POST', body: JSON.stringify(req) });
  }
}

export const coreClient = new CoreClient();
