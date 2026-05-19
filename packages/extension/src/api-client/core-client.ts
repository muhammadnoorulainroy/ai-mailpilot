import { CORE_SERVER_URL, API_PREFIX } from '@ai-mailpilot/shared';
import type {
  HealthResponse,
  AccountListResponse,
  AccountResponse,
  CreateAccountRequest,
  PushEmailsRequest,
  PushEmailsResponse,
} from '@ai-mailpilot/shared';

const TOKEN_KEY = 'core_auth_token';

export class CoreClient {
  private baseUrl: string;
  private authToken: string | null = null;

  constructor(baseUrl: string = `${CORE_SERVER_URL}${API_PREFIX}`) {
    this.baseUrl = baseUrl;
  }

  async loadToken(): Promise<void> {
    const stored = (await browser.storage.local.get(TOKEN_KEY)) as Record<string, unknown>;
    if (typeof stored[TOKEN_KEY] === 'string') {
      this.authToken = stored[TOKEN_KEY];
    }
  }

  async setToken(token: string): Promise<void> {
    this.authToken = token;
    await browser.storage.local.set({ [TOKEN_KEY]: token });
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    if (!this.authToken) await this.loadToken();

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.authToken) headers['Authorization'] = `Bearer ${this.authToken}`;

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers: { ...headers, ...(options?.headers as Record<string, string> | undefined) },
    });

    if (response.status === 401) {
      await this.loadToken();
      throw new Error(
        `Core API 401 ${path}: token rejected. Run mailpilotSetToken("<token-from-config.json>")`,
      );
    }

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Core API ${response.status} ${path}: ${error}`);
    }

    if (response.status === 204) return undefined as T;
    return response.json() as Promise<T>;
  }

  health(): Promise<HealthResponse> {
    return this.request('/health');
  }

  listAccounts(): Promise<AccountListResponse> {
    return this.request('/accounts');
  }

  createAccount(req: CreateAccountRequest): Promise<AccountResponse> {
    return this.request('/accounts', { method: 'POST', body: JSON.stringify(req) });
  }

  getAccount(id: string): Promise<AccountResponse> {
    return this.request(`/accounts/${id}`);
  }

  pushEmails(req: PushEmailsRequest): Promise<PushEmailsResponse> {
    return this.request('/emails/push', { method: 'POST', body: JSON.stringify(req) });
  }
}

export const coreClient = new CoreClient();
