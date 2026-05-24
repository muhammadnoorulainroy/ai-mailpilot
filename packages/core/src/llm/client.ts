/**
 * HTTP client for an OpenAI-compatible LLM endpoint, covering health checks,
 * embeddings, and chat completions with streaming and rate-limit retry handling.
 */
import { z } from 'zod';
import type { LlmConfig } from '../config/schema.js';

/** LLM HTTP error carrying the response status so callers can separate a transient hiccup from a permanent misconfiguration. */
export class LlmApiError extends Error {
  /** Creates the error carrying the HTTP status code and the provider response body. */
  constructor(
    readonly status: number,
    readonly path: string,
    readonly body: string,
  ) {
    super(`LLM API ${status} ${path}: ${body}`);
    this.name = 'LlmApiError';
  }

  /** True for 4xx errors that retrying will not fix, excluding 408 and 429 which can clear on their own. */
  get nonRetryable(): boolean {
    return this.status >= 400 && this.status < 500 && this.status !== 408 && this.status !== 429;
  }
}

const MAX_RATE_LIMIT_RETRIES = 6;

/** Resolve after the given number of milliseconds. */
const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * How long to wait before retrying a 429. Honors the delay the provider suggests in the body
 * ("try again in 1.095s" or "98ms"), falling back to exponential backoff. Padded and clamped so a
 * retry never fires a hair too early or stalls the run too long.
 */
export function rateLimitDelayMs(body: string, attempt: number): number {
  const match = body.match(/try again in ([\d.]+)\s*(ms|s)\b/i);
  if (match) {
    const value = Number.parseFloat(match[1]!);
    const ms = match[2]!.toLowerCase() === 's' ? value * 1000 : value;
    return Math.min(Math.max(Math.ceil(ms) + 250, 500), 30_000);
  }
  return Math.min(1000 * 2 ** attempt, 30_000);
}

const ModelListSchema = z.object({
  data: z.array(z.object({ id: z.string() })),
});

const EmbeddingsSchema = z.object({
  data: z.array(z.object({ embedding: z.array(z.number()), index: z.number().int().optional() })),
  model: z.string().optional(),
});

const ChatCompletionSchema = z.object({
  choices: z.array(
    z.object({
      message: z.object({
        role: z.string(),
        content: z.string(),
      }),
      finish_reason: z.string().nullable().optional(),
    }),
  ),
  model: z.string().optional(),
  usage: z
    .object({
      prompt_tokens: z.number().optional(),
      completion_tokens: z.number().optional(),
      total_tokens: z.number().optional(),
    })
    .optional(),
});

/** A single chat message with its role and text content. */
export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

/** Options for a chat completion request. */
export interface ChatCompletionOptions {
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  responseFormat?: 'text' | 'json_object';
  maxTokens?: number;
  /** Override the request timeout. Bulk callers pass a larger value for slow local models under load. */
  timeoutMs?: number;
  /** Ollama reasoning toggle. False asks a thinking model to answer directly, ignored by servers that lack it. */
  think?: boolean;
  /** Which endpoint to call. 'chat' prefers the cloud chat provider when configured, 'main' uses the local baseUrl. Defaults to 'chat'. */
  provider?: 'main' | 'chat';
}

/** Client for embedding and chat completion calls against an OpenAI-compatible LLM endpoint. */
export interface LlmClient {
  health(): Promise<{ ok: boolean; models: string[] }>;
  embed(text: string, model?: string): Promise<number[]>;
  embedBatch(texts: string[], model?: string): Promise<number[][]>;
  chat(opts: ChatCompletionOptions): Promise<string>;
  chatStream(opts: ChatCompletionOptions): AsyncIterable<string>;
}

const DEFAULT_TIMEOUT_MS = 120_000;
const HEALTH_TIMEOUT_MS = 5_000;
const MAX_ERROR_TEXT = 2_000;

/**
 * Create an LLM client that reads current config on every call so a config
 * PATCH takes effect immediately without rebuilding the client.
 */
export function createLlmClient(getConfig: () => LlmConfig): LlmClient {
  /** Resolve the base URL and auth headers for the chosen endpoint, preferring the cloud chat provider only when it is configured. */
  function endpointFor(kind: 'main' | 'chat'): {
    baseUrl: string;
    headers: Record<string, string>;
  } {
    const config = getConfig();
    const useChat = kind === 'chat' && !!config.chatBaseUrl;
    const baseUrl = (useChat ? config.chatBaseUrl! : config.baseUrl).replace(/\/$/, '');
    const apiKey = useChat ? config.chatApiKey : config.apiKey;
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (apiKey) headers['Authorization'] = `Bearer ${apiKey}`;
    return { baseUrl, headers };
  }

  /** Issue a request, retrying on 429 with the provider-suggested delay up to the retry cap. */
  async function request<T>(
    path: string,
    body?: unknown,
    schema?: z.ZodType<T>,
    timeoutMs: number = DEFAULT_TIMEOUT_MS,
    kind: 'main' | 'chat' = 'main',
  ): Promise<T> {
    for (let attempt = 0; ; attempt++) {
      try {
        return await requestOnce<T>(path, body, schema, timeoutMs, kind);
      } catch (err) {
        if (err instanceof LlmApiError && err.status === 429 && attempt < MAX_RATE_LIMIT_RETRIES) {
          await sleep(rateLimitDelayMs(err.body, attempt));
          continue;
        }
        throw err;
      }
    }
  }

  /** Perform a single fetch with a timeout, mapping aborts to timeout errors and validating the response against the schema. */
  async function requestOnce<T>(
    path: string,
    body?: unknown,
    schema?: z.ZodType<T>,
    timeoutMs: number = DEFAULT_TIMEOUT_MS,
    kind: 'main' | 'chat' = 'main',
  ): Promise<T> {
    const { baseUrl, headers } = endpointFor(kind);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    /** True when the error is an abort raised by the timeout controller. */
    const isAbort = (err: unknown): boolean => err instanceof Error && err.name === 'AbortError';
    try {
      let res: Response;
      try {
        res = await fetch(`${baseUrl}${path}`, {
          method: body ? 'POST' : 'GET',
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        });
      } catch (err) {
        if (isAbort(err)) throw new Error(`LLM API ${path}: timed out after ${timeoutMs}ms`);
        throw err;
      }

      if (!res.ok) {
        const text = (await res.text()).slice(0, MAX_ERROR_TEXT);
        throw new LlmApiError(res.status, path, text);
      }

      let json: unknown;
      try {
        json = await res.json();
      } catch (err) {
        if (isAbort(err)) throw new Error(`LLM API ${path}: timed out after ${timeoutMs}ms`);
        throw new Error(`LLM API ${path}: response was not valid JSON`);
      }
      return schema ? schema.parse(json) : (json as T);
    } finally {
      clearTimeout(timer);
    }
  }

  return {
    /** Probe the endpoint, returning ok plus available model ids, or ok false on any failure. */
    async health() {
      try {
        const result = await request('/models', undefined, ModelListSchema, HEALTH_TIMEOUT_MS);
        return { ok: true, models: result.data.map((m) => m.id) };
      } catch {
        return { ok: false, models: [] };
      }
    },

    /** Return the embedding vector for a single text, defaulting to the configured embedding model. */
    async embed(text, model) {
      const result = await request(
        '/embeddings',
        { input: text, model: model ?? getConfig().embeddingModel },
        EmbeddingsSchema,
      );
      const first = result.data[0];
      if (!first) throw new Error('Embedding API returned no data');
      return first.embedding;
    },

    /** Embed many texts at once, reordering results by their returned index so output aligns with the input order. */
    async embedBatch(texts, model) {
      const result = await request(
        '/embeddings',
        { input: texts, model: model ?? getConfig().embeddingModel },
        EmbeddingsSchema,
      );
      const ordered = result.data.every((d) => typeof d.index === 'number')
        ? [...result.data].sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
        : result.data;
      return ordered.map((d) => d.embedding);
    },

    /** Run a non-streaming chat completion and return the first choice's content. */
    async chat(opts) {
      const result = await request(
        '/chat/completions',
        {
          model: opts.model ?? getConfig().generationModel,
          messages: opts.messages,
          temperature: opts.temperature ?? 0.2,
          ...(opts.responseFormat === 'json_object'
            ? { response_format: { type: 'json_object' } }
            : {}),
          ...(opts.maxTokens ? { max_tokens: opts.maxTokens } : {}),
          ...(opts.think !== undefined ? { think: opts.think } : {}),
        },
        ChatCompletionSchema,
        opts.timeoutMs ?? DEFAULT_TIMEOUT_MS,
        opts.provider ?? 'chat',
      );
      const first = result.choices[0];
      if (!first) throw new Error('Chat API returned no choices');
      return first.message.content;
    },

    /** Stream a chat completion, yielding content deltas as server-sent events arrive until the stream ends or times out. */
    async *chatStream(opts) {
      const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      const { baseUrl, headers } = endpointFor(opts.provider ?? 'chat');
      try {
        const res = await fetch(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            model: opts.model ?? getConfig().generationModel,
            messages: opts.messages,
            temperature: opts.temperature ?? 0.2,
            stream: true,
            ...(opts.maxTokens ? { max_tokens: opts.maxTokens } : {}),
            ...(opts.think !== undefined ? { think: opts.think } : {}),
          }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const text = (await res.text()).slice(0, MAX_ERROR_TEXT);
          throw new LlmApiError(res.status, '/chat/completions', text);
        }
        if (!res.body) throw new Error('LLM API /chat/completions: no stream body');

        /** Parse one SSE line into a done flag or a content delta, ignoring non-data and unparseable lines. */
        const parseSse = (line: string): { done?: boolean; delta?: string } => {
          const t = line.trim();
          if (!t.startsWith('data:')) return {};
          const data = t.slice(5).trim();
          if (data === '[DONE]') return { done: true };
          try {
            const json = JSON.parse(data) as { choices?: Array<{ delta?: { content?: string } }> };
            const d = json.choices?.[0]?.delta?.content;
            if (typeof d === 'string' && d.length > 0) return { delta: d };
          } catch {}
          return {};
        };

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
          let done = false;
          let value: Uint8Array | undefined;
          try {
            ({ done, value } = await reader.read());
          } catch (err) {
            if (err instanceof Error && err.name === 'AbortError') {
              throw new Error(`LLM API /chat/completions: timed out after ${timeoutMs}ms`);
            }
            throw err;
          }
          buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });

          let nl: number;
          while ((nl = buffer.indexOf('\n')) !== -1) {
            const result = parseSse(buffer.slice(0, nl));
            buffer = buffer.slice(nl + 1);
            if (result.done) return;
            if (result.delta) yield result.delta;
          }
          if (done) {
            const result = parseSse(buffer);
            if (result.delta) yield result.delta;
            break;
          }
        }
      } finally {
        clearTimeout(timer);
        controller.abort();
      }
    },
  };
}
