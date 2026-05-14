import { z } from 'zod';
import type { LlmConfig } from '../config/schema.js';

const ModelListSchema = z.object({
  data: z.array(z.object({ id: z.string() })),
});

const EmbeddingsSchema = z.object({
  data: z.array(z.object({ embedding: z.array(z.number()) })),
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

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string };

export interface ChatCompletionOptions {
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  responseFormat?: 'text' | 'json_object';
  maxTokens?: number;
}

export interface LlmClient {
  health(): Promise<{ ok: boolean; models: string[] }>;
  embed(text: string, model?: string): Promise<number[]>;
  embedBatch(texts: string[], model?: string): Promise<number[][]>;
  chat(opts: ChatCompletionOptions): Promise<string>;
}

export function createLlmClient(config: LlmConfig): LlmClient {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (config.apiKey) headers['Authorization'] = `Bearer ${config.apiKey}`;

  const baseUrl = config.baseUrl.replace(/\/$/, '');

  async function request<T>(path: string, body?: unknown, schema?: z.ZodType<T>): Promise<T> {
    const res = await fetch(`${baseUrl}${path}`, {
      method: body ? 'POST' : 'GET',
      headers,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`LLM API ${res.status} ${path}: ${text}`);
    }
    const json = await res.json();
    return schema ? schema.parse(json) : (json as T);
  }

  return {
    async health() {
      try {
        const result = await request('/models', undefined, ModelListSchema);
        return { ok: true, models: result.data.map((m) => m.id) };
      } catch {
        return { ok: false, models: [] };
      }
    },

    async embed(text, model) {
      const result = await request(
        '/embeddings',
        { input: text, model: model ?? config.embeddingModel },
        EmbeddingsSchema,
      );
      const first = result.data[0];
      if (!first) throw new Error('Embedding API returned no data');
      return first.embedding;
    },

    async embedBatch(texts, model) {
      const result = await request(
        '/embeddings',
        { input: texts, model: model ?? config.embeddingModel },
        EmbeddingsSchema,
      );
      return result.data.map((d) => d.embedding);
    },

    async chat(opts) {
      const result = await request(
        '/chat/completions',
        {
          model: opts.model ?? config.generationModel,
          messages: opts.messages,
          temperature: opts.temperature ?? 0.2,
          ...(opts.responseFormat === 'json_object'
            ? { response_format: { type: 'json_object' } }
            : {}),
          ...(opts.maxTokens ? { max_tokens: opts.maxTokens } : {}),
        },
        ChatCompletionSchema,
      );
      const first = result.choices[0];
      if (!first) throw new Error('Chat API returned no choices');
      return first.message.content;
    },
  };
}
