/**
 * Zod schemas and inferred types for application configuration, covering IMAP,
 * LLM provider, and top-level app settings, plus the default config object.
 */
import { z } from 'zod';

/** Zod schema for IMAP connection settings. */
export const ImapConfigSchema = z.object({
  host: z.string().default(''),
  port: z.number().int().positive().default(993),
  user: z.string().default(''),
  password: z.string().default(''),
  tls: z.boolean().default(true),
});

/** Zod schema for LLM provider settings covering embeddings, generation, and chat. */
export const LlmConfigSchema = z.object({
  baseUrl: z.string().url().default('http://localhost:11434/v1'),
  apiKey: z.string().optional(),
  embeddingModel: z.string().default('bge-m3'),
  generationModel: z.string().default('qwen3:8b'),
  chatModel: z.string().optional(),
  embeddingDimensions: z.number().int().positive().default(1024),
  chatTopK: z.number().int().min(1).max(50).nullable().optional(),
  chatSnippetChars: z.number().int().min(200).max(8000).nullable().optional(),
  chatRerank: z.boolean().default(true),
  chatBaseUrl: z.string().url().nullable().optional(),
  chatApiKey: z.string().nullable().optional(),
  categorizeUseChatProvider: z.boolean().default(false),
  priorityUseChatProvider: z.boolean().default(false),
});

/** Zod schema for the full application configuration. */
export const AppConfigSchema = z.object({
  version: z.number().int().default(1),
  locale: z.enum(['en', 'fr']).default('en'),
  autoIndex: z.boolean().default(false),
  indexedFolders: z.array(z.string()).default([]),
  llm: LlmConfigSchema.default(() => LlmConfigSchema.parse({})),
  imap: ImapConfigSchema.optional(),
  authToken: z.string().optional(),
});

/** Application configuration object inferred from AppConfigSchema. */
export type AppConfig = z.infer<typeof AppConfigSchema>;
/** LLM configuration object inferred from LlmConfigSchema. */
export type LlmConfig = z.infer<typeof LlmConfigSchema>;
/** IMAP configuration object inferred from ImapConfigSchema. */
export type ImapConfig = z.infer<typeof ImapConfigSchema>;

/** Default application configuration produced by parsing an empty object. */
export const DEFAULT_CONFIG: AppConfig = AppConfigSchema.parse({});
