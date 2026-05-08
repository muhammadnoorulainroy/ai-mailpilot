import { z } from 'zod';

export const ImapConfigSchema = z.object({
  host: z.string().default(''),
  port: z.number().int().positive().default(993),
  user: z.string().default(''),
  password: z.string().default(''),
  tls: z.boolean().default(true),
});

export const LlmConfigSchema = z.object({
  baseUrl: z.string().url().default('http://localhost:11434/v1'),
  apiKey: z.string().optional(),
  embeddingModel: z.string().default('bge-m3'),
  generationModel: z.string().default('mistral:7b'),
  embeddingDimensions: z.number().int().positive().default(1024),
});

export const AppConfigSchema = z.object({
  version: z.number().int().default(1),
  locale: z.enum(['en', 'fr']).default('en'),
  autoIndex: z.boolean().default(false),
  indexedFolders: z.array(z.string()).default([]),
  llm: LlmConfigSchema.default(() => LlmConfigSchema.parse({})),
  imap: ImapConfigSchema.optional(),
  authToken: z.string().optional(),
});

export type AppConfig = z.infer<typeof AppConfigSchema>;
export type LlmConfig = z.infer<typeof LlmConfigSchema>;
export type ImapConfig = z.infer<typeof ImapConfigSchema>;

export const DEFAULT_CONFIG: AppConfig = AppConfigSchema.parse({});
