/**
 * Loading, validating, persisting, and redacting the app config on disk,
 * including auth token generation and stripping secrets for API responses.
 */
import { readFileSync, writeFileSync, existsSync, chmodSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import { CONFIG_PATH, ensureDirs } from '../util/paths.js';
import { AppConfigSchema, LlmConfigSchema, type AppConfig } from './schema.js';

/**
 * Load the app config from disk, creating a fresh one with an auth token if absent.
 */
export function loadConfig(): AppConfig {
  ensureDirs();

  if (!existsSync(CONFIG_PATH)) {
    const fresh = AppConfigSchema.parse({ authToken: generateToken() });
    saveConfig(fresh);
    return fresh;
  }

  const raw = JSON.parse(readFileSync(CONFIG_PATH, 'utf8')) as Record<string, unknown>;

  if (raw.llm && typeof raw.llm === 'object') {
    raw.llm = LlmConfigSchema.parse(raw.llm);
  }

  const parsed = AppConfigSchema.parse(raw);

  if (!parsed.authToken) {
    parsed.authToken = generateToken();
    saveConfig(parsed);
  }

  return parsed;
}

/**
 * Validate and write the config to disk with owner-only file permissions.
 */
export function saveConfig(config: AppConfig): void {
  ensureDirs();
  const validated = AppConfigSchema.parse(config);
  writeFileSync(CONFIG_PATH, JSON.stringify(validated, null, 2), { encoding: 'utf8', mode: 0o600 });
  try {
    chmodSync(CONFIG_PATH, 0o600);
  } catch {}
}

/**
 * Config shape safe to return over the API, with all secrets stripped.
 */
export interface SafeConfig {
  version: number;
  locale: AppConfig['locale'];
  autoIndex: boolean;
  indexedFolders: string[];
  llm: Omit<AppConfig['llm'], 'apiKey' | 'chatApiKey'> & { chatApiKeySet: boolean };
}

/**
 * Strip every secret before a config is returned over the API. Allowlist-shaped
 * so a newly added secret field is not leaked by default.
 */
export function redactConfig(config: AppConfig): SafeConfig {
  return {
    version: config.version,
    locale: config.locale,
    autoIndex: config.autoIndex,
    indexedFolders: config.indexedFolders,
    llm: {
      baseUrl: config.llm.baseUrl,
      embeddingModel: config.llm.embeddingModel,
      generationModel: config.llm.generationModel,
      chatModel: config.llm.chatModel,
      embeddingDimensions: config.llm.embeddingDimensions,
      chatTopK: config.llm.chatTopK,
      chatSnippetChars: config.llm.chatSnippetChars,
      chatRerank: config.llm.chatRerank,
      chatBaseUrl: config.llm.chatBaseUrl,
      categorizeUseChatProvider: config.llm.categorizeUseChatProvider,
      priorityUseChatProvider: config.llm.priorityUseChatProvider,
      chatApiKeySet: !!config.llm.chatApiKey,
    },
  };
}

/**
 * Generate a random hex auth token for API access.
 */
function generateToken(): string {
  return randomBytes(32).toString('hex');
}
