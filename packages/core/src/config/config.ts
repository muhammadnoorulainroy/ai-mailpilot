import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { randomBytes } from 'node:crypto';
import { CONFIG_PATH, ensureDirs } from '../util/paths.js';
import { AppConfigSchema, LlmConfigSchema, type AppConfig } from './schema.js';

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

export function saveConfig(config: AppConfig): void {
  ensureDirs();
  const validated = AppConfigSchema.parse(config);
  writeFileSync(CONFIG_PATH, JSON.stringify(validated, null, 2), 'utf8');
}

function generateToken(): string {
  return randomBytes(32).toString('hex');
}
