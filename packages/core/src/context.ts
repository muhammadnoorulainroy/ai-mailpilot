import type { Database } from 'better-sqlite3';
import type { Logger } from 'pino';
import { loadConfig } from './config/config.js';
import type { AppConfig } from './config/schema.js';
import { openDatabase } from './db/database.js';
import { createLlmClient, type LlmClient } from './llm/client.js';
import { getLogger } from './util/logger.js';

export interface AppContext {
  config: AppConfig;
  db: Database;
  llm: LlmClient;
  logger: Logger;
}

export function buildContext(): AppContext {
  const logger = getLogger();
  const config = loadConfig();
  const db = openDatabase();
  const llm = createLlmClient(config.llm);

  logger.info({ llmBaseUrl: config.llm.baseUrl }, 'context initialized');

  return { config, db, llm, logger };
}
