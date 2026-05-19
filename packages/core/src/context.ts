import type { Database } from 'better-sqlite3';
import type { Logger } from 'pino';
import { loadConfig } from './config/config.js';
import type { AppConfig } from './config/schema.js';
import { openDatabase } from './db/database.js';
import { createLlmClient, type LlmClient } from './llm/client.js';
import { AccountRepository } from './repositories/account-repository.js';
import { EmailRepository } from './repositories/email-repository.js';
import { EmbeddingRepository } from './repositories/embedding-repository.js';
import { TriageRepository } from './repositories/triage-repository.js';
import { EmbeddingOrchestrator } from './services/embedding-orchestrator.js';
import { TriageService } from './services/triage-service.js';
import { TriageOrchestrator } from './services/triage-orchestrator.js';
import { getLogger } from './util/logger.js';

export interface Repositories {
  accounts: AccountRepository;
  emails: EmailRepository;
  embeddings: EmbeddingRepository;
  triage: TriageRepository;
}

export interface Services {
  embedding: EmbeddingOrchestrator;
  triage: TriageOrchestrator;
}

export interface AppContext {
  config: AppConfig;
  db: Database;
  llm: LlmClient;
  logger: Logger;
  repos: Repositories;
  services: Services;
}

export function buildContext(): AppContext {
  const logger = getLogger();
  const config = loadConfig();
  const db = openDatabase();
  const llm = createLlmClient(config.llm);

  const repos: Repositories = {
    accounts: new AccountRepository(db),
    emails: new EmailRepository(db),
    embeddings: new EmbeddingRepository(db),
    triage: new TriageRepository(db),
  };

  const triageService = new TriageService(llm, logger);

  const services: Services = {
    embedding: new EmbeddingOrchestrator(llm, repos.emails, repos.embeddings, logger),
    triage: new TriageOrchestrator(triageService, repos.triage, logger),
  };

  logger.info({ llmBaseUrl: config.llm.baseUrl }, 'context initialized');

  return { config, db, llm, logger, repos, services };
}
