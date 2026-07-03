/**
 * Application composition root that loads config, opens the database, and wires
 * every repository and service into a single shared AppContext.
 */
import type { Database } from 'better-sqlite3';
import type { Logger } from 'pino';
import { loadConfig } from './config/config.js';
import type { AppConfig } from './config/schema.js';
import { openDatabase } from './db/database.js';
import { createLlmClient, type LlmClient } from './llm/client.js';
import { AccountRepository } from './repositories/account-repository.js';
import { AttachmentRepository } from './repositories/attachment-repository.js';
import { CategoryRepository } from './repositories/category-repository.js';
import { CategoryAliasRepository } from './repositories/category-alias-repository.js';
import { CategoryProposalRepository } from './repositories/category-proposal-repository.js';
import { DiscoveryAuditRepository } from './repositories/discovery-audit-repository.js';
import { ConversationRepository } from './repositories/conversation-repository.js';
import { EmailRepository } from './repositories/email-repository.js';
import { EmailAssistantRepository } from './repositories/email-assistant-repository.js';
import { EmbeddingRepository } from './repositories/embedding-repository.js';
import { FailureRepository } from './repositories/failure-repository.js';
import { CategorizeJobRepository } from './repositories/categorize-job-repository.js';
import { TriageRepository } from './repositories/triage-repository.js';
import { AttachmentService } from './services/attachment-service.js';
import { CategorizationService } from './services/categorization-service.js';
import { CategoryImprovementService } from './services/category-improvement-service.js';
import { CategoryOrchestrator } from './services/category-orchestrator.js';
import { ChatService } from './services/chat-service.js';
import { CorrectionService } from './services/correction-service.js';
import { DashboardService } from './services/dashboard-service.js';
import { EmailAssistantService } from './services/email-assistant-service.js';
import { PriorityService } from './services/priority-service.js';
import { EmbeddingOrchestrator } from './services/embedding-orchestrator.js';
import { LlmCategorizer } from './services/llm-categorizer.js';
import { LlmCategorizeOrchestrator } from './services/llm-categorize-orchestrator.js';
import { TopicDiscoveryService } from './services/topic-discovery-service.js';
import { ResidualDiscoveryService } from './services/residual-discovery-service.js';
import { DiscoveryProposalService } from './services/discovery-proposal-service.js';
import { DiscoveryProposalOrchestrator } from './services/discovery-proposal-orchestrator.js';
import { CategoryCentroidRebuildService } from './services/category-centroid-rebuild-service.js';
import { CategoryHealthService } from './services/category-health-service.js';
import { StructuralProposalService } from './services/structural-proposal-service.js';
import { TriageOrchestrator } from './services/triage-orchestrator.js';
import { TriageService } from './services/triage-service.js';
import { getLogger } from './util/logger.js';

/**
 * Collection of all data-access repositories bound to a single database connection.
 */
export interface Repositories {
  accounts: AccountRepository;
  emails: EmailRepository;
  embeddings: EmbeddingRepository;
  triage: TriageRepository;
  categories: CategoryRepository;
  categoryAliases: CategoryAliasRepository;
  categoryProposals: CategoryProposalRepository;
  discoveryAudit: DiscoveryAuditRepository;
  conversations: ConversationRepository;
  attachments: AttachmentRepository;
  failures: FailureRepository;
  categorizeJobs: CategorizeJobRepository;
  emailAssistant: EmailAssistantRepository;
}

/**
 * Collection of all application services wired with their repositories and the LLM client.
 */
export interface Services {
  embedding: EmbeddingOrchestrator;
  triage: TriageOrchestrator;
  topicDiscovery: TopicDiscoveryService;
  categoryImprovement: CategoryImprovementService;
  discoveryProposal: DiscoveryProposalOrchestrator;
  categoryCentroidRebuild: CategoryCentroidRebuildService;
  categoryHealth: CategoryHealthService;
  structuralProposal: StructuralProposalService;
  category: CategoryOrchestrator;
  llmCategorize: LlmCategorizeOrchestrator;
  correction: CorrectionService;
  dashboard: DashboardService;
  priority: PriorityService;
  emailAssistant: EmailAssistantService;
  chat: ChatService;
  attachment: AttachmentService;
}

/**
 * Fully assembled application context holding config, database, LLM client, logger, repositories, and services.
 */
export interface AppContext {
  config: AppConfig;
  db: Database;
  llm: LlmClient;
  logger: Logger;
  repos: Repositories;
  services: Services;
}

/**
 * Builds the application context by loading config, opening the database, and wiring all repositories and services.
 */
export function buildContext(): AppContext {
  const logger = getLogger();
  const config = loadConfig();
  const db = openDatabase(undefined, logger);
  const llm = createLlmClient(() => config.llm);

  const repos: Repositories = {
    accounts: new AccountRepository(db),
    emails: new EmailRepository(db),
    embeddings: new EmbeddingRepository(db),
    triage: new TriageRepository(db),
    categories: new CategoryRepository(db),
    categoryAliases: new CategoryAliasRepository(db),
    categoryProposals: new CategoryProposalRepository(db),
    discoveryAudit: new DiscoveryAuditRepository(db),
    conversations: new ConversationRepository(db),
    attachments: new AttachmentRepository(db),
    failures: new FailureRepository(db),
    categorizeJobs: new CategorizeJobRepository(db),
    emailAssistant: new EmailAssistantRepository(db),
  };

  const triageService = new TriageService(llm, logger);
  const categorizationService = new CategorizationService(repos.categories, repos.embeddings);
  const llmCategorizer = new LlmCategorizer(llm);

  const residualDiscovery = new ResidualDiscoveryService(repos.embeddings, repos.categories);
  const discoveryProposalService = new DiscoveryProposalService(
    residualDiscovery,
    repos.emails,
    repos.categories,
    llm,
    () => config.llm,
    logger,
  );
  const categoryCentroidRebuild = new CategoryCentroidRebuildService(
    repos.categories,
    repos.embeddings,
    logger,
  );
  const categoryHealth = new CategoryHealthService(repos.categories, repos.embeddings);

  const services: Services = {
    embedding: new EmbeddingOrchestrator(
      llm,
      repos.emails,
      repos.embeddings,
      repos.failures,
      logger,
    ),
    triage: new TriageOrchestrator(triageService, repos.triage, repos.failures, logger),
    topicDiscovery: new TopicDiscoveryService(
      llm,
      repos.emails,
      repos.embeddings,
      repos.categories,
      logger,
      repos.accounts,
      repos.discoveryAudit,
      () => config.llm,
    ),
    categoryImprovement: new CategoryImprovementService(
      db,
      llm,
      repos.emails,
      repos.embeddings,
      repos.categories,
      logger,
      repos.accounts,
      repos.discoveryAudit,
      () => config.llm,
    ),
    discoveryProposal: new DiscoveryProposalOrchestrator(
      db,
      repos.categoryProposals,
      repos.categories,
      repos.emails,
      discoveryProposalService,
      repos.accounts,
      repos.discoveryAudit,
      categoryCentroidRebuild,
      () => config.llm,
      logger,
    ),
    categoryCentroidRebuild,
    categoryHealth,
    structuralProposal: new StructuralProposalService(
      repos.categories,
      repos.categoryProposals,
      categoryHealth,
      logger,
    ),
    category: new CategoryOrchestrator(
      categorizationService,
      repos.emails,
      repos.embeddings,
      repos.categories,
      logger,
    ),
    llmCategorize: new LlmCategorizeOrchestrator(
      llmCategorizer,
      repos.emails,
      repos.embeddings,
      repos.categories,
      repos.categorizeJobs,
      logger,
    ),
    correction: new CorrectionService(db, repos.categories, repos.embeddings),
    dashboard: new DashboardService(repos.emails, repos.triage, repos.categories),
    priority: new PriorityService(repos.triage),
    emailAssistant: new EmailAssistantService(
      llm,
      repos.accounts,
      repos.emails,
      repos.attachments,
      repos.emailAssistant,
      logger,
    ),
    chat: new ChatService(
      llm,
      repos.embeddings,
      repos.emails,
      repos.conversations,
      repos.attachments,
      logger,
    ),
    attachment: new AttachmentService(llm, repos.attachments, logger),
  };

  logger.info({ llmBaseUrl: config.llm.baseUrl }, 'context initialized');

  return { config, db, llm, logger, repos, services };
}
