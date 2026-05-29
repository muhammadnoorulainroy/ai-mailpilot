/**
 * Coordinates bulk embedding-based email categorization runs for an account and
 * exposes their live progress so callers can poll status.
 */
import type { Logger } from 'pino';
import type {
  CategoryAssignment,
  CategoryRepository,
} from '../repositories/category-repository.js';
import type { EmailRepository } from '../repositories/email-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import type { CategorizationService } from './categorization-service.js';

/** Snapshot of an in-flight or finished categorization run for a single account. */
export interface CategorizeProgress {
  status: 'idle' | 'running' | 'completed' | 'error';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  uncategorized: number;
  assigned: number;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

/** Drives bulk embedding-based categorization runs and tracks their progress. */
export class CategoryOrchestrator {
  private progress: CategorizeProgress = this.idle();
  private running = false;

  /** Creates the orchestrator with the repositories and services it coordinates. */
  constructor(
    private service: CategorizationService,
    private emails: EmailRepository,
    private embeddings: EmbeddingRepository,
    private categories: CategoryRepository,
    private logger: Logger,
  ) {}

  /** Returns a copy of the current run progress. */
  getProgress(): CategorizeProgress {
    return { ...this.progress };
  }

  /**
   * Starts a categorization run if one is not already in progress, returning the
   * number of eligible emails and whether a run was started.
   */
  start(
    accountId: string,
    embeddingModelId: string,
    opts: { force?: boolean } = {},
  ): { pending: number; started: boolean } {
    if (this.running) {
      return {
        pending: this.progress.total - this.progress.processed,
        started: false,
      };
    }

    const lockedIds = opts.force
      ? this.categories.getUserAssignedMessageIds(accountId)
      : this.categories.getAssignedMessageIds(accountId);
    const eligible = Math.max(
      0,
      this.embeddings.countForModel(accountId, embeddingModelId) - lockedIds.size,
    );
    if (eligible === 0) {
      this.progress = {
        ...this.idle(),
        status: 'completed',
        accountId,
        modelId: embeddingModelId,
        startedAt: Date.now(),
        completedAt: Date.now(),
      };
      return { pending: 0, started: false };
    }

    this.progress = {
      ...this.idle(),
      status: 'running',
      accountId,
      modelId: embeddingModelId,
      total: eligible,
      startedAt: Date.now(),
    };
    this.running = true;

    void this.runLoop(accountId, embeddingModelId, opts.force ?? false);

    return { pending: eligible, started: true };
  }

  /**
   * Runs the full batch categorization, staging auto-assignments and updating
   * progress. When force is set, existing auto-assignments are swapped rather
   * than appended.
   */
  private async runLoop(
    accountId: string,
    embeddingModelId: string,
    force: boolean,
  ): Promise<void> {
    const now = Date.now();
    const staged: CategoryAssignment[] = [];

    try {
      const lockedIds = force
        ? this.categories.getUserAssignedMessageIds(accountId)
        : this.categories.getAssignedMessageIds(accountId);
      const { matches: matchesByEmail, scored } = this.service.categorizeBatch(
        accountId,
        embeddingModelId,
        { fastGate: true },
        lockedIds,
      );

      for (const [messageId, matches] of matchesByEmail) {
        for (const m of matches) {
          staged.push({
            messageId,
            accountId,
            categoryId: m.categoryId,
            confidence: m.confidence,
            assignedBy: 'auto',
            assignedAt: now,
            method: 'embed',
          });
        }
        this.progress.assigned += matches.length;
      }

      this.progress.processed = scored;
      this.progress.uncategorized = scored - matchesByEmail.size;

      if (force) this.categories.swapAutoAssignments(accountId, staged);
      else this.categories.addAutoAssignments(accountId, staged);

      this.progress = {
        ...this.progress,
        status: 'completed',
        completedAt: Date.now(),
      };
      this.logger.info(
        {
          accountId,
          processed: this.progress.processed,
          assigned: this.progress.assigned,
          uncategorized: this.progress.uncategorized,
        },
        'categorization run complete',
      );
    } catch (err) {
      this.logger.error(
        { err, accountId },
        'categorization run failed; previous assignments preserved',
      );
      this.progress = {
        ...this.progress,
        status: 'error',
        error: err instanceof Error ? err.message : String(err),
        completedAt: Date.now(),
      };
    } finally {
      this.running = false;
    }
  }

  /** Builds a fresh progress snapshot representing an idle, unstarted run. */
  private idle(): CategorizeProgress {
    return {
      status: 'idle',
      accountId: null,
      modelId: null,
      total: 0,
      processed: 0,
      uncategorized: 0,
      assigned: 0,
    };
  }
}
