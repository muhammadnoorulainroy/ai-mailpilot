/**
 * Orchestrates batched, concurrent embedding of unembedded emails, persisting
 * vectors and tracking per-run progress and permanent failures.
 */
import type { Logger } from 'pino';
import type { LlmClient } from '../llm/client.js';
import type { EmailRepository, EmailRow } from '../repositories/email-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import type { FailureRepository } from '../repositories/failure-repository.js';
import { buildEmbeddingInput } from '../util/text.js';

/** Snapshot of an embedding run's lifecycle state and per-email counts. */
export interface EmbeddingProgress {
  status: 'idle' | 'running' | 'completed' | 'completed_with_failures' | 'error';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  failed: number;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

const BATCH_SIZE =
  Number.parseInt(process.env.MAILPILOT_EMBED_BATCH ?? '', 10) > 0
    ? Number.parseInt(process.env.MAILPILOT_EMBED_BATCH ?? '', 10)
    : 32;
const CONCURRENCY =
  Number.parseInt(process.env.MAILPILOT_EMBED_CONCURRENCY ?? '', 10) > 0
    ? Number.parseInt(process.env.MAILPILOT_EMBED_CONCURRENCY ?? '', 10)
    : 3;
const MAX_RETRIES_PER_EMAIL = 1;

/** Drives batched, concurrent embedding of unembedded emails and tracks run progress. */
export class EmbeddingOrchestrator {
  private progress: EmbeddingProgress = this.idle();
  private running = false;

  /** Creates the orchestrator with the embedding client and repositories. */
  constructor(
    private llm: LlmClient,
    private emails: EmailRepository,
    private embeddings: EmbeddingRepository,
    private failures: FailureRepository,
    private logger: Logger,
  ) {}

  /** Returns a copy of the current run progress snapshot. */
  getProgress(): EmbeddingProgress {
    return { ...this.progress };
  }

  /** Reports whether an embedding run is currently in progress. */
  isRunning(): boolean {
    return this.running;
  }

  /**
   * Starts an embedding run for the account and model, skipping permanently failed emails.
   * No-op if a run is already active or nothing is pending.
   */
  start(accountId: string, modelId: string): { pending: number; started: boolean } {
    if (this.running) {
      return { pending: this.progress.total - this.progress.processed, started: false };
    }

    const pending =
      this.emails.countUnembedded(accountId, modelId) -
      this.failures.countPermanentlyFailed(accountId, 'embedding', modelId);
    if (pending <= 0) {
      this.progress = {
        ...this.idle(),
        status: 'completed',
        accountId,
        modelId,
        startedAt: Date.now(),
        completedAt: Date.now(),
      };
      return { pending: 0, started: false };
    }

    this.progress = {
      ...this.idle(),
      status: 'running',
      accountId,
      modelId,
      total: pending,
      startedAt: Date.now(),
    };

    this.running = true;
    void this.runLoop(accountId, modelId);

    return { pending, started: true };
  }

  /**
   * Runs concurrent workers that pull and embed pending batches until none
   * remain, then finalizes the progress status. Skips ids already failed or in flight.
   */
  private async runLoop(accountId: string, modelId: string): Promise<void> {
    const failedThisRun = new Set<string>(
      this.failures.permanentlyFailedIds(accountId, 'embedding', modelId),
    );
    const retryCounts = new Map<string, number>();
    const inFlight = new Set<string>();

    const worker = async (): Promise<void> => {
      while (true) {
        const skip = new Set(failedThisRun);
        for (const id of inFlight) skip.add(id);
        const batch = this.emails.findUnembedded(accountId, modelId, BATCH_SIZE, skip);
        if (batch.length === 0) break;

        for (const e of batch) inFlight.add(e.messageId);
        try {
          await this.processBatch(batch, modelId, failedThisRun, retryCounts);
        } finally {
          for (const e of batch) inFlight.delete(e.messageId);
        }
      }
    };

    try {
      await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

      this.progress = {
        ...this.progress,
        status: this.progress.failed > 0 ? 'completed_with_failures' : 'completed',
        completedAt: Date.now(),
      };
      this.logger.info(
        {
          accountId,
          modelId,
          processed: this.progress.processed,
          failed: this.progress.failed,
          concurrency: CONCURRENCY,
        },
        'embedding run complete',
      );
    } catch (err) {
      this.logger.error({ err, accountId, modelId }, 'embedding run failed');
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

  /**
   * Embeds a batch in one call and saves each vector, falling back to embedding
   * emails one at a time if the batch call fails or returns a mismatched count.
   */
  private async processBatch(
    batch: EmailRow[],
    modelId: string,
    failedThisRun: Set<string>,
    retryCounts: Map<string, number>,
  ): Promise<void> {
    const inputs = batch.map((email) =>
      buildEmbeddingInput({
        subject: email.subject,
        fromAddr: email.fromAddr,
        body: email.body,
        bodyFormat: email.bodyFormat,
      }),
    );

    try {
      const vectors = await this.llm.embedBatch(inputs, modelId);
      if (vectors.length !== batch.length) {
        throw new Error(
          `embedding count mismatch: requested ${batch.length}, got ${vectors.length}`,
        );
      }

      for (let i = 0; i < batch.length; i++) {
        const email = batch[i];
        const vector = vectors[i];
        if (!email || !vector) continue;
        try {
          this.embeddings.saveEmbedding(
            { messageId: email.messageId, accountId: email.accountId, modelId },
            vector,
          );
          this.failures.clearFailure(email.messageId, email.accountId, 'embedding', modelId);
          this.progress.processed += 1;
        } catch (err) {
          this.recordFailure(email, err, modelId, failedThisRun, retryCounts);
        }
      }
    } catch (batchErr) {
      this.logger.warn(
        { err: batchErr, size: batch.length },
        'batch embedding failed, retrying singly',
      );
      for (const email of batch) {
        try {
          const input = buildEmbeddingInput({
            subject: email.subject,
            fromAddr: email.fromAddr,
            body: email.body,
            bodyFormat: email.bodyFormat,
          });
          const vector = await this.llm.embed(input, modelId);
          this.embeddings.saveEmbedding(
            { messageId: email.messageId, accountId: email.accountId, modelId },
            vector,
          );
          this.failures.clearFailure(email.messageId, email.accountId, 'embedding', modelId);
          this.progress.processed += 1;
        } catch (singleErr) {
          this.recordFailure(email, singleErr, modelId, failedThisRun, retryCounts);
        }
      }
    }
  }

  /**
   * Tracks a failed attempt for an email, retrying within the run until the
   * retry cap is hit, after which it marks the email failed and persists the failure.
   */
  private recordFailure(
    email: EmailRow,
    err: unknown,
    modelId: string,
    failedThisRun: Set<string>,
    retryCounts: Map<string, number>,
  ): void {
    const attempts = (retryCounts.get(email.messageId) ?? 0) + 1;
    retryCounts.set(email.messageId, attempts);

    if (attempts > MAX_RETRIES_PER_EMAIL) {
      failedThisRun.add(email.messageId);
      this.progress.failed += 1;
      const persisted = this.failures.recordFailure(
        email.messageId,
        email.accountId,
        'embedding',
        modelId,
        err instanceof Error ? err.message : String(err),
      );
      this.logger.error(
        { err, messageId: email.messageId, attempts, persistedFailures: persisted },
        'embedding failed for this run; recorded',
      );
    } else {
      this.logger.warn(
        { err, messageId: email.messageId, attempts },
        'embedding failed, will retry within run',
      );
    }
  }

  /** Returns a fresh idle progress snapshot with zeroed counts. */
  private idle(): EmbeddingProgress {
    return {
      status: 'idle',
      accountId: null,
      modelId: null,
      total: 0,
      processed: 0,
      failed: 0,
    };
  }
}
