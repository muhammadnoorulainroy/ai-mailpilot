/**
 * Orchestrates a triage run, classifying pending emails in batches with bounded
 * concurrency while tracking progress, retries, and persisted failures.
 */
import type { Logger } from 'pino';
import type { TriageRepository, UnclassifiedEmail } from '../repositories/triage-repository.js';
import type { FailureRepository } from '../repositories/failure-repository.js';
import type { TriageProvider, TriageResult, TriageService } from './triage-service.js';

/**
 * Snapshot of a triage run's status, counts, and per-bucket tallies.
 */
export interface TriageProgress {
  status: 'idle' | 'running' | 'completed' | 'completed_with_failures' | 'error';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  failed: number;
  buckets: { urgent: number; summarize: number; spam: number; personal: number };
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

/**
 * Reads a positive integer from an environment variable, returning the fallback
 * when unset or not a positive number.
 */
function envInt(name: string, fallback: number): number {
  const n = Number.parseInt(process.env[name] ?? '', 10);
  return n > 0 ? n : fallback;
}

const LOCAL_CONCURRENCY = envInt('MAILPILOT_TRIAGE_CONCURRENCY', 2);
const CLOUD_CONCURRENCY = envInt('MAILPILOT_TRIAGE_CLOUD_CONCURRENCY', 8);
const LOCAL_LLM_BATCH_SIZE = envInt('MAILPILOT_TRIAGE_BATCH', 4);
const CLOUD_LLM_BATCH_SIZE = envInt('MAILPILOT_TRIAGE_CLOUD_BATCH', 12);
const MAX_RETRIES_PER_EMAIL = 1;

/**
 * Picks the worker concurrency for the provider, higher for cloud than local.
 */
function runConcurrency(provider: TriageProvider): number {
  return provider === 'chat' ? CLOUD_CONCURRENCY : LOCAL_CONCURRENCY;
}

/**
 * Picks the per-LLM-call batch size for the provider, larger for cloud than local.
 */
function runLlmBatchSize(provider: TriageProvider): number {
  return provider === 'chat' ? CLOUD_LLM_BATCH_SIZE : LOCAL_LLM_BATCH_SIZE;
}

/**
 * Drives a triage run, classifying pending emails in batches with bounded
 * concurrency and tracking progress, retries, and failures.
 */
export class TriageOrchestrator {
  private progress: TriageProgress = this.idle();
  private running = false;

  /** Creates the orchestrator with the triage service, repositories, and logger. */
  constructor(
    private service: TriageService,
    private triage: TriageRepository,
    private failures: FailureRepository,
    private logger: Logger,
  ) {}

  /**
   * Returns a defensive copy of the current triage progress.
   */
  getProgress(): TriageProgress {
    return { ...this.progress, buckets: { ...this.progress.buckets } };
  }

  /**
   * Begins a triage run for the account, returning the pending count and
   * whether a run started. A run already in progress is not restarted.
   */
  start(
    accountId: string,
    modelId: string,
    opts: { force?: boolean; provider?: TriageProvider } = {},
  ): { pending: number; started: boolean } {
    if (this.running) {
      return {
        pending: this.progress.total - this.progress.processed,
        started: false,
      };
    }

    if (opts.force) {
      this.failures.clearForAccount(accountId, 'triage');
    }

    const failedIds = opts.force
      ? new Set<string>()
      : new Set<string>(this.failures.permanentlyFailedIds(accountId, 'triage', modelId));
    const pending = this.triage.countPendingTriage(accountId, opts.force === true, failedIds);
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

    void this.runLoop(accountId, modelId, opts.force === true, opts.provider ?? 'main');

    return { pending, started: true };
  }

  /**
   * Fetches and classifies pending emails in fetch-sized batches until none
   * remain, then records the final completion or error status.
   */
  private async runLoop(
    accountId: string,
    modelId: string,
    force: boolean,
    provider: TriageProvider,
  ): Promise<void> {
    const failedThisRun = new Set<string>(
      this.failures.permanentlyFailedIds(accountId, 'triage', modelId),
    );
    const retryCounts = new Map<string, number>();
    const concurrency = runConcurrency(provider);
    const llmBatchSize = runLlmBatchSize(provider);
    const fetchBatchSize = Math.max(16, concurrency * llmBatchSize * 2);
    const reclassifyBefore = Date.now();

    try {
      while (true) {
        const batch = this.triage.findPendingTriageEmails(
          accountId,
          fetchBatchSize,
          failedThisRun,
          force,
          reclassifyBefore,
        );
        if (batch.length === 0) break;

        await this.processBatchConcurrent(
          batch,
          accountId,
          modelId,
          provider,
          failedThisRun,
          retryCounts,
          concurrency,
          llmBatchSize,
        );
      }

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
          buckets: this.progress.buckets,
          provider,
          concurrency,
          llmBatchSize,
        },
        'triage run complete',
      );
    } catch (err) {
      this.logger.error({ err, accountId, modelId }, 'triage run failed');
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
   * Classify the batch with bounded concurrency, at most CONCURRENCY in flight at once.
   */
  private async processBatchConcurrent(
    batch: UnclassifiedEmail[],
    accountId: string,
    modelId: string,
    provider: TriageProvider,
    failedThisRun: Set<string>,
    retryCounts: Map<string, number>,
    concurrency: number,
    llmBatchSize: number,
  ): Promise<void> {
    const chunks: UnclassifiedEmail[][] = [];
    for (let i = 0; i < batch.length; i += llmBatchSize) {
      chunks.push(batch.slice(i, i + llmBatchSize));
    }

    let cursor = 0;
    const workers: Promise<void>[] = [];

    /** Worker pulling chunks from the shared cursor until all are claimed. */
    const run = async (): Promise<void> => {
      while (cursor < chunks.length) {
        const i = cursor++;
        const chunk = chunks[i]!;
        await this.classifyChunk(chunk, accountId, modelId, provider, failedThisRun, retryCounts);
      }
    };

    for (let i = 0; i < Math.min(concurrency, chunks.length); i++) {
      workers.push(run());
    }
    await Promise.all(workers);
  }

  /**
   * Classifies one chunk as a batch, falling back to per-email classification
   * if the batch call fails or omits any email.
   */
  private async classifyChunk(
    emails: UnclassifiedEmail[],
    accountId: string,
    modelId: string,
    provider: TriageProvider,
    failedThisRun: Set<string>,
    retryCounts: Map<string, number>,
  ): Promise<void> {
    if (emails.length <= 1) {
      const email = emails[0];
      if (email)
        await this.classifyOne(email, accountId, modelId, provider, failedThisRun, retryCounts);
      return;
    }

    try {
      const results = await this.service.classifyBatch(emails, modelId, provider);
      for (const email of emails) {
        const result = results.get(email.messageId);
        if (!result) throw new Error(`triage batch omitted ${email.messageId}`);
      }
      for (const email of emails) {
        this.persistResult(email, accountId, modelId, results.get(email.messageId)!);
      }
    } catch (err) {
      this.logger.warn(
        { err, count: emails.length, provider },
        'batched triage failed; falling back to single-email classification',
      );
      for (const email of emails) {
        await this.classifyOne(email, accountId, modelId, provider, failedThisRun, retryCounts);
      }
    }
  }

  /**
   * Classifies a single email, retrying up to MAX_RETRIES_PER_EMAIL and
   * recording a persisted failure once retries are exhausted.
   */
  private async classifyOne(
    email: UnclassifiedEmail,
    accountId: string,
    modelId: string,
    provider: TriageProvider,
    failedThisRun: Set<string>,
    retryCounts: Map<string, number>,
  ): Promise<void> {
    try {
      const result = await this.service.classify(email, modelId, provider);
      this.persistResult(email, accountId, modelId, result);
    } catch (err) {
      const attempts = (retryCounts.get(email.messageId) ?? 0) + 1;
      retryCounts.set(email.messageId, attempts);

      if (attempts > MAX_RETRIES_PER_EMAIL) {
        failedThisRun.add(email.messageId);
        this.progress.failed += 1;
        const persisted = this.failures.recordFailure(
          email.messageId,
          accountId,
          'triage',
          modelId,
          err instanceof Error ? err.message : String(err),
        );
        this.logger.error(
          {
            err,
            messageId: email.messageId,
            subject: email.subject,
            attempts,
            persistedFailures: persisted,
          },
          'triage failed for this run; recorded',
        );
      } else {
        this.logger.warn(
          { err, messageId: email.messageId, attempts },
          'triage classification failed, will retry',
        );
      }
    }
  }

  /**
   * Stores a classification result, clears any prior failure, and updates the
   * processed count and bucket tally.
   */
  private persistResult(
    email: UnclassifiedEmail,
    accountId: string,
    modelId: string,
    result: TriageResult,
  ): void {
    this.triage.upsert({
      messageId: email.messageId,
      accountId,
      bucket: result.bucket,
      reasoning: result.reasoning,
      metadata: result.metadata,
    });
    this.failures.clearFailure(email.messageId, accountId, 'triage', modelId);
    this.progress.processed += 1;
    this.progress.buckets[result.bucket] += 1;
  }

  /**
   * Builds a fresh idle progress snapshot with zeroed counts and buckets.
   */
  private idle(): TriageProgress {
    return {
      status: 'idle',
      accountId: null,
      modelId: null,
      total: 0,
      processed: 0,
      failed: 0,
      buckets: { urgent: 0, summarize: 0, spam: 0, personal: 0 },
    };
  }
}
