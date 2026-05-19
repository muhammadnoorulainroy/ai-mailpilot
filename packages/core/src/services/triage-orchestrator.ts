import type { Logger } from 'pino';
import type { TriageRepository } from '../repositories/triage-repository.js';
import type { TriageService } from './triage-service.js';

export interface TriageProgress {
  status: 'idle' | 'running' | 'completed' | 'error';
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

const BATCH_SIZE = 8;

export class TriageOrchestrator {
  private progress: TriageProgress = this.idle();
  private running = false;

  constructor(
    private service: TriageService,
    private triage: TriageRepository,
    private logger: Logger,
  ) {}

  getProgress(): TriageProgress {
    return { ...this.progress, buckets: { ...this.progress.buckets } };
  }

  start(accountId: string, modelId: string): { pending: number; started: boolean } {
    if (this.running) {
      return {
        pending: this.progress.total - this.progress.processed,
        started: false,
      };
    }

    const pending = this.triage.countUnclassified(accountId);
    if (pending === 0) {
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

  private async runLoop(accountId: string, modelId: string): Promise<void> {
    try {
      let batch = this.triage.findUnclassifiedEmails(accountId, BATCH_SIZE);
      while (batch.length > 0) {
        for (const email of batch) {
          try {
            const result = await this.service.classify(email, modelId);
            this.triage.upsert({
              messageId: email.messageId,
              accountId,
              bucket: result.bucket,
              reasoning: result.reasoning,
            });
            this.progress.processed += 1;
            this.progress.buckets[result.bucket] += 1;
          } catch (err) {
            this.logger.warn(
              { err, messageId: email.messageId, subject: email.subject },
              'triage classification failed',
            );
            this.progress.failed += 1;
          }
        }
        batch = this.triage.findUnclassifiedEmails(accountId, BATCH_SIZE);
      }

      this.progress = {
        ...this.progress,
        status: 'completed',
        completedAt: Date.now(),
      };
      this.logger.info(
        {
          accountId,
          modelId,
          processed: this.progress.processed,
          failed: this.progress.failed,
          buckets: this.progress.buckets,
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
