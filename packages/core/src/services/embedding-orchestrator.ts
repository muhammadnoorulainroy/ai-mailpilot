import type { Logger } from 'pino';
import type { LlmClient } from '../llm/client.js';
import type { EmailRepository, EmailRow } from '../repositories/email-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import { buildEmbeddingInput } from '../util/text.js';

export interface EmbeddingProgress {
  status: 'idle' | 'running' | 'completed' | 'error';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  failed: number;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

const BATCH_SIZE = 16;

export class EmbeddingOrchestrator {
  private progress: EmbeddingProgress = {
    status: 'idle',
    accountId: null,
    modelId: null,
    total: 0,
    processed: 0,
    failed: 0,
  };

  private running = false;

  constructor(
    private llm: LlmClient,
    private emails: EmailRepository,
    private embeddings: EmbeddingRepository,
    private logger: Logger,
  ) {}

  getProgress(): EmbeddingProgress {
    return { ...this.progress };
  }

  isRunning(): boolean {
    return this.running;
  }

  /**
   * Start embedding all unembedded emails for the given account.
   * Runs in background. Caller polls getProgress() for status.
   * Returns the initial count of pending emails.
   */
  start(accountId: string, modelId: string): { pending: number; started: boolean } {
    if (this.running) {
      return { pending: this.progress.total - this.progress.processed, started: false };
    }

    const pending = this.emails.countUnembedded(accountId, modelId);
    if (pending === 0) {
      this.progress = {
        status: 'completed',
        accountId,
        modelId,
        total: 0,
        processed: 0,
        failed: 0,
        startedAt: Date.now(),
        completedAt: Date.now(),
      };
      return { pending: 0, started: false };
    }

    this.progress = {
      status: 'running',
      accountId,
      modelId,
      total: pending,
      processed: 0,
      failed: 0,
      startedAt: Date.now(),
    };

    this.running = true;
    void this.runLoop(accountId, modelId);

    return { pending, started: true };
  }

  private async runLoop(accountId: string, modelId: string): Promise<void> {
    try {
      let batch: EmailRow[];
      do {
        batch = this.emails.findUnembedded(accountId, modelId, BATCH_SIZE);
        if (batch.length === 0) break;

        await this.processBatch(batch, modelId);
      } while (batch.length > 0);

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

  private async processBatch(batch: EmailRow[], modelId: string): Promise<void> {
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
        this.embeddings.saveEmbedding(
          {
            messageId: email.messageId,
            accountId: email.accountId,
            modelId,
          },
          vector,
        );
        this.progress.processed += 1;
      }
    } catch (err) {
      this.logger.warn({ err, size: batch.length }, 'batch embedding failed, retrying singly');
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
            {
              messageId: email.messageId,
              accountId: email.accountId,
              modelId,
            },
            vector,
          );
          this.progress.processed += 1;
        } catch (singleErr) {
          this.logger.error(
            { err: singleErr, messageId: email.messageId },
            'single embedding failed',
          );
          this.progress.failed += 1;
        }
      }
    }
  }
}
