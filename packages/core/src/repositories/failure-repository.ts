/**
 * Persists per-item embed and triage failures so permanently bad items are retried only a bounded
 * number of times, with helpers to record, clear, and query failures keyed by message and model.
 */
import type { Database, Statement } from 'better-sqlite3';

/** The kind of per-item processing that failed. */
export type FailureKind = 'embedding' | 'triage';

/** Failure count at which an item is treated as permanently bad and no longer retried. */
export const MAX_PERMANENT_FAILURES = 3;

const MAX_ERROR_CHARS = 1000;

/**
 * Tracks per-item embed/triage failures across runs so a permanently bad item is retried only a
 * bounded number of times. Keyed by message, account, kind and model, so switching the model
 * retries items the previous model failed. Rows cascade away when the email or account is deleted.
 */
export class FailureRepository {
  private readonly stmts: {
    record: Statement<unknown[]>;
    clear: Statement<unknown[]>;
    clearForAccount: Statement<unknown[]>;
    permanentIds: Statement<unknown[]>;
    countPermanent: Statement<unknown[]>;
  };

  /** Prepares the reusable statements for recording, clearing, and querying failure rows. */
  constructor(private db: Database) {
    this.stmts = {
      record: db.prepare(
        `INSERT INTO processing_failures (message_id, account_id, kind, model_id, failure_count, last_error, updated_at)
         VALUES (?, ?, ?, ?, 1, ?, ?)
         ON CONFLICT (message_id, account_id, kind, model_id) DO UPDATE SET
           failure_count = failure_count + 1,
           last_error = excluded.last_error,
           updated_at = excluded.updated_at
         RETURNING failure_count`,
      ),
      clear: db.prepare(
        'DELETE FROM processing_failures WHERE message_id = ? AND account_id = ? AND kind = ? AND model_id = ?',
      ),
      clearForAccount: db.prepare(
        'DELETE FROM processing_failures WHERE account_id = ? AND kind = ?',
      ),
      permanentIds: db.prepare(
        'SELECT message_id FROM processing_failures WHERE account_id = ? AND kind = ? AND model_id = ? AND failure_count >= ?',
      ),
      countPermanent: db.prepare(
        'SELECT COUNT(*) AS c FROM processing_failures WHERE account_id = ? AND kind = ? AND model_id = ? AND failure_count >= ?',
      ),
    };
  }

  /** Record one failed run for an item and return its running failure count for this model. */
  recordFailure(
    messageId: string,
    accountId: string,
    kind: FailureKind,
    modelId: string,
    error: string,
  ): number {
    const row = this.stmts.record.get(
      messageId,
      accountId,
      kind,
      modelId,
      error.slice(0, MAX_ERROR_CHARS),
      Date.now(),
    ) as { failure_count: number };
    return row.failure_count;
  }

  /** Forget an item's failure once it finally succeeds, so a recovered item does not count. */
  clearFailure(messageId: string, accountId: string, kind: FailureKind, modelId: string): void {
    this.stmts.clear.run(messageId, accountId, kind, modelId);
  }

  /** Forget all failures for an account and kind across every model. Used by a force re-run. */
  clearForAccount(accountId: string, kind: FailureKind): number {
    return this.stmts.clearForAccount.run(accountId, kind).changes;
  }

  /** Return the message ids that have reached the permanent failure cap for this model. */
  permanentlyFailedIds(
    accountId: string,
    kind: FailureKind,
    modelId: string,
    cap = MAX_PERMANENT_FAILURES,
  ): string[] {
    const rows = this.stmts.permanentIds.all(accountId, kind, modelId, cap) as Array<{
      message_id: string;
    }>;
    return rows.map((r) => r.message_id);
  }

  /** Count the items that have reached the permanent failure cap for this model. */
  countPermanentlyFailed(
    accountId: string,
    kind: FailureKind,
    modelId: string,
    cap = MAX_PERMANENT_FAILURES,
  ): number {
    const row = this.stmts.countPermanent.get(accountId, kind, modelId, cap) as { c: number };
    return row.c;
  }
}
