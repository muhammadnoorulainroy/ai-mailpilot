/**
 * SQLite-backed store for the background Refine (categorize) job, keeping one
 * row per account so progress survives a Core restart.
 */
import type { Database, Statement } from 'better-sqlite3';
import type { LlmCategorizeProgress } from '../services/llm-categorize-orchestrator.js';

interface JobRow {
  account_id: string;
  model_id: string | null;
  status: string;
  total: number;
  processed: number;
  assigned: number;
  uncategorized: number;
  failed: number;
  clusters: number;
  clusters_processed: number;
  gated_clusters: number;
  llm_calls: number;
  error: string | null;
  started_at: number | null;
  completed_at: number | null;
  updated_at: number;
}

/**
 * Persists the background Refine job so progress survives a Core restart.
 * One row per account, since the orchestrator runs one job at a time.
 */
export class CategorizeJobRepository {
  private upsertStmt: Statement;
  private getStmt: Statement;
  private recentStmt: Statement;
  private interruptStmt: Statement;

  /** Prepares the upsert, lookup, and interrupt statements against the job table. */
  constructor(private db: Database) {
    this.upsertStmt = db.prepare(`
      INSERT INTO categorize_jobs (
        account_id, model_id, status, total, processed, assigned, uncategorized, failed,
        clusters, clusters_processed, gated_clusters, llm_calls, error, started_at, completed_at,
        updated_at
      ) VALUES (
        @account_id, @model_id, @status, @total, @processed, @assigned, @uncategorized, @failed,
        @clusters, @clusters_processed, @gated_clusters, @llm_calls, @error, @started_at,
        @completed_at, @updated_at
      )
      ON CONFLICT(account_id) DO UPDATE SET
        model_id = excluded.model_id,
        status = excluded.status,
        total = excluded.total,
        processed = excluded.processed,
        assigned = excluded.assigned,
        uncategorized = excluded.uncategorized,
        failed = excluded.failed,
        clusters = excluded.clusters,
        clusters_processed = excluded.clusters_processed,
        gated_clusters = excluded.gated_clusters,
        llm_calls = excluded.llm_calls,
        error = excluded.error,
        started_at = excluded.started_at,
        completed_at = excluded.completed_at,
        updated_at = excluded.updated_at
    `);
    this.getStmt = db.prepare('SELECT * FROM categorize_jobs WHERE account_id = ?');
    this.recentStmt = db.prepare('SELECT * FROM categorize_jobs ORDER BY updated_at DESC LIMIT 1');
    this.interruptStmt = db.prepare(
      "UPDATE categorize_jobs SET status = 'interrupted', updated_at = ? WHERE status = 'running'",
    );
  }

  /** Upserts the current job progress for an account. */
  save(accountId: string, progress: LlmCategorizeProgress, now: number): void {
    this.upsertStmt.run({
      account_id: accountId,
      model_id: progress.modelId,
      status: progress.status,
      total: progress.total,
      processed: progress.processed,
      assigned: progress.assigned,
      uncategorized: progress.uncategorized,
      failed: progress.failed,
      clusters: progress.clusters,
      clusters_processed: progress.clustersProcessed,
      gated_clusters: progress.gatedClusters,
      llm_calls: progress.llmCalls,
      error: progress.error ?? null,
      started_at: progress.startedAt ?? null,
      completed_at: progress.completedAt ?? null,
      updated_at: now,
    });
  }

  /** Returns the most recently updated job across all accounts, or null if none exists. */
  getMostRecent(): LlmCategorizeProgress | null {
    return toProgress(this.recentStmt.get() as JobRow | undefined);
  }

  /** Returns the stored job progress for an account, or null if none exists. */
  get(accountId: string): LlmCategorizeProgress | null {
    return toProgress(this.getStmt.get(accountId) as JobRow | undefined);
  }

  /**
   * Marks any still-running job as interrupted. Called once at startup so a job
   * left running by a crashed Core is not read as an active job that never completes.
   */
  markRunningInterrupted(now: number): number {
    return this.interruptStmt.run(now).changes;
  }
}

/** Maps a stored job row into a progress object, or null when no row exists. */
function toProgress(row: JobRow | undefined): LlmCategorizeProgress | null {
  if (!row) return null;
  return {
    status: row.status as LlmCategorizeProgress['status'],
    accountId: row.account_id,
    modelId: row.model_id,
    total: row.total,
    processed: row.processed,
    assigned: row.assigned,
    uncategorized: row.uncategorized,
    failed: row.failed,
    clusters: row.clusters,
    clustersProcessed: row.clusters_processed,
    gatedClusters: row.gated_clusters,
    llmCalls: row.llm_calls,
    error: row.error ?? undefined,
    startedAt: row.started_at ?? undefined,
    completedAt: row.completed_at ?? undefined,
  };
}
