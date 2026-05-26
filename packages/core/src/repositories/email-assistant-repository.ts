/**
 * Repository for the email_assistant_summaries cache table, storing and
 * retrieving AI generated email summaries keyed by message and account.
 */
import type { Database, Statement } from 'better-sqlite3';
import type { EmailAssistantSummaryDto } from '@ai-mailpilot/shared';

interface SummaryCacheRow {
  message_id: string;
  account_id: string;
  content_hash: string;
  model_id: string;
  provider: 'local' | 'cloud';
  summary_json: string;
  generated_at: number;
}

/**
 * Persists and retrieves cached AI email summaries keyed by message and account.
 */
export class EmailAssistantRepository {
  private readonly stmts: {
    findSummary: Statement<unknown[]>;
    upsertSummary: Statement<unknown[]>;
  };

  /**
   * Prepares the find and upsert statements once for reuse on the given database.
   */
  constructor(private db: Database) {
    this.stmts = {
      findSummary: db.prepare(
        `SELECT message_id, account_id, content_hash, model_id, provider, summary_json, generated_at
           FROM email_assistant_summaries
          WHERE message_id = ? AND account_id = ?`,
      ),
      upsertSummary: db.prepare(
        `INSERT INTO email_assistant_summaries
           (message_id, account_id, content_hash, model_id, provider, summary_json, generated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(message_id, account_id) DO UPDATE SET
           content_hash = excluded.content_hash,
           model_id = excluded.model_id,
           provider = excluded.provider,
           summary_json = excluded.summary_json,
           generated_at = excluded.generated_at`,
      ),
    };
  }

  /**
   * Returns the cached summary for a message only when it still matches the
   * given content hash and model, otherwise null so the caller regenerates.
   */
  findValidSummary(
    accountId: string,
    messageId: string,
    contentHash: string,
    modelId: string,
  ): EmailAssistantSummaryDto | null {
    const row = this.stmts.findSummary.get(messageId, accountId) as SummaryCacheRow | undefined;
    if (!row || row.content_hash !== contentHash || row.model_id !== modelId) return null;
    try {
      const parsed = JSON.parse(row.summary_json) as EmailAssistantSummaryDto;
      return {
        ...parsed,
        accountId,
        messageId,
        modelId: row.model_id,
        provider: row.provider,
        generatedAt: row.generated_at,
        cached: true,
      };
    } catch {
      return null;
    }
  }

  /**
   * Inserts or updates the cached summary for a message and account.
   */
  saveSummary(contentHash: string, summary: EmailAssistantSummaryDto): void {
    this.stmts.upsertSummary.run(
      summary.messageId,
      summary.accountId,
      contentHash,
      summary.modelId,
      summary.provider,
      JSON.stringify({ ...summary, cached: false }),
      summary.generatedAt,
    );
  }
}
