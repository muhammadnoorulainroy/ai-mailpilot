/**
 * Append-only audit of category discovery and improvement runs so a blocked, failed, skipped, or
 * insufficient run is provable after the fact, including which provider ran and what was read.
 */
import type { Database, Statement } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';

/** Which flow produced the run. */
export type DiscoveryFlow = 'topic_discovery' | 'improve_categories';

/** Outcome of a discovery run. */
export type DiscoveryStatus = 'ok' | 'blocked' | 'failed' | 'insufficient' | 'skipped';

/** A discovery run to record. */
export interface DiscoveryAuditEntry {
  accountId: string;
  flow: DiscoveryFlow;
  accountKind: string;
  provider: 'local' | 'cloud';
  status: DiscoveryStatus;
  modelId?: string | null;
  poolSize?: number;
  sampleSize?: number;
  emailsExposed?: number;
  fieldsRead?: string[];
  redacted?: boolean;
  omittedCategories?: string[];
  error?: string | null;
}

/** A stored discovery audit row. */
export interface DiscoveryAuditRow extends DiscoveryAuditEntry {
  id: string;
  ranAt: number;
}

/** Data access for the discovery audit log. */
export class DiscoveryAuditRepository {
  private readonly stmts: { insert: Statement<unknown[]>; listForAccount: Statement<unknown[]> };

  constructor(db: Database) {
    this.stmts = {
      insert: db.prepare(
        `INSERT INTO discovery_audit
           (id, account_id, ran_at, flow, account_kind, provider, status, model_id,
            pool_size, sample_size, emails_exposed, fields_read, redacted, omitted_categories, error)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ),
      listForAccount: db.prepare(
        'SELECT * FROM discovery_audit WHERE account_id = ? ORDER BY ran_at DESC',
      ),
    };
  }

  /** Record one discovery run. */
  log(entry: DiscoveryAuditEntry): void {
    this.stmts.insert.run(
      randomUUID(),
      entry.accountId,
      Date.now(),
      entry.flow,
      entry.accountKind,
      entry.provider,
      entry.status,
      entry.modelId ?? null,
      entry.poolSize ?? 0,
      entry.sampleSize ?? 0,
      entry.emailsExposed ?? 0,
      JSON.stringify(entry.fieldsRead ?? []),
      entry.redacted ? 1 : 0,
      JSON.stringify(entry.omittedCategories ?? []),
      entry.error ?? null,
    );
  }

  /** Recent runs for an account, newest first. */
  listForAccount(accountId: string): DiscoveryAuditRow[] {
    const rows = this.stmts.listForAccount.all(accountId) as Array<Record<string, unknown>>;
    return rows.map((r) => ({
      id: r.id as string,
      accountId: r.account_id as string,
      ranAt: r.ran_at as number,
      flow: r.flow as DiscoveryFlow,
      accountKind: r.account_kind as string,
      provider: r.provider as 'local' | 'cloud',
      status: r.status as DiscoveryStatus,
      modelId: (r.model_id as string | null) ?? null,
      poolSize: r.pool_size as number,
      sampleSize: r.sample_size as number,
      emailsExposed: r.emails_exposed as number,
      fieldsRead: JSON.parse(r.fields_read as string) as string[],
      redacted: (r.redacted as number) === 1,
      omittedCategories: JSON.parse(r.omitted_categories as string) as string[],
      error: (r.error as string | null) ?? null,
    }));
  }
}
