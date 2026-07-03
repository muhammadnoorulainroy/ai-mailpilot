/**
 * Data access for discovery category proposals (Phase 2c). Each row records one accepted, named
 * cluster awaiting user review: the suggested category it created, the cluster's members and centroid
 * for apply, and the deterministic metrics for the review UI. Lifecycle is pending to applied or
 * dismissed; the row is kept after dismissal so a re-run does not re-propose the same purpose.
 */
import type { Database, Statement } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import { bufferToVector, vectorToBuffer } from '../util/vector.js';

/** Lifecycle state of a proposal. */
export type ProposalStatus = 'pending' | 'applied' | 'dismissed';

/** A stored proposal with its cluster detail decoded. */
export interface CategoryProposal {
  id: string;
  accountId: string;
  categoryId: string;
  runId: string;
  clusterIndex: number;
  label: string;
  description: string;
  canonicalKey: string;
  suggestedKey: string;
  embeddingModelId: string;
  centroid: Float32Array;
  memberIds: string[];
  proposedCount: number;
  cohesion: number;
  separation: number;
  confidence: number;
  evidence: string[];
  status: ProposalStatus;
  createdAt: number;
  appliedAt: number | null;
  dismissedAt: number | null;
}

/** Fields needed to record a new pending proposal. */
export interface CreateProposalInput {
  accountId: string;
  categoryId: string;
  runId: string;
  clusterIndex: number;
  label: string;
  description: string;
  canonicalKey: string;
  suggestedKey: string;
  embeddingModelId: string;
  centroid: Float32Array;
  memberIds: string[];
  proposedCount: number;
  cohesion: number;
  separation: number;
  confidence: number;
  evidence: string[];
}

interface ProposalDbRow {
  id: string;
  account_id: string;
  category_id: string;
  run_id: string;
  cluster_index: number;
  label: string;
  description: string;
  canonical_key: string;
  suggested_key: string;
  embedding_model_id: string;
  centroid: Buffer;
  member_ids: string;
  proposed_count: number;
  cohesion: number;
  separation: number;
  confidence: number;
  evidence: string;
  status: ProposalStatus;
  created_at: number;
  applied_at: number | null;
  dismissed_at: number | null;
}

const COLS =
  'id, account_id, category_id, run_id, cluster_index, label, description, canonical_key, ' +
  'suggested_key, embedding_model_id, centroid, member_ids, proposed_count, cohesion, separation, ' +
  'confidence, evidence, status, created_at, applied_at, dismissed_at';

/** Reads and writes discovery category proposals. */
export class CategoryProposalRepository {
  private readonly stmts: {
    insert: Statement<unknown[]>;
    findById: Statement<unknown[]>;
    listPending: Statement<unknown[]>;
    listForAccount: Statement<unknown[]>;
    resolvedSuggestedKeys: Statement<unknown[]>;
    markApplied: Statement<unknown[]>;
    markDismissed: Statement<unknown[]>;
  };

  constructor(db: Database) {
    this.stmts = {
      insert: db.prepare(
        `INSERT INTO category_proposals
           (${COLS})
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ),
      findById: db.prepare(`SELECT ${COLS} FROM category_proposals WHERE id = ?`),
      listPending: db.prepare(
        `SELECT ${COLS} FROM category_proposals
         WHERE account_id = ? AND status = 'pending'
         ORDER BY confidence DESC, created_at DESC`,
      ),
      listForAccount: db.prepare(
        `SELECT ${COLS} FROM category_proposals WHERE account_id = ? ORDER BY created_at DESC`,
      ),
      resolvedSuggestedKeys: db.prepare(
        `SELECT DISTINCT suggested_key FROM category_proposals
         WHERE account_id = ? AND status IN ('applied', 'dismissed') AND suggested_key <> ''`,
      ),
      markApplied: db.prepare(
        `UPDATE category_proposals SET status = 'applied', applied_at = ? WHERE id = ?`,
      ),
      markDismissed: db.prepare(
        `UPDATE category_proposals SET status = 'dismissed', dismissed_at = ? WHERE id = ?`,
      ),
    };
  }

  /** Record a new pending proposal. */
  create(input: CreateProposalInput): CategoryProposal {
    const now = Date.now();
    const id = randomUUID();
    this.stmts.insert.run(
      id,
      input.accountId,
      input.categoryId,
      input.runId,
      input.clusterIndex,
      input.label,
      input.description,
      input.canonicalKey,
      input.suggestedKey,
      input.embeddingModelId,
      vectorToBuffer(input.centroid),
      JSON.stringify(input.memberIds),
      input.proposedCount,
      input.cohesion,
      input.separation,
      input.confidence,
      JSON.stringify(input.evidence),
      'pending',
      now,
      null,
      null,
    );
    return {
      id,
      ...input,
      status: 'pending',
      createdAt: now,
      appliedAt: null,
      dismissedAt: null,
    };
  }

  /** Look up a proposal by id, or null if not found. */
  findById(id: string): CategoryProposal | null {
    const row = this.stmts.findById.get(id) as ProposalDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  /** Pending proposals for an account, strongest first. */
  listPending(accountId: string): CategoryProposal[] {
    return (this.stmts.listPending.all(accountId) as ProposalDbRow[]).map((r) => this.fromRow(r));
  }

  /** Every proposal for an account regardless of status, newest first. */
  listForAccount(accountId: string): CategoryProposal[] {
    return (this.stmts.listForAccount.all(accountId) as ProposalDbRow[]).map((r) =>
      this.fromRow(r),
    );
  }

  /**
   * Normalized suggested keys of applied or dismissed proposals. A re-run consults these so a purpose
   * the user already accepted or dismissed is not re-proposed even when the model relabels it and the
   * label-derived canonical key drifts. Empty keys are excluded.
   */
  resolvedSuggestedKeys(accountId: string): Set<string> {
    const rows = this.stmts.resolvedSuggestedKeys.all(accountId) as Array<{
      suggested_key: string;
    }>;
    return new Set(rows.map((r) => r.suggested_key));
  }

  /** Mark a proposal applied. */
  markApplied(id: string): void {
    this.stmts.markApplied.run(Date.now(), id);
  }

  /** Mark a proposal dismissed. The row is kept so a re-run can suppress the same purpose. */
  markDismissed(id: string): void {
    this.stmts.markDismissed.run(Date.now(), id);
  }

  private fromRow(row: ProposalDbRow): CategoryProposal {
    return {
      id: row.id,
      accountId: row.account_id,
      categoryId: row.category_id,
      runId: row.run_id,
      clusterIndex: row.cluster_index,
      label: row.label,
      description: row.description,
      canonicalKey: row.canonical_key,
      suggestedKey: row.suggested_key,
      embeddingModelId: row.embedding_model_id,
      centroid: bufferToVector(row.centroid),
      memberIds: JSON.parse(row.member_ids) as string[],
      proposedCount: row.proposed_count,
      cohesion: row.cohesion,
      separation: row.separation,
      confidence: row.confidence,
      evidence: JSON.parse(row.evidence) as string[],
      status: row.status,
      createdAt: row.created_at,
      appliedAt: row.applied_at,
      dismissedAt: row.dismissed_at,
    };
  }
}
