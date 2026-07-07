/**
 * Data access for discovery category proposals. A `new_category` proposal (Phase 2c) records one
 * accepted, named cluster awaiting review: the suggested category it created, the cluster's members
 * and centroid for apply, and the deterministic metrics for the review UI. Phase 3.3 adds STRUCTURAL
 * proposals (split / merge / retire) that carry no single cluster on the parent row (split cluster
 * data lives in category_proposal_children), plus a `suppression_key` so a resolved structural
 * suggestion is not re-proposed. Lifecycle is pending to applied or dismissed; the row is kept after
 * dismissal so a re-run does not re-propose the same purpose. No apply logic lives here.
 */
import type { Database, Statement } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import { bufferToVector, vectorToBuffer } from '../util/vector.js';

/** Lifecycle state of a proposal. */
export type ProposalStatus = 'pending' | 'applied' | 'dismissed';

/** What kind of change a proposal represents. Structural kinds are split, merge, and retire. */
export type ProposalKind = 'new_category' | 'split' | 'merge' | 'retire';

/** A stored proposal with its cluster detail decoded. */
export interface CategoryProposal {
  id: string;
  accountId: string;
  kind: ProposalKind;
  /** The surviving/target category for merge and retire, the source for split, the created one for new_category. */
  categoryId: string;
  /** The merge source (deleted on apply); null for every other kind. Non-cascading, so it may dangle. */
  sourceCategoryId: string | null;
  runId: string;
  clusterIndex: number;
  label: string;
  description: string;
  canonicalKey: string;
  suggestedKey: string;
  /** Stable signature that suppresses re-proposing a resolved structural suggestion. '' for new_category. */
  suppressionKey: string;
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

/** Fields needed to record a new pending new_category proposal (Phase 2c). */
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

/** Fields needed to record a structural (split / merge / retire) proposal. The parent carries no cluster. */
export interface CreateStructuralProposalInput {
  accountId: string;
  kind: 'split' | 'merge' | 'retire';
  /** Target (merge/retire) or source (split) category id. */
  categoryId: string;
  /** Merge source id (deleted on apply); null for split and retire. */
  sourceCategoryId: string | null;
  runId: string;
  label: string;
  description: string;
  canonicalKey: string;
  suppressionKey: string;
  embeddingModelId: string;
  confidence: number;
  evidence: string[];
}

/** One child category of a split proposal, with its own cluster detail. */
export interface CategoryProposalChild {
  id: string;
  proposalId: string;
  label: string;
  description: string;
  canonicalKey: string;
  embeddingModelId: string;
  centroid: Float32Array;
  memberIds: string[];
  proposedCount: number;
  cohesion: number;
  separation: number;
  confidence: number;
  createdAt: number;
}

/** Fields needed to record one split child. */
export interface CreateProposalChildInput {
  proposalId: string;
  label: string;
  description: string;
  canonicalKey: string;
  embeddingModelId: string;
  centroid: Float32Array;
  memberIds: string[];
  proposedCount: number;
  cohesion: number;
  separation: number;
  confidence: number;
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
  kind: ProposalKind;
  source_category_id: string | null;
  suppression_key: string;
}

interface ChildDbRow {
  id: string;
  proposal_id: string;
  label: string;
  description: string;
  canonical_key: string;
  embedding_model_id: string;
  centroid: Buffer;
  member_ids: string;
  proposed_count: number;
  cohesion: number;
  separation: number;
  confidence: number;
  created_at: number;
}

const COLS =
  'id, account_id, category_id, run_id, cluster_index, label, description, canonical_key, ' +
  'suggested_key, embedding_model_id, centroid, member_ids, proposed_count, cohesion, separation, ' +
  'confidence, evidence, status, created_at, applied_at, dismissed_at, ' +
  'kind, source_category_id, suppression_key';

const CHILD_COLS =
  'id, proposal_id, label, description, canonical_key, embedding_model_id, centroid, member_ids, ' +
  'proposed_count, cohesion, separation, confidence, created_at';

/** Reads and writes discovery category proposals and split children. */
export class CategoryProposalRepository {
  private readonly stmts: {
    insert: Statement<unknown[]>;
    findById: Statement<unknown[]>;
    listPending: Statement<unknown[]>;
    listForAccount: Statement<unknown[]>;
    resolvedSuggestedKeys: Statement<unknown[]>;
    resolvedStructuralSuppressionKeys: Statement<unknown[]>;
    markApplied: Statement<unknown[]>;
    markDismissed: Statement<unknown[]>;
    dismissPendingMergesForSource: Statement<unknown[]>;
    insertChild: Statement<unknown[]>;
    listChildren: Statement<unknown[]>;
  };

  constructor(private readonly db: Database) {
    this.stmts = {
      insert: db.prepare(
        `INSERT INTO category_proposals
           (${COLS})
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
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
      resolvedStructuralSuppressionKeys: db.prepare(
        `SELECT DISTINCT suppression_key FROM category_proposals
         WHERE account_id = ? AND kind <> 'new_category'
           AND status IN ('applied', 'dismissed') AND suppression_key <> ''`,
      ),
      markApplied: db.prepare(
        `UPDATE category_proposals SET status = 'applied', applied_at = ? WHERE id = ?`,
      ),
      markDismissed: db.prepare(
        `UPDATE category_proposals SET status = 'dismissed', dismissed_at = ? WHERE id = ?`,
      ),
      dismissPendingMergesForSource: db.prepare(
        `UPDATE category_proposals SET status = 'dismissed', dismissed_at = ?
         WHERE account_id = ? AND kind = 'merge' AND source_category_id = ?
           AND status = 'pending' AND id <> ?`,
      ),
      insertChild: db.prepare(
        `INSERT INTO category_proposal_children (${CHILD_COLS})
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ),
      listChildren: db.prepare(
        // Deterministic, meaningful order (largest child first); created_at collides within a split's
        // transaction, so ordering on it would fall back to the random id and shuffle the children.
        `SELECT ${CHILD_COLS} FROM category_proposal_children WHERE proposal_id = ?
         ORDER BY proposed_count DESC, canonical_key ASC`,
      ),
    };
  }

  /** Record a new pending new_category proposal. */
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
      'new_category',
      null,
      '',
    );
    return {
      id,
      ...input,
      kind: 'new_category',
      sourceCategoryId: null,
      suppressionKey: '',
      status: 'pending',
      createdAt: now,
      appliedAt: null,
      dismissedAt: null,
    };
  }

  /**
   * Record a pending structural (split / merge / retire) proposal. The parent has no single cluster,
   * so its cluster columns are deterministic placeholders (empty centroid, no members, zero metrics);
   * split cluster data is stored per child via createChild. categoryId is the surviving/target category.
   */
  createStructural(input: CreateStructuralProposalInput): CategoryProposal {
    const suppressionKey = input.suppressionKey.trim();
    if (suppressionKey === '') {
      throw new Error('structural proposal requires a non-empty suppressionKey');
    }
    if (input.kind === 'merge') {
      if (!input.sourceCategoryId) {
        throw new Error('merge proposal requires a sourceCategoryId');
      }
      if (input.sourceCategoryId === input.categoryId) {
        throw new Error('merge proposal sourceCategoryId must differ from the target categoryId');
      }
    } else if (input.sourceCategoryId !== null) {
      throw new Error(`${input.kind} proposal must have a null sourceCategoryId`);
    }
    const now = Date.now();
    const id = randomUUID();
    const emptyCentroid = new Float32Array(0);
    this.stmts.insert.run(
      id,
      input.accountId,
      input.categoryId,
      input.runId,
      0,
      input.label,
      input.description,
      input.canonicalKey,
      '',
      input.embeddingModelId,
      vectorToBuffer(emptyCentroid),
      '[]',
      0,
      0,
      0,
      input.confidence,
      JSON.stringify(input.evidence),
      'pending',
      now,
      null,
      null,
      input.kind,
      input.sourceCategoryId,
      suppressionKey,
    );
    return {
      id,
      accountId: input.accountId,
      kind: input.kind,
      categoryId: input.categoryId,
      sourceCategoryId: input.sourceCategoryId,
      runId: input.runId,
      clusterIndex: 0,
      label: input.label,
      description: input.description,
      canonicalKey: input.canonicalKey,
      suggestedKey: '',
      suppressionKey,
      embeddingModelId: input.embeddingModelId,
      centroid: emptyCentroid,
      memberIds: [],
      proposedCount: 0,
      cohesion: 0,
      separation: 0,
      confidence: input.confidence,
      evidence: input.evidence,
      status: 'pending',
      createdAt: now,
      appliedAt: null,
      dismissedAt: null,
    };
  }

  /** Record one child category of a split proposal. Only a pending split proposal may have children. */
  createChild(input: CreateProposalChildInput): CategoryProposalChild {
    const proposal = this.findById(input.proposalId);
    if (!proposal) {
      throw new Error('cannot add a child to a missing proposal');
    }
    if (proposal.kind !== 'split') {
      throw new Error(
        `cannot add a child to a ${proposal.kind} proposal; only split proposals have children`,
      );
    }
    if (proposal.status !== 'pending') {
      throw new Error(`cannot add a child to a ${proposal.status} proposal`);
    }
    const now = Date.now();
    const id = randomUUID();
    this.stmts.insertChild.run(
      id,
      input.proposalId,
      input.label,
      input.description,
      input.canonicalKey,
      input.embeddingModelId,
      vectorToBuffer(input.centroid),
      JSON.stringify(input.memberIds),
      input.proposedCount,
      input.cohesion,
      input.separation,
      input.confidence,
      now,
    );
    return { id, ...input, createdAt: now };
  }

  /**
   * Record a split proposal and its children atomically. The parent row and every child row are
   * written in one transaction, so a split proposal never persists with a missing or partial child
   * set. Returns the created parent proposal.
   */
  createSplit(
    input: CreateStructuralProposalInput,
    children: Array<Omit<CreateProposalChildInput, 'proposalId'>>,
  ): CategoryProposal {
    if (input.kind !== 'split') {
      throw new Error('createSplit requires a split proposal');
    }
    if (children.length < 2) {
      throw new Error('a split proposal requires at least two children');
    }
    const tx = this.db.transaction(() => {
      const proposal = this.createStructural(input);
      for (const child of children) {
        this.createChild({ ...child, proposalId: proposal.id });
      }
      return proposal;
    });
    return tx();
  }

  /** The split children of a proposal, in insertion order. */
  listChildren(proposalId: string): CategoryProposalChild[] {
    return (this.stmts.listChildren.all(proposalId) as ChildDbRow[]).map((r) =>
      this.childFromRow(r),
    );
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
   * Normalized suggested keys of applied or dismissed new_category proposals. A re-run consults these
   * so a purpose the user already accepted or dismissed is not re-proposed even when the model
   * relabels it and the label-derived canonical key drifts. Empty keys are excluded.
   */
  resolvedSuggestedKeys(accountId: string): Set<string> {
    const rows = this.stmts.resolvedSuggestedKeys.all(accountId) as Array<{
      suggested_key: string;
    }>;
    return new Set(rows.map((r) => r.suggested_key));
  }

  /**
   * Suppression keys of applied or dismissed STRUCTURAL (split/merge/retire) proposals. A re-run
   * consults these so the same structural suggestion is not re-proposed once resolved. Empty keys and
   * new_category proposals are excluded.
   */
  resolvedStructuralSuppressionKeys(accountId: string): Set<string> {
    const rows = this.stmts.resolvedStructuralSuppressionKeys.all(accountId) as Array<{
      suppression_key: string;
    }>;
    return new Set(rows.map((r) => r.suppression_key));
  }

  /** Mark a proposal applied. */
  markApplied(id: string): void {
    this.stmts.markApplied.run(Date.now(), id);
  }

  /** Mark a proposal dismissed. The row is kept so a re-run can suppress the same purpose. */
  markDismissed(id: string): void {
    this.stmts.markDismissed.run(Date.now(), id);
  }

  /**
   * Dismiss every other pending merge proposal that names the given source category. Used after a
   * merge deletes its source: those siblings can never apply (source_category_id is non-cascading, so
   * their source is gone) and would otherwise linger in the queue. Returns the number dismissed.
   */
  dismissPendingMergesForSource(
    accountId: string,
    sourceCategoryId: string,
    exceptProposalId: string,
  ): number {
    return this.stmts.dismissPendingMergesForSource.run(
      Date.now(),
      accountId,
      sourceCategoryId,
      exceptProposalId,
    ).changes;
  }

  private fromRow(row: ProposalDbRow): CategoryProposal {
    return {
      id: row.id,
      accountId: row.account_id,
      kind: row.kind,
      categoryId: row.category_id,
      sourceCategoryId: row.source_category_id,
      runId: row.run_id,
      clusterIndex: row.cluster_index,
      label: row.label,
      description: row.description,
      canonicalKey: row.canonical_key,
      suggestedKey: row.suggested_key,
      suppressionKey: row.suppression_key,
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

  private childFromRow(row: ChildDbRow): CategoryProposalChild {
    return {
      id: row.id,
      proposalId: row.proposal_id,
      label: row.label,
      description: row.description,
      canonicalKey: row.canonical_key,
      embeddingModelId: row.embedding_model_id,
      centroid: bufferToVector(row.centroid),
      memberIds: JSON.parse(row.member_ids) as string[],
      proposedCount: row.proposed_count,
      cohesion: row.cohesion,
      separation: row.separation,
      confidence: row.confidence,
      createdAt: row.created_at,
    };
  }
}
