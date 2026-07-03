/**
 * SQLite-backed data access for categories, their per-email assignments, the no-category
 * decisions an LLM has made, and per-category embedding centroids.
 */
import type { Database, Statement } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';
import { EMBEDDING_DIM } from '../db/schema.js';
import { EmbeddingDimensionError } from './embedding-repository.js';
import { bufferToVector, vectorToBuffer } from '../util/vector.js';
import { canonicalizeModelId } from '../util/model-id.js';
import { normalizeForMatch } from '../util/text.js';

/** Where a category came from. */
export type CategorySource = 'auto' | 'user' | 'imported';

/** Lifecycle state of a category. Only active categories are shown to the user and assigned to. */
export type CategoryStatus = 'active' | 'suggested' | 'retired';

/** Who created an email to category assignment. */
export type AssignedBy = 'user' | 'auto';

const RECONCILE_KEEP_MIN_ASSIGNMENTS = 3;

/**
 * How an auto assignment was produced. 'embed' is the nearest-centroid pass, 'llm' is an LLM
 * judgement, 'gate' is a confident embedding match with no LLM call. Null for user corrections.
 */
export type AssignmentMethod = 'embed' | 'llm' | 'gate' | 'proposal';

/**
 * The base canonical key for a label, before per-account collision suffixing. Deterministic and
 * frozen: callers that need to check whether a purpose already exists (discovery dedup) derive the
 * same base and look it up with findByCanonicalKey.
 */
export function canonicalKeyBase(label: string): string {
  return normalizeForMatch(label).replace(/\s+/g, '_').slice(0, 60) || 'category';
}

/** A category as stored, with timestamps. */
export interface CategoryRow {
  id: string;
  accountId: string;
  label: string;
  description: string | null;
  source: CategorySource;
  canonicalKey: string;
  status: CategoryStatus;
  firstSeenAt: number;
  retiredAt: number | null;
  createdAt: number;
  updatedAt: number;
}

/** Fields needed to create a category. */
export interface CreateCategoryInput {
  accountId: string;
  label: string;
  description?: string;
  source: CategorySource;
  status?: CategoryStatus;
  /** Frozen identity key. Derived deterministically from the label when omitted. */
  canonicalKey?: string;
}

/** Editable fields for an existing category. */
export interface UpdateCategoryInput {
  label?: string;
  description?: string | null;
}

/** A single email to category assignment with its confidence and provenance. */
export interface CategoryAssignment {
  messageId: string;
  accountId: string;
  categoryId: string;
  confidence: number;
  assignedBy: AssignedBy;
  assignedAt: number;
  method?: AssignmentMethod | null;
}

/**
 * One of an email's category memberships, with how that membership was decided so the UI can
 * show provenance per chip.
 */
export interface EmailMembership {
  id: string;
  label: string;
  confidence: number;
  assignedBy: AssignedBy;
  method: AssignmentMethod | null;
}

/** A category along with how many emails are assigned to it. */
export interface CategoryWithCount extends CategoryRow {
  emailCount: number;
}

/** A stored category centroid and the count of emails that contributed to it. */
export interface CentroidEntry {
  categoryId: string;
  label: string;
  vector: Float32Array;
  emailCount: number;
}

interface CategoryDbRow {
  id: string;
  account_id: string;
  label: string;
  description: string | null;
  source: CategorySource;
  canonical_key: string;
  status: CategoryStatus;
  first_seen_at: number;
  retired_at: number | null;
  created_at: number;
  updated_at: number;
}

/** Data access for categories, their email assignments, and per-category centroids. */
export class CategoryRepository {
  private readonly stmts: {
    insertCategory: Statement<unknown[]>;
    updateCategory: Statement<unknown[]>;
    deleteCategory: Statement<unknown[]>;
    findById: Statement<unknown[]>;
    findByLabel: Statement<unknown[]>;
    findByCanonicalKey: Statement<unknown[]>;
    selectCanonicalKeys: Statement<unknown[]>;
    listForAccount: Statement<unknown[]>;
    listByStatus: Statement<unknown[]>;
    setStatus: Statement<unknown[]>;
    retireCategory: Statement<unknown[]>;
    listAutoAssignments: Statement<unknown[]>;
    countEmailsForCategory: Statement<unknown[]>;
    clearAuto: Statement<unknown[]>;
    countUncategorized: Statement<unknown[]>;
    deleteAutoEC: Statement<unknown[]>;
    deleteForEmail: Statement<unknown[]>;
    insertEC: Statement<unknown[]>;
    findCentroidRowId: Statement<unknown[]>;
    updateCentroidVec: Statement<unknown[]>;
    updateCentroidIndex: Statement<unknown[]>;
    insertCentroidVec: Statement<unknown[]>;
    insertCentroidIndex: Statement<unknown[]>;
    listCentroids: Statement<unknown[]>;
    getCentroidByCategory: Statement<unknown[]>;
    selectUserAssigned: Statement<unknown[]>;
    selectAssigned: Statement<unknown[]>;
    selectLlmProtected: Statement<unknown[]>;
    selectCategoryMembersByAssignedBy: Statement<unknown[]>;
    listAutoWithUserFlag: Statement<unknown[]>;
    selectECByEmail: Statement<unknown[]>;
    selectECByEmailWithLabel: Statement<unknown[]>;
    insertDecision: Statement<unknown[]>;
    deleteDecision: Statement<unknown[]>;
    deleteDecisionsForEmail: Statement<unknown[]>;
    selectDecisions: Statement<unknown[]>;
    clearDecisionsForAccount: Statement<unknown[]>;
    selectCorrectionExamples: Statement<unknown[]>;
  };

  /** Prepare every statement this repository reuses against the given database. */
  constructor(private db: Database) {
    this.stmts = {
      insertCategory: db.prepare(
        `INSERT INTO categories (id, account_id, label, description, source, canonical_key, status, first_seen_at, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ),
      updateCategory: db.prepare(
        'UPDATE categories SET label = ?, description = ?, updated_at = ? WHERE id = ?',
      ),
      deleteCategory: db.prepare('DELETE FROM categories WHERE id = ?'),
      findById: db.prepare(
        'SELECT id, account_id, label, description, source, canonical_key, status, first_seen_at, retired_at, created_at, updated_at FROM categories WHERE id = ?',
      ),
      findByLabel: db.prepare(
        'SELECT id, account_id, label, description, source, canonical_key, status, first_seen_at, retired_at, created_at, updated_at FROM categories WHERE account_id = ? AND label = ?',
      ),
      findByCanonicalKey: db.prepare(
        'SELECT id, account_id, label, description, source, canonical_key, status, first_seen_at, retired_at, created_at, updated_at FROM categories WHERE account_id = ? AND canonical_key = ?',
      ),
      selectCanonicalKeys: db.prepare('SELECT canonical_key FROM categories WHERE account_id = ?'),
      listForAccount: db.prepare(
        `SELECT c.id, c.account_id, c.label, c.description, c.source,
                c.canonical_key, c.status, c.first_seen_at, c.retired_at, c.created_at, c.updated_at,
                COALESCE(COUNT(ec.message_id), 0) AS email_count
           FROM categories c
           LEFT JOIN email_categories ec ON ec.category_id = c.id
          WHERE c.account_id = ?
          GROUP BY c.id
          ORDER BY c.label ASC`,
      ),
      listByStatus: db.prepare(
        `SELECT c.id, c.account_id, c.label, c.description, c.source,
                c.canonical_key, c.status, c.first_seen_at, c.retired_at, c.created_at, c.updated_at,
                COALESCE(COUNT(ec.message_id), 0) AS email_count
           FROM categories c
           LEFT JOIN email_categories ec ON ec.category_id = c.id
          WHERE c.account_id = ? AND c.status = ?
          GROUP BY c.id
          ORDER BY c.label ASC`,
      ),
      setStatus: db.prepare(
        'UPDATE categories SET status = ?, retired_at = ?, updated_at = ? WHERE id = ?',
      ),
      retireCategory: db.prepare(
        "UPDATE categories SET status = 'retired', retired_at = ?, updated_at = ? WHERE id = ?",
      ),
      listAutoAssignments: db.prepare(
        `SELECT ec.message_id, ec.category_id
           FROM email_categories ec
           JOIN categories c ON c.id = ec.category_id
          WHERE ec.account_id = ? AND ec.assigned_by = 'auto' AND c.status = 'active'`,
      ),
      countEmailsForCategory: db.prepare(
        'SELECT COUNT(*) AS c FROM email_categories WHERE category_id = ?',
      ),
      clearAuto: db.prepare("DELETE FROM categories WHERE account_id = ? AND source = 'auto'"),
      countUncategorized: db.prepare(
        `SELECT COUNT(*) AS c FROM emails e
          LEFT JOIN email_categories ec
            ON ec.message_id = e.message_id AND ec.account_id = e.account_id
         WHERE e.account_id = ? AND ec.message_id IS NULL`,
      ),
      deleteAutoEC: db.prepare(
        "DELETE FROM email_categories WHERE account_id = ? AND assigned_by = 'auto'",
      ),
      deleteForEmail: db.prepare(
        'DELETE FROM email_categories WHERE message_id = ? AND account_id = ?',
      ),
      insertEC: db.prepare(
        `INSERT INTO email_categories (message_id, account_id, category_id, confidence, assigned_by, assigned_at, method)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT (message_id, account_id, category_id) DO UPDATE SET
           confidence = excluded.confidence,
           assigned_by = excluded.assigned_by,
           assigned_at = excluded.assigned_at,
           method = excluded.method`,
      ),
      findCentroidRowId: db.prepare(
        'SELECT rowid FROM category_embedding_index WHERE category_id = ? AND model_id = ?',
      ),
      updateCentroidVec: db.prepare('UPDATE category_embeddings SET embedding = ? WHERE rowid = ?'),
      updateCentroidIndex: db.prepare(
        'UPDATE category_embedding_index SET email_count = ?, updated_at = ? WHERE rowid = ?',
      ),
      insertCentroidVec: db.prepare('INSERT INTO category_embeddings (embedding) VALUES (?)'),
      insertCentroidIndex: db.prepare(
        `INSERT INTO category_embedding_index (rowid, category_id, model_id, email_count, updated_at)
         VALUES (?, ?, ?, ?, ?)`,
      ),
      listCentroids: db.prepare(
        `SELECT cei.category_id, c.label, ce.embedding, cei.email_count
           FROM category_embeddings ce
           JOIN category_embedding_index cei ON cei.rowid = ce.rowid
           JOIN categories c ON c.id = cei.category_id
          WHERE c.account_id = ? AND cei.model_id = ? AND c.status = 'active'`,
      ),
      getCentroidByCategory: db.prepare(
        `SELECT ce.embedding, cei.email_count
           FROM category_embeddings ce
           JOIN category_embedding_index cei ON cei.rowid = ce.rowid
          WHERE cei.category_id = ? AND cei.model_id = ?`,
      ),
      selectUserAssigned: db.prepare(
        "SELECT DISTINCT message_id FROM email_categories WHERE account_id = ? AND assigned_by = 'user'",
      ),
      selectAssigned: db.prepare(
        'SELECT DISTINCT message_id FROM email_categories WHERE account_id = ?',
      ),
      selectCategoryMembersByAssignedBy: db.prepare(
        'SELECT message_id FROM email_categories WHERE account_id = ? AND category_id = ? AND assigned_by = ?',
      ),
      selectLlmProtected: db.prepare(
        "SELECT DISTINCT message_id FROM email_categories WHERE account_id = ? AND (assigned_by = 'user' OR method IN ('llm', 'gate'))",
      ),
      insertDecision: db.prepare(
        `INSERT INTO llm_category_decisions (message_id, account_id, model_id, decided_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT (message_id, account_id, model_id) DO UPDATE SET decided_at = excluded.decided_at`,
      ),
      deleteDecision: db.prepare(
        'DELETE FROM llm_category_decisions WHERE message_id = ? AND account_id = ? AND model_id = ?',
      ),
      deleteDecisionsForEmail: db.prepare(
        'DELETE FROM llm_category_decisions WHERE message_id = ? AND account_id = ?',
      ),
      selectDecisions: db.prepare(
        'SELECT message_id FROM llm_category_decisions WHERE account_id = ? AND model_id = ?',
      ),
      clearDecisionsForAccount: db.prepare(
        'DELETE FROM llm_category_decisions WHERE account_id = ?',
      ),
      selectCorrectionExamples: db.prepare(
        `SELECT ec.message_id, e.subject, c.label
           FROM email_categories ec
           JOIN emails e ON e.message_id = ec.message_id AND e.account_id = ec.account_id
           JOIN categories c ON c.id = ec.category_id
          WHERE ec.account_id = ? AND ec.assigned_by = 'user'
            AND ec.message_id IN (
              SELECT message_id FROM email_categories
               WHERE account_id = ? AND assigned_by = 'user'
               GROUP BY message_id
               ORDER BY MAX(assigned_at) DESC
               LIMIT ?
            )
          ORDER BY ec.assigned_at DESC`,
      ),
      listAutoWithUserFlag: db.prepare(
        `SELECT c.id, c.label,
                EXISTS(SELECT 1 FROM email_categories ec
                        WHERE ec.category_id = c.id AND ec.assigned_by = 'user') AS has_user,
                (SELECT COUNT(*) FROM email_categories ec
                        WHERE ec.category_id = c.id) AS assigned_count
           FROM categories c
          WHERE c.account_id = ? AND c.source = 'auto'`,
      ),
      selectECByEmail: db.prepare(
        `SELECT message_id, account_id, category_id, confidence, assigned_by, assigned_at
           FROM email_categories
          WHERE message_id = ? AND account_id = ?
          ORDER BY confidence DESC`,
      ),
      selectECByEmailWithLabel: db.prepare(
        `SELECT ec.message_id, ec.account_id, ec.category_id, ec.confidence,
                ec.assigned_by, ec.assigned_at, ec.method, c.label
           FROM email_categories ec
           JOIN categories c ON c.id = ec.category_id
          WHERE ec.message_id = ? AND ec.account_id = ?
          ORDER BY ec.confidence DESC`,
      ),
    };
  }

  /** Insert a new category and return the stored row. Freezes a deterministic canonical key. */
  create(input: CreateCategoryInput): CategoryRow {
    const now = Date.now();
    const row: CategoryRow = {
      id: randomUUID(),
      accountId: input.accountId,
      label: input.label,
      description: input.description ?? null,
      source: input.source,
      canonicalKey: input.canonicalKey ?? this.deriveCanonicalKey(input.accountId, input.label),
      status: input.status ?? 'active',
      firstSeenAt: now,
      retiredAt: null,
      createdAt: now,
      updatedAt: now,
    };
    this.stmts.insertCategory.run(
      row.id,
      row.accountId,
      row.label,
      row.description,
      row.source,
      row.canonicalKey,
      row.status,
      row.firstSeenAt,
      row.createdAt,
      row.updatedAt,
    );
    return row;
  }

  /**
   * A deterministic, per-account-unique canonical key for a label, matching the migration
   * backfill. Suffixes on collision so two labels never share a key.
   */
  private deriveCanonicalKey(accountId: string, label: string): string {
    const taken = new Set(
      (this.stmts.selectCanonicalKeys.all(accountId) as Array<{ canonical_key: string }>).map(
        (r) => r.canonical_key,
      ),
    );
    const base = canonicalKeyBase(label);
    if (!taken.has(base)) return base;
    for (let i = 2; ; i++) {
      const candidate = `${base}_${i}`;
      if (!taken.has(candidate)) return candidate;
    }
  }

  /** Apply partial edits to a category, returning the updated row or null if it does not exist. */
  update(id: string, input: UpdateCategoryInput): CategoryRow | null {
    const existing = this.findById(id);
    if (!existing) return null;
    const next: CategoryRow = {
      ...existing,
      label: input.label ?? existing.label,
      description: input.description === undefined ? existing.description : input.description,
      updatedAt: Date.now(),
    };
    this.stmts.updateCategory.run(next.label, next.description, next.updatedAt, id);
    return next;
  }

  /** Number of emails assigned to a category. */
  countEmails(categoryId: string): number {
    const row = this.stmts.countEmailsForCategory.get(categoryId) as { c: number } | undefined;
    return row?.c ?? 0;
  }

  /** Delete a category, returning whether a row was removed. */
  delete(id: string): boolean {
    return this.stmts.deleteCategory.run(id).changes > 0;
  }

  /** Look up a category by id, or null if not found. */
  findById(id: string): CategoryRow | null {
    const row = this.stmts.findById.get(id) as CategoryDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  /** Look up a category by account and label, or null if not found. */
  findByLabel(accountId: string, label: string): CategoryRow | null {
    const row = this.stmts.findByLabel.get(accountId, label) as CategoryDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  /** All categories for an account with their email counts, ordered by label. Every status. */
  listForAccount(accountId: string): CategoryWithCount[] {
    const rows = this.stmts.listForAccount.all(accountId) as Array<
      CategoryDbRow & { email_count: number }
    >;
    return rows.map((r) => ({ ...this.fromRow(r), emailCount: r.email_count }));
  }

  /** Every category regardless of status. Named for intent; same result as listForAccount. */
  listAll(accountId: string): CategoryWithCount[] {
    return this.listForAccount(accountId);
  }

  /** Categories with the given status, with email counts, ordered by label. */
  private listByStatus(accountId: string, status: CategoryStatus): CategoryWithCount[] {
    const rows = this.stmts.listByStatus.all(accountId, status) as Array<
      CategoryDbRow & { email_count: number }
    >;
    return rows.map((r) => ({ ...this.fromRow(r), emailCount: r.email_count }));
  }

  /** Active categories only: the ones shown to the user and eligible for assignment. */
  listActive(accountId: string): CategoryWithCount[] {
    return this.listByStatus(accountId, 'active');
  }

  /** Suggested (dormant) categories awaiting user approval. */
  listSuggested(accountId: string): CategoryWithCount[] {
    return this.listByStatus(accountId, 'suggested');
  }

  /** Retired categories, kept for history and to preserve their assignments. */
  listRetired(accountId: string): CategoryWithCount[] {
    return this.listByStatus(accountId, 'retired');
  }

  /** Non-user (auto) assignments to active categories. Used to detect low-confidence residual mail. */
  listAutoAssignments(accountId: string): Array<{ messageId: string; categoryId: string }> {
    const rows = this.stmts.listAutoAssignments.all(accountId) as Array<{
      message_id: string;
      category_id: string;
    }>;
    return rows.map((r) => ({ messageId: r.message_id, categoryId: r.category_id }));
  }

  /** Look up a category by its frozen canonical key, or null if none exists. */
  findByCanonicalKey(accountId: string, canonicalKey: string): CategoryRow | null {
    const row = this.stmts.findByCanonicalKey.get(accountId, canonicalKey) as
      | CategoryDbRow
      | undefined;
    return row ? this.fromRow(row) : null;
  }

  /** Set a category's status. Retiring stamps retired_at; any other status clears it. */
  setStatus(id: string, status: CategoryStatus): void {
    const retiredAt = status === 'retired' ? Date.now() : null;
    this.stmts.setStatus.run(status, retiredAt, Date.now(), id);
  }

  /**
   * Retire a category: it keeps its id, assignments, and history but stops being shown or
   * assigned to. Phase 2 only, driven by a user-approved retirement proposal. Nothing in the
   * discovery path calls this automatically.
   */
  retire(id: string): void {
    this.stmts.retireCategory.run(Date.now(), Date.now(), id);
  }

  /** Delete all auto categories for an account, returning the number removed. */
  clearAutoForAccount(accountId: string): number {
    return this.stmts.clearAuto.run(accountId).changes;
  }

  /**
   * Retired. This cleared and recreated all auto categories, destroying their ids and any
   * assignments, folders, and corrections tied to them. Use reconcileAutoCategories, which
   * preserves identity.
   */
  replaceAutoCategories(): never {
    throw new Error('replaceAutoCategories is retired; use reconcileAutoCategories');
  }

  /**
   * Reconcile an account's auto categories with a freshly discovered set, preserving
   * identity. A category whose label still matches a discovered topic keeps its id so
   * tags, folders, and corrections tied to it stay valid, and gets its centroid and
   * description refreshed. New topics are created. Obsolete auto categories are NEVER deleted or
   * retired here; they stay active and are returned in `omitted` as candidates for a Phase 2
   * user-approved retirement. Returns the live count and the omitted ids.
   */
  reconcileAutoCategories(
    accountId: string,
    modelId: string,
    topics: ReadonlyArray<{
      label: string;
      description: string;
      centroid: Float32Array;
      emailCount: number;
    }>,
  ): { live: number; omitted: string[] } {
    const tx = this.db.transaction(() => {
      const existing = this.stmts.listAutoWithUserFlag.all(accountId) as Array<{
        id: string;
        label: string;
        has_user: number;
        assigned_count: number;
      }>;
      const byLabel = new Map(existing.map((e) => [e.label.trim().toLowerCase(), e]));
      const keptIds = new Set<string>();
      const seenLabels = new Set<string>();

      for (const t of topics) {
        const key = t.label.trim().toLowerCase();
        if (seenLabels.has(key)) continue;
        seenLabels.add(key);

        const match = byLabel.get(key);
        if (match) {
          this.update(match.id, { description: t.description });
          this.saveCentroid(match.id, modelId, t.centroid, t.emailCount);
          keptIds.add(match.id);
        } else {
          const row = this.create({
            accountId,
            label: t.label,
            description: t.description,
            source: 'auto',
          });
          this.saveCentroid(row.id, modelId, t.centroid, t.emailCount);
          keptIds.add(row.id);
        }
      }

      const omitted: string[] = [];
      for (const e of existing) {
        if (keptIds.has(e.id)) continue;
        // Phase 1: never delete and never retire. Keep it active and record that this run did
        // not re-find it, so a Phase 2 user-approved flow can propose retiring it.
        keptIds.add(e.id);
        if (e.has_user === 0 && e.assigned_count < RECONCILE_KEEP_MIN_ASSIGNMENTS) {
          omitted.push(e.id);
        }
      }
      return { live: keptIds.size, omitted };
    });
    return tx();
  }

  /** Number of emails in an account that have no category. */
  countUncategorized(accountId: string): number {
    const row = this.stmts.countUncategorized.get(accountId) as { c: number } | undefined;
    return row?.c ?? 0;
  }

  /** Atomically swap all auto assignments for an account. Used by a full re-run. */
  swapAutoAssignments(accountId: string, assignments: CategoryAssignment[]): void {
    const insert = this.stmts.insertEC;
    const deleteAuto = this.stmts.deleteAutoEC;

    const tx = this.db.transaction((items: CategoryAssignment[]) => {
      deleteAuto.run(accountId);
      for (const a of items) {
        insert.run(
          a.messageId,
          a.accountId,
          a.categoryId,
          a.confidence,
          a.assignedBy,
          a.assignedAt,
          a.method ?? null,
        );
      }
    });
    tx(assignments);
  }

  /**
   * Add assignments without removing anything else. Use for emails that currently have
   * no assignment, so a re-run only fills new mail and never reshuffles existing results.
   */
  addAutoAssignments(accountId: string, assignments: CategoryAssignment[]): void {
    const insert = this.stmts.insertEC;
    const tx = this.db.transaction((items: CategoryAssignment[]) => {
      for (const a of items) {
        insert.run(
          a.messageId,
          a.accountId,
          a.categoryId,
          a.confidence,
          a.assignedBy,
          a.assignedAt,
          a.method ?? null,
        );
      }
    });
    tx(assignments);
  }

  /**
   * Replace the auto assignments for many emails that share one decision in a single
   * transaction. Each email gets the same category set with descending confidence, first
   * is primary. An empty categoryIds leaves the emails uncategorized. Callers must
   * exclude user-locked emails.
   */
  bulkReplaceForCluster(
    accountId: string,
    messageIds: string[],
    categoryIds: string[],
    method: AssignmentMethod | null = null,
  ): void {
    const insert = this.stmts.insertEC;
    const del = this.stmts.deleteForEmail;
    const now = Date.now();
    const tx = this.db.transaction(() => {
      for (const messageId of messageIds) {
        del.run(messageId, accountId);
        categoryIds.forEach((categoryId, idx) => {
          insert.run(
            messageId,
            accountId,
            categoryId,
            Math.max(0.5, 1 - idx * 0.1),
            'auto',
            now,
            method,
          );
        });
      }
    });
    tx();
  }

  /** Replace assignments for a single email. */
  replaceEmailAssignments(
    messageId: string,
    accountId: string,
    assignments: CategoryAssignment[],
  ): void {
    const insert = this.stmts.insertEC;
    const del = this.stmts.deleteForEmail;
    const tx = this.db.transaction((items: CategoryAssignment[]) => {
      del.run(messageId, accountId);
      for (const a of items) {
        insert.run(
          a.messageId,
          a.accountId,
          a.categoryId,
          a.confidence,
          a.assignedBy,
          a.assignedAt,
          a.method ?? null,
        );
      }
    });
    tx(assignments);
  }

  /**
   * Move one email's AUTO assignment from sourceCategoryId to targetCategoryId, preserving its
   * confidence, method, and assigned_at. Only the auto (message, account, source) row is touched, so
   * a user assignment is never moved and the email keeps its memberships in every other category.
   * Returns true when a row moved (false when the email had no auto assignment on the source, or a
   * target row already existed). Used by split apply to relocate a source member into its child.
   */
  moveAutoAssignment(
    messageId: string,
    accountId: string,
    sourceCategoryId: string,
    targetCategoryId: string,
  ): boolean {
    const tx = this.db.transaction((): boolean => {
      const inserted = this.db
        .prepare(
          `INSERT INTO email_categories (message_id, account_id, category_id, confidence, assigned_by, assigned_at, method)
             SELECT message_id, account_id, ?, confidence, assigned_by, assigned_at, method
               FROM email_categories
              WHERE message_id = ? AND account_id = ? AND category_id = ? AND assigned_by = 'auto'
           ON CONFLICT (message_id, account_id, category_id) DO NOTHING`,
        )
        .run(targetCategoryId, messageId, accountId, sourceCategoryId).changes;
      if (inserted === 0) return false;
      this.db
        .prepare(
          `DELETE FROM email_categories
            WHERE message_id = ? AND account_id = ? AND category_id = ? AND assigned_by = 'auto'`,
        )
        .run(messageId, accountId, sourceCategoryId);
      return true;
    });
    return tx();
  }

  /**
   * Emails assigned to a category, newest first. Joins on the emails table so the UI can
   * render subject, from, and date without extra round-trips.
   */
  listEmailsForCategory(
    categoryId: string,
    limit = 200,
    offset = 0,
  ): Array<{
    messageId: string;
    accountId: string;
    folder: string;
    subject: string | null;
    fromAddr: string | null;
    date: number | null;
    hasAttachments: boolean;
    confidence: number;
    assignedBy: AssignedBy;
    method: AssignmentMethod | null;
    categories: EmailMembership[];
  }> {
    const rows = this.db
      .prepare(
        `SELECT e.message_id, e.account_id, e.folder, e.subject, e.from_addr,
                e.date, e.has_attachments,
                ec.confidence, ec.assigned_by, ec.method
           FROM email_categories ec
           JOIN emails e
             ON e.message_id = ec.message_id AND e.account_id = ec.account_id
          WHERE ec.category_id = ?
          ORDER BY COALESCE(e.date, e.indexed_at) DESC, e.message_id DESC
          LIMIT ? OFFSET ?`,
      )
      .all(categoryId, limit, offset) as Array<{
      message_id: string;
      account_id: string;
      folder: string;
      subject: string | null;
      from_addr: string | null;
      date: number | null;
      has_attachments: number;
      confidence: number;
      assigned_by: AssignedBy;
      method: AssignmentMethod | null;
    }>;

    const membershipsByMessage = this.membershipsFor(
      rows.length > 0 ? rows[0]!.account_id : '',
      rows.map((r) => r.message_id),
    );

    return rows.map((r) => ({
      messageId: r.message_id,
      accountId: r.account_id,
      folder: r.folder,
      subject: r.subject,
      fromAddr: r.from_addr,
      date: r.date,
      hasAttachments: r.has_attachments === 1,
      confidence: r.confidence,
      assignedBy: r.assigned_by,
      method: r.method,
      categories: membershipsByMessage.get(r.message_id) ?? [],
    }));
  }

  /** All category memberships for the given message ids, grouped by message id for the UI. */
  private membershipsFor(accountId: string, messageIds: string[]): Map<string, EmailMembership[]> {
    const result = new Map<string, EmailMembership[]>();
    if (messageIds.length === 0) return result;

    const placeholders = messageIds.map(() => '?').join(',');
    const rows = this.db
      .prepare(
        `SELECT ec.message_id, c.id, c.label, ec.confidence, ec.assigned_by, ec.method
           FROM email_categories ec
           JOIN categories c ON c.id = ec.category_id
          WHERE ec.account_id = ? AND ec.message_id IN (${placeholders})
          ORDER BY c.label ASC`,
      )
      .all(accountId, ...messageIds) as Array<{
      message_id: string;
      id: string;
      label: string;
      confidence: number;
      assigned_by: AssignedBy;
      method: AssignmentMethod | null;
    }>;

    for (const row of rows) {
      const list = result.get(row.message_id) ?? [];
      list.push({
        id: row.id,
        label: row.label,
        confidence: row.confidence,
        assignedBy: row.assigned_by,
        method: row.method,
      });
      result.set(row.message_id, list);
    }
    return result;
  }

  /**
   * Move every email assigned to sourceId over to targetId, then delete the source
   * category. When an email is already in the target, USER provenance dominates: the
   * winning row is the user one if either side is user, otherwise the higher-confidence
   * row; the winner's method and assigned_at are kept, and confidence is the max. A user
   * assignment is never overwritten by an auto one, even at equal or higher auto confidence.
   * Runs in a single transaction so a half-merge can't leak.
   */
  mergeInto(sourceId: string, targetId: string): { reassigned: number } {
    if (sourceId === targetId) return { reassigned: 0 };

    // The incoming (source) row wins the conflict when it is a user row over an auto row,
    // or when both share provenance and it has strictly higher confidence. Otherwise the
    // existing (target) row is kept. Reused for assigned_by, method, and assigned_at.
    const takeExcluded =
      `(excluded.assigned_by = 'user' AND assigned_by = 'auto') ` +
      `OR (excluded.assigned_by = assigned_by AND excluded.confidence > confidence)`;

    const tx = this.db.transaction((src: string, dst: string): number => {
      const moved = this.db
        .prepare(
          `INSERT INTO email_categories (message_id, account_id, category_id, confidence, assigned_by, method, assigned_at)
             SELECT message_id, account_id, ?, confidence, assigned_by, method, assigned_at
               FROM email_categories
              WHERE category_id = ?
           ON CONFLICT (message_id, account_id, category_id) DO UPDATE SET
             assigned_by = CASE WHEN ${takeExcluded} THEN excluded.assigned_by ELSE assigned_by END,
             method = CASE WHEN ${takeExcluded} THEN excluded.method ELSE method END,
             assigned_at = CASE WHEN ${takeExcluded} THEN excluded.assigned_at ELSE assigned_at END,
             confidence = MAX(confidence, excluded.confidence)`,
        )
        .run(dst, src).changes;

      this.stmts.deleteCategory.run(src);
      return moved;
    });

    return { reassigned: tx(sourceId, targetId) };
  }

  /**
   * The single primary category for every categorized email in an account, used to file
   * each email into exactly one folder. Only ACTIVE categories are considered, so a retired
   * or suggested category is never chosen as a primary (and an email whose only assignments
   * are to non-active categories does not appear). Primary is highest confidence, then a user
   * assignment over auto, then label order as a stable tiebreak.
   */
  getPrimaryCategoryPerEmail(accountId: string): Array<{ messageId: string; categoryId: string }> {
    const rows = this.db
      .prepare(
        `SELECT message_id, category_id FROM (
           SELECT ec.message_id AS message_id, ec.category_id AS category_id,
                  ROW_NUMBER() OVER (
                    PARTITION BY ec.message_id
                    ORDER BY ec.confidence DESC, (ec.assigned_by = 'user') DESC, c.label ASC
                  ) AS rn
             FROM email_categories ec
             JOIN categories c ON c.id = ec.category_id
            WHERE ec.account_id = ? AND c.status = 'active'
         ) WHERE rn = 1`,
      )
      .all(accountId) as Array<{ message_id: string; category_id: string }>;
    return rows.map((r) => ({ messageId: r.message_id, categoryId: r.category_id }));
  }

  /** Centroid and contributing-email count for a single category, or null if none. */
  getCentroid(
    categoryId: string,
    modelId: string,
  ): { vector: Float32Array; emailCount: number } | null {
    const canonical = canonicalizeModelId(modelId);
    const row = this.stmts.getCentroidByCategory.get(categoryId, canonical) as
      | { embedding: Buffer; email_count: number }
      | undefined;
    if (!row) return null;
    return { vector: bufferToVector(row.embedding), emailCount: row.email_count };
  }

  /** Message ids the user has manually categorized. Auto-categorize must not touch them. */
  getUserAssignedMessageIds(accountId: string): Set<string> {
    const rows = this.stmts.selectUserAssigned.all(accountId) as Array<{ message_id: string }>;
    return new Set(rows.map((r) => r.message_id));
  }

  /** Message ids assigned to one category with the given provenance ('user' or 'auto'). */
  listCategoryMemberIds(accountId: string, categoryId: string, assignedBy: AssignedBy): string[] {
    const rows = this.stmts.selectCategoryMembersByAssignedBy.all(
      accountId,
      categoryId,
      assignedBy,
    ) as Array<{ message_id: string }>;
    return rows.map((r) => r.message_id);
  }

  /** Message ids that already have any category. Skip set for the incremental fast pass. */
  getAssignedMessageIds(accountId: string): Set<string> {
    const rows = this.stmts.selectAssigned.all(accountId) as Array<{ message_id: string }>;
    return new Set(rows.map((r) => r.message_id));
  }

  /**
   * Message ids the incremental AI pass must not touch: user corrections and prior AI
   * results. Embed-only and uncategorized emails stay eligible for AI upgrade.
   */
  getLlmProtectedMessageIds(accountId: string): Set<string> {
    const rows = this.stmts.selectLlmProtected.all(accountId) as Array<{ message_id: string }>;
    return new Set(rows.map((r) => r.message_id));
  }

  /** Record that the LLM with this model judged these emails to fit no category. */
  recordNoneDecisions(accountId: string, modelId: string, messageIds: string[]): void {
    if (messageIds.length === 0) return;
    const now = Date.now();
    const tx = this.db.transaction(() => {
      for (const id of messageIds) this.stmts.insertDecision.run(id, accountId, modelId, now);
    });
    tx();
  }

  /** Forget the no-category decision for emails that now have a category. */
  clearDecisions(accountId: string, modelId: string, messageIds: string[]): void {
    if (messageIds.length === 0) return;
    const tx = this.db.transaction(() => {
      for (const id of messageIds) this.stmts.deleteDecision.run(id, accountId, modelId);
    });
    tx();
  }

  /** Emails the LLM with this model already judged uncategorizable. The incremental pass skips them. */
  getNoneDecisionIds(accountId: string, modelId: string): Set<string> {
    const rows = this.stmts.selectDecisions.all(accountId, modelId) as Array<{
      message_id: string;
    }>;
    return new Set(rows.map((r) => r.message_id));
  }

  /** Clear all no-category decisions for an account so the AI pass re-evaluates them. */
  clearNoneDecisions(accountId: string): number {
    return this.stmts.clearDecisionsForAccount.run(accountId).changes;
  }

  /** Forget any no-category decision across all models for one email when the user edits it directly. */
  clearDecisionsForEmail(messageId: string, accountId: string): void {
    this.stmts.deleteDecisionsForEmail.run(messageId, accountId);
  }

  /** Recent emails the user filed themselves, grouped with their labels, for LLM few-shot examples. */
  getUserCorrectionExamples(
    accountId: string,
    limit: number,
  ): Array<{ subject: string | null; labels: string[] }> {
    const rows = this.stmts.selectCorrectionExamples.all(accountId, accountId, limit) as Array<{
      message_id: string;
      subject: string | null;
      label: string;
    }>;
    const byMsg = new Map<string, { subject: string | null; labels: string[] }>();
    for (const r of rows) {
      const e = byMsg.get(r.message_id) ?? { subject: r.subject, labels: [] };
      if (!e.labels.includes(r.label)) e.labels.push(r.label);
      byMsg.set(r.message_id, e);
    }
    return [...byMsg.values()];
  }

  /** Category assignments for one email, each with its category label, highest confidence first. */
  getEmailCategoriesWithLabels(
    messageId: string,
    accountId: string,
  ): Array<CategoryAssignment & { label: string }> {
    const rows = this.stmts.selectECByEmailWithLabel.all(messageId, accountId) as Array<{
      message_id: string;
      account_id: string;
      category_id: string;
      confidence: number;
      assigned_by: AssignedBy;
      assigned_at: number;
      method: AssignmentMethod | null;
      label: string;
    }>;
    return rows.map((r) => ({
      messageId: r.message_id,
      accountId: r.account_id,
      categoryId: r.category_id,
      confidence: r.confidence,
      assignedBy: r.assigned_by,
      assignedAt: r.assigned_at,
      method: r.method,
      label: r.label,
    }));
  }

  /** Category assignments for one email, highest confidence first. */
  getEmailCategories(messageId: string, accountId: string): CategoryAssignment[] {
    const rows = this.stmts.selectECByEmail.all(messageId, accountId) as Array<{
      message_id: string;
      account_id: string;
      category_id: string;
      confidence: number;
      assigned_by: AssignedBy;
      assigned_at: number;
    }>;
    return rows.map((r) => ({
      messageId: r.message_id,
      accountId: r.account_id,
      categoryId: r.category_id,
      confidence: r.confidence,
      assignedBy: r.assigned_by,
      assignedAt: r.assigned_at,
    }));
  }

  /** Store or update a category's centroid vector and its contributing-email count. */
  saveCentroid(
    categoryId: string,
    modelId: string,
    vector: ArrayLike<number>,
    emailCount: number,
  ): void {
    if (vector.length !== EMBEDDING_DIM) {
      throw new EmbeddingDimensionError(vector.length);
    }
    const canonical = canonicalizeModelId(modelId);
    const f32 = vector instanceof Float32Array ? vector : Float32Array.from(vector);
    const buf = vectorToBuffer(f32);
    const now = Date.now();

    const tx = this.db.transaction(() => {
      const existing = this.stmts.findCentroidRowId.get(categoryId, canonical) as
        | { rowid: number }
        | undefined;

      if (existing) {
        this.stmts.updateCentroidVec.run(buf, existing.rowid);
        this.stmts.updateCentroidIndex.run(emailCount, now, existing.rowid);
        return;
      }

      const ins = this.stmts.insertCentroidVec.run(buf);
      const rowid = Number(ins.lastInsertRowid);
      this.stmts.insertCentroidIndex.run(rowid, categoryId, canonical, emailCount, now);
    });
    tx();
  }

  /** All category centroids for an account and model, with their labels and email counts. */
  getCentroidEntries(accountId: string, modelId: string): CentroidEntry[] {
    const canonical = canonicalizeModelId(modelId);
    const rows = this.stmts.listCentroids.all(accountId, canonical) as Array<{
      category_id: string;
      label: string;
      embedding: Buffer;
      email_count: number;
    }>;
    return rows.map((r) => ({
      categoryId: r.category_id,
      label: r.label,
      emailCount: r.email_count,
      vector: bufferToVector(r.embedding),
    }));
  }

  /** Map a snake-case database row to the camel-case CategoryRow shape. */
  private fromRow(row: CategoryDbRow): CategoryRow {
    return {
      id: row.id,
      accountId: row.account_id,
      label: row.label,
      description: row.description,
      source: row.source,
      canonicalKey: row.canonical_key,
      status: row.status,
      firstSeenAt: row.first_seen_at,
      retiredAt: row.retired_at,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }
}
