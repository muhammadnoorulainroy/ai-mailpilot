/**
 * Aliases let a category answer to French/English and wording variants (Factures, Billing,
 * Payment Receipts) without ever changing its visible label. Lookup is by normalized text, so
 * accents and case do not matter.
 */
import type { Database, Statement } from 'better-sqlite3';
import { normalizeForMatch } from '../util/text.js';
import type { CategoryRow, CategorySource, CategoryStatus } from './category-repository.js';

/** Where an alias came from. */
export type AliasSource = 'auto' | 'user' | 'imported' | 'seed';

interface JoinedCategoryRow {
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

/** Data access for category aliases. */
export class CategoryAliasRepository {
  private readonly stmts: {
    insert: Statement<unknown[]>;
    findByAlias: Statement<unknown[]>;
    listForCategory: Statement<unknown[]>;
    remove: Statement<unknown[]>;
  };

  constructor(db: Database) {
    this.stmts = {
      insert: db.prepare(
        `INSERT INTO category_aliases (account_id, category_id, alias, normalized_alias, source, created_at)
         SELECT ?, ?, ?, ?, ?, ?
         WHERE EXISTS (SELECT 1 FROM categories WHERE id = ? AND account_id = ?)
         ON CONFLICT (account_id, normalized_alias) DO NOTHING`,
      ),
      findByAlias: db.prepare(
        `SELECT c.id, c.account_id, c.label, c.description, c.source, c.canonical_key, c.status,
                c.first_seen_at, c.retired_at, c.created_at, c.updated_at
           FROM category_aliases a
           JOIN categories c ON c.id = a.category_id AND c.account_id = a.account_id
          WHERE a.account_id = ? AND a.normalized_alias = ? AND c.status = 'active'`,
      ),
      listForCategory: db.prepare(
        'SELECT alias FROM category_aliases WHERE category_id = ? ORDER BY alias ASC',
      ),
      remove: db.prepare(
        'DELETE FROM category_aliases WHERE account_id = ? AND normalized_alias = ?',
      ),
    };
  }

  /** Record an alias for a category. A duplicate normalized alias in the account is ignored. */
  addAlias(
    accountId: string,
    categoryId: string,
    alias: string,
    source: AliasSource = 'auto',
  ): void {
    this.stmts.insert.run(
      accountId,
      categoryId,
      alias,
      normalizeForMatch(alias),
      source,
      Date.now(),
      categoryId,
      accountId,
    );
  }

  /** Resolve free text to the category it is an alias of, or null. Case and accent insensitive. */
  findByAlias(accountId: string, text: string): CategoryRow | null {
    const row = this.stmts.findByAlias.get(accountId, normalizeForMatch(text)) as
      | JoinedCategoryRow
      | undefined;
    if (!row) return null;
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

  /** All alias strings for a category. */
  listForCategory(categoryId: string): string[] {
    return (this.stmts.listForCategory.all(categoryId) as Array<{ alias: string }>).map(
      (r) => r.alias,
    );
  }

  /** Remove an alias by its text. */
  removeAlias(accountId: string, alias: string): void {
    this.stmts.remove.run(accountId, normalizeForMatch(alias));
  }
}
