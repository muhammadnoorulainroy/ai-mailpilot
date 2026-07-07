/**
 * Repository and types for persisting mail account records in SQLite,
 * exposing create, lookup, list, delete, and upsert operations.
 */
import type { Database, Statement } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';

/** Classification of an account by its origin or purpose. */
export type AccountKind = 'personal' | 'work' | 'institutional';

/** A stored mail account record. */
export interface Account {
  id: string;
  address: string;
  displayName: string | null;
  kind: AccountKind;
  excludeFromDiscovery: boolean;
  discoveryEnabled: boolean;
  createdAt: number;
}

/** Fields required to create a new account. */
export interface CreateAccountInput {
  address: string;
  displayName?: string;
  kind: AccountKind;
  discoveryEnabled?: boolean;
}

interface AccountDbRow {
  id: string;
  address: string;
  display_name: string | null;
  kind: AccountKind;
  exclude_from_discovery: number;
  created_at: number;
}

/** Data access for account records backed by SQLite prepared statements. */
export class AccountRepository {
  private readonly stmts: {
    insert: Statement<unknown[]>;
    findById: Statement<unknown[]>;
    findByAddress: Statement<unknown[]>;
    list: Statement<unknown[]>;
    delete: Statement<unknown[]>;
    updateMeta: Statement<unknown[]>;
    selectEligibility: Statement<unknown[]>;
    setExclude: Statement<unknown[]>;
  };

  /** Prepares the SQL statements used for all account operations. */
  constructor(db: Database) {
    this.stmts = {
      insert: db.prepare(
        `INSERT INTO accounts
           (id, address, display_name, kind, exclude_from_discovery, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`,
      ),
      findById: db.prepare(
        `SELECT id, address, display_name, kind, exclude_from_discovery, created_at
           FROM accounts WHERE id = ?`,
      ),
      findByAddress: db.prepare(
        `SELECT id, address, display_name, kind, exclude_from_discovery, created_at
           FROM accounts WHERE address = ?`,
      ),
      list: db.prepare(
        `SELECT id, address, display_name, kind, exclude_from_discovery, created_at
           FROM accounts ORDER BY created_at ASC`,
      ),
      delete: db.prepare('DELETE FROM accounts WHERE id = ?'),
      updateMeta: db.prepare('UPDATE accounts SET display_name = ?, kind = ? WHERE id = ?'),
      selectEligibility: db.prepare(
        'SELECT kind, exclude_from_discovery FROM accounts WHERE id = ?',
      ),
      setExclude: db.prepare('UPDATE accounts SET exclude_from_discovery = ? WHERE id = ?'),
    };
  }

  /** Inserts a new account with a generated id and returns it. */
  create(input: CreateAccountInput): Account {
    const discoveryEnabled = input.discoveryEnabled ?? input.kind !== 'personal';
    const account: Account = {
      id: randomUUID(),
      address: input.address,
      displayName: input.displayName ?? null,
      kind: input.kind,
      excludeFromDiscovery: !discoveryEnabled,
      discoveryEnabled,
      createdAt: Date.now(),
    };
    this.stmts.insert.run(
      account.id,
      account.address,
      account.displayName,
      account.kind,
      account.excludeFromDiscovery ? 1 : 0,
      account.createdAt,
    );
    return account;
  }

  /** Returns the account with the given id, or null if none exists. */
  findById(id: string): Account | null {
    const row = this.stmts.findById.get(id) as AccountDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  /** Returns the account matching the given address, or null if none exists. */
  findByAddress(address: string): Account | null {
    const row = this.stmts.findByAddress.get(address) as AccountDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  /** Returns all accounts ordered by creation time ascending. */
  list(): Account[] {
    const rows = this.stmts.list.all() as AccountDbRow[];
    return rows.map((r) => this.fromRow(r));
  }

  /** Deletes the account with the given id and returns whether a row was removed. */
  delete(id: string): boolean {
    return this.stmts.delete.run(id).changes > 0;
  }

  /**
   * Creates the account if its address is unseen, otherwise refreshes detected
   * metadata while preserving an existing display name when none is supplied.
   */
  upsertByAddress(input: CreateAccountInput): Account {
    const existing = this.findByAddress(input.address);
    if (!existing) return this.create(input);

    const displayName = input.displayName ?? existing.displayName;
    const becamePersonal = existing.kind !== 'personal' && input.kind === 'personal';
    if (existing.displayName !== displayName || existing.kind !== input.kind) {
      this.stmts.updateMeta.run(displayName, input.kind, existing.id);
    }
    if (input.discoveryEnabled !== undefined) {
      this.setExcludeFromDiscovery(existing.id, !input.discoveryEnabled);
    } else if (becamePersonal) {
      this.setExcludeFromDiscovery(existing.id, true);
    }
    return this.findById(existing.id) ?? { ...existing, displayName, kind: input.kind };
  }

  /** Whether discovery may run for this account. Personal accounts can opt in explicitly. */
  isDiscoveryEligible(id: string): boolean {
    const row = this.stmts.selectEligibility.get(id) as
      | { kind: AccountKind; exclude_from_discovery: number }
      | undefined;
    if (!row) return false;
    return row.exclude_from_discovery === 0;
  }

  /** Set whether this account is excluded from discovery. */
  setExcludeFromDiscovery(id: string, excluded: boolean): void {
    this.stmts.setExclude.run(excluded ? 1 : 0, id);
  }

  /** Positive user-facing update: enable or disable AI discovery for this account. */
  setDiscoveryEnabled(id: string, enabled: boolean): Account | null {
    this.setExcludeFromDiscovery(id, !enabled);
    return this.findById(id);
  }

  /** Maps a raw database row to an Account, converting snake_case columns. */
  private fromRow(row: AccountDbRow): Account {
    return {
      id: row.id,
      address: row.address,
      displayName: row.display_name,
      kind: row.kind,
      excludeFromDiscovery: row.exclude_from_discovery !== 0,
      discoveryEnabled: row.exclude_from_discovery === 0,
      createdAt: row.created_at,
    };
  }
}
