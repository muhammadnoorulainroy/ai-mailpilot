import type { Database } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';

export type AccountKind = 'personal' | 'work' | 'institutional';

export interface Account {
  id: string;
  address: string;
  displayName: string | null;
  kind: AccountKind;
  createdAt: number;
}

export interface CreateAccountInput {
  address: string;
  displayName?: string;
  kind: AccountKind;
}

export class AccountRepository {
  constructor(private db: Database) {}

  create(input: CreateAccountInput): Account {
    const account: Account = {
      id: randomUUID(),
      address: input.address,
      displayName: input.displayName ?? null,
      kind: input.kind,
      createdAt: Date.now(),
    };

    this.db
      .prepare(
        'INSERT INTO accounts (id, address, display_name, kind, created_at) VALUES (?, ?, ?, ?, ?)',
      )
      .run(account.id, account.address, account.displayName, account.kind, account.createdAt);

    return account;
  }

  findById(id: string): Account | null {
    const row = this.db
      .prepare('SELECT id, address, display_name, kind, created_at FROM accounts WHERE id = ?')
      .get(id) as
      | {
          id: string;
          address: string;
          display_name: string | null;
          kind: AccountKind;
          created_at: number;
        }
      | undefined;

    return row ? this.fromRow(row) : null;
  }

  findByAddress(address: string): Account | null {
    const row = this.db
      .prepare(
        'SELECT id, address, display_name, kind, created_at FROM accounts WHERE address = ?',
      )
      .get(address) as
      | {
          id: string;
          address: string;
          display_name: string | null;
          kind: AccountKind;
          created_at: number;
        }
      | undefined;

    return row ? this.fromRow(row) : null;
  }

  list(): Account[] {
    const rows = this.db
      .prepare(
        'SELECT id, address, display_name, kind, created_at FROM accounts ORDER BY created_at ASC',
      )
      .all() as Array<{
      id: string;
      address: string;
      display_name: string | null;
      kind: AccountKind;
      created_at: number;
    }>;

    return rows.map((r) => this.fromRow(r));
  }

  delete(id: string): boolean {
    const result = this.db.prepare('DELETE FROM accounts WHERE id = ?').run(id);
    return result.changes > 0;
  }

  upsertByAddress(input: CreateAccountInput): Account {
    const existing = this.findByAddress(input.address);
    if (existing) return existing;
    return this.create(input);
  }

  private fromRow(row: {
    id: string;
    address: string;
    display_name: string | null;
    kind: AccountKind;
    created_at: number;
  }): Account {
    return {
      id: row.id,
      address: row.address,
      displayName: row.display_name,
      kind: row.kind,
      createdAt: row.created_at,
    };
  }
}
