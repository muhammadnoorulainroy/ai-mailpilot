/**
 * Repository for persisting chat conversations in the conversations table, storing each
 * conversation's turns and rolling summary as a single JSON history blob per row.
 */
import type { Database, Statement } from 'better-sqlite3';
import { randomUUID } from 'node:crypto';

/** A retrieved source attached to a chat turn, used to cite the email it came from. */
export interface StoredChatSource {
  messageId: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  score: number;
  attachmentName?: string;
}

/** A single user or assistant message in a conversation, with optional cited sources. */
export interface StoredChatTurn {
  role: 'user' | 'assistant';
  content: string;
  at: number;
  sources?: StoredChatSource[];
}

/** A stored chat conversation with its turns and rolling summary of folded-out turns. */
export interface Conversation {
  id: string;
  accountId: string;
  turns: StoredChatTurn[];
  summary: string;
  summarizedCount: number;
  updatedAt: number;
}

/** Lightweight conversation entry for sidebar listings, carrying a short preview. */
export interface ConversationSummary {
  id: string;
  updatedAt: number;
  preview: string;
}

interface ConversationDbRow {
  id: string;
  account_id: string;
  history_json: string;
  updated_at: number;
}

interface HistoryBlob {
  turns: StoredChatTurn[];
  summary: string;
  summarizedCount: number;
}

const PREVIEW_CHARS = 80;

/**
 * Persist chat conversations in the conversations table with history as JSON, one row each.
 * The server is the source of truth so records survive reloads and the LLM context can be
 * rebuilt from stored turns. The blob also carries the rolling summary so long chats stay
 * coherent without a separate column.
 */
export class ConversationRepository {
  private readonly stmts: {
    insert: Statement<unknown[]>;
    get: Statement<unknown[]>;
    getScoped: Statement<unknown[]>;
    update: Statement<unknown[]>;
    updateHistory: Statement<unknown[]>;
    listForAccount: Statement<unknown[]>;
    delete: Statement<unknown[]>;
    deleteScoped: Statement<unknown[]>;
  };

  /** Prepare the reusable statements for conversation reads, writes, and deletes. */
  constructor(private db: Database) {
    this.stmts = {
      insert: db.prepare(
        'INSERT INTO conversations (id, account_id, history_json, updated_at) VALUES (?, ?, ?, ?)',
      ),
      get: db.prepare(
        'SELECT id, account_id, history_json, updated_at FROM conversations WHERE id = ?',
      ),
      getScoped: db.prepare(
        'SELECT id, account_id, history_json, updated_at FROM conversations WHERE id = ? AND account_id = ?',
      ),
      update: db.prepare('UPDATE conversations SET history_json = ?, updated_at = ? WHERE id = ?'),
      updateHistory: db.prepare('UPDATE conversations SET history_json = ? WHERE id = ?'),
      listForAccount: db.prepare(
        'SELECT id, account_id, history_json, updated_at FROM conversations WHERE account_id = ? ORDER BY updated_at DESC LIMIT ?',
      ),
      delete: db.prepare('DELETE FROM conversations WHERE id = ?'),
      deleteScoped: db.prepare('DELETE FROM conversations WHERE id = ? AND account_id = ?'),
    };
  }

  /** Create an empty conversation for the account and persist it. */
  create(accountId: string): Conversation {
    const id = randomUUID();
    const now = Date.now();
    this.stmts.insert.run(id, accountId, JSON.stringify(this.blob([], '', 0)), now);
    return { id, accountId, turns: [], summary: '', summarizedCount: 0, updatedAt: now };
  }

  /** Load a conversation by id, or null if it does not exist. */
  get(id: string): Conversation | null {
    const row = this.stmts.get.get(id) as ConversationDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  /** Load a conversation only if it belongs to the account, preventing cross-account access. */
  getForAccount(id: string, accountId: string): Conversation | null {
    const row = this.stmts.getScoped.get(id, accountId) as ConversationDbRow | undefined;
    return row ? this.fromRow(row) : null;
  }

  /** Delete a conversation only if it belongs to the account. Returns whether a row was removed. */
  deleteForAccount(id: string, accountId: string): boolean {
    return this.stmts.deleteScoped.run(id, accountId).changes > 0;
  }

  /** Append turns and bump updated_at, preserving the rolling summary. */
  append(id: string, turns: StoredChatTurn[]): void {
    const existing = this.get(id);
    if (!existing) return;
    const merged = [...existing.turns, ...turns];
    this.stmts.update.run(
      JSON.stringify(this.blob(merged, existing.summary, existing.summarizedCount)),
      Date.now(),
      id,
    );
  }

  /**
   * Replace the rolling summary and the count of turns it covers without bumping updated_at.
   * Monotonic: a write that would move summarizedCount backwards is ignored so a stale
   * concurrent request can never regress the summary.
   */
  updateSummary(id: string, summary: string, summarizedCount: number): void {
    const existing = this.get(id);
    if (!existing || summarizedCount < existing.summarizedCount) return;
    this.stmts.updateHistory.run(
      JSON.stringify(this.blob(existing.turns, summary, summarizedCount)),
      id,
    );
  }

  /** List recent conversations for an account, most recently updated first. */
  listForAccount(accountId: string, limit = 30): ConversationSummary[] {
    const rows = this.stmts.listForAccount.all(accountId, limit) as ConversationDbRow[];
    return rows.map((r) => {
      const { turns } = this.parseHistory(r.history_json);
      const firstUser = turns.find((t) => t.role === 'user');
      return {
        id: r.id,
        updatedAt: r.updated_at,
        preview: (firstUser?.content ?? '(empty)').slice(0, PREVIEW_CHARS),
      };
    });
  }

  /** Delete a conversation by id. Returns whether a row was removed. */
  delete(id: string): boolean {
    return this.stmts.delete.run(id).changes > 0;
  }

  /** Build the history blob that is serialized into the history_json column. */
  private blob(turns: StoredChatTurn[], summary: string, summarizedCount: number): HistoryBlob {
    return { turns, summary, summarizedCount };
  }

  /** Map a raw database row to a Conversation, parsing its stored history. */
  private fromRow(row: ConversationDbRow): Conversation {
    const { turns, summary, summarizedCount } = this.parseHistory(row.history_json);
    return {
      id: row.id,
      accountId: row.account_id,
      turns,
      summary,
      summarizedCount,
      updatedAt: row.updated_at,
    };
  }

  /** Parse a stored history blob, tolerating malformed or partial JSON with safe defaults. */
  private parseHistory(json: string): HistoryBlob {
    try {
      const parsed = JSON.parse(json) as Partial<HistoryBlob>;
      const turns = Array.isArray(parsed.turns) ? parsed.turns : [];
      const summary = typeof parsed.summary === 'string' ? parsed.summary : '';
      const summarizedCount =
        typeof parsed.summarizedCount === 'number'
          ? Math.max(0, Math.min(parsed.summarizedCount, turns.length))
          : 0;
      return { turns, summary, summarizedCount };
    } catch {
      return { turns: [], summary: '', summarizedCount: 0 };
    }
  }
}
