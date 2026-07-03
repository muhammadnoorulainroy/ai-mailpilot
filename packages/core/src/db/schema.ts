/**
 * Defines the SQLite database schema as an ordered sequence of versioned
 * migrations that build the tables, indexes, triggers, and vector tables.
 */
import type { Migration } from './migrations.js';
import { normalizeForMatch } from '../util/text.js';

/** Dimensionality of the embedding vectors used by all vec0 tables. */
export const EMBEDDING_DIM = 1024;

/**
 * Deterministic, per-account-unique canonical key derived from a label. Frozen once assigned so
 * a category keeps its identity across discovery runs even when its cluster shifts.
 */
function canonicalKeyFor(label: string, taken: Set<string>): string {
  const base = normalizeForMatch(label).replace(/\s+/g, '_').slice(0, 60) || 'category';
  if (!taken.has(base)) return base;
  for (let i = 2; ; i++) {
    const candidate = `${base}_${i}`;
    if (!taken.has(candidate)) return candidate;
  }
}

/** Ordered list of database migrations applied in sequence to build the schema. */
export const migrations: Migration[] = [
  {
    version: 1,
    name: 'initial_schema',
    up: (db) => {
      db.exec(`
        CREATE TABLE accounts (
          id TEXT PRIMARY KEY,
          address TEXT NOT NULL,
          display_name TEXT,
          kind TEXT NOT NULL CHECK (kind IN ('personal', 'work', 'institutional')),
          created_at INTEGER NOT NULL
        );

        CREATE TABLE categories (
          id TEXT PRIMARY KEY,
          account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
          label TEXT NOT NULL,
          description TEXT,
          source TEXT NOT NULL CHECK (source IN ('auto', 'user', 'imported')),
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL,
          UNIQUE (account_id, label)
        );

        CREATE INDEX idx_categories_account ON categories(account_id);

        CREATE TABLE emails (
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
          folder TEXT NOT NULL,
          subject TEXT,
          from_addr TEXT,
          date INTEGER,
          has_attachments INTEGER NOT NULL DEFAULT 0,
          indexed_at INTEGER NOT NULL,
          PRIMARY KEY (message_id, account_id)
        );

        CREATE INDEX idx_emails_account_folder ON emails(account_id, folder);
        CREATE INDEX idx_emails_date ON emails(account_id, date DESC);

        CREATE TABLE email_categories (
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          category_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
          confidence REAL NOT NULL,
          assigned_by TEXT NOT NULL CHECK (assigned_by IN ('user', 'auto')),
          assigned_at INTEGER NOT NULL,
          PRIMARY KEY (message_id, account_id, category_id),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );

        CREATE TABLE triage (
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          bucket TEXT NOT NULL CHECK (bucket IN ('urgent', 'summarize', 'spam', 'personal')),
          reasoning TEXT,
          classified_at INTEGER NOT NULL,
          PRIMARY KEY (message_id, account_id),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );

        CREATE TABLE awaiting_response (
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          marked_urgent_at INTEGER NOT NULL,
          replied_at INTEGER,
          PRIMARY KEY (message_id, account_id),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );

        CREATE TABLE drafts (
          id TEXT PRIMARY KEY,
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          draft_body TEXT NOT NULL,
          model_used TEXT NOT NULL,
          approved INTEGER NOT NULL DEFAULT 0,
          created_at INTEGER NOT NULL,
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );

        CREATE TABLE mailing_lists (
          id TEXT PRIMARY KEY,
          account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
          list_address TEXT NOT NULL,
          display_name TEXT,
          rules_json TEXT NOT NULL DEFAULT '{}',
          UNIQUE (account_id, list_address)
        );

        CREATE TABLE conversations (
          id TEXT PRIMARY KEY,
          account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
          history_json TEXT NOT NULL,
          updated_at INTEGER NOT NULL
        );
      `);
    },
  },
  {
    version: 2,
    name: 'vector_tables',
    up: (db) => {
      db.exec(`
        CREATE VIRTUAL TABLE email_embeddings USING vec0(
          embedding FLOAT[${EMBEDDING_DIM}]
        );

        CREATE TABLE email_embedding_index (
          rowid INTEGER PRIMARY KEY,
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          model_id TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          UNIQUE (message_id, account_id, model_id),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );

        CREATE INDEX idx_email_embedding_lookup ON email_embedding_index(message_id, account_id, model_id);

        CREATE VIRTUAL TABLE category_embeddings USING vec0(
          embedding FLOAT[${EMBEDDING_DIM}]
        );

        CREATE TABLE category_embedding_index (
          rowid INTEGER PRIMARY KEY,
          category_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
          model_id TEXT NOT NULL,
          email_count INTEGER NOT NULL DEFAULT 0,
          updated_at INTEGER NOT NULL,
          UNIQUE (category_id, model_id)
        );
      `);
    },
  },
  {
    version: 3,
    name: 'email_body',
    up: (db) => {
      db.exec(`
        ALTER TABLE emails ADD COLUMN body TEXT;
        ALTER TABLE emails ADD COLUMN body_format TEXT DEFAULT 'text';
      `);
    },
  },
  {
    version: 4,
    name: 'vec_cascade_triggers',
    up: (db) => {
      db.exec(`
        DELETE FROM email_embeddings WHERE rowid NOT IN (SELECT rowid FROM email_embedding_index);
        DELETE FROM category_embeddings WHERE rowid NOT IN (SELECT rowid FROM category_embedding_index);

        CREATE TRIGGER trg_email_embedding_index_delete
        AFTER DELETE ON email_embedding_index
        BEGIN
          DELETE FROM email_embeddings WHERE rowid = OLD.rowid;
        END;

        CREATE TRIGGER trg_category_embedding_index_delete
        AFTER DELETE ON category_embedding_index
        BEGIN
          DELETE FROM category_embeddings WHERE rowid = OLD.rowid;
        END;
      `);
    },
  },
  {
    version: 5,
    name: 'canonicalize_model_ids',
    up: (db) => {
      db.exec(`
        UPDATE email_embedding_index
           SET model_id = SUBSTR(model_id, 1, LENGTH(model_id) - 7)
         WHERE model_id LIKE '%:latest';

        UPDATE category_embedding_index
           SET model_id = SUBSTR(model_id, 1, LENGTH(model_id) - 7)
         WHERE model_id LIKE '%:latest';
      `);
    },
  },
  {
    version: 6,
    name: 'email_body_fetched',
    up: (db) => {
      db.exec(`
        ALTER TABLE emails ADD COLUMN body_fetched INTEGER NOT NULL DEFAULT 1;
        UPDATE emails SET body_fetched = 0 WHERE body IS NULL;
      `);
    },
  },
  {
    version: 7,
    name: 'assignment_method',
    up: (db) => {
      db.exec(`ALTER TABLE email_categories ADD COLUMN method TEXT;`);
    },
  },
  {
    version: 8,
    name: 'email_fts',
    up: (db) => {
      db.exec(`
        CREATE VIRTUAL TABLE email_fts USING fts5(
          subject, from_addr, body,
          content='emails',
          content_rowid='rowid',
          tokenize='unicode61 remove_diacritics 2'
        );

        INSERT INTO email_fts(rowid, subject, from_addr, body)
          SELECT rowid, subject, from_addr, body FROM emails;

        CREATE TRIGGER trg_emails_fts_insert AFTER INSERT ON emails BEGIN
          INSERT INTO email_fts(rowid, subject, from_addr, body)
          VALUES (new.rowid, new.subject, new.from_addr, new.body);
        END;

        CREATE TRIGGER trg_emails_fts_delete AFTER DELETE ON emails BEGIN
          INSERT INTO email_fts(email_fts, rowid, subject, from_addr, body)
          VALUES ('delete', old.rowid, old.subject, old.from_addr, old.body);
        END;

        CREATE TRIGGER trg_emails_fts_update AFTER UPDATE ON emails
        WHEN old.subject IS NOT new.subject
          OR old.from_addr IS NOT new.from_addr
          OR old.body IS NOT new.body
        BEGIN
          INSERT INTO email_fts(email_fts, rowid, subject, from_addr, body)
          VALUES ('delete', old.rowid, old.subject, old.from_addr, old.body);
          INSERT INTO email_fts(rowid, subject, from_addr, body)
          VALUES (new.rowid, new.subject, new.from_addr, new.body);
        END;
      `);
    },
  },
  {
    version: 9,
    name: 'attachments',
    up: (db) => {
      db.exec(`
        CREATE TABLE attachments (
          id TEXT PRIMARY KEY,
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          filename TEXT NOT NULL,
          content_type TEXT,
          part_name TEXT NOT NULL,
          size INTEGER,
          status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'extracted', 'empty', 'unsupported', 'error')),
          error TEXT,
          char_count INTEGER NOT NULL DEFAULT 0,
          indexed_at INTEGER NOT NULL,
          UNIQUE (message_id, account_id, part_name),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_attachments_message ON attachments(message_id, account_id);

        CREATE TABLE attachment_chunks (
          attachment_id TEXT NOT NULL REFERENCES attachments(id) ON DELETE CASCADE,
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          chunk_index INTEGER NOT NULL,
          text TEXT NOT NULL,
          created_at INTEGER NOT NULL
        );
        CREATE INDEX idx_attachment_chunks_attachment ON attachment_chunks(attachment_id);
        CREATE INDEX idx_attachment_chunks_account ON attachment_chunks(account_id);

        CREATE VIRTUAL TABLE attachment_chunk_embeddings USING vec0(embedding FLOAT[${EMBEDDING_DIM}]);

        CREATE TABLE attachment_chunk_embedding_index (
          rowid INTEGER PRIMARY KEY,
          chunk_rowid INTEGER NOT NULL,
          account_id TEXT NOT NULL,
          model_id TEXT NOT NULL,
          created_at INTEGER NOT NULL,
          UNIQUE (chunk_rowid, model_id)
        );
        CREATE INDEX idx_attachment_chunk_emb_lookup
          ON attachment_chunk_embedding_index(chunk_rowid, model_id);

        CREATE VIRTUAL TABLE attachment_fts USING fts5(
          text,
          content='attachment_chunks',
          content_rowid='rowid',
          tokenize='unicode61 remove_diacritics 2'
        );

        -- vec0 tables do not FK-cascade; drop the vector when its index row goes.
        CREATE TRIGGER trg_attachment_chunk_emb_delete
        AFTER DELETE ON attachment_chunk_embedding_index BEGIN
          DELETE FROM attachment_chunk_embeddings WHERE rowid = OLD.rowid;
        END;

        -- A chunk deletion cascades to its embedding index row and its FTS entry.
        CREATE TRIGGER trg_attachment_chunks_delete
        AFTER DELETE ON attachment_chunks BEGIN
          DELETE FROM attachment_chunk_embedding_index WHERE chunk_rowid = OLD.rowid;
          INSERT INTO attachment_fts(attachment_fts, rowid, text) VALUES ('delete', OLD.rowid, OLD.text);
        END;

        CREATE TRIGGER trg_attachment_chunks_insert
        AFTER INSERT ON attachment_chunks BEGIN
          INSERT INTO attachment_fts(rowid, text) VALUES (new.rowid, new.text);
        END;
      `);
    },
  },
  {
    version: 10,
    name: 'processing_failures',
    up: (db) => {
      db.exec(`
        CREATE TABLE processing_failures (
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          kind TEXT NOT NULL CHECK (kind IN ('embedding', 'triage')),
          failure_count INTEGER NOT NULL DEFAULT 0,
          last_error TEXT,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY (message_id, account_id, kind),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_processing_failures_account ON processing_failures(account_id, kind, failure_count);
      `);
    },
  },
  {
    version: 11,
    name: 'processing_failures_model_scoped',
    up: (db) => {
      db.exec(`
        DROP TABLE IF EXISTS processing_failures;
        CREATE TABLE processing_failures (
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          kind TEXT NOT NULL CHECK (kind IN ('embedding', 'triage')),
          model_id TEXT NOT NULL,
          failure_count INTEGER NOT NULL DEFAULT 0,
          last_error TEXT,
          updated_at INTEGER NOT NULL,
          PRIMARY KEY (message_id, account_id, kind, model_id),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_processing_failures_account
          ON processing_failures(account_id, kind, model_id, failure_count);
      `);
    },
  },
  {
    version: 12,
    name: 'llm_category_decisions',
    up: (db) => {
      db.exec(`
        CREATE TABLE llm_category_decisions (
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          model_id TEXT NOT NULL,
          decided_at INTEGER NOT NULL,
          PRIMARY KEY (message_id, account_id, model_id),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_llm_decisions_account ON llm_category_decisions(account_id, model_id);
      `);
    },
  },
  {
    version: 13,
    name: 'categorize_jobs',
    up: (db) => {
      db.exec(`
        CREATE TABLE categorize_jobs (
          account_id TEXT PRIMARY KEY,
          model_id TEXT,
          status TEXT NOT NULL,
          total INTEGER NOT NULL DEFAULT 0,
          processed INTEGER NOT NULL DEFAULT 0,
          assigned INTEGER NOT NULL DEFAULT 0,
          uncategorized INTEGER NOT NULL DEFAULT 0,
          failed INTEGER NOT NULL DEFAULT 0,
          clusters INTEGER NOT NULL DEFAULT 0,
          clusters_processed INTEGER NOT NULL DEFAULT 0,
          error TEXT,
          started_at INTEGER,
          updated_at INTEGER NOT NULL
        );
      `);
    },
  },
  {
    version: 14,
    name: 'categorize_jobs_completed_at',
    up: (db) => {
      db.exec('ALTER TABLE categorize_jobs ADD COLUMN completed_at INTEGER;');
    },
  },
  {
    version: 15,
    name: 'categorize_jobs_call_counters',
    up: (db) => {
      db.exec(`
        ALTER TABLE categorize_jobs ADD COLUMN gated_clusters INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE categorize_jobs ADD COLUMN llm_calls INTEGER NOT NULL DEFAULT 0;
      `);
    },
  },
  {
    version: 16,
    name: 'triage_metadata_and_resolution',
    up: (db) => {
      db.exec(`
        ALTER TABLE triage ADD COLUMN metadata TEXT;
        ALTER TABLE triage ADD COLUMN action_required INTEGER NOT NULL DEFAULT 0;
        ALTER TABLE triage ADD COLUMN deadline_at INTEGER;
        ALTER TABLE triage ADD COLUMN dismissed_at INTEGER;
        ALTER TABLE triage ADD COLUMN done_at INTEGER;
        ALTER TABLE triage ADD COLUMN snoozed_until INTEGER;
      `);
    },
  },
  {
    version: 17,
    name: 'email_assistant_summaries',
    up: (db) => {
      db.exec(`
        CREATE TABLE email_assistant_summaries (
          message_id TEXT NOT NULL,
          account_id TEXT NOT NULL,
          content_hash TEXT NOT NULL,
          model_id TEXT NOT NULL,
          provider TEXT NOT NULL CHECK (provider IN ('local', 'cloud')),
          summary_json TEXT NOT NULL,
          generated_at INTEGER NOT NULL,
          PRIMARY KEY (message_id, account_id),
          FOREIGN KEY (message_id, account_id) REFERENCES emails(message_id, account_id) ON DELETE CASCADE
        );
        CREATE INDEX idx_email_assistant_account ON email_assistant_summaries(account_id);
      `);
    },
  },
  {
    version: 18,
    name: 'discovery_foundation',
    up: (db) => {
      db.exec(`
        ALTER TABLE categories ADD COLUMN canonical_key TEXT;
        ALTER TABLE categories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'
          CHECK (status IN ('active', 'suggested', 'retired'));
        ALTER TABLE categories ADD COLUMN first_seen_at INTEGER;
        ALTER TABLE categories ADD COLUMN retired_at INTEGER;

        ALTER TABLE accounts ADD COLUMN exclude_from_discovery INTEGER NOT NULL DEFAULT 0;

        CREATE TABLE category_aliases (
          account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
          category_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
          alias TEXT NOT NULL,
          normalized_alias TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'auto' CHECK (source IN ('auto', 'user', 'imported', 'seed')),
          created_at INTEGER NOT NULL,
          PRIMARY KEY (account_id, normalized_alias)
        );
        CREATE INDEX idx_category_aliases_category ON category_aliases(category_id);

        CREATE TABLE discovery_audit (
          id TEXT PRIMARY KEY,
          account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
          ran_at INTEGER NOT NULL,
          flow TEXT NOT NULL CHECK (flow IN ('topic_discovery', 'improve_categories')),
          account_kind TEXT NOT NULL,
          provider TEXT NOT NULL CHECK (provider IN ('local', 'cloud')),
          status TEXT NOT NULL CHECK (status IN ('ok', 'blocked', 'failed', 'insufficient', 'skipped')),
          model_id TEXT,
          pool_size INTEGER NOT NULL DEFAULT 0,
          sample_size INTEGER NOT NULL DEFAULT 0,
          emails_exposed INTEGER NOT NULL DEFAULT 0,
          fields_read TEXT NOT NULL DEFAULT '[]',
          redacted INTEGER NOT NULL DEFAULT 0,
          omitted_categories TEXT NOT NULL DEFAULT '[]',
          error TEXT
        );
        CREATE INDEX idx_discovery_audit_account ON discovery_audit(account_id, ran_at);
      `);

      // Freeze a deterministic canonical_key on every existing category and stamp first_seen_at.
      const rows = db
        .prepare(
          `SELECT id, account_id, label, created_at FROM categories ORDER BY account_id, created_at, id`,
        )
        .all() as Array<{ id: string; account_id: string; label: string; created_at: number }>;
      const taken = new Map<string, Set<string>>();
      const upd = db.prepare(
        `UPDATE categories SET canonical_key = ?, first_seen_at = ? WHERE id = ?`,
      );
      for (const r of rows) {
        const seen = taken.get(r.account_id) ?? new Set<string>();
        const key = canonicalKeyFor(r.label, seen);
        seen.add(key);
        taken.set(r.account_id, seen);
        upd.run(key, r.created_at, r.id);
      }

      db.exec(
        `CREATE UNIQUE INDEX idx_categories_canonical ON categories(account_id, canonical_key);`,
      );

      // A unique index still allows multiple NULLs in SQLite; enforce non-null with triggers
      // rather than a table rebuild (PRAGMA foreign_keys cannot toggle inside this transaction).
      db.exec(`
        CREATE TRIGGER trg_categories_canonical_not_null_insert
        BEFORE INSERT ON categories WHEN NEW.canonical_key IS NULL
        BEGIN SELECT RAISE(ABORT, 'canonical_key must not be null'); END;

        CREATE TRIGGER trg_categories_canonical_not_null_update
        BEFORE UPDATE ON categories WHEN NEW.canonical_key IS NULL
        BEGIN SELECT RAISE(ABORT, 'canonical_key must not be null'); END;
      `);
    },
  },
  {
    version: 19,
    name: 'category_proposals',
    up: (db) => {
      db.exec(`
        CREATE TABLE category_proposals (
          id TEXT PRIMARY KEY,
          account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
          category_id TEXT NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
          run_id TEXT NOT NULL,
          cluster_index INTEGER NOT NULL,
          label TEXT NOT NULL,
          description TEXT NOT NULL,
          canonical_key TEXT NOT NULL,
          suggested_key TEXT NOT NULL DEFAULT '',
          embedding_model_id TEXT NOT NULL,
          centroid BLOB NOT NULL,
          member_ids TEXT NOT NULL,
          proposed_count INTEGER NOT NULL,
          cohesion REAL NOT NULL,
          separation REAL NOT NULL,
          confidence REAL NOT NULL,
          evidence TEXT NOT NULL DEFAULT '[]',
          status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'applied', 'dismissed')),
          created_at INTEGER NOT NULL,
          applied_at INTEGER,
          dismissed_at INTEGER
        );
        CREATE INDEX idx_category_proposals_account_status
          ON category_proposals(account_id, status, created_at DESC);
        CREATE INDEX idx_category_proposals_category ON category_proposals(category_id);
      `);

      // Rebuild discovery_audit to widen the flow CHECK to the new discovery_proposal run. SQLite
      // cannot alter a CHECK in place, so copy the append-only table through a new definition.
      db.exec(`
        CREATE TABLE discovery_audit_new (
          id TEXT PRIMARY KEY,
          account_id TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
          ran_at INTEGER NOT NULL,
          flow TEXT NOT NULL
            CHECK (flow IN ('topic_discovery', 'improve_categories', 'discovery_proposal')),
          account_kind TEXT NOT NULL,
          provider TEXT NOT NULL CHECK (provider IN ('local', 'cloud')),
          status TEXT NOT NULL
            CHECK (status IN ('ok', 'blocked', 'failed', 'insufficient', 'skipped')),
          model_id TEXT,
          pool_size INTEGER NOT NULL DEFAULT 0,
          sample_size INTEGER NOT NULL DEFAULT 0,
          emails_exposed INTEGER NOT NULL DEFAULT 0,
          fields_read TEXT NOT NULL DEFAULT '[]',
          redacted INTEGER NOT NULL DEFAULT 0,
          omitted_categories TEXT NOT NULL DEFAULT '[]',
          error TEXT
        );
        INSERT INTO discovery_audit_new SELECT * FROM discovery_audit;
        DROP TABLE discovery_audit;
        ALTER TABLE discovery_audit_new RENAME TO discovery_audit;
        CREATE INDEX idx_discovery_audit_account ON discovery_audit(account_id, ran_at);
      `);
    },
  },
  {
    version: 20,
    name: 'structural_proposals',
    up: (db) => {
      // Phase 3.3 foundation: let category_proposals carry structural (split/merge/retire) proposals.
      // kind defaults to 'new_category' so existing rows keep their behavior. source_category_id is a
      // plain (non-cascading) column: a merge deletes the source, and it must NOT cascade-delete the
      // proposal (whose category_id points at the surviving target). suppression_key keeps a re-run
      // from re-proposing a resolved structural suggestion. The parent's cluster columns are left as
      // deterministic placeholders for structural kinds; split cluster data lives in the children table.
      db.exec(`
        ALTER TABLE category_proposals ADD COLUMN kind TEXT NOT NULL DEFAULT 'new_category'
          CHECK (kind IN ('new_category', 'split', 'merge', 'retire'));
        ALTER TABLE category_proposals ADD COLUMN source_category_id TEXT;
        ALTER TABLE category_proposals ADD COLUMN suppression_key TEXT NOT NULL DEFAULT '';

        CREATE TABLE category_proposal_children (
          id TEXT PRIMARY KEY,
          proposal_id TEXT NOT NULL REFERENCES category_proposals(id) ON DELETE CASCADE,
          label TEXT NOT NULL,
          description TEXT NOT NULL DEFAULT '',
          canonical_key TEXT NOT NULL,
          embedding_model_id TEXT NOT NULL,
          centroid BLOB NOT NULL,
          member_ids TEXT NOT NULL DEFAULT '[]',
          proposed_count INTEGER NOT NULL DEFAULT 0,
          cohesion REAL NOT NULL DEFAULT 0,
          separation REAL NOT NULL DEFAULT 0,
          confidence REAL NOT NULL DEFAULT 0,
          created_at INTEGER NOT NULL
        );
        CREATE INDEX idx_proposal_children_proposal ON category_proposal_children(proposal_id);
      `);
    },
  },
];
