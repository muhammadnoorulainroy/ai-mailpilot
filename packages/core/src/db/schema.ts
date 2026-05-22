/**
 * Defines the SQLite database schema as an ordered sequence of versioned
 * migrations that build the tables, indexes, triggers, and vector tables.
 */
import type { Migration } from './migrations.js';

/** Dimensionality of the embedding vectors used by all vec0 tables. */
export const EMBEDDING_DIM = 1024;

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
];
