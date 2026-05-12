import type { Migration } from './migrations.js';

export const EMBEDDING_DIM = 1024;

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
];
