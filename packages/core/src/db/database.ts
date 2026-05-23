/**
 * Database connection setup for the core package. Opens the SQLite database,
 * loads the vector extension, applies encryption for persistent stores, and runs migrations.
 */
import BetterSqlite3, { type Database } from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import type { Logger } from 'pino';
import { DB_PATH, DB_KEY_PATH, ensureDirs } from '../util/paths.js';
import { runMigrations } from './migrations.js';
import { migrations } from './schema.js';
import { resolveDbKey, applyDbKey, ensureEncrypted } from './encryption.js';

/**
 * Opens the SQLite database at the given path, loading the vector extension and
 * applying migrations. Persistent databases are encrypted, while in-memory ones are not.
 */
export function openDatabase(path: string = DB_PATH, logger?: Logger): Database {
  ensureDirs();

  if (path === ':memory:') {
    const mem = new BetterSqlite3(path);
    mem.pragma('foreign_keys = ON');
    sqliteVec.load(mem);
    runMigrations(mem, migrations);
    return mem;
  }

  const { keyHex } = resolveDbKey(DB_KEY_PATH);
  ensureEncrypted(path, keyHex, logger);

  const db = new BetterSqlite3(path);
  applyDbKey(db, keyHex);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
  db.pragma('synchronous = NORMAL');

  sqliteVec.load(db);

  const applied = runMigrations(db, migrations);
  if (applied.length > 0) {
    logger?.info({ applied }, 'database migrations applied');
  }

  return db;
}

export type { Database } from 'better-sqlite3';
