import BetterSqlite3, { type Database } from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { DB_PATH, ensureDirs } from '../util/paths.js';
import { runMigrations } from './migrations.js';
import { migrations } from './schema.js';

export function openDatabase(path: string = DB_PATH): Database {
  ensureDirs();

  const db = new BetterSqlite3(path);
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');
  db.pragma('synchronous = NORMAL');

  sqliteVec.load(db);

  const applied = runMigrations(db, migrations);
  if (applied.length > 0) {
    console.log(`Applied migrations: ${applied.join(', ')}`);
  }

  return db;
}

export type { Database } from 'better-sqlite3';
