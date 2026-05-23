/**
 * SQLCipher key management and at-rest encryption for the local database. Resolves the database
 * key, unlocks connections, and migrates a plaintext database to encrypted in place with
 * self-healing recovery from an interrupted migration.
 */
import { randomBytes } from 'node:crypto';
import {
  readFileSync,
  writeFileSync,
  existsSync,
  copyFileSync,
  unlinkSync,
  rmSync,
  chmodSync,
  openSync,
  readSync,
  closeSync,
} from 'node:fs';
import BetterSqlite3, { type Database } from 'better-sqlite3';
import type { Logger } from 'pino';

const BACKUP_SUFFIX = '.plain.bak';
const KEY_RE = /^[0-9a-f]{64}$/i;
const SQLITE_MAGIC = 'SQLite format 3\0';

/** Load the database key, generating and persisting it on first run. */
export function resolveDbKey(keyPath: string): { keyHex: string; created: boolean } {
  if (existsSync(keyPath)) {
    const keyHex = readFileSync(keyPath, 'utf8').trim();
    if (KEY_RE.test(keyHex)) return { keyHex, created: false };
    throw new Error(
      `database key file ${keyPath} is malformed; move it aside to reset (this discards the encrypted db)`,
    );
  }
  const keyHex = randomBytes(32).toString('hex');
  writeFileSync(keyPath, keyHex, { encoding: 'utf8', mode: 0o600 });
  try {
    chmodSync(keyPath, 0o600);
  } catch {}
  return { keyHex, created: true };
}

/**
 * Unlock a connection. Must run before any other statement. The raw key skips the passphrase
 * KDF since it is already 256 bits of entropy.
 */
export function applyDbKey(db: Database, keyHex: string): void {
  db.pragma("cipher='sqlcipher'");
  db.pragma(`key="x'${keyHex}'"`);
}

/** Whether the file is an unencrypted SQLite database, read by its header magic without opening. */
export function isPlaintextDatabase(dbPath: string): boolean {
  const fd = openSync(dbPath, 'r');
  try {
    const buf = Buffer.alloc(16);
    readSync(fd, buf, 0, 16, 0);
    return buf.toString('latin1') === SQLITE_MAGIC;
  } finally {
    closeSync(fd);
  }
}

/** Whether the database opens and reads cleanly with the given key, returning false on any failure. */
function opensWithKey(dbPath: string, keyHex: string): boolean {
  try {
    const db = new BetterSqlite3(dbPath, { readonly: true });
    try {
      applyDbKey(db, keyHex);
      db.prepare('SELECT count(*) FROM sqlite_master').get();
      return true;
    } finally {
      db.close();
    }
  } catch {
    return false;
  }
}

/**
 * Make the database at `dbPath` encrypted with `keyHex`, idempotently and self-healingly.
 * Recovers from an interrupted prior migration using the plaintext backup, encrypts a plaintext
 * database in place, and leaves an already-encrypted or not-yet-created database untouched.
 */
export function ensureEncrypted(dbPath: string, keyHex: string, logger?: Logger): void {
  const backup = dbPath + BACKUP_SUFFIX;

  if (existsSync(backup)) {
    if (existsSync(dbPath) && opensWithKey(dbPath, keyHex)) {
      unlinkSync(backup);
    } else {
      logger?.warn({ dbPath }, 'recovering interrupted database encryption from backup');
      copyFileSync(backup, dbPath);
      rmSync(dbPath + '-wal', { force: true });
      rmSync(dbPath + '-shm', { force: true });
    }
  }

  if (!existsSync(dbPath)) return;
  if (!isPlaintextDatabase(dbPath)) return;

  migratePlaintextDb(dbPath, keyHex, logger);
}

/**
 * One-time, in-place migration of a plaintext database to encrypted. Flushes the WAL so the
 * safety backup is complete, backs up, rekeys, then removes the backup on success. An
 * interrupted rekey leaves the backup for ensureEncrypted to recover from.
 */
export function migratePlaintextDb(dbPath: string, keyHex: string, logger?: Logger): void {
  const pre = new BetterSqlite3(dbPath);
  try {
    pre.prepare('SELECT count(*) FROM sqlite_master').get();
    pre.pragma('wal_checkpoint(TRUNCATE)');
    pre.pragma('journal_mode = DELETE');
  } catch (err) {
    pre.close();
    throw new Error(
      `expected a plaintext database at ${dbPath} to encrypt, but it is unreadable; aborting: ${String(err)}`,
    );
  } finally {
    pre.close();
  }

  const backup = dbPath + BACKUP_SUFFIX;
  copyFileSync(dbPath, backup);
  try {
    chmodSync(backup, 0o600);
  } catch {}
  logger?.warn(
    { dbPath, backup },
    'encrypting existing database (one-time migration; may take a moment). A temporary plaintext backup exists until this completes.',
  );

  const db = new BetterSqlite3(dbPath);
  try {
    db.pragma("cipher='sqlcipher'");
    db.pragma(`rekey="x'${keyHex}'"`);
  } finally {
    db.close();
  }

  unlinkSync(backup);
  logger?.info('database encryption complete');
}
