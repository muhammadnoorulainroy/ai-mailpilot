/**
 * Tests for the SQLCipher database encryption helpers: key generation and reuse,
 * malformed key handling, in-place plaintext to encrypted migration, and ensureEncrypted.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, writeFileSync, existsSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import BetterSqlite3 from 'better-sqlite3';
import {
  resolveDbKey,
  applyDbKey,
  migratePlaintextDb,
  ensureEncrypted,
  isPlaintextDatabase,
} from '../src/db/encryption.js';

describe('database encryption', () => {
  const dirs: string[] = [];
  /** Creates a tracked temp directory and registers it for cleanup in afterEach. */
  const tmp = (): string => {
    const d = mkdtempSync(join(tmpdir(), 'mp-enc-'));
    dirs.push(d);
    return d;
  };
  afterEach(() => {
    for (const d of dirs) rmSync(d, { recursive: true, force: true });
    dirs.length = 0;
  });

  it('generates a 256-bit key once and reuses it', () => {
    const keyPath = join(tmp(), 'db.key');
    const a = resolveDbKey(keyPath);
    expect(a.created).toBe(true);
    expect(a.keyHex).toMatch(/^[0-9a-f]{64}$/);

    const b = resolveDbKey(keyPath);
    expect(b.created).toBe(false);
    expect(b.keyHex).toBe(a.keyHex);
  });

  it('rejects a malformed key file rather than silently discarding the encrypted db', () => {
    const keyPath = join(tmp(), 'db.key');
    writeFileSync(keyPath, 'not-a-real-key');
    expect(() => resolveDbKey(keyPath)).toThrow(/malformed/);
  });

  it('migrates a plaintext db to encrypted in place: data preserved, no plaintext, no key no read', () => {
    const dir = tmp();
    const dbPath = join(dir, 'mailpilot.db');

    const plain = new BetterSqlite3(dbPath);
    plain.pragma('journal_mode = WAL');
    plain.exec('CREATE TABLE e(id INTEGER PRIMARY KEY, body TEXT)');
    plain.prepare('INSERT INTO e(body) VALUES (?)').run('PLAINTEXT_MARKER');
    plain.pragma('wal_checkpoint(TRUNCATE)');
    plain.close();
    expect(readFileSync(dbPath).includes(Buffer.from('PLAINTEXT_MARKER'))).toBe(true);

    const { keyHex } = resolveDbKey(join(dir, 'db.key'));
    migratePlaintextDb(dbPath, keyHex);

    expect(readFileSync(dbPath).includes(Buffer.from('PLAINTEXT_MARKER'))).toBe(false);
    expect(existsSync(dbPath + '.plain.bak')).toBe(false);

    const noKey = new BetterSqlite3(dbPath);
    expect(() => noKey.prepare('SELECT count(*) c FROM e').get()).toThrow();
    noKey.close();

    const keyed = new BetterSqlite3(dbPath);
    applyDbKey(keyed, keyHex);
    const row = keyed.prepare('SELECT body FROM e WHERE id = 1').get() as { body: string };
    expect(row.body).toBe('PLAINTEXT_MARKER');
    keyed.close();
  });

  it('ensureEncrypted migrates a plaintext db even when the key already exists', () => {
    const dir = tmp();
    const dbPath = join(dir, 'mailpilot.db');
    const plain = new BetterSqlite3(dbPath);
    plain.exec('CREATE TABLE e(id INTEGER PRIMARY KEY, body TEXT)');
    plain.prepare('INSERT INTO e(body) VALUES (?)').run('STILL_PLAINTEXT');
    plain.close();

    const { keyHex } = resolveDbKey(join(dir, 'db.key'));
    expect(isPlaintextDatabase(dbPath)).toBe(true);

    ensureEncrypted(dbPath, keyHex);

    expect(isPlaintextDatabase(dbPath)).toBe(false);
    const keyed = new BetterSqlite3(dbPath);
    applyDbKey(keyed, keyHex);
    expect((keyed.prepare('SELECT body FROM e WHERE id=1').get() as { body: string }).body).toBe('STILL_PLAINTEXT');
    keyed.close();
  });

  it('ensureEncrypted is a no-op on an already-encrypted db', () => {
    const dir = tmp();
    const dbPath = join(dir, 'mailpilot.db');
    const { keyHex } = resolveDbKey(join(dir, 'db.key'));
    const enc = new BetterSqlite3(dbPath);
    applyDbKey(enc, keyHex);
    enc.exec('CREATE TABLE e(id INTEGER PRIMARY KEY)');
    enc.close();
    expect(isPlaintextDatabase(dbPath)).toBe(false);

    expect(() => ensureEncrypted(dbPath, keyHex)).not.toThrow();
    expect(isPlaintextDatabase(dbPath)).toBe(false);
  });
});
