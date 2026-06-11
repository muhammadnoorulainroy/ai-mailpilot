/**
 * Tests for FailureRepository covering failure counting up to the permanent cap,
 * per model and per kind scoping, clearing on success or by account, and cascade
 * deletion when the related email is removed.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { FailureRepository, MAX_PERMANENT_FAILURES } from '../src/repositories/failure-repository.js';

const A = 'model-a';
const B = 'model-b';

describe('FailureRepository', () => {
  let db: Database;
  let accounts: AccountRepository;
  let emails: EmailRepository;
  let failures: FailureRepository;
  let accountId: string;

  beforeEach(() => {
    db = openDatabase(':memory:');
    accounts = new AccountRepository(db);
    emails = new EmailRepository(db);
    failures = new FailureRepository(db);
    accountId = accounts.create({ address: 'a@x.y', kind: 'work' }).id;
    emails.upsertBatch([
      { messageId: 'm1', accountId, folder: 'INBOX', subject: 'one' },
      { messageId: 'm2', accountId, folder: 'INBOX', subject: 'two' },
    ]);
  });
  afterEach(() => db.close());

  it('counts up and becomes permanent only at the cap', () => {
    for (let i = 1; i < MAX_PERMANENT_FAILURES; i++) {
      const n = failures.recordFailure('m1', accountId, 'embedding', A, 'boom');
      expect(n).toBe(i);
      expect(failures.permanentlyFailedIds(accountId, 'embedding', A)).toEqual([]);
    }
    expect(failures.recordFailure('m1', accountId, 'embedding', A, 'boom')).toBe(MAX_PERMANENT_FAILURES);
    expect(failures.permanentlyFailedIds(accountId, 'embedding', A)).toEqual(['m1']);
    expect(failures.countPermanentlyFailed(accountId, 'embedding', A)).toBe(1);
  });

  it('retries under a different model: failures are scoped per model_id', () => {
    for (let i = 0; i < MAX_PERMANENT_FAILURES; i++) failures.recordFailure('m1', accountId, 'embedding', A, 'x');
    expect(failures.countPermanentlyFailed(accountId, 'embedding', A)).toBe(1);
    expect(failures.countPermanentlyFailed(accountId, 'embedding', B)).toBe(0);
    expect(failures.permanentlyFailedIds(accountId, 'embedding', B)).toEqual([]);
  });

  it('keeps embedding and triage failures separate', () => {
    for (let i = 0; i < MAX_PERMANENT_FAILURES; i++) failures.recordFailure('m1', accountId, 'triage', A, 'x');
    expect(failures.countPermanentlyFailed(accountId, 'triage', A)).toBe(1);
    expect(failures.countPermanentlyFailed(accountId, 'embedding', A)).toBe(0);
  });

  it('clears a single item (on success) and a whole account+kind (force)', () => {
    for (let i = 0; i < MAX_PERMANENT_FAILURES; i++) failures.recordFailure('m1', accountId, 'embedding', A, 'x');
    for (let i = 0; i < MAX_PERMANENT_FAILURES; i++) failures.recordFailure('m2', accountId, 'embedding', A, 'x');
    failures.clearFailure('m1', accountId, 'embedding', A);
    expect(failures.permanentlyFailedIds(accountId, 'embedding', A)).toEqual(['m2']);
    failures.clearForAccount(accountId, 'embedding');
    expect(failures.countPermanentlyFailed(accountId, 'embedding', A)).toBe(0);
  });

  it('cascades away when the email is deleted', () => {
    for (let i = 0; i < MAX_PERMANENT_FAILURES; i++) failures.recordFailure('m1', accountId, 'embedding', A, 'x');
    emails.delete('m1', accountId);
    expect(failures.countPermanentlyFailed(accountId, 'embedding', A)).toBe(0);
  });
});
