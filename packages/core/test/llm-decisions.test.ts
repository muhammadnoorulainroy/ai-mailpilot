/**
 * Tests for durable "no category" LLM decisions in CategoryRepository, covering
 * recording and reading decisions, scoping by model, clearing on category
 * assignment or retry, per-email clears, cascade on email delete, and
 * surfacing recent user corrections as few-shot examples.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';

const MODEL = 'qwen3:8b';

describe('LLM no-category decisions (durable, so the AI pass does not reprocess them)', () => {
  let db: Database;
  let accounts: AccountRepository;
  let emails: EmailRepository;
  let categories: CategoryRepository;
  let accountId: string;

  beforeEach(() => {
    db = openDatabase(':memory:');
    accounts = new AccountRepository(db);
    emails = new EmailRepository(db);
    categories = new CategoryRepository(db);
    accountId = accounts.create({ address: 'a@x.y', kind: 'work' }).id;
    emails.upsertBatch([
      { messageId: 'm1', accountId, folder: 'INBOX', subject: 'one' },
      { messageId: 'm2', accountId, folder: 'INBOX', subject: 'two' },
    ]);
  });
  afterEach(() => db.close());

  it('records and reads "no category" decisions, so they become protected/locked', () => {
    categories.recordNoneDecisions(accountId, MODEL, ['m1', 'm2']);
    expect([...categories.getNoneDecisionIds(accountId, MODEL)].sort()).toEqual(['m1', 'm2']);
  });

  it('scopes decisions by model, so switching the model re-evaluates them', () => {
    categories.recordNoneDecisions(accountId, MODEL, ['m1']);
    expect(categories.getNoneDecisionIds(accountId, MODEL).has('m1')).toBe(true);
    expect(categories.getNoneDecisionIds(accountId, 'other-model').has('m1')).toBe(false);
  });

  it('clears a decision once the email gets a category, and clears all on retry', () => {
    categories.recordNoneDecisions(accountId, MODEL, ['m1', 'm2']);
    categories.clearDecisions(accountId, MODEL, ['m1']);
    expect([...categories.getNoneDecisionIds(accountId, MODEL)]).toEqual(['m2']);
    categories.clearNoneDecisions(accountId);
    expect(categories.getNoneDecisionIds(accountId, MODEL).size).toBe(0);
  });

  it('clears decisions for one email across all models when the user edits it directly', () => {
    categories.recordNoneDecisions(accountId, MODEL, ['m1']);
    categories.recordNoneDecisions(accountId, 'other-model', ['m1']);
    categories.clearDecisionsForEmail('m1', accountId);
    expect(categories.getNoneDecisionIds(accountId, MODEL).has('m1')).toBe(false);
    expect(categories.getNoneDecisionIds(accountId, 'other-model').has('m1')).toBe(false);
  });

  it('cascades away when the email is deleted', () => {
    categories.recordNoneDecisions(accountId, MODEL, ['m1']);
    emails.delete('m1', accountId);
    expect(categories.getNoneDecisionIds(accountId, MODEL).size).toBe(0);
  });

  it('returns recent user corrections as few-shot examples grouped by email', () => {
    const cat = categories.create({ accountId, label: 'Bills', source: 'user' });
    categories.replaceEmailAssignments('m1', accountId, [
      {
        messageId: 'm1',
        accountId,
        categoryId: cat.id,
        confidence: 1,
        assignedBy: 'user',
        assignedAt: Date.now(),
      },
    ]);
    expect(categories.getUserCorrectionExamples(accountId, 5)).toEqual([
      { subject: 'one', labels: ['Bills'] },
    ]);
  });
});
