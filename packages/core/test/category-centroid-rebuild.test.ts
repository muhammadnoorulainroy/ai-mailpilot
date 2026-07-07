/**
 * Phase 3.1 tests: correction-based centroid rebuild. Rebuild recomputes a category's stored centroid
 * from its trusted (user-confirmed) member embeddings, deterministically. It prefers user members,
 * uses auto only as an explicit zero-user fallback, leaves the centroid untouched when data is thin,
 * is account-isolated, and never changes labels, status, source, key, or assignments.
 */
import { describe, it, expect } from 'vitest';
import type { Logger } from 'pino';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { rankCategories } from '../src/services/categorization-service.js';
import { cosineSimilarity } from '../src/util/vector.js';
import {
  CategoryCentroidRebuildService,
  MIN_TRUSTED_REBUILD,
} from '../src/services/category-centroid-rebuild-service.js';

const MODEL = 'bge-m3';
const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

function setup() {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const categories = new CategoryRepository(db);
  const emails = new EmailRepository(db);
  const embeddings = new EmbeddingRepository(db);
  const service = new CategoryCentroidRebuildService(categories, embeddings, silentLogger);
  return { db, accounts, categories, emails, embeddings, service };
}

type Harness = ReturnType<typeof setup>;

function userMember(
  h: Harness,
  accountId: string,
  categoryId: string,
  id: string,
  vec: Float32Array,
) {
  h.emails.upsertBatch([{ messageId: id, accountId, folder: 'INBOX', subject: id }]);
  h.embeddings.saveEmbedding({ messageId: id, accountId, modelId: MODEL }, vec);
  h.categories.replaceEmailAssignments(id, accountId, [
    {
      messageId: id,
      accountId,
      categoryId,
      confidence: 1,
      assignedBy: 'user',
      assignedAt: 1,
      method: null,
    },
  ]);
}

function autoMember(
  h: Harness,
  accountId: string,
  categoryId: string,
  id: string,
  vec: Float32Array,
) {
  h.emails.upsertBatch([{ messageId: id, accountId, folder: 'INBOX', subject: id }]);
  h.embeddings.saveEmbedding({ messageId: id, accountId, modelId: MODEL }, vec);
  h.categories.addAutoAssignments(accountId, [
    {
      messageId: id,
      accountId,
      categoryId,
      confidence: 0.6,
      assignedBy: 'auto',
      assignedAt: 1,
      method: 'embed',
    },
  ]);
}

/** A user assignment with no stored embedding for the model, so the rebuild must skip it. */
function userMemberNoEmbedding(h: Harness, accountId: string, categoryId: string, id: string) {
  h.emails.upsertBatch([{ messageId: id, accountId, folder: 'INBOX', subject: id }]);
  h.categories.replaceEmailAssignments(id, accountId, [
    {
      messageId: id,
      accountId,
      categoryId,
      confidence: 1,
      assignedBy: 'user',
      assignedAt: 1,
      method: null,
    },
  ]);
}

describe('CategoryCentroidRebuildService', () => {
  it('rebuilds a centroid from user-confirmed member embeddings', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const cat = h.categories.create({ accountId: acc.id, label: 'Bills', source: 'user' });
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) userMember(h, acc.id, cat.id, `u${i}`, axis(0));

    const res = h.service.rebuild(acc.id, cat.id, MODEL);

    expect(res).toMatchObject({
      status: 'rebuilt',
      vectorsUsed: MIN_TRUSTED_REBUILD,
      usedAutoFallback: false,
    });
    const stored = h.categories.getCentroid(cat.id, MODEL)!;
    expect(cosineSimilarity(stored.vector, axis(0))).toBeGreaterThan(0.99);
    expect(stored.emailCount).toBe(MIN_TRUSTED_REBUILD);
    h.db.close();
  });

  it('leaves the existing centroid unchanged when trusted data is below the floor', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const cat = h.categories.create({ accountId: acc.id, label: 'Bills', source: 'user' });
    h.categories.saveCentroid(cat.id, MODEL, axis(1), 5);
    for (let i = 0; i < MIN_TRUSTED_REBUILD - 1; i++)
      userMember(h, acc.id, cat.id, `u${i}`, axis(0));

    const res = h.service.rebuild(acc.id, cat.id, MODEL);

    expect(res).toMatchObject({
      status: 'insufficient_trusted_data',
      vectorsUsed: MIN_TRUSTED_REBUILD - 1,
      usedAutoFallback: false,
    });
    const stored = h.categories.getCentroid(cat.id, MODEL)!;
    expect(cosineSimilarity(stored.vector, axis(1))).toBeGreaterThan(0.99);
    expect(stored.emailCount).toBe(5);
    h.db.close();
  });

  it('uses only user vectors and ignores auto assignments when trusted data suffices', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const cat = h.categories.create({ accountId: acc.id, label: 'Bills', source: 'user' });
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) userMember(h, acc.id, cat.id, `u${i}`, axis(0));
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) autoMember(h, acc.id, cat.id, `a${i}`, axis(1));

    const res = h.service.rebuild(acc.id, cat.id, MODEL, { allowAutoFallback: true });

    expect(res).toMatchObject({
      status: 'rebuilt',
      vectorsUsed: MIN_TRUSTED_REBUILD,
      usedAutoFallback: false,
    });
    const stored = h.categories.getCentroid(cat.id, MODEL)!;
    expect(cosineSimilarity(stored.vector, axis(0))).toBeGreaterThan(0.99);
    expect(cosineSimilarity(stored.vector, axis(1))).toBeLessThan(0.1);
    h.db.close();
  });

  it('falls back to auto vectors only when explicitly allowed and there are zero user vectors', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });

    // Zero user, enough auto, no fallback flag: unchanged (no centroid written).
    const catA = h.categories.create({ accountId: acc.id, label: 'A', source: 'auto' });
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) autoMember(h, acc.id, catA.id, `a${i}`, axis(1));
    expect(h.service.rebuild(acc.id, catA.id, MODEL).status).toBe('insufficient_trusted_data');
    expect(h.categories.getCentroid(catA.id, MODEL)).toBeNull();

    // Same category, fallback allowed: rebuilt from auto and flagged.
    const resB = h.service.rebuild(acc.id, catA.id, MODEL, { allowAutoFallback: true });
    expect(resB).toMatchObject({
      status: 'rebuilt',
      vectorsUsed: MIN_TRUSTED_REBUILD,
      usedAutoFallback: true,
    });
    expect(
      cosineSimilarity(h.categories.getCentroid(catA.id, MODEL)!.vector, axis(1)),
    ).toBeGreaterThan(0.99);

    // Below-floor user members present: auto fallback must NOT fire even when allowed.
    const catC = h.categories.create({ accountId: acc.id, label: 'C', source: 'user' });
    for (let i = 0; i < MIN_TRUSTED_REBUILD - 1; i++)
      userMember(h, acc.id, catC.id, `cu${i}`, axis(0));
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) autoMember(h, acc.id, catC.id, `ca${i}`, axis(1));
    const resC = h.service.rebuild(acc.id, catC.id, MODEL, { allowAutoFallback: true });
    expect(resC).toMatchObject({ status: 'insufficient_trusted_data', usedAutoFallback: false });
    expect(h.categories.getCentroid(catC.id, MODEL)).toBeNull();
    h.db.close();
  });

  it('does not auto-fall-back when the category has user members but no usable user embeddings', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const cat = h.categories.create({ accountId: acc.id, label: 'Bills', source: 'user' });
    // An existing centroid that must survive untouched.
    h.categories.saveCentroid(cat.id, MODEL, axis(1), 5);
    // User-confirmed members whose embeddings are missing for this model (zero usable user vectors,
    // but the category still HAS user members).
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) userMemberNoEmbedding(h, acc.id, cat.id, `u${i}`);
    // Plenty of auto members with embeddings.
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) autoMember(h, acc.id, cat.id, `a${i}`, axis(0));

    const res = h.service.rebuild(acc.id, cat.id, MODEL, { allowAutoFallback: true });

    expect(res).toMatchObject({ status: 'insufficient_trusted_data', usedAutoFallback: false });
    const stored = h.categories.getCentroid(cat.id, MODEL)!;
    expect(cosineSimilarity(stored.vector, axis(1))).toBeGreaterThan(0.99);
    expect(stored.emailCount).toBe(5);
    h.db.close();
  });

  it('skips members with no embedding, and a model with no embeddings fails safely', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const cat = h.categories.create({ accountId: acc.id, label: 'Bills', source: 'user' });
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) userMember(h, acc.id, cat.id, `u${i}`, axis(0));
    userMemberNoEmbedding(h, acc.id, cat.id, 'noemb');

    const res = h.service.rebuild(acc.id, cat.id, MODEL);
    expect(res).toMatchObject({ status: 'rebuilt', vectorsUsed: MIN_TRUSTED_REBUILD });
    expect(
      cosineSimilarity(h.categories.getCentroid(cat.id, MODEL)!.vector, axis(0)),
    ).toBeGreaterThan(0.99);

    // A different model has no stored embeddings, so there is nothing to rebuild from and no crash.
    expect(h.service.rebuild(acc.id, cat.id, 'other-model').status).toBe(
      'insufficient_trusted_data',
    );
    h.db.close();
  });

  it('is strictly isolated by account', () => {
    const h = setup();
    const a = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const b = h.accounts.create({ address: 'b@x.com', kind: 'work' });
    const catA = h.categories.create({ accountId: a.id, label: 'Shared', source: 'user' });
    const catB = h.categories.create({ accountId: b.id, label: 'Shared', source: 'user' });
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) userMember(h, a.id, catA.id, `a${i}`, axis(0));
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) userMember(h, b.id, catB.id, `b${i}`, axis(2));

    h.service.rebuild(a.id, catA.id, MODEL);
    expect(
      cosineSimilarity(h.categories.getCentroid(catA.id, MODEL)!.vector, axis(0)),
    ).toBeGreaterThan(0.99);
    // B is untouched by A's rebuild.
    expect(h.categories.getCentroid(catB.id, MODEL)).toBeNull();
    // Rebuilding B's category under account A is rejected, not cross-read.
    expect(h.service.rebuild(a.id, catB.id, MODEL).status).toBe('category_not_found');
    // B's own rebuild is independent.
    h.service.rebuild(b.id, catB.id, MODEL);
    expect(
      cosineSimilarity(h.categories.getCentroid(catB.id, MODEL)!.vector, axis(2)),
    ).toBeGreaterThan(0.99);
    h.db.close();
  });

  it('changes only the centroid, never labels, status, source, key, or assignments', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const cat = h.categories.create({
      accountId: acc.id,
      label: 'Bills',
      description: 'money',
      source: 'user',
    });
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) userMember(h, acc.id, cat.id, `u${i}`, axis(0));
    const before = h.categories.findById(cat.id)!;
    const assignmentsBefore = h.categories.getEmailCategories('u0', acc.id);

    h.service.rebuild(acc.id, cat.id, MODEL);

    const after = h.categories.findById(cat.id)!;
    expect(after.label).toBe(before.label);
    expect(after.description).toBe(before.description);
    expect(after.status).toBe(before.status);
    expect(after.source).toBe(before.source);
    expect(after.canonicalKey).toBe(before.canonicalKey);
    expect(h.categories.getEmailCategories('u0', acc.id)).toEqual(assignmentsBefore);
    expect(h.categories.listCategoryMemberIds(acc.id, cat.id, 'user').sort()).toEqual([
      'u0',
      'u1',
      'u2',
    ]);
    h.db.close();
  });

  it('makes the rebuilt centroid the one nearest-centroid ranking reads', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const cat = h.categories.create({ accountId: acc.id, label: 'Bills', source: 'user' });
    // The stored centroid had drifted off-topic (axis 1) before the user corrected members.
    h.categories.saveCentroid(cat.id, MODEL, axis(1), 4);
    for (let i = 0; i < MIN_TRUSTED_REBUILD; i++) userMember(h, acc.id, cat.id, `u${i}`, axis(0));
    // A distractor category anchored on axis 1, so the ranking is a real comparison.
    const other = h.categories.create({ accountId: acc.id, label: 'Other', source: 'user' });
    h.categories.saveCentroid(other.id, MODEL, axis(1), 3);

    h.service.rebuild(acc.id, cat.id, MODEL);

    const entries = h.categories.getCentroidEntries(acc.id, MODEL);
    const entry = entries.find((e) => e.categoryId === cat.id)!;
    expect(cosineSimilarity(entry.vector, axis(0))).toBeGreaterThan(0.99);
    // An axis-0 email now ranks the corrected category first through the embed-match read path.
    const ranked = rankCategories(axis(0), entries);
    expect(ranked[0]!.categoryId).toBe(cat.id);
    h.db.close();
  });
});
