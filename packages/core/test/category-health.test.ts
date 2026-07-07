/**
 * Phase 3.2 tests: read-only category health metrics. The sensor computes size, cohesion, separation,
 * overlap, drift, and coverage per active category from members and stored centroids, deterministically
 * and without any write. Covers cohesion tightness, separation/overlap from stored centroids, drift on a
 * stale centroid, counts and coverage, strict read-only behavior, account isolation, and the null cases.
 */
import { describe, it, expect } from 'vitest';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { cosineSimilarity } from '../src/util/vector.js';
import { CategoryHealthService } from '../src/services/category-health-service.js';

const MODEL = 'bge-m3';

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

/** Unit vector at cosine `target` to axis 0, with the remaining mass on `otherDim` (default 1). */
function unitNearAxis0(target: number, otherDim = 1): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[0] = target;
  v[otherDim] = Math.sqrt(1 - target * target);
  return v;
}

function setup() {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const categories = new CategoryRepository(db);
  const emails = new EmailRepository(db);
  const embeddings = new EmbeddingRepository(db);
  const service = new CategoryHealthService(categories, embeddings);
  return { db, accounts, categories, emails, embeddings, service };
}

type Harness = ReturnType<typeof setup>;

let seq = 0;
function member(
  h: Harness,
  accountId: string,
  categoryId: string,
  vec: Float32Array | null,
  assignedBy: 'user' | 'auto',
) {
  const id = `m${seq++}`;
  h.emails.upsertBatch([{ messageId: id, accountId, folder: 'INBOX', subject: id }]);
  if (vec) h.embeddings.saveEmbedding({ messageId: id, accountId, modelId: MODEL }, vec);
  if (assignedBy === 'user') {
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
  } else {
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
  return id;
}

const find = (health: ReturnType<CategoryHealthService['metricsForAccount']>, id: string) =>
  health.find((x) => x.categoryId === id)!;

describe('CategoryHealthService', () => {
  it('reports high cohesion for a tight category and lower cohesion for a spread one', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const tight = h.categories.create({ accountId: acc.id, label: 'Tight', source: 'user' });
    for (let i = 0; i < 4; i++) member(h, acc.id, tight.id, axis(0), 'user');
    const spread = h.categories.create({ accountId: acc.id, label: 'Spread', source: 'user' });
    member(h, acc.id, spread.id, axis(0), 'user');
    member(h, acc.id, spread.id, axis(0), 'user');
    member(h, acc.id, spread.id, axis(1), 'user');
    member(h, acc.id, spread.id, axis(1), 'user');

    const health = h.service.metricsForAccount(acc.id, MODEL);
    expect(find(health, tight.id).cohesion!).toBeGreaterThan(0.99);
    expect(find(health, spread.id).cohesion!).toBeLessThan(0.8);
    expect(find(health, spread.id).cohesion!).toBeLessThan(find(health, tight.id).cohesion!);
    h.db.close();
  });

  it('measures separation, overlap, and the nearest category from stored centroids', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const a = h.categories.create({ accountId: acc.id, label: 'A', source: 'user' });
    const b = h.categories.create({ accountId: acc.id, label: 'B', source: 'user' });
    const c = h.categories.create({ accountId: acc.id, label: 'C', source: 'user' });
    h.categories.saveCentroid(a.id, MODEL, axis(0), 3);
    h.categories.saveCentroid(b.id, MODEL, axis(1), 3);
    // C is close to A (cos 0.9) with its off-axis mass on axis 2, so it stays orthogonal to B.
    h.categories.saveCentroid(c.id, MODEL, unitNearAxis0(0.9, 2), 3);

    const health = h.service.metricsForAccount(acc.id, MODEL);
    const ha = find(health, a.id);
    expect(ha.nearestCategoryId).toBe(c.id);
    expect(ha.overlap!).toBeCloseTo(0.9, 3);
    expect(ha.separation!).toBeCloseTo(0.1, 3);
    // B is far from both A and C.
    expect(find(health, b.id).overlap!).toBeLessThan(0.1);
    h.db.close();
  });

  it('reports high drift for a stale stored centroid and near-zero for a fresh one', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const stale = h.categories.create({ accountId: acc.id, label: 'Stale', source: 'user' });
    h.categories.saveCentroid(stale.id, MODEL, axis(1), 5); // stored points away from members
    for (let i = 0; i < 4; i++) member(h, acc.id, stale.id, axis(0), 'user');
    const fresh = h.categories.create({ accountId: acc.id, label: 'Fresh', source: 'user' });
    for (let i = 0; i < 4; i++) member(h, acc.id, fresh.id, axis(2), 'user');
    h.categories.saveCentroid(fresh.id, MODEL, axis(2), 4); // stored matches members

    const health = h.service.metricsForAccount(acc.id, MODEL);
    expect(find(health, stale.id).drift!).toBeGreaterThan(0.9);
    expect(find(health, fresh.id).drift!).toBeLessThan(0.01);
    h.db.close();
  });

  it('counts members and computes coverage as the share of assigned members', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const a = h.categories.create({ accountId: acc.id, label: 'A', source: 'user' });
    for (let i = 0; i < 3; i++) member(h, acc.id, a.id, axis(0), 'user');
    for (let i = 0; i < 2; i++) member(h, acc.id, a.id, axis(0), 'auto');
    const b = h.categories.create({ accountId: acc.id, label: 'B', source: 'user' });
    for (let i = 0; i < 5; i++) member(h, acc.id, b.id, axis(1), 'user');

    const ha = find(h.service.metricsForAccount(acc.id, MODEL), a.id);
    expect(ha.size).toBe(5);
    expect(ha.userMemberCount).toBe(3);
    expect(ha.coverage).toBeCloseTo(0.5, 5); // 5 of 10 total members
    h.db.close();
  });

  it('writes nothing: the stored centroid, assignments, and categories are unchanged', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const cat = h.categories.create({ accountId: acc.id, label: 'A', source: 'user' });
    const members: string[] = [];
    for (let i = 0; i < 3; i++) members.push(member(h, acc.id, cat.id, axis(0), 'user'));
    h.categories.saveCentroid(cat.id, MODEL, axis(1), 5); // deliberately stale
    const before = h.categories.getCentroid(cat.id, MODEL)!;

    h.service.metricsForAccount(acc.id, MODEL);

    const after = h.categories.getCentroid(cat.id, MODEL)!;
    // The stale centroid is NOT rebuilt: same direction and same email count.
    expect(cosineSimilarity(after.vector, before.vector)).toBeGreaterThan(0.999);
    expect(after.emailCount).toBe(5);
    expect(h.categories.listCategoryMemberIds(acc.id, cat.id, 'user').sort()).toEqual(
      members.sort(),
    );
    expect(h.categories.listActive(acc.id)).toHaveLength(1);
    h.db.close();
  });

  it('is isolated by account', () => {
    const h = setup();
    const a = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const b = h.accounts.create({ address: 'b@x.com', kind: 'work' });
    const catA = h.categories.create({ accountId: a.id, label: 'A', source: 'user' });
    for (let i = 0; i < 3; i++) member(h, a.id, catA.id, axis(0), 'user');
    const catB = h.categories.create({ accountId: b.id, label: 'B', source: 'user' });
    for (let i = 0; i < 3; i++) member(h, b.id, catB.id, axis(1), 'user');

    const health = h.service.metricsForAccount(a.id, MODEL);
    expect(health.map((x) => x.categoryId)).toEqual([catA.id]);
    expect(find(health, catA.id).size).toBe(3);
    h.db.close();
  });

  it('handles categories with no stored centroid, no members, or missing embeddings', () => {
    const h = setup();
    const acc = h.accounts.create({ address: 'a@x.com', kind: 'work' });
    const noCentroid = h.categories.create({
      accountId: acc.id,
      label: 'NoCentroid',
      source: 'user',
    });
    for (let i = 0; i < 3; i++) member(h, acc.id, noCentroid.id, axis(0), 'user');
    const empty = h.categories.create({ accountId: acc.id, label: 'Empty', source: 'user' });
    const noEmb = h.categories.create({ accountId: acc.id, label: 'NoEmb', source: 'user' });
    for (let i = 0; i < 3; i++) member(h, acc.id, noEmb.id, null, 'user'); // members without embeddings

    const health = h.service.metricsForAccount(acc.id, MODEL);

    const nc = find(health, noCentroid.id);
    expect(nc.cohesion!).toBeGreaterThan(0.99); // members present, so cohesion is defined
    expect(nc.separation).toBeNull(); // no stored centroid
    expect(nc.overlap).toBeNull();
    expect(nc.drift).toBeNull();

    const em = find(health, empty.id);
    expect(em.size).toBe(0);
    expect(em.cohesion).toBeNull();
    expect(em.coverage).toBe(0);

    const ne = find(health, noEmb.id);
    expect(ne.size).toBe(3);
    expect(ne.userMemberCount).toBe(3);
    expect(ne.cohesion).toBeNull(); // members counted, but no usable embeddings
    expect(ne.drift).toBeNull();
    h.db.close();
  });
});
