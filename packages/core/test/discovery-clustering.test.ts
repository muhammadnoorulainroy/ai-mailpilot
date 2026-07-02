/**
 * Phase 2a tests: deterministic bounded-coreset residual clustering with cohesion/separation
 * metrics (including a synthetic ~20k-vector performance check), and residual selection that
 * excludes user-assigned mail and uses fresh cosine (never stored confidence) for low-confidence.
 */
import { describe, it, expect } from 'vitest';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { mulberry32 } from '../src/util/rand.js';
import {
  clusterResidual,
  MAX_LEADERS,
  MIN_CLUSTER_SIZE,
  type ClusterPoint,
} from '../src/services/discovery-clustering.js';
import {
  ResidualDiscoveryService,
  RESIDUAL_COSINE_FLOOR,
} from '../src/services/residual-discovery-service.js';

/** Unit vector pointing along one axis (already normalized). */
function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

/** Normalize in place and return. */
function normalize(v: Float32Array): Float32Array {
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i]! * v[i]!;
  n = Math.sqrt(n);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i]! /= n;
  return v;
}

/** `count` unit vectors tightly clustered around axis `dim`, deterministic via `rnd`. */
function blob(dim: number, count: number, rnd: () => number, noise = 0.01): ClusterPoint[] {
  const pts: ClusterPoint[] = [];
  for (let i = 0; i < count; i++) {
    const v = new Float32Array(EMBEDDING_DIM);
    v[dim] = 1;
    for (let d = 0; d < EMBEDDING_DIM; d++) v[d]! += (rnd() - 0.5) * noise;
    pts.push({ messageId: `${dim}-${i}`, vector: normalize(v) });
  }
  return pts;
}

describe('clusterResidual', () => {
  it('is deterministic for the same input and seed', () => {
    const rnd = mulberry32(1);
    const points = [...blob(0, 20, rnd), ...blob(1, 20, rnd)];
    const a = clusterResidual(points, 42);
    const b = clusterResidual(points, 42);
    expect(a.map((c) => c.memberIds)).toEqual(b.map((c) => c.memberIds));
  });

  it('separates two well-separated blobs into two clusters with high cohesion and separation', () => {
    const rnd = mulberry32(2);
    const points = [...blob(0, 30, rnd), ...blob(1, 30, rnd)];
    const clusters = clusterResidual(points, 7);
    expect(clusters).toHaveLength(2);
    for (const c of clusters) {
      expect(c.size).toBe(30);
      expect(c.cohesion).toBeGreaterThan(0.9);
      expect(c.separation).toBeGreaterThan(0.5);
    }
  });

  it('drops clusters below MIN_CLUSTER_SIZE and keeps a rare-but-coherent pocket at the threshold', () => {
    const rnd = mulberry32(3);
    const points = [
      ...blob(0, 40, rnd), // big cluster
      ...blob(5, MIN_CLUSTER_SIZE, rnd), // rare coherent pocket, exactly at the floor: kept
      ...blob(9, MIN_CLUSTER_SIZE - 1, rnd), // too small: dropped
    ];
    const clusters = clusterResidual(points, 11);
    const sizes = clusters.map((c) => c.size).sort((a, b) => a - b);
    expect(sizes).toEqual([MIN_CLUSTER_SIZE, 40]);
  });

  it('never exceeds MAX_LEADERS clusters even with many distinct blobs', () => {
    const rnd = mulberry32(4);
    const points: ClusterPoint[] = [];
    for (let d = 0; d < MAX_LEADERS + 40; d++) points.push(...blob(d, MIN_CLUSTER_SIZE, rnd));
    const clusters = clusterResidual(points, 5);
    expect(clusters.length).toBeLessThanOrEqual(MAX_LEADERS);
  });

  it('returns nothing for an empty residual set', () => {
    expect(clusterResidual([], 1)).toEqual([]);
  });

  it('clusters ~20k vectors within a bounded budget and assigns every point', () => {
    const rnd = mulberry32(99);
    const BLOBS = 40;
    const PER = 500;
    const points: ClusterPoint[] = [];
    for (let d = 0; d < BLOBS; d++) points.push(...blob(d, PER, rnd));
    expect(points).toHaveLength(BLOBS * PER);

    const t0 = Date.now();
    const clusters = clusterResidual(points, 123);
    const durationMs = Date.now() - t0;

    console.log(`[perf] clusterResidual ${points.length} vectors -> ${clusters.length} clusters in ${durationMs}ms`);
    expect(clusters.length).toBeGreaterThan(0);
    expect(clusters.length).toBeLessThanOrEqual(MAX_LEADERS);
    expect(clusters.reduce((s, c) => s + c.size, 0)).toBe(BLOBS * PER);
    expect(durationMs).toBeLessThan(10_000);
  });
});

describe('ResidualDiscoveryService.selectResidual', () => {
  it('includes uncategorized and low-confidence auto mail, excludes user-assigned and confident auto mail', () => {
    const db = openDatabase(':memory:');
    const accounts = new AccountRepository(db);
    const categories = new CategoryRepository(db);
    const emails = new EmailRepository(db);
    const embeddings = new EmbeddingRepository(db);
    const acc = accounts.create({ address: 'w@x.com', kind: 'work' });

    // Active category A with a centroid on axis 0.
    const catA = categories.create({ accountId: acc.id, label: 'A', source: 'auto' });
    categories.saveCentroid(catA.id, 'bge-m3', axis(0), 10);

    const ids = ['uncat', 'confident', 'lowconf', 'usercorrected'];
    emails.upsertBatch(ids.map((id) => ({ messageId: id, accountId: acc.id, folder: 'INBOX' })));
    embeddings.saveEmbedding({ messageId: 'uncat', accountId: acc.id, modelId: 'bge-m3' }, axis(2)); // no assignment
    embeddings.saveEmbedding({ messageId: 'confident', accountId: acc.id, modelId: 'bge-m3' }, axis(0)); // cosine 1 to A
    embeddings.saveEmbedding({ messageId: 'lowconf', accountId: acc.id, modelId: 'bge-m3' }, axis(1)); // cosine ~0 to A
    embeddings.saveEmbedding({ messageId: 'usercorrected', accountId: acc.id, modelId: 'bge-m3' }, axis(1)); // cosine ~0 to A

    // Auto assignments to A. 'confident' matches, 'lowconf' does not. Stored confidence is a lie (1.0).
    for (const id of ['confident', 'lowconf']) {
      categories.replaceEmailAssignments(id, acc.id, [
        {
          messageId: id,
          accountId: acc.id,
          categoryId: catA.id,
          confidence: 1.0,
          assignedBy: 'auto',
          assignedAt: 1,
          method: 'gate',
        },
      ]);
    }
    // A user assignment to A, even though its embedding is far from A's centroid.
    categories.replaceEmailAssignments('usercorrected', acc.id, [
      {
        messageId: 'usercorrected',
        accountId: acc.id,
        categoryId: catA.id,
        confidence: 1.0,
        assignedBy: 'user',
        assignedAt: 1,
        method: null,
      },
    ]);

    const svc = new ResidualDiscoveryService(embeddings, categories);
    const residual = new Set(svc.selectResidual(acc.id, 'bge-m3').map((p) => p.messageId));

    expect(residual.has('uncat')).toBe(true); // uncategorized
    expect(residual.has('lowconf')).toBe(true); // fresh cosine below floor, despite stored confidence 1.0
    expect(residual.has('confident')).toBe(false); // fresh cosine 1.0, above floor
    expect(residual.has('usercorrected')).toBe(false); // user-assigned is never residual
    expect(RESIDUAL_COSINE_FLOOR).toBeGreaterThan(0);
    expect(RESIDUAL_COSINE_FLOOR).toBeLessThan(1);
    db.close();
  });
});
