/**
 * Cluster-representative discovery sampling (hybrid): a VARIETY half (one representative per largest
 * content cluster) plus a VOLUME half (sender-mixed sample of the rest). Verifies full coverage when
 * the budget allows, that the largest clusters get proportionally more of the sample, that smaller
 * distinct clusters are still represented when one cluster dominates (the anti-dilution property that
 * plain sender-mixing loses), determinism, and the fallbacks (too few embeddings, tiny pool).
 */
import { describe, it, expect } from 'vitest';
import {
  clusterRepresentativeSample,
  mixedSampleBySender,
} from '../src/services/topic-discovery-service.js';
import type { EmailSummary } from '../src/repositories/email-repository.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';

/**
 * A unit vector pointing along axis k, so same-k vectors are identical (cosine 1) and different-k are
 * orthogonal (cosine 0). Full-length because the centroid math iterates EMBEDDING_DIM dimensions.
 */
function vec(k: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[k] = 1;
  return v;
}

interface ClusterSpec {
  cluster: number;
  sender: string;
  count: number;
}

/** Build a pool where each email carries a known content cluster and sender, plus lookups for both. */
function buildPool(spec: ClusterSpec[]): {
  pool: EmailSummary[];
  vectorOf: (id: string) => Float32Array | null;
  clusterOf: (id: string) => number;
} {
  const pool: EmailSummary[] = [];
  const clusterById = new Map<string, number>();
  const vecById = new Map<string, Float32Array>();
  let i = 0;
  for (const s of spec) {
    for (let c = 0; c < s.count; c++) {
      const messageId = `m${i++}`;
      pool.push({
        messageId,
        accountId: 'a',
        folder: '/INBOX',
        subject: `subject ${s.cluster}`,
        fromAddr: s.sender,
        date: null,
        hasAttachments: false,
        indexedAt: 0,
      });
      clusterById.set(messageId, s.cluster);
      vecById.set(messageId, vec(s.cluster));
    }
  }
  return {
    pool,
    vectorOf: (id) => vecById.get(id) ?? null,
    clusterOf: (id) => clusterById.get(id)!,
  };
}

const distinctClusters = (sample: EmailSummary[], clusterOf: (id: string) => number): number =>
  new Set(sample.map((e) => clusterOf(e.messageId))).size;

const EIGHT_CLUSTERS: ClusterSpec[] = [
  { cluster: 0, sender: 'a@x.com', count: 40 },
  { cluster: 1, sender: 'b@x.com', count: 40 },
  { cluster: 2, sender: 'c@x.com', count: 20 },
  { cluster: 3, sender: 'd@x.com', count: 20 },
  { cluster: 4, sender: 'e@x.com', count: 10 },
  { cluster: 5, sender: 'f@x.com', count: 10 },
  { cluster: 6, sender: 'g@x.com', count: 5 },
  { cluster: 7, sender: 'h@x.com', count: 5 },
];

describe('clusterRepresentativeSample (hybrid)', () => {
  it('covers every content cluster and is deterministic', () => {
    const { pool, vectorOf, clusterOf } = buildPool(EIGHT_CLUSTERS);
    const sample = clusterRepresentativeSample(pool, vectorOf, 20, 123);
    expect(sample).toHaveLength(20);
    // varietyBudget = 10 >= 8 clusters, so the variety half alone represents every cluster.
    expect(distinctClusters(sample, clusterOf)).toBe(8);

    const again = clusterRepresentativeSample(pool, vectorOf, 20, 123);
    expect(again.map((e) => e.messageId)).toEqual(sample.map((e) => e.messageId));
  });

  it('gives the largest cluster proportionally more of the sample than the smallest', () => {
    const { pool, vectorOf, clusterOf } = buildPool(EIGHT_CLUSTERS);
    const sample = clusterRepresentativeSample(pool, vectorOf, 20, 123);
    const counts = new Map<number, number>();
    for (const e of sample) {
      const c = clusterOf(e.messageId);
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    // The volume half is sender-frequency-weighted, so the 40-email cluster outweighs the 5-email one.
    expect(counts.get(0)!).toBeGreaterThan(counts.get(7)!);
  });

  it('still represents smaller distinct clusters when one cluster dominates the sender', () => {
    // One loud sender whose mail is one huge content cluster plus several small distinct ones.
    const spec: ClusterSpec[] = [
      { cluster: 0, sender: 'jobs@linkedin.com', count: 1000 },
      { cluster: 1, sender: 'jobs@linkedin.com', count: 15 },
      { cluster: 2, sender: 'jobs@linkedin.com', count: 12 },
      { cluster: 3, sender: 'jobs@linkedin.com', count: 10 },
      { cluster: 4, sender: 'jobs@linkedin.com', count: 8 },
      { cluster: 5, sender: 'jobs@linkedin.com', count: 6 },
    ];
    const { pool, vectorOf, clusterOf } = buildPool(spec);

    const hybrid = clusterRepresentativeSample(pool, vectorOf, 8, 7);
    const covered = new Set(hybrid.map((e) => clusterOf(e.messageId)));
    // varietyBudget = 4: the four largest clusters each get a representative, not just the huge one.
    for (const c of [0, 1, 2, 3]) expect(covered.has(c)).toBe(true);

    // Plain sender-mixing drowns in the dominant cluster, so it covers fewer distinct kinds.
    const senderSample = mixedSampleBySender(pool, 8, 7);
    expect(distinctClusters(hybrid, clusterOf)).toBeGreaterThan(
      distinctClusters(senderSample, clusterOf),
    );
  });

  it('falls back to sender-mixing when too few emails carry an embedding', () => {
    const { pool } = buildPool([{ cluster: 0, sender: 'a@x.com', count: 100 }]);
    // Fewer stored embeddings than the sample size, so clustering cannot be trusted.
    const sparse = (id: string): Float32Array | null =>
      ['m0', 'm1', 'm2'].includes(id) ? vec(0) : null;
    const sample = clusterRepresentativeSample(pool, sparse, 20, 55);
    const expected = mixedSampleBySender(pool, 20, 55);
    expect(sample.map((e) => e.messageId)).toEqual(expected.map((e) => e.messageId));
  });

  it('returns the whole pool when it is already at or below the sample size', () => {
    const { pool, vectorOf } = buildPool([{ cluster: 0, sender: 'a@x.com', count: 5 }]);
    const sample = clusterRepresentativeSample(pool, vectorOf, 20, 1);
    expect(sample).toHaveLength(5);
    expect(new Set(sample.map((e) => e.messageId))).toEqual(new Set(pool.map((e) => e.messageId)));
  });
});
