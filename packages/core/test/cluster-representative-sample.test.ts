/**
 * Cluster-representative discovery sampling: pick the LLM sample by content variety instead of by
 * sender, so every distinct KIND of mail is represented (including a loud sender's different message
 * types). Verifies full cluster coverage, proportional fill for the largest clusters, that it beats
 * sender-mixing on intra-sender variety, determinism, and the fallbacks (too few embeddings, tiny pool).
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

describe('clusterRepresentativeSample', () => {
  it('covers every content cluster and is deterministic', () => {
    const { pool, vectorOf, clusterOf } = buildPool(EIGHT_CLUSTERS);
    const sample = clusterRepresentativeSample(pool, vectorOf, 20, 123);
    expect(sample).toHaveLength(20);
    expect(distinctClusters(sample, clusterOf)).toBe(8); // one representative from every cluster

    const again = clusterRepresentativeSample(pool, vectorOf, 20, 123);
    expect(again.map((e) => e.messageId)).toEqual(sample.map((e) => e.messageId));
  });

  it('gives the largest clusters proportionally more of the sample', () => {
    const { pool, vectorOf, clusterOf } = buildPool(EIGHT_CLUSTERS);
    const sample = clusterRepresentativeSample(pool, vectorOf, 20, 123);
    const counts = new Map<number, number>();
    for (const e of sample) {
      const c = clusterOf(e.messageId);
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    // The four largest clusters (the two 40s and two 20s) each get an extra pick; the tail gets one.
    expect(counts.get(0)).toBe(3); // a 40-email cluster
    expect(counts.get(7)).toBe(2); // a 5-email cluster
  });

  it('captures intra-sender variety that sender-mixing collapses', () => {
    const spec: ClusterSpec[] = [];
    // One loud sender whose mail spans 12 distinct content clusters, plus a few quiet single-cluster
    // senders. Sender-mixing gives the loud sender one diversity slot; content clustering does not.
    for (let k = 0; k < 12; k++) spec.push({ cluster: k, sender: 'jobs@linkedin.com', count: 8 });
    for (let k = 12; k < 15; k++) spec.push({ cluster: k, sender: `q${k}@x.com`, count: 5 });
    const { pool, vectorOf, clusterOf } = buildPool(spec);

    const clusterSample = clusterRepresentativeSample(pool, vectorOf, 10, 7);
    const senderSample = mixedSampleBySender(pool, 10, 7);

    expect(distinctClusters(clusterSample, clusterOf)).toBe(10); // ten distinct kinds in ten slots
    expect(distinctClusters(clusterSample, clusterOf)).toBeGreaterThan(
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
