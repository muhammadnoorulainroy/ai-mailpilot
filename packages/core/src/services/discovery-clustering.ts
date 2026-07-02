/**
 * On-device residual clustering for discovery (Phase 2a). Pure functions: given residual email
 * embeddings, produce candidate clusters with cohesion and separation metrics, deterministically
 * and within a bounded cost budget. No LLM, no I/O, no persistence, no category creation.
 *
 * Clustering is purely by embedding content. Sender or domain is never used here as cluster logic;
 * it is only ever a sampling/diversity signal upstream. This keeps a loud sender from becoming a
 * category on its own.
 */
import { dot, runningMeanUpdate } from '../util/vector.js';
import { seededShuffle } from '../util/rand.js';

/** A residual email to cluster, identified by message id and its unit-normalized embedding. */
export interface ClusterPoint {
  messageId: string;
  vector: Float32Array;
}

/** A discovered candidate cluster with deterministic quality metrics. Not persisted in 2a. */
export interface DiscoveredCluster {
  memberIds: string[];
  size: number;
  centroid: Float32Array;
  /** Mean cosine of members to the centroid, in [0, 1]. Higher is tighter. Sampled for big clusters. */
  cohesion: number;
  /** Cosine gap to the nearest other cluster centroid, in [0, 1]. Higher is better separated. */
  separation: number;
}

/** Leaders are grown from at most this many points, bounding the O(coreset x leaders) build cost. */
export const CLUSTER_CORESET_MAX = 8000;
/** Hard cap on the number of clusters (leaders), bounding per-point comparison cost. */
export const MAX_LEADERS = 256;
/** A point joins an existing leader when its cosine to that leader is at least this. */
export const CLUSTER_JOIN_COSINE = 0.75;
/** Clusters smaller than this are dropped as noise rather than proposed. */
export const MIN_CLUSTER_SIZE = 3;
/** Cohesion is averaged over at most this many members per cluster, to stay bounded. */
export const COHESION_SAMPLE = 256;

interface Leader {
  centroid: Float32Array;
  memberIds: string[];
  count: number;
}

/**
 * Cluster residual points into candidate clusters. Deterministic for a given seed: the point order
 * and the bounded coreset are seeded, so identical input yields identical clusters. Cost is bounded
 * by the coreset cap and the leader cap, so it stays laptop-safe on large inboxes. Vectors are
 * assumed unit-normalized (bge-m3 output), so dot product equals cosine similarity.
 */
export function clusterResidual(
  points: readonly ClusterPoint[],
  seed: number,
): DiscoveredCluster[] {
  if (points.length === 0) return [];

  const ordered = seededShuffle(points, seed);
  const coreset = ordered.slice(0, CLUSTER_CORESET_MAX);
  const leaders: Leader[] = [];

  /** Nearest leader by cosine, opening a new one only when allowed and under the cap; else fold in. */
  const nearestLeader = (p: ClusterPoint, allowNew: boolean): Leader | null => {
    let best: Leader | null = null;
    let bestCos = -Infinity;
    for (const l of leaders) {
      const cos = dot(l.centroid, p.vector);
      if (cos >= CLUSTER_JOIN_COSINE) return l; // early exit: close enough to join
      if (cos > bestCos) {
        bestCos = cos;
        best = l;
      }
    }
    if (allowNew && leaders.length < MAX_LEADERS) {
      const l: Leader = { centroid: p.vector, memberIds: [], count: 0 };
      leaders.push(l);
      return l;
    }
    return best; // overflow: fold into the nearest existing leader
  };

  const absorb = (l: Leader | null, p: ClusterPoint): void => {
    if (!l) return;
    l.centroid = runningMeanUpdate(l.centroid, l.count, p.vector);
    l.count += 1;
    l.memberIds.push(p.messageId);
  };

  // Grow leaders on the bounded coreset, then assign the rest without opening new leaders.
  for (const p of coreset) absorb(nearestLeader(p, true), p);
  for (let i = CLUSTER_CORESET_MAX; i < ordered.length; i++) {
    absorb(nearestLeader(ordered[i]!, false), ordered[i]!);
  }

  const kept = leaders.filter((l) => l.memberIds.length >= MIN_CLUSTER_SIZE);
  const byId = new Map(points.map((p) => [p.messageId, p.vector]));

  return kept.map((l, idx) => ({
    memberIds: l.memberIds,
    size: l.memberIds.length,
    centroid: l.centroid,
    cohesion: cohesionOf(l, byId, seed + idx + 1),
    separation: separationOf(l, kept),
  }));
}

/** Mean cosine of a bounded sample of members to the cluster centroid, in [0, 1]. */
function cohesionOf(l: Leader, byId: Map<string, Float32Array>, seed: number): number {
  const ids =
    l.memberIds.length > COHESION_SAMPLE
      ? seededShuffle(l.memberIds, seed).slice(0, COHESION_SAMPLE)
      : l.memberIds;
  let sum = 0;
  let n = 0;
  for (const id of ids) {
    const v = byId.get(id);
    if (!v) continue;
    sum += dot(l.centroid, v);
    n += 1;
  }
  if (n === 0) return 0;
  const c = sum / n;
  return c < 0 ? 0 : c > 1 ? 1 : c;
}

/** Cosine gap from a cluster centroid to the nearest other cluster centroid, in [0, 1]. */
function separationOf(l: Leader, all: Leader[]): number {
  let nearest = -Infinity;
  for (const other of all) {
    if (other === l) continue;
    const cos = dot(l.centroid, other.centroid);
    if (cos > nearest) nearest = cos;
  }
  if (nearest === -Infinity) return 1; // only cluster: maximally separated
  const sep = 1 - nearest;
  return sep < 0 ? 0 : sep > 1 ? 1 : sep;
}
