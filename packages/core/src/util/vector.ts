/**
 * Vector math helpers operating on Float32Array.
 * Centralized to avoid number[] to Float32Array conversion roundtrips.
 */

import { EMBEDDING_DIM } from '../db/schema.js';

/**
 * View a buffer of float32 little-endian bytes as a Float32Array without copying.
 */
export function bufferToVector(buf: Buffer): Float32Array {
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

/**
 * View a Float32Array as a buffer of float32 little-endian bytes for storage.
 */
export function vectorToBuffer(vec: Float32Array): Buffer {
  return Buffer.from(vec.buffer, vec.byteOffset, vec.byteLength);
}

/**
 * Euclidean distance between two equal-length numeric vectors.
 */
export function l2Distance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    const d = (a[i] as number) - (b[i] as number);
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/**
 * For unit-normalized vectors, cosine similarity = 1 - L2distance^2 / 2.
 * bge-m3 returns normalized vectors so this is exact.
 */
export function cosineFromL2Distance(d: number): number {
  const cos = 1 - (d * d) / 2;
  if (cos < 0) return 0;
  if (cos > 1) return 1;
  return cos;
}

/**
 * Compute the L2-normalized centroid of a set of unit vectors.
 * Returns null if no valid vector is provided.
 */
export function meanNormalize(vectors: Float32Array[]): Float32Array | null {
  if (vectors.length === 0) return null;
  const dim = EMBEDDING_DIM;
  const sum = new Float32Array(dim);

  for (const v of vectors) {
    if (v.length !== dim) continue;
    for (let i = 0; i < dim; i++) sum[i] += v[i];
  }

  const inv = 1 / vectors.length;
  let normSq = 0;
  for (let i = 0; i < dim; i++) {
    sum[i] *= inv;
    normSq += sum[i] * sum[i];
  }

  const norm = Math.sqrt(normSq);
  if (norm > 0) {
    const invNorm = 1 / norm;
    for (let i = 0; i < dim; i++) sum[i] *= invNorm;
  }
  return sum;
}

/**
 * Fold one new vector into an existing centroid that summarizes count vectors.
 * Re-normalizing each step approximates the mean direction rather than the exact
 * arithmetic mean, which is what we want for cosine matching. Used by the learning
 * loop so filing an email drifts the centroid toward it. A starting count of 0 seeds
 * the centroid with vec directly.
 */
export function runningMeanUpdate(
  centroid: ArrayLike<number>,
  count: number,
  vec: ArrayLike<number>,
): Float32Array {
  const dim = EMBEDDING_DIM;
  const out = new Float32Array(dim);
  const denom = count + 1;
  let normSq = 0;
  for (let i = 0; i < dim; i++) {
    const val = ((centroid[i] as number) * count + (vec[i] as number)) / denom;
    out[i] = val;
    normSq += val * val;
  }
  const norm = Math.sqrt(normSq);
  if (norm > 0) {
    const inv = 1 / norm;
    for (let i = 0; i < dim; i++) out[i] *= inv;
  }
  return out;
}

/**
 * Assert that a vector has the expected EMBEDDING_DIM. Throws otherwise.
 */
export function assertEmbeddingDim(vec: ArrayLike<number>): void {
  if (vec.length !== EMBEDDING_DIM) {
    throw new Error(
      `expected ${EMBEDDING_DIM}-dim vector, got ${vec.length}. Use a 1024-dim embedding model (e.g. bge-m3).`,
    );
  }
}
