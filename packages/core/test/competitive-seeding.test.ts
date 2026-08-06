/**
 * Competitive centroid seeding: each email seeds only its single nearest topic (above a floor), and
 * topics that gather too few real members are dropped. This is the fix for centroids that were seeded
 * from loose lexical neighbours of the label text (a "Course Materials" that is really one
 * newsletter). Verifies competitive routing, dropping thin/empty topics, the floor, and determinism.
 */
import { describe, it, expect } from 'vitest';
import { seedCentroidsCompetitive } from '../src/services/topic-discovery-service.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';

/** A unit vector with the given (index, weight) components. */
function vecOn(pairs: Array<[number, number]>): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  for (const [i, val] of pairs) v[i] = val;
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i]! * v[i]!;
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

const topic = (label: string, axis: number) => ({
  topic: { label, description: label.toLowerCase() },
  vec: vecOn([[axis, 1]]),
});

/** `count` emails pointing along `axis`, ids prefixed with `tag`. */
function emails(tag: string, axis: number, count: number) {
  return Array.from({ length: count }, (_, i) => ({
    messageId: `${tag}${i}`,
    vector: vecOn([[axis, 1]]),
  }));
}

describe('seedCentroidsCompetitive', () => {
  const kept = [topic('A', 0), topic('B', 1), topic('C', 2)];

  it('routes each email to its single nearest topic and drops thin/empty topics', () => {
    const entries = [
      ...emails('a', 0, 15), // topic A: 15 -> survives
      ...emails('b', 1, 12), // topic B: 12 -> survives
      ...emails('c', 2, 5), // topic C: 5 -> below MIN_CATEGORY_MEMBERS -> dropped
      // 3 emails closer to A than B: competition must give them to A only, not both.
      { messageId: 'x0', vector: vecOn([[0, 0.8], [1, 0.3]]) },
      { messageId: 'x1', vector: vecOn([[0, 0.8], [1, 0.3]]) },
      { messageId: 'x2', vector: vecOn([[0, 0.8], [1, 0.3]]) },
      // 2 emails below the seeding floor (cosine 0.4 to A, on an unused axis otherwise): unassigned.
      { messageId: 'lo0', vector: vecOn([[0, 0.4], [9, 0.9]]) },
      { messageId: 'lo1', vector: vecOn([[0, 0.4], [9, 0.9]]) },
    ];
    const staged = seedCentroidsCompetitive(kept, entries);

    expect(staged.map((s) => s.label)).toEqual(['A', 'B']); // C dropped as thin
    const byLabel = new Map(staged.map((s) => [s.label, s.emailCount]));
    expect(byLabel.get('A')).toBe(18); // 15 + the 3 competitive emails
    expect(byLabel.get('B')).toBe(12); // competition did NOT also add them to B
  });

  it('is deterministic regardless of input order', () => {
    const entries = [...emails('a', 0, 15), ...emails('b', 1, 12), ...emails('c', 2, 5)];
    const forward = seedCentroidsCompetitive(kept, entries);
    const reversed = seedCentroidsCompetitive(kept, [...entries].reverse());
    expect(reversed.map((s) => ({ label: s.label, n: s.emailCount }))).toEqual(
      forward.map((s) => ({ label: s.label, n: s.emailCount })),
    );
  });
});
