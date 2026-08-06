/**
 * Integration test for the Python clustering transport: it spawns the real sidecar (ml/cluster.py) on
 * a tiny two-blob batch (small enough to skip UMAP, so it runs in seconds), and checks the parsed
 * cluster defs, determinism, and the missing-script error path. Skips when Python or its ML packages
 * are not installed, so the suite stays green in environments without the sidecar.
 */
import { describe, it, expect } from 'vitest';
import { spawnSync } from 'node:child_process';
import { ClusteringClient } from '../src/services/clustering-client.js';

const PY = process.env.MAILPILOT_PYTHON ?? 'python';
const READY = (() => {
  try {
    return spawnSync(PY, ['-c', 'import numpy, sklearn'], { timeout: 30_000 }).status === 0;
  } catch {
    return false;
  }
})();

const DIM = 32;
/** Two orthogonal blobs of 15 near-identical points each (30 rows), each row lightly jittered. */
function blobs(): Float32Array[] {
  const out: Float32Array[] = [];
  for (let b = 0; b < 2; b++) {
    for (let i = 0; i < 15; i++) {
      const v = new Float32Array(DIM);
      v[b] = 1;
      v[8 + i] = 0.01; // unique tiny component so points are not exact duplicates
      out.push(v);
    }
  }
  return out;
}

describe.skipIf(!READY)('ClusteringClient (real sidecar)', () => {
  it('clusters a tiny batch into valid cluster definitions', async () => {
    const res = await new ClusteringClient().cluster(blobs(), DIM, {
      targetClusters: 2,
      timeoutMs: 60_000,
    });
    expect(res.count).toBe(30);
    expect(res.clusterCount).toBeGreaterThanOrEqual(1);
    for (const c of res.clusters) {
      expect(c.representatives.length).toBeGreaterThan(0);
      expect(c.centroid).toHaveLength(DIM);
      expect(c.prototypes.length).toBeGreaterThanOrEqual(1);
      for (const i of c.representatives) {
        expect(i).toBeGreaterThanOrEqual(0);
        expect(i).toBeLessThan(30);
      }
    }
  }, 90_000);

  it('is deterministic for a fixed seed', async () => {
    const client = new ClusteringClient();
    const a = await client.cluster(blobs(), DIM, { targetClusters: 2, timeoutMs: 60_000 });
    const b = await client.cluster(blobs(), DIM, { targetClusters: 2, timeoutMs: 60_000 });
    expect(b.clusters.map((c) => c.representatives)).toEqual(
      a.clusters.map((c) => c.representatives),
    );
    expect(b.clusters.map((c) => c.centroid)).toEqual(a.clusters.map((c) => c.centroid));
  }, 90_000);
});

describe('ClusteringClient error handling', () => {
  it('throws when the sidecar script is missing', async () => {
    const client = new ClusteringClient({ script: '/nonexistent/cluster.py' });
    expect(client.available()).toBe(false);
    await expect(client.cluster(blobs(), DIM)).rejects.toThrow(/not found/);
  });
});
