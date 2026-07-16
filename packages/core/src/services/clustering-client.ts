/**
 * Transport to the Python clustering sidecar (ml/cluster.py). Writes the embedding batch to a temp
 * float32 file, spawns the sidecar, and parses the returned clusters. Stateless and privacy-preserving:
 * only vectors cross the boundary, never plaintext email or the database. The sidecar is optional -
 * callers must handle it being unavailable (Python or its packages not installed).
 */
import { spawn } from 'node:child_process';
import {
  mkdtempSync,
  openSync,
  writeSync,
  closeSync,
  readFileSync,
  rmSync,
  existsSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** One discovered cluster: a category definition Core names and matches against. */
export interface DiscoveredClusterDef {
  id: number;
  size: number;
  /** Row indices into the input vectors, nearest the centroid, for naming the cluster. */
  representatives: number[];
  /** Unit centroid in the original embedding space. */
  centroid: number[];
  /** Unit sub-centroids for multi-prototype (nearest-prototype) matching. */
  prototypes: number[][];
}

export interface ClusterResult {
  count: number;
  sampledCount: number;
  clusterCount: number;
  clusters: DiscoveredClusterDef[];
  metrics: { noiseFraction: number; biggestClusterFraction: number; silhouette: number | null };
}

export interface ClusterOptions {
  seed?: number;
  targetClusters?: number;
  /** Max time to allow the sidecar, in ms. UMAP on a large inbox can take minutes. */
  timeoutMs?: number;
}

/** Raised when the sidecar cannot run or returns nothing usable, so callers can fall back cleanly. */
export class ClusteringUnavailableError extends Error {}

const DEFAULT_SCRIPT = fileURLToPath(new URL('../../../../ml/cluster.py', import.meta.url));
const DEFAULT_TIMEOUT_MS = 600_000;

/** Runs the Python clustering sidecar over a batch of embedding vectors. */
export class ClusteringClient {
  private readonly python: string;
  private readonly script: string;

  constructor(opts: { python?: string; script?: string } = {}) {
    this.python = opts.python ?? process.env.MAILPILOT_PYTHON ?? 'python';
    this.script = opts.script ?? process.env.MAILPILOT_CLUSTER_SCRIPT ?? DEFAULT_SCRIPT;
  }

  /** True when the sidecar script is present. Does not verify Python packages (that surfaces at run). */
  available(): boolean {
    return existsSync(this.script);
  }

  /**
   * Cluster `vectors` (each length `dim`). Returns the sidecar's cluster definitions. Throws
   * ClusteringUnavailableError when the sidecar is missing, fails, or returns malformed output.
   */
  async cluster(
    vectors: readonly Float32Array[],
    dim: number,
    options: ClusterOptions = {},
  ): Promise<ClusterResult> {
    if (!this.available()) {
      throw new ClusteringUnavailableError(`clustering script not found at ${this.script}`);
    }
    const dir = mkdtempSync(join(tmpdir(), 'mailpilot-cluster-'));
    const inPath = join(dir, 'vectors.f32');
    const outPath = join(dir, 'clusters.json');
    try {
      writeVectors(inPath, vectors, dim);
      await this.run(
        [
          this.script,
          '--input',
          inPath,
          '--count',
          String(vectors.length),
          '--dim',
          String(dim),
          '--seed',
          String(options.seed ?? 42),
          '--target-clusters',
          String(options.targetClusters ?? 15),
          '--output',
          outPath,
        ],
        options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      );
      return parseResult(readFileSync(outPath, 'utf8'));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }

  /** Spawn the sidecar, rejecting on spawn failure, non-zero exit, or timeout. */
  private run(args: string[], timeoutMs: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const child = spawn(this.python, args, { stdio: ['ignore', 'ignore', 'pipe'] });
      let stderr = '';
      const timer = setTimeout(() => {
        child.kill('SIGKILL');
        reject(new ClusteringUnavailableError(`clustering timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      child.stderr?.on('data', (d) => {
        stderr += String(d);
      });
      child.on('error', (err) => {
        clearTimeout(timer);
        reject(
          new ClusteringUnavailableError(`could not start clustering sidecar: ${err.message}`),
        );
      });
      child.on('close', (code) => {
        clearTimeout(timer);
        if (code === 0) resolve();
        else
          reject(
            new ClusteringUnavailableError(`clustering exited ${code}: ${stderr.slice(-500)}`),
          );
      });
    });
  }
}

/** Write the vectors as a contiguous little-endian float32 buffer the sidecar reads with np.fromfile. */
function writeVectors(path: string, vectors: readonly Float32Array[], dim: number): void {
  const fd = openSync(path, 'w');
  try {
    for (const v of vectors) {
      if (v.length !== dim) throw new Error(`vector length ${v.length} != dim ${dim}`);
      writeSync(fd, Buffer.from(v.buffer, v.byteOffset, v.byteLength));
    }
  } finally {
    closeSync(fd);
  }
}

/** Parse and shape-check the sidecar's JSON so a truncated/garbled result fails loudly, not silently. */
function parseResult(raw: string): ClusterResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw new ClusteringUnavailableError('clustering returned invalid JSON');
  }
  const r = parsed as ClusterResult;
  if (!r || !Array.isArray(r.clusters)) {
    throw new ClusteringUnavailableError('clustering result missing clusters');
  }
  return r;
}
