/**
 * Transport to the persistent Python cross-encoder reranking sidecar (ml/rerank_server.py). The
 * sidecar loads a multilingual cross-encoder once and stays warm; this client spawns it lazily on
 * first use and serves rerank requests over newline-delimited JSON on stdin/stdout. A cross-encoder
 * ranks a (query, document) pair jointly, far more reliably than a weak local LLM listwise reranker.
 *
 * Optional and fail-safe: if Python, sentence-transformers, or the model is missing, or the process
 * crashes or times out, `rerank` resolves to null and the caller falls back to fusion order. Requests
 * are serialized (one in flight) since the sidecar holds a single model. Privacy-preserving: only the
 * query and candidate snippets cross the boundary.
 */
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { createInterface, type Interface } from 'node:readline';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import type { Logger } from 'pino';

const DEFAULT_SCRIPT = fileURLToPath(new URL('../../../../ml/rerank_server.py', import.meta.url));
const DEFAULT_TIMEOUT_MS = 30_000;
/** First start may download the model (hundreds of MB), so the readiness wait is generous. */
const START_TIMEOUT_MS = 300_000;

/** Options for constructing a RerankerClient. */
export interface RerankerClientOptions {
  python?: string;
  script?: string;
  model?: string;
  logger?: Logger;
}

/** A pending request awaiting the sidecar's next response line. */
interface Pending {
  resolve: (scores: number[] | null) => void;
  timer: ReturnType<typeof setTimeout>;
}

/**
 * Runs the persistent Python cross-encoder reranking sidecar. `rerank` returns per-document scores
 * (higher is more relevant), or null when the sidecar is unavailable so the caller keeps fusion order.
 */
export class RerankerClient {
  private readonly python: string;
  private readonly script: string;
  private readonly model?: string;
  private readonly logger?: Logger;

  private proc: ChildProcessWithoutNullStreams | null = null;
  private rl: Interface | null = null;
  private starting: Promise<boolean> | null = null;
  /** Serializes requests: the sidecar holds one model and answers one line at a time. */
  private queue: Promise<unknown> = Promise.resolve();
  private pending: Pending | null = null;
  /** Set once the sidecar proves unavailable, so we stop trying for the process lifetime. */
  private disabled = false;

  constructor(opts: RerankerClientOptions = {}) {
    this.python = opts.python ?? process.env.MAILPILOT_PYTHON ?? 'python';
    this.script = opts.script ?? process.env.MAILPILOT_RERANK_SCRIPT ?? DEFAULT_SCRIPT;
    this.model = opts.model ?? process.env.MAILPILOT_RERANK_MODEL;
    this.logger = opts.logger;
  }

  /** True when the sidecar script is present and has not already proven unavailable. */
  available(): boolean {
    return !this.disabled && existsSync(this.script);
  }

  /**
   * Score each document against the query. Returns scores aligned to `documents`, or null when the
   * sidecar is unavailable, times out, or errors, so the caller falls back to fusion order.
   */
  async rerank(
    query: string,
    documents: string[],
    timeoutMs: number = DEFAULT_TIMEOUT_MS,
  ): Promise<number[] | null> {
    if (!this.available() || documents.length === 0) return null;
    const run = this.queue.then(() => this.rerankOne(query, documents, timeoutMs));
    // Keep the chain alive even if this request rejects, so one failure does not wedge the queue.
    this.queue = run.catch(() => undefined);
    return run;
  }

  /** Stop the sidecar (called on Core shutdown). */
  dispose(): void {
    this.disabled = true;
    this.teardown();
  }

  /** Spawn the sidecar and resolve once it prints its readiness line; caches the result. */
  private ensureStarted(): Promise<boolean> {
    if (this.starting) return this.starting;
    this.starting = new Promise<boolean>((resolve) => {
      let proc: ChildProcessWithoutNullStreams;
      try {
        proc = spawn(this.python, [this.script], {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: this.model ? { ...process.env, MAILPILOT_RERANK_MODEL: this.model } : process.env,
        });
      } catch (err) {
        this.logger?.warn({ err }, 'reranker: could not spawn sidecar');
        this.disabled = true;
        resolve(false);
        return;
      }

      const startTimer = setTimeout(() => {
        this.logger?.warn('reranker: sidecar did not become ready in time');
        this.disabled = true;
        this.teardown();
        resolve(false);
      }, START_TIMEOUT_MS);

      const rl = createInterface({ input: proc.stdout });
      let ready = false;
      rl.on('line', (line) => {
        if (!ready) {
          clearTimeout(startTimer);
          let msg: { ready?: boolean; error?: string; model?: string } = {};
          try {
            msg = JSON.parse(line) as typeof msg;
          } catch {
            /* ignore a non-JSON banner line */
            return;
          }
          if (msg.ready) {
            ready = true;
            this.logger?.info({ model: msg.model }, 'reranker: sidecar ready');
            resolve(true);
          } else {
            this.logger?.warn({ error: msg.error }, 'reranker: sidecar failed to load model');
            this.disabled = true;
            this.teardown();
            resolve(false);
          }
          return;
        }
        this.onResponse(line);
      });

      proc.stderr.on('data', (d) =>
        this.logger?.debug({ stderr: String(d).slice(-300) }, 'reranker'),
      );
      proc.on('exit', (code) => {
        this.logger?.warn({ code }, 'reranker: sidecar exited');
        this.failPending();
        this.teardown();
        // A crash disables the client for this Core lifetime; fusion order takes over.
        this.disabled = true;
      });

      this.proc = proc;
      this.rl = rl;
    });
    return this.starting;
  }

  /** Send one request and await its response line, enforcing a timeout. */
  private async rerankOne(
    query: string,
    documents: string[],
    timeoutMs: number,
  ): Promise<number[] | null> {
    const ok = await this.ensureStarted();
    if (!ok || !this.proc) return null;
    return new Promise<number[] | null>((resolve) => {
      const timer = setTimeout(() => {
        if (this.pending) {
          this.pending = null;
          this.logger?.warn('reranker: request timed out');
          resolve(null);
        }
      }, timeoutMs);
      this.pending = { resolve, timer };
      try {
        this.proc!.stdin.write(JSON.stringify({ query, documents }) + '\n');
      } catch (err) {
        clearTimeout(timer);
        this.pending = null;
        this.logger?.warn({ err }, 'reranker: write failed');
        resolve(null);
      }
    });
  }

  /** Resolve the in-flight request with the sidecar's parsed scores, or null on a bad line. */
  private onResponse(line: string): void {
    const p = this.pending;
    if (!p) return;
    this.pending = null;
    clearTimeout(p.timer);
    try {
      const msg = JSON.parse(line) as { scores?: number[]; error?: string };
      if (Array.isArray(msg.scores)) p.resolve(msg.scores);
      else {
        this.logger?.warn({ error: msg.error }, 'reranker: request error');
        p.resolve(null);
      }
    } catch {
      p.resolve(null);
    }
  }

  /** Fail any in-flight request (used when the sidecar exits). */
  private failPending(): void {
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.resolve(null);
      this.pending = null;
    }
  }

  /** Tear down the process and reader; a later call re-spawns unless disabled. */
  private teardown(): void {
    this.rl?.close();
    this.rl = null;
    if (this.proc) {
      this.proc.removeAllListeners('exit');
      this.proc.kill('SIGKILL');
      this.proc = null;
    }
    this.starting = null;
  }
}
