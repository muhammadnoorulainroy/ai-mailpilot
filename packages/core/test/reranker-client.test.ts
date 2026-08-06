/**
 * Transport tests for RerankerClient against a FAKE sidecar (a Node script implementing the same
 * newline-JSON protocol), so the persistent-process, request/response, and fallback logic is covered
 * without the real model. The fake returns descending scores (first document highest).
 */
import { describe, it, expect, afterEach } from 'vitest';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { RerankerClient } from '../src/services/reranker-client.js';

/** A fake sidecar: prints a ready line, then for each request echoes descending scores. */
const FAKE_SIDECAR = `
process.stdout.write(JSON.stringify({ ready: true, model: 'fake' }) + '\\n');
let buf = '';
process.stdin.on('data', (d) => {
  buf += d;
  let i;
  while ((i = buf.indexOf('\\n')) >= 0) {
    const line = buf.slice(0, i);
    buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    const req = JSON.parse(line);
    const scores = req.documents.map((_, k) => req.documents.length - k);
    process.stdout.write(JSON.stringify({ scores }) + '\\n');
  }
});
`;

/** A fake sidecar that fails to load its model (prints not-ready and exits). */
const FAILING_SIDECAR = `process.stdout.write(JSON.stringify({ ready: false, error: 'no model' }) + '\\n');`;

let dir: string | null = null;
const clients: RerankerClient[] = [];

function writeScript(body: string): string {
  dir = mkdtempSync(join(tmpdir(), 'mailpilot-rerank-test-'));
  const path = join(dir, 'fake.cjs');
  writeFileSync(path, body, 'utf8');
  return path;
}

function makeClient(script: string): RerankerClient {
  const c = new RerankerClient({ python: process.execPath, script });
  clients.push(c);
  return c;
}

afterEach(() => {
  for (const c of clients.splice(0)) c.dispose();
  if (dir) rmSync(dir, { recursive: true, force: true });
  dir = null;
});

describe('RerankerClient', () => {
  it('returns scores aligned to the documents from a warm sidecar', async () => {
    const client = makeClient(writeScript(FAKE_SIDECAR));
    const scores = await client.rerank('q', ['a', 'b', 'c']);
    expect(scores).toEqual([3, 2, 1]);
  });

  it('reuses the same warm process across calls', async () => {
    const client = makeClient(writeScript(FAKE_SIDECAR));
    expect(await client.rerank('q', ['a', 'b'])).toEqual([2, 1]);
    expect(await client.rerank('q2', ['x'])).toEqual([1]);
  });

  it('returns null (fallback) when the sidecar script is missing', async () => {
    const client = makeClient(join(tmpdir(), 'does-not-exist-rerank.cjs'));
    expect(client.available()).toBe(false);
    expect(await client.rerank('q', ['a'])).toBeNull();
  });

  it('returns null for an empty document list without spawning', async () => {
    const client = makeClient(writeScript(FAKE_SIDECAR));
    expect(await client.rerank('q', [])).toBeNull();
  });

  it('disables and returns null when the sidecar cannot load its model', async () => {
    const client = makeClient(writeScript(FAILING_SIDECAR));
    expect(await client.rerank('q', ['a'])).toBeNull();
  });
});
