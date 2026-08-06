/**
 * Maintenance CLI: re-embed attachment chunks under the configured embedding model, filling in each
 * chunk's document context.
 *
 * Run this after changing the embedding model. Chunk vectors are keyed by model id and the KNN
 * search filters on it, so chunks left on the old model are invisible to the dense arm and
 * attachment retrieval silently degrades to keyword-only.
 *
 * Safety: refuses to run while Core is up (it holds the database) and backs the database up before
 * mutating. Defaults to a dry run; nothing is written unless you pass --execute.
 *
 * Usage:
 *   npx tsx packages/core/scripts/reembed-attachment-chunks.ts [--execute] [--account <id>]
 *
 * MAILPILOT_DB points the run at a copy.
 */
import { existsSync, copyFileSync, statSync } from 'node:fs';
import net from 'node:net';
import BetterSqlite3 from 'better-sqlite3';
import { CORE_SERVER_URL, CORE_SERVER_HOST, CORE_SERVER_PORT } from '@ai-mailpilot/shared';
import { openDatabase } from '../src/db/database.js';
import { DB_PATH, DB_KEY_PATH } from '../src/util/paths.js';
import { applyDbKey, resolveDbKey } from '../src/db/encryption.js';
import { loadConfig } from '../src/config/config.js';
import { createLlmClient } from '../src/llm/client.js';
import { reembedAttachmentChunks } from '../src/maintenance/reembed-attachment-chunks.js';

interface Options {
  execute: boolean;
  account: string | null;
}

function parseArgs(argv: string[]): Options {
  const o: Options = { execute: false, account: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === '--execute' || a === '--yes') o.execute = true;
    else if (a === '--account') o.account = argv[++i] ?? null;
    else {
      console.error(`Unknown argument: ${a}`);
      process.exit(2);
    }
  }
  return o;
}

/** True when something is listening on the Core port, meaning Core likely holds the database. */
function coreIsRunning(): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = net.connect({ host: CORE_SERVER_HOST, port: CORE_SERVER_PORT });
    const finish = (up: boolean): void => {
      socket.destroy();
      resolve(up);
    };
    socket.setTimeout(1500);
    socket.once('connect', () => finish(true));
    socket.once('timeout', () => finish(false));
    socket.once('error', () => finish(false));
  });
}

/** Copy the DB and its sidecars to a timestamped snapshot and confirm the copy opens. */
function backupDatabase(dbPath: string, keyHex: string): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const backupPath = `${dbPath}.bak-${stamp}`;
  copyFileSync(dbPath, backupPath);
  for (const suffix of ['-wal', '-shm']) {
    if (existsSync(dbPath + suffix)) copyFileSync(dbPath + suffix, backupPath + suffix);
  }
  if (statSync(backupPath).size === 0) throw new Error(`backup at ${backupPath} is empty`);
  const probe = new BetterSqlite3(backupPath, { readonly: true });
  try {
    applyDbKey(probe, keyHex);
    probe.prepare('SELECT count(*) FROM sqlite_master').get();
  } finally {
    probe.close();
  }
  return backupPath;
}

async function main(): Promise<void> {
  const opts = parseArgs(process.argv.slice(2));
  const dbPath = process.env.MAILPILOT_DB || DB_PATH;
  const isLive = dbPath === DB_PATH;

  if (isLive && (await coreIsRunning())) {
    console.error(
      [
        'Core is running and is holding the database.',
        `Stop it first (close the extension in Thunderbird, or stop the process on ${CORE_SERVER_URL}).`,
      ].join('\n'),
    );
    process.exit(1);
  }
  if (!existsSync(dbPath)) {
    console.error(`No database at ${dbPath}.`);
    process.exit(1);
  }

  const config = loadConfig();
  const modelId = config.llm.embeddingModel;
  const llm = createLlmClient(() => config.llm);

  const health = await llm.health();
  if (!health.ok) {
    console.error(`LLM endpoint ${config.llm.baseUrl} is unreachable; start it and retry.`);
    process.exit(1);
  }

  console.log(`Database:   ${dbPath}${isLive ? '' : '  (copy)'}`);
  console.log(`Embedding:  ${modelId}  via ${config.llm.baseUrl}`);
  console.log(`Mode:       ${opts.execute ? 'EXECUTE' : 'DRY RUN'}`);
  console.log();

  const { keyHex } = resolveDbKey(DB_KEY_PATH);
  const probe = new BetterSqlite3(dbPath, { readonly: true });
  applyDbKey(probe, keyHex);
  const pending = probe
    .prepare(
      `SELECT count(*) AS c FROM attachment_chunks c
        WHERE NOT EXISTS (SELECT 1 FROM attachment_chunk_embedding_index ei
                           WHERE ei.chunk_rowid = c.rowid AND ei.model_id = ?)`,
    )
    .get(modelId) as { c: number };
  const total = probe.prepare('SELECT count(*) AS c FROM attachment_chunks').get() as { c: number };
  probe.close();

  console.log(`attachment chunks:        ${total.c}`);
  console.log(`missing a ${modelId} vector: ${pending.c}`);
  console.log();

  if (pending.c === 0) {
    console.log('Nothing to do: every chunk already has a vector for this model.');
    return;
  }
  if (!opts.execute) {
    console.log('Dry run only. Re-run with --execute to embed (a backup is taken first).');
    return;
  }

  if (isLive) {
    const backupPath = backupDatabase(dbPath, keyHex);
    console.log(`Backup written: ${backupPath}\n`);
  }

  const db = openDatabase(dbPath);
  const started = Date.now();
  try {
    const result = await reembedAttachmentChunks(db, llm, modelId, {
      accountId: opts.account ?? undefined,
      onProgress: (done, all) => {
        if (done % 200 === 0 || done === all) {
          const pct = ((done / all) * 100).toFixed(0);
          console.log(`  ${done}/${all}  (${pct}%)`);
        }
      },
    });
    const secs = ((Date.now() - started) / 1000).toFixed(0);
    console.log();
    console.log('=== DONE ===');
    console.log(`  scanned          ${result.scanned}`);
    console.log(`  embedded         ${result.embedded}`);
    console.log(`  failed           ${result.failed}`);
    console.log(`  contexts written ${result.contextsWritten}`);
    console.log(`  elapsed          ${secs}s`);
  } finally {
    db.close();
  }
}

await main();
