/**
 * Evaluation instrument: rebuilds every active category's centroid set from its member embeddings
 * using the shipped CategoryCentroidRebuildService, and reports how far each centroid moved.
 *
 * Pairs with threshold-sweep.ts. Sweeping before and after a rebuild isolates one variable, centroid
 * freshness, and the --multi-prototype flag isolates a second, whether sub-prototypes earn their keep.
 *
 * No model calls: a rebuild is the normalized mean of stored member vectors. Rebuilds fall back to
 * auto members only for categories with no user-confirmed members, which is the service's own rule.
 *
 * Usage:
 *   npx tsx packages/core/scripts/rebuild-centroids.ts --db <path> [--multi-prototype] [--dry-run]
 *
 * Writes to the database it is given. Run it against a COPY, never the live file.
 */
import BetterSqlite3 from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import pino from 'pino';
import { DB_KEY_PATH } from '../src/util/paths.js';
import { applyDbKey, resolveDbKey } from '../src/db/encryption.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { CategoryCentroidRebuildService } from '../src/services/category-centroid-rebuild-service.js';

interface Options {
  db: string | null;
  account: string | null;
  model: string | null;
  multiPrototype: boolean;
  dryRun: boolean;
}

function parseArgs(argv: string[]): Options {
  const o: Options = { db: null, account: null, model: null, multiPrototype: false, dryRun: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === '--db') o.db = argv[++i] ?? null;
    else if (a === '--account') o.account = argv[++i] ?? null;
    else if (a === '--model') o.model = argv[++i] ?? null;
    else if (a === '--multi-prototype') o.multiPrototype = true;
    else if (a === '--dry-run') o.dryRun = true;
    else {
      console.error(`Unknown argument: ${a}`);
      process.exit(2);
    }
  }
  if (!o.db) {
    console.error('--db is required. Point it at a COPY of mailpilot.db, not the live file.');
    process.exit(2);
  }
  return o;
}

const opts = parseArgs(process.argv.slice(2));

const { keyHex } = resolveDbKey(DB_KEY_PATH);
const db = new BetterSqlite3(opts.db!);
applyDbKey(db, keyHex);
sqliteVec.load(db);
db.pragma('foreign_keys = ON');

const accounts = new AccountRepository(db);
const categories = new CategoryRepository(db);
const embeddings = new EmbeddingRepository(db);

const account = opts.account ? accounts.findById(opts.account) : (accounts.list()[0] ?? null);
if (!account) {
  console.error('No account found in this database.');
  process.exit(1);
}

const modelId =
  opts.model ??
  (
    db
      .prepare(
        `SELECT model_id, COUNT(*) AS n FROM email_embedding_index
          WHERE account_id = ? GROUP BY model_id ORDER BY n DESC LIMIT 1`,
      )
      .get(account.id) as { model_id: string } | undefined
  )?.model_id;
if (!modelId) {
  console.error('No embedding model found for this account.');
  process.exit(1);
}

const logger = pino({ level: 'silent' });
const service = new CategoryCentroidRebuildService(
  categories,
  embeddings,
  logger,
  () => opts.multiPrototype,
);

const toF32 = (buf: Buffer): Float32Array =>
  new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

/** Cosine between two unit-ish vectors, used only to report how far a centroid moved. */
function cosine(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d === 0 ? 0 : dot / d;
}

const aggregateStmt = db.prepare(
  `SELECT ce.embedding AS embedding
     FROM category_embedding_index cei
     JOIN category_embeddings ce ON ce.rowid = cei.rowid
    WHERE cei.category_id = ? AND cei.model_id = ? AND cei.prototype_index = 0`,
);

function aggregateFor(categoryId: string): Float32Array | null {
  const row = aggregateStmt.get(categoryId, modelId) as { embedding: Buffer } | undefined;
  return row ? toF32(row.embedding) : null;
}

const active = categories.listActive(account.id);

console.log(`account: ${account.address}`);
console.log(`embedding model: ${modelId}`);
console.log(`multi-prototype: ${opts.multiPrototype}`);
console.log(`active categories: ${active.length}`);
console.log(opts.dryRun ? 'MODE: dry run, nothing written\n' : 'MODE: writing\n');

interface Row {
  category: string;
  status: string;
  vectors: number;
  fallback: string;
  subs: number;
  'cosine before/after': string;
  moved: string;
}

const rows: Row[] = [];
let rebuilt = 0;
let skipped = 0;
let created = 0;

for (const category of active) {
  const before = aggregateFor(category.id);
  const userMembers = categories.listCategoryMemberIds(account.id, category.id, 'user');

  if (opts.dryRun) {
    const autoMembers =
      userMembers.length === 0
        ? categories.listCategoryMemberIds(account.id, category.id, 'auto')
        : [];
    rows.push({
      category: category.label,
      status: 'dry-run',
      vectors: userMembers.length || autoMembers.length,
      fallback: userMembers.length === 0 ? 'auto' : 'user',
      subs: 0,
      'cosine before/after': 'n/a',
      moved: before ? '' : 'no prior centroid',
    });
    continue;
  }

  const result = service.rebuild(account.id, category.id, modelId, { allowAutoFallback: true });
  const after = result.status === 'rebuilt' ? aggregateFor(category.id) : before;

  if (result.status === 'rebuilt') rebuilt += 1;
  else skipped += 1;
  if (result.status === 'rebuilt' && !before) created += 1;

  const sim = before && after ? cosine(before, after) : null;
  rows.push({
    category: category.label,
    status: result.status,
    vectors: result.vectorsUsed,
    fallback: result.usedAutoFallback ? 'auto' : 'user',
    subs: result.subPrototypeCount,
    'cosine before/after': sim === null ? 'n/a' : sim.toFixed(4),
    moved: !before ? 'new centroid' : sim !== null && sim < 0.95 ? 'MOVED' : '',
  });
}

rows.sort((a, b) => b.vectors - a.vectors);
console.table(rows);

console.log(`\nrebuilt: ${rebuilt}, skipped: ${skipped}, newly created centroids: ${created}`);

const protoCounts = db
  .prepare(
    `SELECT cei.prototype_index AS idx, COUNT(*) AS n
       FROM category_embedding_index cei
       JOIN categories c ON c.id = cei.category_id
      WHERE c.account_id = ? AND cei.model_id = ? AND c.status = 'active'
      GROUP BY cei.prototype_index ORDER BY idx`,
  )
  .all(account.id, modelId);
console.log('prototype vectors by index:', JSON.stringify(protoCounts));

db.close();
