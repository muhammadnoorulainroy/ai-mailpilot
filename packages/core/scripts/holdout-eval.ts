/**
 * Evaluation instrument: 5-fold cross-validation of the fast-pass gate.
 *
 * threshold-sweep.ts scores every message against centroids that were built from those same messages,
 * so its agreement figure is in-sample and flatters the cheap path. This script removes that: the
 * corpus is split into 5 deterministic folds, centroids are rebuilt from 4 folds using the shipped
 * meanNormalize, and only the held-out fold is scored. Every labelled message is scored exactly once,
 * against centroids that never saw it.
 *
 * Ranking uses the production rankCategories. Centroids are aggregate-only (prototype 0), matching the
 * shipped default; the sub-prototype delta is measured separately and in-sample.
 *
 * No model calls. Agreement is still against the LLM pass's recorded decision rather than human
 * ground truth, so this measures how well a nearest-centroid rule generalises to mail it did not see,
 * not how often either pass is correct.
 *
 * Usage:
 *   npx tsx packages/core/scripts/holdout-eval.ts --db <path> [--folds 5] [--out <dir>]
 *
 * Read-only. Run it against a COPY of the database.
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import BetterSqlite3 from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { DB_KEY_PATH } from '../src/util/paths.js';
import { applyDbKey, resolveDbKey } from '../src/db/encryption.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { meanNormalize } from '../src/util/vector.js';
import { rankCategories } from '../src/services/categorization-service.js';
import type { CentroidEntry } from '../src/repositories/category-repository.js';

interface Options {
  db: string | null;
  account: string | null;
  model: string | null;
  folds: number;
  out: string;
}

function parseArgs(argv: string[]): Options {
  const o: Options = { db: null, account: null, model: null, folds: 5, out: '.' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === '--db') o.db = argv[++i] ?? null;
    else if (a === '--account') o.account = argv[++i] ?? null;
    else if (a === '--model') o.model = argv[++i] ?? null;
    else if (a === '--folds') o.folds = Number(argv[++i]);
    else if (a === '--out') o.out = argv[++i] ?? '.';
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

const accounts = new AccountRepository(db);
const account = opts.account ? accounts.findById(opts.account) : (accounts.list()[0] ?? null);
if (!account) {
  console.error('No account found.');
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
  console.error('No embedding model found.');
  process.exit(1);
}

/** Stable string hash, so fold assignment is identical on every run and on every machine. */
function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

const labels = new Map<string, string>();
for (const r of db
  .prepare(`SELECT id, label FROM categories WHERE account_id = ? AND status = 'active'`)
  .all(account.id) as Array<{ id: string; label: string }>) {
  labels.set(r.id, r.label);
}

// Recorded LLM decisions, restricted to categories that are still active.
const llmSet = new Map<string, Set<string>>();
for (const r of db
  .prepare(
    `SELECT ec.message_id AS message_id, ec.category_id AS category_id
       FROM email_categories ec
       JOIN categories c ON c.id = ec.category_id
      WHERE ec.account_id = ? AND ec.method = 'llm' AND ec.assigned_by = 'auto'
        AND c.status = 'active'`,
  )
  .all(account.id) as Array<{ message_id: string; category_id: string }>) {
  let s = llmSet.get(r.message_id);
  if (!s) llmSet.set(r.message_id, (s = new Set()));
  s.add(r.category_id);
}

// Messages the user corrected by hand are excluded, matching production behaviour.
const userLocked = new Set<string>(
  (
    db
      .prepare(
        `SELECT DISTINCT message_id FROM email_categories
          WHERE account_id = ? AND assigned_by = 'user'`,
      )
      .all(account.id) as Array<{ message_id: string }>
  ).map((r) => r.message_id),
);

const toF32 = (buf: Buffer): Float32Array =>
  new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

interface Labelled {
  messageId: string;
  vector: Float32Array;
  categories: Set<string>;
  fold: number;
}

const t0 = Date.now();
const corpus: Labelled[] = [];
for (const row of db
  .prepare(
    `SELECT i.message_id AS message_id, e.embedding AS embedding
       FROM email_embedding_index i
       JOIN email_embeddings e ON e.rowid = i.rowid
      WHERE i.account_id = ? AND i.model_id = ?`,
  )
  .iterate(account.id, modelId) as Iterable<{ message_id: string; embedding: Buffer }>) {
  const cats = llmSet.get(row.message_id);
  if (!cats || cats.size === 0) continue;
  if (userLocked.has(row.message_id)) continue;
  corpus.push({
    messageId: row.message_id,
    vector: toF32(row.embedding),
    categories: cats,
    fold: fnv1a(row.message_id) % opts.folds,
  });
}

console.log(`account: ${account.address}`);
console.log(`embedding model: ${modelId}`);
console.log(`labelled messages: ${corpus.length} (loaded in ${Date.now() - t0} ms)`);
console.log(`folds: ${opts.folds}\n`);

const MIN_CONFIDENCE = 0.78;
const MIN_MARGIN = 0.1;
/** Fewest members a fold's training split needs before that category gets a centroid at all. */
const MIN_TRAIN_MEMBERS = 3;

interface Tally {
  fired: number;
  scored: number;
  comparable: number;
  agree: number;
}
const empty = (): Tally => ({ fired: 0, scored: 0, comparable: 0, agree: 0 });

/** Build one aggregate centroid per category from the given training members. */
function centroidsFrom(train: Labelled[]): CentroidEntry[] {
  const byCategory = new Map<string, Float32Array[]>();
  for (const m of train) {
    for (const categoryId of m.categories) {
      let arr = byCategory.get(categoryId);
      if (!arr) byCategory.set(categoryId, (arr = []));
      arr.push(m.vector);
    }
  }
  const out: CentroidEntry[] = [];
  for (const [categoryId, vectors] of byCategory) {
    if (vectors.length < MIN_TRAIN_MEMBERS) continue;
    const vector = meanNormalize(vectors);
    if (!vector) continue;
    out.push({
      categoryId,
      label: labels.get(categoryId) ?? categoryId,
      emailCount: vectors.length,
      vector,
    });
  }
  return out;
}

/** Score a message set against centroids and tally gate behaviour at the shipped constants. */
function evaluate(
  items: Labelled[],
  centroids: CentroidEntry[],
  perCategory?: Map<string, Tally>,
): Tally {
  const t = empty();
  for (const m of items) {
    const ranked = rankCategories(m.vector, centroids);
    t.scored += 1;
    const top1 = ranked[0];
    if (!top1) continue;
    const top2 = ranked[1];
    if (top1.confidence < MIN_CONFIDENCE) continue;
    if (top2 && top1.confidence - top2.confidence < MIN_MARGIN) continue;
    t.fired += 1;
    t.comparable += 1;
    const hit = m.categories.has(top1.categoryId);
    if (hit) t.agree += 1;
    if (perCategory) {
      let pc = perCategory.get(top1.categoryId);
      if (!pc) perCategory.set(top1.categoryId, (pc = empty()));
      pc.fired += 1;
      pc.comparable += 1;
      if (hit) pc.agree += 1;
    }
  }
  return t;
}

const pct = (n: number, d: number): string => (d === 0 ? 'n/a' : ((n / d) * 100).toFixed(2));

const foldRows: Array<Record<string, string | number>> = [];
const total = empty();
const perCategory = new Map<string, Tally>();
let inSampleFold0: Tally | null = null;

for (let k = 0; k < opts.folds; k++) {
  const train = corpus.filter((m) => m.fold !== k);
  const test = corpus.filter((m) => m.fold === k);
  const centroids = centroidsFrom(train);
  const t = evaluate(test, centroids, perCategory);

  total.fired += t.fired;
  total.scored += t.scored;
  total.comparable += t.comparable;
  total.agree += t.agree;

  // Fold 0 also scored in-sample, so the optimism of the in-sample figure is visible as a pair.
  if (k === 0) inSampleFold0 = evaluate(train, centroids);

  foldRows.push({
    fold: k,
    train: train.length,
    test: test.length,
    centroids: centroids.length,
    fired: t.fired,
    'fires %': pct(t.fired, t.scored),
    'agrees %': pct(t.agree, t.comparable),
  });
}

console.table(foldRows);

const rates = foldRows.map((r) => Number(r['agrees %']));
const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
const spread = Math.max(...rates) - Math.min(...rates);

console.log(`\n=== held-out result at the shipped operating point (0.78, 0.10) ===`);
console.log(`labelled messages scored   ${total.scored}`);
console.log(`fast path takes            ${total.fired} (${pct(total.fired, total.scored)}%)`);
console.log(
  `escalates to the LLM       ${total.scored - total.fired} (${pct(total.scored - total.fired, total.scored)}%)`,
);
console.log(`agreement, pooled          ${pct(total.agree, total.comparable)}%`);
console.log(`agreement, mean of folds   ${mean.toFixed(2)}% (spread ${spread.toFixed(2)} points)`);

if (inSampleFold0) {
  console.log(`\n=== the cost of measuring in-sample (fold 0 centroids, both splits) ===`);
  console.log(
    `in-sample  (training split, ${inSampleFold0.scored} msgs): fires ${pct(inSampleFold0.fired, inSampleFold0.scored)}%, agrees ${pct(inSampleFold0.agree, inSampleFold0.comparable)}%`,
  );
  const f0 = foldRows[0]!;
  console.log(
    `held-out   (test split, ${f0.test} msgs): fires ${f0['fires %']}%, agrees ${f0['agrees %']}%`,
  );
}

const catRows = [...perCategory.entries()]
  .map(([id, t]) => ({
    category: labels.get(id) ?? id,
    'gate wins': t.fired,
    'agrees %': pct(t.agree, t.comparable),
  }))
  .sort((a, b) => b['gate wins'] - a['gate wins']);
console.log(`\n=== held-out behaviour by winning category ===`);
console.table(catRows);

mkdirSync(opts.out, { recursive: true });
const csv = [
  'fold,train,test,centroids,fired,fires_pct,agrees_pct',
  ...foldRows.map((r) =>
    [r.fold, r.train, r.test, r.centroids, r.fired, r['fires %'], r['agrees %']].join(','),
  ),
].join('\n');
writeFileSync(join(opts.out, 'holdout-folds.csv'), csv + '\n', 'utf8');
const catCsv = [
  'category,gate_wins,agrees_pct',
  ...catRows.map((r) => `"${r.category}",${r['gate wins']},${r['agrees %']}`),
].join('\n');
writeFileSync(join(opts.out, 'holdout-by-category.csv'), catCsv + '\n', 'utf8');
console.log(`\nwrote ${join(opts.out, 'holdout-folds.csv')}`);
console.log(`wrote ${join(opts.out, 'holdout-by-category.csv')}`);

console.log(`\n=== caveats ===`);
console.log(`1. Agreement is against the LLM pass's recorded decision, not human ground truth.`);
console.log(`2. Only messages carrying an LLM decision to an active category are scored.`);
console.log(`3. Centroids are aggregate-only; the sub-prototype delta is measured separately.`);
console.log(`4. n is one mailbox, and one category holds most of the mail.`);

db.close();
