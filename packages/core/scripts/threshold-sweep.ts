/**
 * Evaluation instrument: characterises the fast-pass gate's operating region.
 *
 * The two-pass categoriser accepts an embedding-only assignment when the top category clears an
 * absolute confidence floor (FAST_GATE_MIN_CONFIDENCE, 0.78) and beats the runner-up by a margin
 * (FAST_GATE_MIN_MARGIN, 0.10). Both constants were set by hand. This script sweeps them over the
 * real corpus and reports, at each point, how much mail the cheap path would take and how often it
 * would agree with the LLM pass that actually ran.
 *
 * No model calls. Ranking uses the production rankCategories, and the swept gate is checked against
 * the production gateFastAssignment at the shipped constants before any sweep runs, so the swept
 * predicate is known to be the same function.
 *
 * Agreement is measured against the stored decisions of the LLM pass (email_categories.method =
 * 'llm'), which is a recorded decision rather than human ground truth. See the caveats printed at
 * the end of the run.
 *
 * Usage:
 *   npx tsx packages/core/scripts/threshold-sweep.ts --db <path> [--account <id>] [--out <dir>]
 *
 * Run it against a COPY of the database, never the live file.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import BetterSqlite3 from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { DB_KEY_PATH } from '../src/util/paths.js';
import { applyDbKey, resolveDbKey } from '../src/db/encryption.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { rankCategories, type CategoryMatch } from '../src/services/categorization-service.js';
import { gateFastAssignment } from '../src/services/categorize-strategy.js';

interface Options {
  db: string | null;
  account: string | null;
  out: string;
  model: string | null;
  multiPrototype: boolean;
}

function parseArgs(argv: string[]): Options {
  const o: Options = { db: null, account: null, out: '.', model: null, multiPrototype: true };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === '--db') o.db = argv[++i] ?? null;
    else if (a === '--account') o.account = argv[++i] ?? null;
    else if (a === '--out') o.out = argv[++i] ?? '.';
    else if (a === '--model') o.model = argv[++i] ?? null;
    else if (a === '--single-prototype') o.multiPrototype = false;
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

/** The shipped gate, with its two constants lifted into parameters. */
function gateAt(ranked: TopTwo, minConfidence: number, minMargin: number): boolean {
  if (ranked.top1 === null) return false;
  if (ranked.top1 < minConfidence) return false;
  if (ranked.top2 !== null && ranked.top1 - ranked.top2 < minMargin) return false;
  return true;
}

/** Per-email cache: only the two best confidences and the winning category are needed for a sweep. */
interface TopTwo {
  messageId: string;
  top1: number | null;
  top2: number | null;
  top1Category: string | null;
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

const accountList = accounts.list();
const account = opts.account
  ? (accountList.find((a) => a.id === opts.account) ?? null)
  : (accountList[0] ?? null);
if (!account) {
  console.error('No matching account.');
  process.exit(1);
}

// The model the centroids were actually built under, so the sweep never compares across models.
const modelRow = db
  .prepare(
    `SELECT cei.model_id AS model_id, COUNT(*) AS n
       FROM category_embedding_index cei
       JOIN categories c ON c.id = cei.category_id
      WHERE c.account_id = ? AND cei.prototype_index = 0
      GROUP BY cei.model_id ORDER BY n DESC LIMIT 1`,
  )
  .get(account.id) as { model_id: string; n: number } | undefined;

const modelId = opts.model ?? modelRow?.model_id;
if (!modelId) {
  console.error('No category centroids exist for this account. Run discovery first.');
  process.exit(1);
}

const prototypes = categories.getEffectivePrototypeEntries(account.id, modelId, opts.multiPrototype);
const activeCategories = categories.listActive(account.id);
const withCentroid = new Set(prototypes.map((p) => p.categoryId));

console.log('=== setup ===');
console.log(`account            ${account.address} (${account.kind})`);
console.log(`embedding model    ${modelId}`);
console.log(`multi-prototype    ${opts.multiPrototype}`);
console.log(`active categories  ${activeCategories.length}`);
console.log(`prototype vectors  ${prototypes.length}`);
console.log(
  `categories without a centroid  ${activeCategories.filter((c) => !withCentroid.has(c.id)).length}`,
);

// Recorded LLM decisions, used as the comparison set. Multi-label, so a set per message.
const llmRows = db
  .prepare(
    `SELECT message_id, category_id, confidence
       FROM email_categories
      WHERE account_id = ? AND method = 'llm' AND assigned_by = 'auto'`,
  )
  .all(account.id) as Array<{ message_id: string; category_id: string; confidence: number }>;

const llmSet = new Map<string, Set<string>>();
const llmPrimary = new Map<string, { id: string; confidence: number }>();
for (const r of llmRows) {
  let s = llmSet.get(r.message_id);
  if (!s) {
    s = new Set();
    llmSet.set(r.message_id, s);
  }
  s.add(r.category_id);
  const cur = llmPrimary.get(r.message_id);
  if (!cur || r.confidence > cur.confidence) {
    llmPrimary.set(r.message_id, { id: r.category_id, confidence: r.confidence });
  }
}
console.log(`llm-decided messages           ${llmSet.size}`);

// Messages a user corrected by hand are excluded: the gate never runs on them in production.
const userLocked = categories.getUserAssignedMessageIds(account.id);
console.log(`user-locked messages (skipped) ${userLocked.size}`);

console.log('\n=== ranking corpus ===');
const started = Date.now();
const entries = embeddings.listForAccount(account.id, modelId);
console.log(`embeddings loaded  ${entries.length} in ${Date.now() - started} ms`);

const cache: TopTwo[] = [];
let mismatches = 0;
const rankStart = Date.now();
for (let i = 0; i < entries.length; i++) {
  const e = entries[i]!;
  if (userLocked.has(e.messageId)) continue;
  const ranked: CategoryMatch[] = rankCategories(e.vector, prototypes);
  const t: TopTwo = {
    messageId: e.messageId,
    top1: ranked[0]?.confidence ?? null,
    top2: ranked[1]?.confidence ?? null,
    top1Category: ranked[0]?.categoryId ?? null,
  };
  cache.push(t);

  // Equivalence check against the production gate at the shipped constants.
  const shipped = gateFastAssignment(ranked) !== null;
  const swept = gateAt(t, 0.78, 0.1);
  if (shipped !== swept) mismatches += 1;

  if ((i + 1) % 10000 === 0) {
    console.log(`  ranked ${i + 1}/${entries.length}`);
  }
}
console.log(`ranked ${cache.length} messages in ${Date.now() - rankStart} ms`);
if (mismatches > 0) {
  console.error(
    `ABORT: swept gate disagreed with gateFastAssignment on ${mismatches} messages. The sweep is not measuring the shipped predicate.`,
  );
  process.exit(1);
}
console.log('swept gate matches gateFastAssignment on every message at (0.78, 0.10)');

/** One row of the sweep. */
interface Point {
  minConfidence: number;
  minMargin: number;
  fired: number;
  fireRate: number;
  escalationRate: number;
  comparable: number;
  agreeInSet: number;
  agreeInSetRate: number;
  agreePrimary: number;
  agreePrimaryRate: number;
}

const confidences = [
  0.5, 0.55, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88,
  0.9, 0.92, 0.95,
];
const margins = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3];

const points: Point[] = [];
for (const mc of confidences) {
  for (const mm of margins) {
    let fired = 0;
    let comparable = 0;
    let agreeInSet = 0;
    let agreePrimary = 0;
    for (const t of cache) {
      if (!gateAt(t, mc, mm)) continue;
      fired += 1;
      const set = llmSet.get(t.messageId);
      if (!set || !t.top1Category) continue;
      comparable += 1;
      if (set.has(t.top1Category)) agreeInSet += 1;
      if (llmPrimary.get(t.messageId)?.id === t.top1Category) agreePrimary += 1;
    }
    points.push({
      minConfidence: mc,
      minMargin: mm,
      fired,
      fireRate: fired / cache.length,
      escalationRate: 1 - fired / cache.length,
      comparable,
      agreeInSet,
      agreeInSetRate: comparable ? agreeInSet / comparable : 0,
      agreePrimary,
      agreePrimaryRate: comparable ? agreePrimary / comparable : 0,
    });
  }
}

mkdirSync(opts.out, { recursive: true });
const csv = [
  'min_confidence,min_margin,fired,fire_rate,escalation_rate,comparable,agree_in_set,agree_in_set_rate,agree_primary,agree_primary_rate',
  ...points.map((p) =>
    [
      p.minConfidence,
      p.minMargin,
      p.fired,
      p.fireRate.toFixed(6),
      p.escalationRate.toFixed(6),
      p.comparable,
      p.agreeInSet,
      p.agreeInSetRate.toFixed(6),
      p.agreePrimary,
      p.agreePrimaryRate.toFixed(6),
    ].join(','),
  ),
].join('\n');
writeFileSync(join(opts.out, 'threshold-sweep.csv'), csv, 'utf8');

// Distribution of the two quantities the gate tests, for the report's figure.
const top1s = cache.map((c) => c.top1 ?? 0).sort((a, b) => a - b);
const gaps = cache
  .map((c) => (c.top1 === null ? 0 : c.top1 - (c.top2 ?? 0)))
  .sort((a, b) => a - b);
const pct = (arr: number[], p: number): number => arr[Math.floor((arr.length - 1) * p)] ?? 0;
const distCsv = [
  'quantile,top1_confidence,top1_minus_top2',
  ...[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].map(
    (q) => `${q},${pct(top1s, q).toFixed(6)},${pct(gaps, q).toFixed(6)}`,
  ),
].join('\n');
writeFileSync(join(opts.out, 'threshold-distribution.csv'), distCsv, 'utf8');

const at = (mc: number, mm: number): Point =>
  points.find((p) => p.minConfidence === mc && p.minMargin === mm)!;
const shippedPoint = at(0.78, 0.1);

console.log('\n=== margin held at the shipped 0.10, sweeping the confidence floor ===');
console.table(
  confidences.map((mc) => {
    const p = at(mc, 0.1);
    return {
      min_confidence: mc,
      fired: p.fired,
      'fire %': (p.fireRate * 100).toFixed(1),
      'escalation %': (p.escalationRate * 100).toFixed(1),
      comparable: p.comparable,
      'agree in set %': (p.agreeInSetRate * 100).toFixed(1),
      'agree primary %': (p.agreePrimaryRate * 100).toFixed(1),
    };
  }),
);

console.log('\n=== confidence held at the shipped 0.78, sweeping the margin ===');
console.table(
  margins.map((mm) => {
    const p = at(0.78, mm);
    return {
      min_margin: mm,
      fired: p.fired,
      'fire %': (p.fireRate * 100).toFixed(1),
      'escalation %': (p.escalationRate * 100).toFixed(1),
      comparable: p.comparable,
      'agree in set %': (p.agreeInSetRate * 100).toFixed(1),
      'agree primary %': (p.agreePrimaryRate * 100).toFixed(1),
    };
  }),
);

console.log('\n=== distribution of what the gate tests ===');
console.table(
  [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].map((q) => ({
    quantile: q,
    top1_confidence: pct(top1s, q).toFixed(4),
    top1_minus_top2: pct(gaps, q).toFixed(4),
  })),
);

console.log('\n=== shipped operating point (0.78, 0.10) ===');
console.log(`messages scored        ${cache.length}`);
console.log(
  `fast path takes        ${shippedPoint.fired} (${(shippedPoint.fireRate * 100).toFixed(2)}%)`,
);
console.log(
  `escalates to the LLM   ${cache.length - shippedPoint.fired} (${(shippedPoint.escalationRate * 100).toFixed(2)}%)`,
);
console.log(`comparable to an LLM decision  ${shippedPoint.comparable}`);
console.log(
  `agreement with the LLM label set  ${(shippedPoint.agreeInSetRate * 100).toFixed(2)}%`,
);
console.log(
  `agreement with the LLM primary    ${(shippedPoint.agreePrimaryRate * 100).toFixed(2)}%`,
);

console.log(`\nwrote ${join(opts.out, 'threshold-sweep.csv')}`);
console.log(`wrote ${join(opts.out, 'threshold-distribution.csv')}`);

console.log(`
=== caveats to carry into the report ===
1. Agreement is measured against the LLM pass's recorded decision, not human ground truth. It says
   how often the cheap path reproduces the expensive path, not how often either is correct.
2. The LLM decisions were written by earlier runs, some of them under a different taxonomy and a
   different model. Treat agreement as indicative, not as a controlled comparison.
3. Centroids are the ones stored now. A category whose centroid was rebuilt after those decisions
   were made will rank differently than it did at the time.
4. Messages the user corrected by hand are excluded, matching production behaviour.
5. n is one mailbox, one user, one account.
`);

db.close();
