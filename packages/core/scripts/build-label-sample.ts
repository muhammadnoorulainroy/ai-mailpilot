/**
 * Evaluation instrument: draws a stratified sample for hand-labelling and freezes every arm's
 * prediction before a single human label exists.
 *
 * The sweep and the cross-validation both measure the cheap path against the expensive path. Neither
 * can say whether either is right. This script produces the ground-truth instrument: a blinded
 * labelling sheet, plus a key file holding what each arm predicted, written now so no arm can be
 * tuned after seeing the answers.
 *
 * Three strata, because they answer different questions:
 *   uniform      - an unbiased random draw. The only stratum that estimates population accuracy.
 *   per-category - a fixed quota per active category. Too thin for a precision estimate, enough to
 *                  catch a broken category, which is how Travel and Accommodation would have shown up.
 *   disagreement - messages where the fast gate and the LLM pass chose differently. The most
 *                  informative labels available, because they adjudicate the two passes directly.
 *
 * Only the uniform stratum may be pooled for an overall figure. The scorer enforces this.
 *
 * The labeller never sees a prediction, a confidence, or a stratum. Anchoring a human to the model's
 * answer would make the whole measurement worthless.
 *
 * No model calls. Snippets are built with the production preprocessForEmbedding, so the labeller
 * reads roughly what the embedding model read.
 *
 * Usage:
 *   npx tsx packages/core/scripts/build-label-sample.ts --db <path> --out <dir>
 *     [--uniform 150] [--per-category 12] [--disagreement 40] [--seed mailpilot-eval-1]
 *
 * Read-only against the database. Output contains real message content, so it is written outside
 * the repository and must not be committed.
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import BetterSqlite3 from 'better-sqlite3';
import * as sqliteVec from 'sqlite-vec';
import { DB_KEY_PATH } from '../src/util/paths.js';
import { applyDbKey, resolveDbKey } from '../src/db/encryption.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import { preprocessForEmbedding } from '../src/util/text.js';
import { rankCategories } from '../src/services/categorization-service.js';
import { gateFastAssignment } from '../src/services/categorize-strategy.js';

/** Documented here only for the README; the gate itself is applied by gateFastAssignment. */
const GATE_MIN_CONFIDENCE = 0.78;
const GATE_MIN_MARGIN = 0.1;

interface Options {
  db: string | null;
  account: string | null;
  model: string | null;
  out: string;
  uniform: number;
  perCategory: number;
  disagreement: number;
  seed: string;
  snippetChars: number;
}

function parseArgs(argv: string[]): Options {
  const o: Options = {
    db: null,
    account: null,
    model: null,
    out: '.',
    uniform: 150,
    perCategory: 12,
    disagreement: 40,
    seed: 'mailpilot-eval-1',
    snippetChars: 700,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === '--db') o.db = argv[++i] ?? null;
    else if (a === '--account') o.account = argv[++i] ?? null;
    else if (a === '--model') o.model = argv[++i] ?? null;
    else if (a === '--out') o.out = argv[++i] ?? '.';
    else if (a === '--uniform') o.uniform = Number(argv[++i]);
    else if (a === '--per-category') o.perCategory = Number(argv[++i]);
    else if (a === '--disagreement') o.disagreement = Number(argv[++i]);
    else if (a === '--seed') o.seed = argv[++i] ?? o.seed;
    else if (a === '--snippet-chars') o.snippetChars = Number(argv[++i]);
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

/** Stable string hash. Sampling must be reproducible and auditable, so no Math.random anywhere. */
function fnv1a(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}
const rank = (salt: string, id: string): number => fnv1a(`${opts.seed}:${salt}:${id}`);

const { keyHex } = resolveDbKey(DB_KEY_PATH);
const db = new BetterSqlite3(opts.db!);
applyDbKey(db, keyHex);
sqliteVec.load(db);

const accounts = new AccountRepository(db);
const categories = new CategoryRepository(db);
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

const active = categories.listActive(account.id);
const labelOf = new Map(active.map((c) => [c.id, c.label]));
const centroids = categories.getEffectivePrototypeEntries(account.id, modelId, true);

// ---------------------------------------------------------------------------
// Stored assignments, split by how they were made.
// ---------------------------------------------------------------------------
interface Assignment {
  categoryId: string;
  method: string | null;
  assignedBy: string;
  confidence: number | null;
}
const assignments = new Map<string, Assignment[]>();
for (const r of db
  .prepare(
    `SELECT message_id, category_id, method, assigned_by, confidence
       FROM email_categories WHERE account_id = ?`,
  )
  .all(account.id) as Array<{
  message_id: string;
  category_id: string;
  method: string | null;
  assigned_by: string;
  confidence: number | null;
}>) {
  let arr = assignments.get(r.message_id);
  if (!arr) assignments.set(r.message_id, (arr = []));
  arr.push({
    categoryId: r.category_id,
    method: r.method,
    assignedBy: r.assigned_by,
    confidence: r.confidence,
  });
}

const llmPick = (id: string): string | null =>
  assignments
    .get(id)
    ?.find((a) => a.method === 'llm' && labelOf.has(a.categoryId))?.categoryId ?? null;

// ---------------------------------------------------------------------------
// Score every embedded message once: nearest centroid, and the shipped gate.
// ---------------------------------------------------------------------------
const toF32 = (buf: Buffer): Float32Array =>
  new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);

interface Scored {
  messageId: string;
  nearest: string | null;
  gate: string | null;
  top1: number | null;
  top2: number | null;
}
const scored = new Map<string, Scored>();
const t0 = Date.now();
for (const row of db
  .prepare(
    `SELECT i.message_id AS message_id, e.embedding AS embedding
       FROM email_embedding_index i
       JOIN email_embeddings e ON e.rowid = i.rowid
      WHERE i.account_id = ? AND i.model_id = ?`,
  )
  .iterate(account.id, modelId) as Iterable<{ message_id: string; embedding: Buffer }>) {
  const ranked = rankCategories(toF32(row.embedding), centroids);
  const t1 = ranked[0] ?? null;
  const t2 = ranked[1] ?? null;
  scored.set(row.message_id, {
    messageId: row.message_id,
    nearest: t1?.categoryId ?? null,
    gate: gateFastAssignment(ranked)?.categoryId ?? null,
    top1: t1?.confidence ?? null,
    top2: t2?.confidence ?? null,
  });
}
console.log(`scored ${scored.size} embedded messages in ${Date.now() - t0} ms`);

// ---------------------------------------------------------------------------
// Strata.
// ---------------------------------------------------------------------------
const pool = [...scored.keys()];
const chosen = new Map<string, string>(); // messageId -> stratum

const uniform = [...pool].sort((a, b) => rank('uniform', a) - rank('uniform', b));
for (const id of uniform.slice(0, opts.uniform)) chosen.set(id, 'uniform');

for (const category of active) {
  const members = (
    db
      .prepare(
        `SELECT message_id FROM email_categories
          WHERE account_id = ? AND category_id = ? AND assigned_by = 'auto'`,
      )
      .all(account.id, category.id) as Array<{ message_id: string }>
  )
    .map((r) => r.message_id)
    .filter((id) => scored.has(id) && !chosen.has(id))
    .sort((a, b) => rank(`cat:${category.id}`, a) - rank(`cat:${category.id}`, b));
  for (const id of members.slice(0, opts.perCategory)) chosen.set(id, `category:${category.label}`);
}

const disagreeing = pool
  .filter((id) => {
    if (chosen.has(id)) return false;
    const g = scored.get(id)?.gate;
    const l = llmPick(id);
    return g !== null && g !== undefined && l !== null && g !== l;
  })
  .sort((a, b) => rank('disagree', a) - rank('disagree', b));
for (const id of disagreeing.slice(0, opts.disagreement)) chosen.set(id, 'disagreement');

// Interleave, so the labeller never sees a run of one category and drifts into answering by rhythm.
const sampleIds = [...chosen.keys()].sort((a, b) => rank('shuffle', a) - rank('shuffle', b));

// ---------------------------------------------------------------------------
// Keyword baseline, derived only from messages that are NOT in the sample.
// ---------------------------------------------------------------------------
const STOP = new Set(
  ('the a an and or but if then for to of in on at by with from as is are was were be been being ' +
    'this that these those it its you your we our they their he she his her i me my not no yes do ' +
    'does did have has had will would can could should may might must about into over under out up ' +
    'down more most other some such only own same so than too very just re fw fwd please thanks ' +
    'thank hi hello dear regards best sincerely email mail message sent view click here view online ' +
    'unsubscribe com www http https org net new all any get got also').split(/\s+/),
);

function tokens(subject: string, from: string, body: string): Set<string> {
  const out = new Set<string>();
  const domain = from.split('@')[1]?.toLowerCase().replace(/[^a-z0-9.]/g, '');
  if (domain) out.add(`domain:${domain}`);
  for (const raw of `${subject} ${body}`.toLowerCase().split(/[^a-z0-9]+/)) {
    if (raw.length < 3 || raw.length > 24) continue;
    if (STOP.has(raw)) continue;
    if (/^\d+$/.test(raw)) continue;
    out.add(raw);
  }
  return out;
}

const bodyStmt = db.prepare(
  `SELECT subject, from_addr, body, body_format FROM emails WHERE message_id = ? AND account_id = ?`,
);
interface EmailRow {
  subject: string | null;
  from_addr: string | null;
  body: string | null;
  body_format: string | null;
}
const snippetOf = (r: EmailRow): string =>
  r.body
    ? preprocessForEmbedding(r.body, {
        format: r.body_format === 'html' ? 'html' : 'text',
        maxChars: opts.snippetChars,
      })
    : '';

const docFreq = new Map<string, number>();
const catFreq = new Map<string, Map<string, number>>();
const catDocs = new Map<string, number>();
let trainingDocs = 0;
for (const [messageId, arr] of assignments) {
  if (chosen.has(messageId)) continue;
  const cat = arr.find((a) => a.method === 'llm' && labelOf.has(a.categoryId))?.categoryId;
  if (!cat) continue;
  const row = bodyStmt.get(messageId, account.id) as EmailRow | undefined;
  if (!row) continue;
  const ts = tokens(row.subject ?? '', row.from_addr ?? '', snippetOf(row));
  trainingDocs += 1;
  catDocs.set(cat, (catDocs.get(cat) ?? 0) + 1);
  let cf = catFreq.get(cat);
  if (!cf) catFreq.set(cat, (cf = new Map()));
  for (const t of ts) {
    docFreq.set(t, (docFreq.get(t) ?? 0) + 1);
    cf.set(t, (cf.get(t) ?? 0) + 1);
  }
}

const KEYWORDS_PER_CATEGORY = 25;
const MIN_DOC_FREQ = 20;
const MIN_CATEGORY_HITS = 5;
const keywords = new Map<string, string[]>();
for (const [cat, cf] of catFreq) {
  const size = catDocs.get(cat) ?? 0;
  if (size === 0) continue;
  const terms = [...cf.entries()]
    .filter(([t, n]) => (docFreq.get(t) ?? 0) >= MIN_DOC_FREQ && n >= MIN_CATEGORY_HITS)
    .map(([t, n]) => {
      const pInCat = n / size;
      const pOverall = (docFreq.get(t) ?? 1) / trainingDocs;
      return { t, score: Math.log((pInCat + 1e-6) / (pOverall + 1e-6)) };
    })
    .sort((a, b) => b.score - a.score || a.t.localeCompare(b.t))
    .slice(0, KEYWORDS_PER_CATEGORY)
    .map((x) => x.t);
  keywords.set(cat, terms);
}

const majorityCategory =
  [...catDocs.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0]?.[0] ?? null;

function keywordPredict(ts: Set<string>): string | null {
  let best: string | null = null;
  let bestHits = 0;
  for (const [cat, terms] of keywords) {
    let hits = 0;
    for (const t of terms) if (ts.has(t)) hits += 1;
    if (hits > bestHits) {
      bestHits = hits;
      best = cat;
    }
  }
  return bestHits > 0 ? best : majorityCategory;
}

// ---------------------------------------------------------------------------
// Build the rows.
// ---------------------------------------------------------------------------
interface SampleItem {
  n: number;
  messageId: string;
  subject: string;
  from: string;
  date: string;
  folder: string;
  snippet: string;
}
interface KeyItem {
  n: number;
  messageId: string;
  stratum: string;
  delivered: string[];
  deliveredMethods: string[];
  llm: string | null;
  embedNearest: string | null;
  fastGate: string | null;
  keyword: string | null;
  majority: string | null;
  top1: number | null;
  top2: number | null;
}

const metaStmt = db.prepare(
  `SELECT subject, from_addr, date, folder, body, body_format
     FROM emails WHERE message_id = ? AND account_id = ?`,
);

const items: SampleItem[] = [];
const keys: KeyItem[] = [];
let n = 0;
for (const messageId of sampleIds) {
  const row = metaStmt.get(messageId, account.id) as
    | (EmailRow & { date: number | null; folder: string | null })
    | undefined;
  if (!row) continue;
  n += 1;
  const snippet = snippetOf(row);
  items.push({
    n,
    messageId,
    subject: row.subject ?? '(no subject)',
    from: row.from_addr ?? '(unknown sender)',
    date: row.date ? new Date(row.date).toISOString().slice(0, 10) : '',
    folder: row.folder ?? '',
    snippet,
  });
  const s = scored.get(messageId)!;
  const delivered = assignments.get(messageId) ?? [];
  keys.push({
    n,
    messageId,
    stratum: chosen.get(messageId)!,
    delivered: delivered.map((a) => a.categoryId),
    deliveredMethods: delivered.map((a) => a.method ?? 'null'),
    llm: llmPick(messageId),
    embedNearest: s.nearest,
    fastGate: s.gate,
    keyword: keywordPredict(tokens(row.subject ?? '', row.from_addr ?? '', snippet)),
    majority: majorityCategory,
    top1: s.top1,
    top2: s.top2,
  });
}

// ---------------------------------------------------------------------------
// Emit.
// ---------------------------------------------------------------------------
mkdirSync(opts.out, { recursive: true });

const catList = active.map((c) => ({ id: c.id, label: c.label }));

const csvCell = (v: string): string => `"${v.replace(/"/g, '""')}"`;
const sampleCsv = [
  'n,message_id,date,from,subject,snippet,true_category,second_category,notes',
  ...items.map((it) =>
    [
      it.n,
      csvCell(it.messageId),
      it.date,
      csvCell(it.from),
      csvCell(it.subject),
      csvCell(it.snippet.replace(/\s+/g, ' ').slice(0, 500)),
      '',
      '',
      '',
    ].join(','),
  ),
].join('\n');
writeFileSync(join(opts.out, 'sample.csv'), sampleCsv + '\n', 'utf8');

writeFileSync(
  join(opts.out, 'sample-keys.json'),
  JSON.stringify(
    {
      generatedFor: account.address,
      seed: opts.seed,
      embeddingModel: modelId,
      gate: { minConfidence: GATE_MIN_CONFIDENCE, minMargin: GATE_MIN_MARGIN },
      categories: catList,
      keywordBaseline: {
        trainingDocs,
        minDocFreq: MIN_DOC_FREQ,
        termsPerCategory: KEYWORDS_PER_CATEGORY,
        terms: Object.fromEntries([...keywords].map(([k, v]) => [labelOf.get(k) ?? k, v])),
      },
      strata: Object.fromEntries(
        [...new Set(keys.map((k) => k.stratum))].map((s) => [
          s,
          keys.filter((k) => k.stratum === s).length,
        ]),
      ),
      items: keys,
    },
    null,
    2,
  ),
  'utf8',
);

const html = buildLabeller(items, catList.map((c) => c.label), opts.seed);
writeFileSync(join(opts.out, 'labeller.html'), html, 'utf8');

const strataCounts = [...new Set(keys.map((k) => k.stratum))]
  .map((s) => ({ stratum: s, n: keys.filter((k) => k.stratum === s).length }))
  .sort((a, b) => b.n - a.n);

writeFileSync(
  join(opts.out, 'README.md'),
  `# Categorisation ground-truth sample

Generated from a copy of the mailbox database. Seed \`${opts.seed}\`, so this exact sample can be
regenerated and audited.

**This directory contains real message content. Do not commit it.**

## How to label

Open \`labeller.html\` in a browser. It runs entirely locally, saves after every keystroke to the
browser's own storage, and never sends anything anywhere. Press the number keys to pick a category,
\`0\` for none of them, \`u\` for unsure, and the arrow keys to move. Export when done.

\`sample.csv\` holds the same items if you would rather label in a spreadsheet. Fill in
\`true_category\` using the exact category name, or the literal \`NONE\` when nothing fits, or \`UNSURE\`
when you cannot tell. A blank means not yet labelled, so it is skipped rather than counted.

## What you must not do

Do not open \`sample-keys.json\` until labelling is finished. It holds what every arm predicted. The
measurement depends on the labels being made without seeing them.

## When labelling is finished

Export \`labels.csv\` from the page, then:

\`\`\`
npx tsx packages/core/scripts/score-labels.ts --labels labels.csv --keys sample-keys.json
\`\`\`

## Strata

${strataCounts.map((s) => `- \`${s.stratum}\`: ${s.n}`).join('\n')}

Only the \`uniform\` stratum is a random draw from the mailbox, so it is the only one that may be
pooled into an overall accuracy figure. The per-category quotas exist to catch a category that is
badly broken. The disagreement stratum is deliberately over-sampled from cases where the fast gate
and the LLM pass chose differently, because those labels decide which pass to believe.

## Arms frozen at ${new Date().toISOString().slice(0, 10)}

| Arm | What it is |
|---|---|
| majority | Always predict the largest category. The floor any method must beat. |
| keyword | Top ${KEYWORDS_PER_CATEGORY} distinctive terms per category plus sender domain, derived from ${trainingDocs} labelled messages that are not in this sample. |
| embedNearest | Nearest category centroid, no gate, always answers. |
| fastGate | The shipped gateFastAssignment, confidence ${GATE_MIN_CONFIDENCE} and margin ${GATE_MIN_MARGIN}. Abstains rather than guessing. |
| llm | The LLM decision pass's recorded choice. |
| delivered | What the system actually stored, whichever pass produced it. |

All six were computed and written before any label existed, so none of them can be tuned to the
answers.
`,
  'utf8',
);

console.log(`\nsampled ${items.length} messages`);
console.table(strataCounts);
console.log(`keyword baseline trained on ${trainingDocs} messages outside the sample`);
console.log(`\nwrote ${join(opts.out, 'labeller.html')}`);
console.log(`wrote ${join(opts.out, 'sample.csv')}`);
console.log(`wrote ${join(opts.out, 'sample-keys.json')}  (do not open until labelling is done)`);
console.log(`wrote ${join(opts.out, 'README.md')}`);

db.close();

/** Self-contained local labelling page. No network, no external assets, no predictions shown. */
function buildLabeller(
  data: SampleItem[],
  cats: Array<{ id: string; label: string }>,
  seed: string,
): string {
  const payload = JSON.stringify({ seed, items: data, categories: cats }).replace(/</g, '\\u003c');
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Categorisation labelling</title>
<style>
  :root { --bg:#fbfbfa; --fg:#1a1a18; --muted:#6b6b66; --line:#e2e2dd; --accent:#2b5f8a; --card:#fff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --fg:#e8e8e4; --muted:#9a9a94; --line:#2e2e34; --accent:#7fb2d9; --card:#1e1e24; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
  header { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 20px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; z-index:5; }
  h1 { font-size:15px; font-weight:600; margin:0; }
  .progress { flex:1; min-width:180px; height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
  .progress > i { display:block; height:100%; background:var(--accent); width:0; transition:width .15s; }
  .count { color:var(--muted); font-variant-numeric:tabular-nums; font-size:13px; }
  button { font:inherit; padding:6px 12px; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:6px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  main { max-width:860px; margin:0 auto; padding:20px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px 20px; }
  .meta { color:var(--muted); font-size:13px; margin-bottom:2px; word-break:break-word; }
  .subject { font-size:18px; font-weight:600; margin:8px 0 14px; word-break:break-word; }
  .snippet { white-space:pre-wrap; font-size:14px; max-height:340px; overflow:auto; padding:12px; background:var(--bg); border:1px solid var(--line); border-radius:6px; word-break:break-word; }
  .cats { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:6px; margin-top:16px; }
  .cats label { display:flex; align-items:center; gap:8px; padding:7px 10px; border:1px solid var(--line); border-radius:6px; cursor:pointer; font-size:14px; }
  .cats label:hover { border-color:var(--accent); }
  .cats label.on { border-color:var(--accent); background:color-mix(in srgb, var(--accent) 12%, transparent); }
  .cats kbd { font:12px ui-monospace, monospace; color:var(--muted); border:1px solid var(--line); border-radius:4px; padding:0 5px; }
  .row { display:flex; gap:10px; margin-top:16px; align-items:center; flex-wrap:wrap; }
  input[type=text] { flex:1; min-width:200px; padding:7px 10px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); font:inherit; }
  .hint { color:var(--muted); font-size:12.5px; margin-top:14px; }
  .done { text-align:center; padding:60px 20px; }
</style>
</head>
<body>
<header>
  <h1>Categorisation labelling</h1>
  <div class="progress"><i id="bar"></i></div>
  <span class="count" id="count"></span>
  <button id="prev">&larr; Prev</button>
  <button id="next">Next &rarr;</button>
  <button id="jump">First unlabelled</button>
  <button id="export">Export CSV</button>
</header>
<main id="main"></main>
<script>
const DATA = ${payload};
const KEY = 'mailpilot-labels-' + DATA.seed;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { labels = {}; }
let idx = 0;

function save() {
  try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {}
}
function labelled() { return DATA.items.filter(it => labels[it.messageId] && labels[it.messageId].primary).length; }

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function render() {
  const it = DATA.items[idx];
  const main = document.getElementById('main');
  if (!it) { main.innerHTML = '<div class="done">Nothing to show.</div>'; return; }
  const cur = labels[it.messageId] || {};
  const opts = DATA.categories.map((c, i) => {
    const k = i + 1;
    const on = cur.primary === c ? ' on' : '';
    return '<label class="' + on.trim() + '" data-pick="' + esc(c) + '">' +
      (k <= 9 ? '<kbd>' + k + '</kbd>' : '<kbd>&nbsp;</kbd>') + '<span>' + esc(c) + '</span></label>';
  }).join('');
  main.innerHTML =
    '<div class="card">' +
      '<div class="meta">' + esc(it.date) + ' &middot; ' + esc(it.from) + '</div>' +
      '<div class="meta">' + esc(it.folder) + '</div>' +
      '<div class="subject">' + esc(it.subject) + '</div>' +
      '<div class="snippet">' + esc(it.snippet || '(no body stored)') + '</div>' +
      '<div class="cats">' + opts +
        '<label class="' + (cur.primary === 'NONE' ? 'on' : '') + '" data-pick="NONE"><kbd>0</kbd><span>None of these</span></label>' +
        '<label class="' + (cur.primary === 'UNSURE' ? 'on' : '') + '" data-pick="UNSURE"><kbd>u</kbd><span>Unsure</span></label>' +
      '</div>' +
      '<div class="row">' +
        '<input type="text" id="second" placeholder="Second category, only if genuinely both" value="' + esc(cur.second || '') + '">' +
        '<input type="text" id="notes" placeholder="Notes, optional" value="' + esc(cur.notes || '') + '">' +
      '</div>' +
      '<div class="hint">Number keys pick a category. <kbd>0</kbd> none, <kbd>u</kbd> unsure, arrow keys move, ' +
        'picking a category moves on automatically. Progress is saved in this browser only.</div>' +
    '</div>';

  main.querySelectorAll('[data-pick]').forEach(el => {
    el.addEventListener('click', () => pick(el.getAttribute('data-pick')));
  });
  ['second','notes'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => {
      const rec = labels[it.messageId] || (labels[it.messageId] = {});
      rec[id] = el.value;
      save();
    });
  });
  document.getElementById('count').textContent = (idx + 1) + ' / ' + DATA.items.length + '  (' + labelled() + ' done)';
  document.getElementById('bar').style.width = (labelled() / DATA.items.length * 100) + '%';
}

function pick(value) {
  const it = DATA.items[idx];
  const rec = labels[it.messageId] || (labels[it.messageId] = {});
  rec.primary = value;
  save();
  if (idx < DATA.items.length - 1) { idx++; }
  render();
}

function move(d) { idx = Math.max(0, Math.min(DATA.items.length - 1, idx + d)); render(); }

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowRight') { move(1); e.preventDefault(); }
  else if (e.key === 'ArrowLeft') { move(-1); e.preventDefault(); }
  else if (e.key === '0') { pick('NONE'); }
  else if (e.key.toLowerCase() === 'u') { pick('UNSURE'); }
  else if (/^[1-9]$/.test(e.key)) {
    const c = DATA.categories[Number(e.key) - 1];
    if (c) pick(c);
  }
});

document.getElementById('prev').onclick = () => move(-1);
document.getElementById('next').onclick = () => move(1);
document.getElementById('jump').onclick = () => {
  const i = DATA.items.findIndex(it => !(labels[it.messageId] && labels[it.messageId].primary));
  idx = i === -1 ? DATA.items.length - 1 : i;
  render();
};
document.getElementById('export').onclick = () => {
  const q = s => '"' + String(s == null ? '' : s).replace(/"/g, '""') + '"';
  const lines = ['n,message_id,true_category,second_category,notes'];
  DATA.items.forEach(it => {
    const r = labels[it.messageId] || {};
    lines.push([it.n, q(it.messageId), q(r.primary || ''), q(r.second || ''), q(r.notes || '')].join(','));
  });
  const blob = new Blob([lines.join('\\n') + '\\n'], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'labels.csv';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
};

const first = DATA.items.findIndex(it => !(labels[it.messageId] && labels[it.messageId].primary));
idx = first === -1 ? 0 : first;
render();
</script>
</body>
</html>
`;
}
