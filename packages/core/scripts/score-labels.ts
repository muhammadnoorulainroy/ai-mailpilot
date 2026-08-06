/**
 * Evaluation instrument: scores every arm against the hand labels.
 *
 * Reads the labels exported from labeller.html and the key file frozen by build-label-sample.ts, and
 * reports the ablation. It refuses to pool the stratified samples into an overall figure, because only
 * the uniform stratum is a random draw; quota strata would bias any pooled number toward small
 * categories and the disagreement stratum is deliberately adversarial.
 *
 * Label conventions:
 *   a category name  the message belongs there
 *   NONE             no category fits. An arm that named one is wrong; an arm that abstained is right.
 *   UNSURE           excluded from every accuracy figure and reported separately.
 *
 * Confidence intervals are Wilson score intervals at 95%, which behave sensibly at these sample sizes
 * and near 0 and 1, where the normal approximation does not.
 *
 * Usage:
 *   npx tsx packages/core/scripts/score-labels.ts --labels <labels.csv> --keys <sample-keys.json>
 */
import { readFileSync } from 'node:fs';

interface Options {
  labels: string | null;
  keys: string | null;
}

function parseArgs(argv: string[]): Options {
  const o: Options = { labels: null, keys: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]!;
    if (a === '--labels') o.labels = argv[++i] ?? null;
    else if (a === '--keys') o.keys = argv[++i] ?? null;
    else {
      console.error(`Unknown argument: ${a}`);
      process.exit(2);
    }
  }
  if (!o.labels || !o.keys) {
    console.error('--labels and --keys are both required.');
    process.exit(2);
  }
  return o;
}

const opts = parseArgs(process.argv.slice(2));

/** Minimal RFC4180 reader: handles quoted fields, doubled quotes and embedded newlines. */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i]!;
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else quoted = false;
      } else field += c;
      continue;
    }
    if (c === '"') quoted = true;
    else if (c === ',') {
      row.push(field);
      field = '';
    } else if (c === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else if (c !== '\r') field += c;
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((v) => v.trim() !== ''));
}

interface KeyItem {
  n: number;
  messageId: string;
  stratum: string;
  delivered: string[];
  llm: string | null;
  embedNearest: string | null;
  fastGate: string | null;
  keyword: string | null;
  majority: string | null;
}
interface Keys {
  categories: Array<{ id: string; label: string }>;
  strata: Record<string, number>;
  items: KeyItem[];
}

const keys = JSON.parse(readFileSync(opts.keys!, 'utf8')) as Keys;
const idOf = new Map(keys.categories.map((c) => [c.label.toLowerCase(), c.id]));
const keyByMessage = new Map(keys.items.map((k) => [k.messageId, k]));

const rows = parseCsv(readFileSync(opts.labels!, 'utf8'));
const header = rows[0]!.map((h) => h.trim().toLowerCase());
const col = (name: string): number => header.indexOf(name);
const cMessage = col('message_id');
const cTrue = col('true_category');
if (cMessage === -1 || cTrue === -1) {
  console.error(`labels file must have message_id and true_category columns, found: ${header}`);
  process.exit(2);
}

type Truth = { kind: 'category'; id: string } | { kind: 'none' } | { kind: 'unsure' };

const truth = new Map<string, Truth>();
const unrecognised = new Map<string, number>();
for (const r of rows.slice(1)) {
  const messageId = r[cMessage]?.trim();
  const raw = (r[cTrue] ?? '').trim();
  if (!messageId || raw === '') continue;
  const upper = raw.toUpperCase();
  if (upper === 'UNSURE') truth.set(messageId, { kind: 'unsure' });
  else if (upper === 'NONE') truth.set(messageId, { kind: 'none' });
  else {
    const id = idOf.get(raw.toLowerCase());
    if (id) truth.set(messageId, { kind: 'category', id });
    else unrecognised.set(raw, (unrecognised.get(raw) ?? 0) + 1);
  }
}

if (unrecognised.size > 0) {
  console.log('=== labels that match no category, ignored ===');
  console.table([...unrecognised].map(([label, n]) => ({ label, n })));
}

/** Wilson score interval at 95%, returned as percentage points. */
function wilson(hits: number, n: number): { lo: number; hi: number } {
  if (n === 0) return { lo: 0, hi: 0 };
  const z = 1.959963985;
  const p = hits / n;
  const d = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / d;
  const half = ((z / d) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) as number;
  return { lo: Math.max(0, (centre - half) * 100), hi: Math.min(100, (centre + half) * 100) };
}

const ARMS = ['majority', 'keyword', 'embedNearest', 'fastGate', 'llm', 'delivered'] as const;
type Arm = (typeof ARMS)[number];

/** What an arm predicted: a category id, or null meaning it abstained. */
function predict(k: KeyItem, arm: Arm): string | null {
  if (arm === 'delivered') return k.delivered[0] ?? null;
  return k[arm];
}
/** Full predicted set, so a multi-label arm can also be scored generously. */
function predictSet(k: KeyItem, arm: Arm): string[] {
  if (arm === 'delivered') return k.delivered;
  const p = k[arm];
  return p ? [p] : [];
}

interface Score {
  /** Labelled and not unsure. */
  scored: number;
  /** The arm named a category. */
  answered: number;
  /** The arm declined to name one. */
  abstained: number;
  /** Right, counting a correct abstention on a none-of-these message. */
  correct: number;
  /** Right among the messages where it did name a category. */
  correctAnswered: number;
  /** As correct, but a multi-label arm only has to contain the true category. */
  correctInSet: number;
}

function scoreArm(items: KeyItem[], arm: Arm): Score {
  const s: Score = {
    scored: 0,
    answered: 0,
    abstained: 0,
    correct: 0,
    correctAnswered: 0,
    correctInSet: 0,
  };
  for (const k of items) {
    const t = truth.get(k.messageId);
    if (!t || t.kind === 'unsure') continue;
    s.scored += 1;
    const p = predict(k, arm);
    const set = predictSet(k, arm);
    if (p === null) {
      s.abstained += 1;
      // Abstaining is the right answer when no category fits, and a miss when one does.
      if (t.kind === 'none') {
        s.correct += 1;
        s.correctInSet += 1;
      }
      continue;
    }
    s.answered += 1;
    if (t.kind === 'none') continue;
    if (p === t.id) {
      s.correct += 1;
      s.correctAnswered += 1;
    }
    if (set.includes(t.id)) s.correctInSet += 1;
  }
  return s;
}

const pct = (n: number, d: number): string => (d === 0 ? 'n/a' : ((n / d) * 100).toFixed(1));

function report(title: string, items: KeyItem[], note: string): void {
  const withTruth = items.filter((k) => truth.has(k.messageId));
  const unsure = withTruth.filter((k) => truth.get(k.messageId)!.kind === 'unsure').length;
  const none = withTruth.filter((k) => truth.get(k.messageId)!.kind === 'none').length;
  console.log(`\n=== ${title} ===`);
  console.log(note);
  console.log(
    `sampled ${items.length}, labelled ${withTruth.length}, of which unsure ${unsure} and none-of-these ${none}`,
  );
  if (withTruth.length - unsure === 0) {
    console.log('nothing scoreable yet.');
    return;
  }
  const table = ARMS.map((arm) => {
    const s = scoreArm(withTruth, arm);
    const ci = wilson(s.correct, s.scored);
    return {
      arm,
      scored: s.scored,
      answered: s.answered,
      abstained: s.abstained,
      'accuracy %': pct(s.correct, s.scored),
      '95% CI': `${ci.lo.toFixed(1)} to ${ci.hi.toFixed(1)}`,
      'precision when answering %': pct(s.correctAnswered, s.answered),
      'in-set %': pct(s.correctInSet, s.scored),
    };
  });
  console.table(table);
}

const uniform = keys.items.filter((k) => k.stratum === 'uniform');
const disagreement = keys.items.filter((k) => k.stratum === 'disagreement');
const quota = keys.items.filter((k) => k.stratum.startsWith('category:'));

report(
  'population accuracy',
  uniform,
  'Uniform random draw. This is the only stratum that estimates accuracy on the mailbox as a whole.',
);

report(
  'the adversarial stratum',
  disagreement,
  'Messages where the fast gate and the LLM pass chose differently. Not a population estimate.\n' +
    'Read it as: when the two passes disagree, which one should be believed.',
);

// Per-category precision, from the quota strata, where truth is compared against what was delivered.
console.log(`\n=== per-category precision, from the quota strata ===`);
console.log(
  'Sampled from what the system assigned, so this is precision, not recall. Twelve per category is\n' +
    'too thin for a tight estimate; it is a screen for a category that is badly wrong.',
);
const catRows: Array<Record<string, string | number>> = [];
for (const category of keys.categories) {
  const items = quota.filter((k) => k.stratum === `category:${category.label}`);
  const withTruth = items.filter((k) => {
    const t = truth.get(k.messageId);
    return t && t.kind !== 'unsure';
  });
  if (withTruth.length === 0) continue;
  const hits = withTruth.filter((k) => {
    const t = truth.get(k.messageId)!;
    return t.kind === 'category' && t.id === category.id;
  }).length;
  const ci = wilson(hits, withTruth.length);
  catRows.push({
    category: category.label,
    labelled: withTruth.length,
    correct: hits,
    'precision %': pct(hits, withTruth.length),
    '95% CI': `${ci.lo.toFixed(0)} to ${ci.hi.toFixed(0)}`,
  });
}
if (catRows.length > 0) console.table(catRows);
else console.log('no quota labels yet.');

// Overall progress, so the labeller knows how far along they are.
const total = keys.items.length;
const done = keys.items.filter((k) => truth.has(k.messageId)).length;
console.log(`\nlabelled ${done} of ${total} sampled messages (${pct(done, total)}%)`);
console.log(
  `\nReminder for the report: every arm was frozen before labelling began, and only the uniform\n` +
    `stratum carries a population estimate. n is one mailbox.`,
);