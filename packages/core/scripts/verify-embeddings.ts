/**
 * Diagnostic script: inspect embedding quality for the synced emails.
 *
 * Runs:
 *   - Sanity: vector dimensions, magnitudes, all-zero check
 *   - Self-similarity: each email vs itself should be distance ~0
 *   - Sample KNN: pick N random emails, show top-3 nearest neighbors
 *   - Query tests: embed natural language queries and show closest matches
 *   - Preprocessing inspection: what text actually went to the model for a sample
 *
 * Usage: npx tsx packages/core/scripts/verify-embeddings.ts [accountId]
 */
import { openDatabase } from '../src/db/database.js';
import { loadConfig } from '../src/config/config.js';
import { createLlmClient } from '../src/llm/client.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { EmbeddingRepository } from '../src/repositories/embedding-repository.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { buildEmbeddingInput, preprocessForEmbedding } from '../src/util/text.js';

const db = openDatabase();
const config = loadConfig();
const llm = createLlmClient(() => config.llm);
const accounts = new AccountRepository(db);
const emails = new EmailRepository(db);
const embeddings = new EmbeddingRepository(db);

const modelId = config.llm.embeddingModel;

const accountId = process.argv[2] ?? accounts.list()[0]?.id;
if (!accountId) {
  console.error('No account found. Run a sync first or pass an account ID as argument.');
  process.exit(1);
}

const account = accounts.findById(accountId);
if (!account) {
  console.error(`Account ${accountId} not found.`);
  process.exit(1);
}

console.log(`Account: ${account.address} (${account.kind})`);
console.log(`Model:   ${modelId}`);
console.log();

// ─── 1. Vector sanity ────────────────────────────────────────────────────

console.log('=== 1. Vector sanity ===');
const indexed = db
  .prepare<[string, string], { message_id: string }>(
    'SELECT message_id FROM email_embedding_index WHERE account_id = ? AND model_id = ?',
  )
  .all(accountId, modelId);

console.log(`Total embeddings: ${indexed.length}`);

const sampleIds = indexed.slice(0, Math.min(50, indexed.length)).map((r) => r.message_id);
let nonZero = 0;
let dimMismatch = 0;
const magnitudes: number[] = [];

for (const messageId of sampleIds) {
  const vec = embeddings.getEmbedding({ messageId, accountId, modelId });
  if (!vec) continue;
  if (vec.length !== 1024) dimMismatch++;
  const sum = vec.reduce((a, b) => a + b * b, 0);
  const mag = Math.sqrt(sum);
  magnitudes.push(mag);
  if (mag > 1e-6) nonZero++;
}

magnitudes.sort((a, b) => a - b);
const min = magnitudes[0]!;
const max = magnitudes[magnitudes.length - 1]!;
const median = magnitudes[Math.floor(magnitudes.length / 2)]!;
console.log(`Dimension mismatches: ${dimMismatch}`);
console.log(`Non-zero vectors:    ${nonZero}/${magnitudes.length}`);
console.log(`Magnitude min:       ${min.toFixed(4)}`);
console.log(`Magnitude median:    ${median.toFixed(4)}`);
console.log(`Magnitude max:       ${max.toFixed(4)}`);
console.log();

// ─── 2. Self-similarity ──────────────────────────────────────────────────

console.log('=== 2. Self-similarity check ===');
console.log('Each email should be closest to itself with distance ~0\n');

const checks = sampleIds.slice(0, 5);
for (const messageId of checks) {
  const vec = embeddings.getEmbedding({ messageId, accountId, modelId });
  if (!vec) continue;
  const hits = embeddings.search(accountId, modelId, vec, 1);
  const email = emails.findById(messageId, accountId);
  const subj = (email?.subject ?? '').slice(0, 50);
  const top = hits[0];
  const ok = top?.messageId === messageId && (top?.distance ?? 1) < 0.01;
  console.log(
    `${ok ? 'OK ' : 'FAIL'} "${subj}..." -> top1=${top?.messageId.slice(0, 15)} dist=${top?.distance?.toFixed(4)}`,
  );
}
console.log();

// ─── 3. Sample KNN ───────────────────────────────────────────────────────

console.log('=== 3. Sample KNN (3 random emails, top 3 neighbors each) ===\n');

const allEmails = emails.list({ accountId, limit: 500 });
const shuffled = [...allEmails].sort(() => Math.random() - 0.5);
const samples = shuffled.slice(0, 3);

for (const sample of samples) {
  const vec = embeddings.getEmbedding({
    messageId: sample.messageId,
    accountId,
    modelId,
  });
  if (!vec) continue;

  console.log(`Query: "${(sample.subject ?? '').slice(0, 70)}"`);
  console.log(`  from: ${sample.fromAddr}`);

  const hits = embeddings.search(accountId, modelId, vec, 4);
  for (const hit of hits) {
    if (hit.messageId === sample.messageId) continue;
    const e = emails.findById(hit.messageId, accountId);
    console.log(
      `   d=${hit.distance.toFixed(3)}  "${(e?.subject ?? '').slice(0, 55)}"  <- ${(e?.fromAddr ?? '').slice(0, 35)}`,
    );
  }
  console.log();
}

// ─── 4. Natural language queries ────────────────────────────────────────

console.log('=== 4. Natural language query tests ===\n');

const queries = [
  'urgent message that needs my reply',
  'student application or admission',
  'invoice or payment receipt',
  'meeting or calendar invitation',
  'newsletter or promotional email',
];

for (const q of queries) {
  const qVec = await llm.embed(q, modelId);
  const hits = embeddings.search(accountId, modelId, qVec, 3);
  console.log(`Query: "${q}"`);
  for (const hit of hits) {
    const e = emails.findById(hit.messageId, accountId);
    console.log(
      `   d=${hit.distance.toFixed(3)}  "${(e?.subject ?? '').slice(0, 55)}"  <- ${(e?.fromAddr ?? '').slice(0, 35)}`,
    );
  }
  console.log();
}

// ─── 5. Preprocessing inspection ─────────────────────────────────────────

console.log('=== 5. Preprocessing sample (what we actually embed) ===\n');

const inspectSample = samples[0];
if (inspectSample) {
  console.log(`Subject:  ${inspectSample.subject}`);
  console.log(`From:     ${inspectSample.fromAddr}`);
  console.log(`Format:   ${inspectSample.bodyFormat}`);
  console.log(`Raw body length:  ${inspectSample.body?.length ?? 0} chars`);
  if (inspectSample.body) {
    const preprocessed = preprocessForEmbedding(inspectSample.body, {
      format: inspectSample.bodyFormat,
    });
    console.log(`Preprocessed:     ${preprocessed.length} chars`);
    const full = buildEmbeddingInput({
      subject: inspectSample.subject,
      fromAddr: inspectSample.fromAddr,
      body: inspectSample.body,
      bodyFormat: inspectSample.bodyFormat,
    });
    console.log(`---`);
    console.log(full.slice(0, 600));
    if (full.length > 600) console.log(`... [truncated at 600 of ${full.length} chars]`);
  }
}

db.close();
