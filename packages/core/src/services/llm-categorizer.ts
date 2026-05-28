/**
 * LLM-based email categorizer. Given each email and a short list of candidate categories ranked by
 * embedding similarity, the model picks which candidates fit, and the batched JSON reply is parsed
 * back into validated category IDs.
 */
import type { LlmClient } from '../llm/client.js';
import { parseLlmJson, stripThink } from '../util/json-llm.js';
import { preprocessForEmbedding } from '../util/text.js';

/** A category an email may be filed into. */
export interface CategoryCandidate {
  id: string;
  label: string;
  description: string | null;
}

/** The email fields used to categorize it. */
export interface EmailForCategorization {
  subject: string | null;
  fromAddr: string | null;
  body: string | null;
  bodyFormat: 'text' | 'html';
}

/**
 * One email plus the shortlist of categories it may be filed into, ranked by embedding
 * similarity. The LLM is the final judge, but only over this shortlist.
 */
export interface DecisionInput {
  email: EmailForCategorization;
  candidates: CategoryCandidate[];
}

/** A past user categorization, shown to the LLM as a few-shot style example. */
export interface CorrectionExample {
  subject: string | null;
  labels: string[];
}

const BODY_SNIPPET_CHARS = 900;
const TOKENS_PER_EMAIL = 40;
const TIMEOUT_MS =
  Number.parseInt(process.env.MAILPILOT_LLM_TIMEOUT_MS ?? '', 10) > 0
    ? Number.parseInt(process.env.MAILPILOT_LLM_TIMEOUT_MS ?? '', 10)
    : 180_000;

const SYSTEM_PROMPT = `You file emails into the user's existing categories by reading what each is actually about.

Rules:
- Each email lists its OWN candidate categories. Choose ONLY from that email's list. Never invent a category or use a label not in its list.
- Judge by the email's PURPOSE and CONTENT, not just surface keywords or the sender. A payment receipt or invoice is a purchase/financial email, not "education" or "support". A grade notice is academic. A login or verification code is security.
- Purpose boundaries (decide by what the email DOES, not who sent it):
  - A promotion, discount, deal, coupon, offer, or marketing newsletter is MARKETING, even from a delivery, travel, ride, food, or bank brand. A "deal" or "offer" is NOT a shipment, trip, or transaction.
  - SHIPPING/DELIVERIES is only an actual order status: shipped, out for delivery, tracking, delivered.
  - TRAVEL is only an actual booking/reservation: flight, hotel, itinerary, check-in. A travel-brand promo is marketing.
  - A bank's promotional email is MARKETING, not a banking transaction; banking is real transfers, statements, and card activity.
- Most emails belong to exactly ONE category. Add a second only if it genuinely fits a DIFFERENT second purpose.
- If none of that email's candidates fit, return an empty list for that email.`;

/** Produce a trimmed plain-text snippet of the email body for the prompt. */
function buildSnippet(email: EmailForCategorization): string {
  return email.body
    ? preprocessForEmbedding(email.body, { format: email.bodyFormat, maxChars: BODY_SNIPPET_CHARS })
    : '';
}

/** Render the candidate categories as a bulleted label and description list for the prompt. */
function buildCategoryList(candidates: CategoryCandidate[]): string {
  return candidates
    .map((c) => `- ${c.label}${c.description ? `: ${c.description}` : ''}`)
    .join('\n');
}

/** Build a lookup from a normalized lowercase label to its category ID. */
function labelMap(candidates: CategoryCandidate[]): Map<string, string> {
  return new Map(candidates.map((c) => [c.label.trim().toLowerCase(), c.id]));
}

/** Map model-returned labels to deduplicated category IDs, dropping anything not in byLabel. */
function labelsToIds(labels: unknown[], byLabel: Map<string, string>): string[] {
  const ids: string[] = [];
  for (const label of labels) {
    if (typeof label !== 'string') continue;
    const id = byLabel.get(label.trim().toLowerCase());
    if (id && !ids.includes(id)) ids.push(id);
  }
  return ids;
}

/** Narrow a value to a plain object record, returning null for arrays and non-objects. */
function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** Read a numeric position the model may have attached to an answer object via index, id, or email. */
function explicitIndex(o: Record<string, unknown>): number | undefined {
  return [o.index, o.id, o.email].find((x) => typeof x === 'number') as number | undefined;
}

/**
 * Pull per-email answers out of one container shape into a 1-based index map. Handles arrays
 * (positional or carrying their own index) and objects keyed by number string, detecting whether an
 * array's explicit indices are zero based so they line up with the expected count.
 */
function answersFromContainer(obj: unknown, count: number): Map<number, unknown> {
  const map = new Map<number, unknown>();
  if (Array.isArray(obj)) {
    const explicit = obj.map((item) => {
      const o = asRecord(item);
      return o ? explicitIndex(o) : undefined;
    });
    const zeroBased =
      explicit.some((x) => x === 0) &&
      explicit.every((x) => x === undefined || (x >= 0 && x < count));
    obj.forEach((item, idx) => {
      const o = asRecord(item);
      let key = idx + 1;
      let value: unknown = item;
      if (o && !Array.isArray(item)) {
        const ex = explicit[idx];
        key =
          ex === undefined ? idx + 1 : zeroBased ? ex + 1 : ex >= 1 && ex <= count ? ex : idx + 1;
        value = o.categories ?? o.labels ?? o.category ?? o.label;
      }
      if ((Array.isArray(item) || typeof item === 'string' || o) && !map.has(key))
        map.set(key, value);
    });
  } else {
    const rec = asRecord(obj);
    if (rec)
      for (let i = 1; i <= count; i++) if (rec[String(i)] !== undefined) map.set(i, rec[String(i)]);
  }
  return map;
}

/** Unwrap a common emails or results envelope and extract the indexed answers, falling back to the wrapper itself. */
function extractIndexedAnswers(parsed: unknown, count: number): Map<number, unknown> {
  const wrapper = asRecord(parsed);
  let container: unknown = parsed;
  if (wrapper) {
    if (asRecord(wrapper.emails) || Array.isArray(wrapper.emails)) container = wrapper.emails;
    else if (asRecord(wrapper.results) || Array.isArray(wrapper.results))
      container = wrapper.results;
  }
  let map = answersFromContainer(container, count);
  if (map.size === 0 && wrapper && container !== wrapper)
    map = answersFromContainer(wrapper, count);
  return map;
}

/** Coerce a single answer entry into a label array, treating a blank string as an empty list and rejecting other shapes with null. */
function toLabelArray(entry: unknown): unknown[] | null {
  if (Array.isArray(entry)) return entry;
  if (typeof entry === 'string') return entry.trim() === '' ? [] : [entry];
  return null;
}

/**
 * Parse a batched reply into one result per email, aligned by 1-based index. Each result is a list
 * of valid category IDs, [] when the model validly returned no labels, or null when there was no
 * usable answer. The caller must RETRY a null, never record it as a real "no category fits". Tolerant
 * of common output shapes, but still validates every label against that email's OWN candidate list.
 */
export function resolveBatchLabels(
  raw: string,
  count: number,
  candidates: CategoryCandidate[] | CategoryCandidate[][],
): (string[] | null)[] {
  const out: (string[] | null)[] = Array.from({ length: count }, () => null);
  let parsed: unknown;
  try {
    parsed = parseLlmJson(stripThink(raw));
  } catch {
    return out;
  }
  if (!parsed || typeof parsed !== 'object') return out;

  const answers = extractIndexedAnswers(parsed, count);
  const perEmail = Array.isArray(candidates[0]);
  const sharedMap = perEmail ? null : labelMap(candidates as CategoryCandidate[]);
  for (let i = 0; i < count; i++) {
    const entry = answers.get(i + 1);
    if (entry === undefined) continue;
    const labels = toLabelArray(entry);
    if (labels === null) continue;
    if (labels.length === 0) {
      out[i] = [];
      continue;
    }
    const byLabel = perEmail
      ? labelMap((candidates as CategoryCandidate[][])[i] ?? [])
      : sharedMap!;
    const ids = labelsToIds(labels, byLabel);
    out[i] = ids.length > 0 ? ids : null;
  }
  return out;
}

/** Uses an LLM to file emails into their candidate categories. */
export class LlmCategorizer {
  /** Store the LLM client used for every categorization call. */
  constructor(private llm: LlmClient) {}

  /**
   * Decide categories for one email from its shortlist. Returns chosen category IDs, best-fit
   * first, an empty array means none of these fit.
   */
  async decide(
    email: EmailForCategorization,
    candidates: CategoryCandidate[],
    modelId: string,
    provider: 'main' | 'chat' = 'main',
  ): Promise<string[]> {
    const [ids] = await this.decideBatch([{ email, candidates }], [], modelId, provider);
    return ids ?? [];
  }

  /**
   * Decide categories for several emails in a single LLM call. Each email carries its own shortlist
   * so the model judges only the plausible few. Optional examples are past user categorizations
   * shown as few-shot style guidance. Returns one ID list per input entry, in order.
   */
  async decideBatch(
    entries: DecisionInput[],
    examples: CorrectionExample[],
    modelId: string,
    provider: 'main' | 'chat' = 'main',
  ): Promise<(string[] | null)[]> {
    if (entries.length === 0) return [];

    const blocks = entries
      .map((entry, i) => {
        const snippet = buildSnippet(entry.email);
        const catList = buildCategoryList(entry.candidates) || '- (no candidates)';
        return (
          `[${i + 1}]\n` +
          `Candidate categories for this email:\n${catList}\n` +
          `Subject: ${entry.email.subject ?? '(none)'}\n` +
          `From: ${entry.email.fromAddr ?? '(unknown)'}\n` +
          `Body: ${snippet || '(empty)'}`
        );
      })
      .join('\n\n');

    const examplesBlock =
      examples.length > 0
        ? `Examples of how the user files their own mail (match this style):\n` +
          examples
            .map(
              (e) => `- "${(e.subject ?? '').slice(0, 80)}" -> ${e.labels.join(', ') || '(none)'}`,
            )
            .join('\n') +
          `\n\n`
        : '';

    const userPrompt =
      examplesBlock +
      `EMAILS (${entries.length}). Each lists the ONLY categories it may use:\n${blocks}\n\n` +
      `For EACH numbered email choose from ITS OWN candidate list only. Output ONLY JSON mapping ` +
      `each number to its labels, with an entry for every number, e.g. {"1": ["Label"], "2": []}.`;

    const system = provider === 'main' ? `/no_think\n${SYSTEM_PROMPT}` : SYSTEM_PROMPT;
    const raw = await this.llm.chat({
      model: modelId,
      provider,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: userPrompt },
      ],
      responseFormat: 'json_object',
      temperature: 0,
      maxTokens: TOKENS_PER_EMAIL * entries.length + 80,
      timeoutMs: TIMEOUT_MS,
      think: provider === 'main' ? false : undefined,
    });

    return resolveBatchLabels(
      raw,
      entries.length,
      entries.map((e) => e.candidates),
    );
  }
}
