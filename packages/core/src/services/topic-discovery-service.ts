/**
 * Topic discovery for an account's inbox. Samples email subjects, asks the LLM for a purpose-based
 * taxonomy, then embeds, merges, and reconciles the result into auto categories with centroids.
 */
import { z } from 'zod';
import type { Logger } from 'pino';
import { LlmApiError, type LlmClient } from '../llm/client.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmailRepository, EmailSummary } from '../repositories/email-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import { parseLlmJson, stripCodeFence } from '../util/json-llm.js';
import { cosineFromL2Distance, l2Distance, meanNormalize } from '../util/vector.js';
import { purposeSignature } from './categorize-strategy.js';
import type { AccountRepository } from '../repositories/account-repository.js';
import type { DiscoveryAuditRepository } from '../repositories/discovery-audit-repository.js';
import type { LlmConfig } from '../config/schema.js';
import { assertDiscoveryLocal, discoveryProvider } from './discovery-guard.js';
import { stableHash, seededShuffle } from '../util/rand.js';

const TWIN_COSINE_STRONG = 0.93;
const TWIN_COSINE_SAME_PURPOSE = 0.9;

/**
 * True when two category labels name the same purpose, by label-token overlap or label-embedding
 * cosine. Categories sharing a known purpose signature merge at a lower cosine, distinct known
 * purposes never merge.
 */
export function areTwins(
  labelA: string,
  vecA: Float32Array,
  labelB: string,
  vecB: Float32Array,
): boolean {
  if (isNearDuplicateLabel(labelA, [labelB])) return true;
  const sigA = purposeSignature(labelA);
  const sigB = purposeSignature(labelB);
  if (sigA !== null && sigB !== null && sigA !== sigB) return false;
  const cos = cosineFromL2Distance(l2Distance(vecA, vecB));
  const threshold = sigA !== null && sigA === sigB ? TWIN_COSINE_SAME_PURPOSE : TWIN_COSINE_STRONG;
  return cos >= threshold;
}

const TopicListSchema = z.object({
  topics: z
    .array(
      z.object({
        label: z.string().min(1).max(80),
        description: z.string().min(1).max(300),
      }),
    )
    .min(2)
    .max(20),
});

/** A single discovered topic with its label and one-sentence purpose description. */
export type DiscoveredTopic = z.infer<typeof TopicListSchema>['topics'][number];

const SingleTopicSchema = z.object({
  label: z.string().min(1).max(80),
  description: z.string().min(1).max(300),
});

/** Remove any reasoning block the model emitted, including an unterminated one at the end. */
function stripThink(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<think>[\s\S]*$/i, '');
}

/**
 * Pull complete topic objects out of a possibly truncated JSON answer. Scans the topics array,
 * balancing braces while respecting strings, and keeps each fully-closed object that validates so a
 * cut-off answer still yields usable topics instead of failing the whole run.
 */
export function salvageTopics(raw: string): DiscoveredTopic[] {
  const text = stripThink(stripCodeFence(raw));
  const openers = Array.from(text.matchAll(/"topics"\s*:\s*\[/g));
  const last = openers.length > 0 ? openers[openers.length - 1] : undefined;
  const start =
    last && last.index !== undefined ? last.index + last[0].length - 1 : text.indexOf('[');
  if (start === -1) return [];

  const out: DiscoveredTopic[] = [];
  let depth = 0;
  let objStart = -1;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length && out.length < 20; i++) {
    const ch = text[i]!;
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
    } else if (ch === '{') {
      if (depth === 0) objStart = i;
      depth += 1;
    } else if (ch === '}') {
      if (depth > 0) {
        depth -= 1;
        if (depth === 0 && objStart !== -1) {
          try {
            const parsed = SingleTopicSchema.safeParse(JSON.parse(text.slice(objStart, i + 1)));
            if (parsed.success) out.push(parsed.data);
          } catch {}
          objStart = -1;
        }
      }
    } else if (ch === ']' && depth === 0) {
      break;
    }
  }
  return out;
}

/**
 * Outcome of a discovery run. A status of 'insufficient_categories' means too few concrete topics
 * were produced to form a safe taxonomy, so the existing categories were kept.
 */
export interface DiscoveryResult {
  status: 'ok' | 'insufficient_categories';
  topicsCreated: number;
  emailsSampled: number;
  centroidsComputed: number;
}

const SAMPLE_SIZE = 90;
const TOPIC_OUTPUT_TOKENS = 1500;
const RECENT_POOL = 600;
const HISTORICAL_POOL = 1000;
const UNCATEGORIZED_POOL = 400;
const TOP_DOMAIN_COUNT = 10;
const PER_DOMAIN_EXAMPLES = 20;
const SMALL_INBOX_MAX = 200;
const MODEST_INBOX_MAX = 2000;
const MIN_CATEGORIES_LARGE = 8;
const TARGET_TOPIC_COUNT = '10 to 16';
const ASSIGNMENT_THRESHOLD = 1.0;

/** Minimum number of concrete categories required to accept a taxonomy, scaled to inbox size. */
function minCategoriesFor(inboxSize: number): number {
  if (inboxSize < SMALL_INBOX_MAX) return 2;
  if (inboxSize < MODEST_INBOX_MAX) return 4;
  return MIN_CATEGORIES_LARGE;
}

const SYSTEM_PROMPT = `You analyze a sample of email subjects and identify recurring TOPICS in the user's inbox.

A topic is a recurring theme of communication defined by its PURPOSE, NOT a single email's content.

Each topic is a BROAD purpose that groups many senders. Any email clearly belongs to one topic, not several.

COMMON PURPOSES to draw from. Include any that the sample clearly shows, named in your own words; do NOT invent a category with no evidence in the sample, and do NOT force one that the inbox does not contain:
- Developer code reviews - GitHub, GitLab, Bitbucket pull requests, CI runs, commits, repository notifications
- Developer platforms and tokens - developer tool accounts, API keys, token expiry, deploy and platform notices
- Travel and accommodation - flights, hotels, bookings, travel deals from any travel site
- Banking and transactions - bank transfers, balances, statements, card activity
- Security and sign-in - login alerts, password resets, verification codes, account security
- Job opportunities - job listings and openings from any board or recruiter
- Professional networking - connection requests, profile views, network updates
- Social media updates - posts, mentions, comments, follows, and activity from social networks
- Receipts and invoices - payment confirmations, invoices, billing statements for things bought
- Shipping and deliveries - order shipment, tracking, delivery updates
- Course grades - grades, marks, exam results
- Course materials - assignments, lectures, schedules from courses
- Health and insurance - medical, pharmacy, insurance claims and policies
- Marketing and promotions - marketing, digests, opt-in newsletters and offers

GUIDELINES:
- Identify ${TARGET_TOPIC_COUNT} topics that together cover most of the inbox. Aim for at least 8 distinct concrete topics for a large inbox.
- Cover the high-volume senders listed below: each should have a fitting topic from the common purposes above or one you name.
- Every topic must be PURPOSE-based. NEVER name a topic after a single sender, brand, or service. Use one broad "Developer Code Reviews" for all GitHub/GitLab/CI mail, never "GitHub Notifications".
- Do NOT split one purpose into small variants (e.g. one "Travel & Accommodation", not separate "Flight Bookings" + "Hotel Bookings" + "Travel Deals").
- Make topics MUTUALLY EXCLUSIVE: no two may overlap or be near-synonyms. If two ideas overlap, merge them into one broader topic.
- BANNED: vague catch-all topics where every word is generic, such as "Technical Support", "Service Announcements", "Account Notifications", "General Updates", "Notifications", "Miscellaneous". Each topic needs a concrete anchor word naming a real purpose (banking, invoice, job, course, security, developer, travel, insurance, shipping, etc.).
- Topic descriptions: ONE short sentence (under 20 words) naming the concrete PURPOSE only. Do NOT name specific senders, brands, apps, or services in the description.
- Topic labels: 2-4 words, Title Case, no quotes.
- Output ONLY valid JSON, no extra text, no code fences, no markdown.

OUTPUT FORMAT:
{"topics": [{"label": "Receipts & Invoices", "description": "Payment confirmations and invoices for purchases and subscriptions."}, ...]}`;

const STRICTER_FEEDBACK = `\n\nReturn ONLY a compact JSON object with at most 12 topics. Keep each description under 10 words. No markdown, no code fences, no extra text.`;

interface StagedTopic {
  label: string;
  description: string;
  centroid: Float32Array;
  emailCount: number;
}

/**
 * Discovers recurring topics in an account's inbox by sampling email subjects, asking the LLM for a
 * purpose-based taxonomy, then computing embedding centroids and reconciling with existing
 * categories.
 */
export class TopicDiscoveryService {
  private running = false;

  /** Wire up the repositories, LLM client, and logger the discovery run depends on. */
  constructor(
    private llm: LlmClient,
    private emails: EmailRepository,
    private embeddings: EmbeddingRepository,
    private categories: CategoryRepository,
    private logger: Logger,
    private accounts?: AccountRepository,
    private audit?: DiscoveryAuditRepository,
    private getConfig?: () => LlmConfig,
  ) {}

  /**
   * Run topic discovery for one account and rebuild its auto categories. Single-flight, two
   * overlapping runs would clobber each other.
   */
  async discover(
    accountId: string,
    embeddingModelId: string,
    generationModelId: string,
  ): Promise<DiscoveryResult> {
    if (this.running) {
      throw new Error('topic discovery is already running');
    }
    this.running = true;
    try {
      return await this.runDiscovery(accountId, embeddingModelId, generationModelId);
    } catch (err) {
      const cfg = this.getConfig?.();
      const provider = cfg ? discoveryProvider(cfg) : 'main';
      this.audit?.log({
        accountId,
        flow: 'topic_discovery',
        accountKind: this.accounts?.findById(accountId)?.kind ?? 'unknown',
        provider: provider === 'main' ? 'local' : 'cloud',
        status: 'failed',
        modelId: generationModelId,
        error: String(err),
      });
      throw err;
    } finally {
      this.running = false;
    }
  }

  /**
   * The full discovery pipeline: sample subjects, request topics with vague-label and stricter
   * retries, embed and merge them, reconcile with existing categories, then compute centroids and
   * rebuild auto categories.
   */
  private async runDiscovery(
    accountId: string,
    embeddingModelId: string,
    generationModelId: string,
  ): Promise<DiscoveryResult> {
    const cfg = this.getConfig?.();
    const provider = cfg ? discoveryProvider(cfg) : 'main';
    const accountKind = this.accounts?.findById(accountId)?.kind ?? 'unknown';
    if (this.accounts && !this.accounts.isDiscoveryEligible(accountId)) {
      this.audit?.log({
        accountId,
        flow: 'topic_discovery',
        accountKind,
        provider: provider === 'main' ? 'local' : 'cloud',
        status: 'skipped',
        modelId: generationModelId,
      });
      return { status: 'ok', topicsCreated: 0, emailsSampled: 0, centroidsComputed: 0 };
    }

    const auditNoData = (poolSize: number): void =>
      this.audit?.log({
        accountId,
        flow: 'topic_discovery',
        accountKind,
        provider: provider === 'main' ? 'local' : 'cloud',
        status: 'insufficient',
        modelId: generationModelId,
        poolSize,
        sampleSize: 0,
        emailsExposed: 0,
      });

    const senders = this.emails.listSenders(accountId);
    const inboxSize = senders.length;
    if (inboxSize === 0) {
      auditNoData(0);
      return { status: 'ok', topicsCreated: 0, emailsSampled: 0, centroidsComputed: 0 };
    }
    const freq = domainFrequency(senders);

    const pool = this.buildDiscoveryPool(
      accountId,
      freq.slice(0, TOP_DOMAIN_COUNT).map(([d]) => d),
    );
    if (pool.length === 0) {
      auditNoData(pool.length);
      return { status: 'ok', topicsCreated: 0, emailsSampled: 0, centroidsComputed: 0 };
    }

    const sample = mixedSampleBySender(
      pool,
      SAMPLE_SIZE,
      stableHash(`${accountId}|${embeddingModelId}`),
    );
    this.logger.info(
      { accountId, sampleSize: sample.length, pool: pool.length, inboxSize },
      'topic discovery: sampling',
    );

    const sampleText = sample
      .map((e, i) => {
        const subject = (e.subject ?? '(no subject)').slice(0, 64);
        const from = (e.fromAddr ?? 'unknown').slice(0, 40);
        return `${i + 1}. "${subject}" - ${from}`;
      })
      .join('\n');

    const topDomains = freq
      .slice(0, 12)
      .map(([d, c]) => `${d} (${c})`)
      .join(', ');

    const userPrompt =
      `Inbox sample (${sample.length} emails):\n\n${sampleText}\n\n` +
      `Highest-volume senders by domain: ${topDomains}\n\n` +
      `Identify ${TARGET_TOPIC_COUNT} recurring topics. Make sure each high-volume sender above has a fitting topic.`;

    const insufficient = (): DiscoveryResult => {
      this.audit?.log({
        accountId,
        flow: 'topic_discovery',
        accountKind,
        provider: provider === 'main' ? 'local' : 'cloud',
        status: 'insufficient',
        modelId: generationModelId,
        poolSize: pool.length,
        sampleSize: sample.length,
        emailsExposed: sample.length,
        fieldsRead: ['subject', 'from_addr'],
      });
      return {
        status: 'insufficient_categories',
        topicsCreated: 0,
        emailsSampled: sample.length,
        centroidsComputed: 0,
      };
    };

    let topics: DiscoveredTopic[];
    try {
      topics = await this.requestTopics(generationModelId, userPrompt);
    } catch (err) {
      if (err instanceof LlmApiError && err.nonRetryable) throw err;
      this.logger.warn(
        { accountId, err },
        'topic discovery: first pass unusable, retrying with a stricter prompt',
      );
      try {
        topics = await this.requestTopics(generationModelId, userPrompt + STRICTER_FEEDBACK);
      } catch (retryErr) {
        if (retryErr instanceof LlmApiError && retryErr.nonRetryable) throw retryErr;
        this.logger.warn(
          { accountId, err: retryErr },
          'topic discovery: stricter retry produced no usable topics, keeping existing taxonomy',
        );
        return insufficient();
      }
    }
    const vague = topics.filter((t) => isVagueTopicLabel(t.label)).map((t) => t.label);
    if (vague.length > 0) {
      this.logger.info({ accountId, vague }, 'topic discovery: vague labels found, retrying');
      const feedback =
        `\n\nThe previous attempt produced these vague, banned labels: ${vague.join(', ')}. ` +
        `Do NOT use them or any catch-all. Replace each with a concrete purpose-based topic.`;
      try {
        topics = await this.requestTopics(generationModelId, userPrompt + feedback);
      } catch (err) {
        this.logger.warn({ accountId, err }, 'topic discovery: retry failed, keeping first pass');
      }
    }

    const concrete = dedupeNearLabels(topics.filter((t) => !isVagueTopicLabel(t.label)));
    const minCategories = minCategoriesFor(inboxSize);
    if (concrete.length < minCategories) {
      this.logger.warn(
        { accountId, usable: concrete.length, min: minCategories },
        'topic discovery: too few concrete categories, keeping existing taxonomy',
      );
      return insufficient();
    }

    const embedded: Array<{ topic: DiscoveredTopic; vec: Float32Array }> = [];
    for (const topic of concrete) {
      embedded.push({
        topic,
        vec: Float32Array.from(
          await this.llm.embed(`${topic.label}: ${topic.description}`, embeddingModelId),
        ),
      });
    }

    const kept = mergeOverlappingTopics(embedded, brandTokens(freq));
    if (kept.length < concrete.length) {
      const keptSet = new Set(kept.map((k) => k.topic));
      this.logger.info(
        {
          accountId,
          before: concrete.length,
          after: kept.length,
          dropped: concrete.filter((c) => !keptSet.has(c)).map((c) => c.label),
        },
        'topic discovery: merged overlapping categories',
      );
    }
    if (kept.length < minCategories) {
      this.logger.warn(
        { accountId, usable: kept.length, min: minCategories },
        'topic discovery: too much overlap after merge, keeping existing taxonomy',
      );
      return insufficient();
    }
    this.logger.info(
      { accountId, count: kept.length, labels: kept.map((k) => k.topic.label) },
      'topic discovery: concrete topics accepted',
    );

    await this.reconcileWithExisting(accountId, embeddingModelId, kept);

    const allEntries = this.embeddings.listForAccount(accountId, embeddingModelId);
    const vectorsByMsg = new Map<string, Float32Array>();
    for (const e of allEntries) vectorsByMsg.set(e.messageId, e.vector);

    const staged: StagedTopic[] = [];
    for (const { topic, vec } of kept) {
      const matched = this.embeddings
        .search(accountId, embeddingModelId, vec, 50)
        .filter((h) => h.distance < ASSIGNMENT_THRESHOLD);

      const matchedVectors: Float32Array[] = [];
      for (const h of matched) {
        const v = vectorsByMsg.get(h.messageId);
        if (v) matchedVectors.push(v);
      }

      const centroid = matchedVectors.length > 0 ? meanNormalize(matchedVectors) : null;
      staged.push(
        centroid
          ? {
              label: topic.label,
              description: topic.description,
              centroid,
              emailCount: matched.length,
            }
          : {
              label: topic.label,
              description: topic.description,
              centroid: vec,
              emailCount: 0,
            },
      );
    }

    const { live: centroidsComputed, omitted } = this.categories.reconcileAutoCategories(
      accountId,
      embeddingModelId,
      staged,
    );

    this.audit?.log({
      accountId,
      flow: 'topic_discovery',
      accountKind,
      provider: provider === 'main' ? 'local' : 'cloud',
      status: 'ok',
      modelId: generationModelId,
      poolSize: pool.length,
      sampleSize: sample.length,
      emailsExposed: sample.length,
      fieldsRead: ['subject', 'from_addr'],
      omittedCategories: omitted,
    });

    return {
      status: 'ok',
      topicsCreated: staged.length,
      emailsSampled: sample.length,
      centroidsComputed,
    };
  }

  /**
   * Merge existing auto-category twins and fold each newly discovered twin into the existing category
   * it duplicates, so re-discovery never leaves two categories for one purpose. User categories are
   * never touched. Compares label-embeddings (the same signal mergeOverlappingTopics uses), not the
   * collapsible email centroids. Mutates `kept` in place by renaming twin topics to the existing label.
   */
  private async reconcileWithExisting(
    accountId: string,
    embeddingModelId: string,
    kept: Array<{ topic: DiscoveredTopic; vec: Float32Array }>,
  ): Promise<void> {
    // Active auto categories only: never merge into or rename a suggested proposal or a retired
    // category, so discovery proposals awaiting review are left untouched.
    const existing = this.categories
      .listForAccount(accountId)
      .filter((c) => c.source === 'auto' && c.status === 'active');
    if (existing.length === 0) return;

    const embedded: Array<{ id: string; label: string; vec: Float32Array; emailCount: number }> =
      [];
    for (const c of existing) {
      const vec = Float32Array.from(
        await this.llm.embed(`${c.label}: ${c.description ?? ''}`, embeddingModelId),
      );
      embedded.push({ id: c.id, label: c.label, vec, emailCount: c.emailCount });
    }

    const survivors: typeof embedded = [];
    for (const cat of [...embedded].sort((a, b) => b.emailCount - a.emailCount)) {
      const twin = survivors.find((s) => areTwins(s.label, s.vec, cat.label, cat.vec));
      if (twin) {
        this.categories.mergeInto(cat.id, twin.id);
        this.logger.info(
          { accountId, merged: cat.label, into: twin.label },
          'topic discovery: merged existing semantic twin',
        );
      } else {
        survivors.push(cat);
      }
    }

    const claimed = new Set<string>();
    for (const k of kept) {
      const twin = survivors.find(
        (e) => !claimed.has(e.id) && areTwins(e.label, e.vec, k.topic.label, k.vec),
      );
      if (twin && twin.label.trim().toLowerCase() !== k.topic.label.trim().toLowerCase()) {
        claimed.add(twin.id);
        this.logger.info(
          { accountId, discovered: k.topic.label, reuse: twin.label },
          'topic discovery: discovered topic reuses existing semantic twin',
        );
        k.topic = { ...k.topic, label: twin.label };
      }
    }
  }

  /**
   * Assemble the candidate email pool to sample from, mixing recent, random historical,
   * uncategorized, and per-top-domain examples, deduplicated by message id.
   */
  private buildDiscoveryPool(accountId: string, topDomains: string[]): EmailSummary[] {
    const recent = this.emails.listSummaries({ accountId, limit: RECENT_POOL });
    const historical = this.emails.listSummariesRandom(accountId, HISTORICAL_POOL);
    const uncategorized = this.emails.listUncategorizedSummaries(accountId, UNCATEGORIZED_POOL);
    const domainExamples: EmailSummary[] = [];
    for (const domain of topDomains) {
      domainExamples.push(
        ...this.emails.listSummariesByDomain(accountId, domain, PER_DOMAIN_EXAMPLES),
      );
    }
    return dedupeById([...recent, ...historical, ...domainExamples, ...uncategorized]);
  }

  /** Send one topic-discovery prompt to the LLM and parse the JSON answer into topics. */
  private async requestTopics(model: string, userPrompt: string): Promise<DiscoveredTopic[]> {
    const cfg = this.getConfig?.();
    const provider = cfg ? discoveryProvider(cfg) : 'main';
    if (cfg) assertDiscoveryLocal(cfg, provider);
    const raw = await this.llm.chat({
      model,
      provider,
      messages: [
        { role: 'system', content: `/no_think\n${SYSTEM_PROMPT}` },
        { role: 'user', content: userPrompt },
      ],
      responseFormat: 'json_object',
      temperature: 0.2,
      maxTokens: TOPIC_OUTPUT_TOKENS,
      think: false,
    });
    return this.parseTopics(raw);
  }

  /**
   * Parse the model's raw answer into topics, falling back to salvaging objects from a truncated
   * response. Throws when fewer than two usable topics result.
   */
  private parseTopics(raw: string): DiscoveredTopic[] {
    const cleaned = stripThink(raw);
    try {
      const result = TopicListSchema.safeParse(parseLlmJson(cleaned));
      if (result.success) return result.data.topics;
    } catch {}
    const salvaged = salvageTopics(cleaned);
    this.logger.info(
      { rawLength: raw.length, salvaged: salvaged.length },
      'topic discovery: strict parse failed, used salvage',
    );
    if (salvaged.length >= 2) return salvaged;
    throw new Error('topic discovery: response had no usable topics');
  }
}

const LABEL_STOPWORDS = new Set([
  'and',
  'the',
  'your',
  'my',
  'our',
  'of',
  'for',
  'to',
  'from',
  'a',
  'an',
  're',
  'with',
  'on',
  'in',
  'at',
  'by',
  'or',
]);

export const VAGUE_TOKENS = new Set([
  'support',
  'help',
  'helpdesk',
  'assistance',
  'assist',
  'technical',
  'notification',
  'notifications',
  'alert',
  'alerts',
  'update',
  'updates',
  'general',
  'important',
  'service',
  'services',
  'system',
  'systems',
  'account',
  'accounts',
  'announcement',
  'announcements',
  'info',
  'information',
  'news',
  'message',
  'messages',
  'mail',
  'mails',
  'email',
  'emails',
  'misc',
  'miscellaneous',
  'other',
  'others',
  'uncategorized',
  'various',
  'status',
  'communication',
  'communications',
  'correspondence',
  'request',
  'requests',
  'inquiry',
  'inquiries',
  'query',
  'queries',
  'reminder',
  'reminders',
  'item',
  'items',
  'stuff',
  'thing',
  'things',
  'customer',
  'tech',
  'center',
  'centre',
  'daily',
  'weekly',
  'monthly',
  'quarterly',
  'company',
  'report',
  'reports',
  'inbox',
  'action',
  'actions',
  'activity',
  'recent',
  'latest',
  'online',
  'portal',
  'platform',
  'summary',
  'summaries',
  'detail',
  'details',
  'digest',
  'feed',
  'hub',
]);

/** Lowercase word tokens of a label with common stopwords removed. */
export function significantTokens(label: string): string[] {
  const tokens = label.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  return tokens.filter((t) => !LABEL_STOPWORDS.has(t));
}

/**
 * A label is vague when it has no concrete anchor: every significant word is a generic catch-all
 * term. Pattern based, so variants like "Technical Assistance" or "Account Notifications" are
 * caught without listing each one, while concrete categories are left alone.
 */
export function isVagueTopicLabel(label: string): boolean {
  const significant = significantTokens(label);
  if (significant.length === 0) return true;
  return significant.every((t) => VAGUE_TOKENS.has(t));
}

/** Jaccard similarity between two token sets, the intersection size over the union size. */
function jaccard(a: Set<string>, b: Set<string>): number {
  let inter = 0;
  for (const t of a) if (b.has(t)) inter += 1;
  const union = a.size + b.size - inter;
  return union === 0 ? 0 : inter / union;
}

/** True when a label significantly overlaps any of the given labels, used to reject near-twins of existing categories. */
export function isNearDuplicateLabel(label: string, others: string[]): boolean {
  const tok = new Set(significantTokens(label));
  if (tok.size === 0) return false;
  return others.some((o) => jaccard(tok, new Set(significantTokens(o))) >= 0.6);
}

const OVERLAP_PURPOSE = 0.9;
const OVERLAP_SENDER = 0.8;

const GENERIC_DOMAIN_LABELS = new Set([
  'mail',
  'email',
  'mailer',
  'smtp',
  'news',
  'info',
  'no',
  'noreply',
  'reply',
  'notification',
  'notifications',
  'em',
  'send',
  'sendgrid',
  'mailgun',
  'amazonses',
]);

const PURPOSE_DOMAIN_LABELS = new Set([
  'bank',
  'banking',
  'job',
  'jobs',
  'career',
  'careers',
  'travel',
  'insurance',
  'health',
  'shipping',
  'delivery',
  'security',
  'developer',
  'course',
  'courses',
  'invoice',
  'invoices',
  'receipt',
  'receipts',
  'shop',
  'store',
  'social',
]);

/** Brand tokens from the top sender domains, so a category named after a sender (GitHub) is detectable. */
export function brandTokens(freq: Array<[string, number]>): Set<string> {
  const brands = new Set<string>();
  for (const [domain] of freq.slice(0, 15)) {
    const parts = domain.split('.').filter(Boolean);
    const brand = (parts.length >= 2 ? parts[parts.length - 2]! : (parts[0] ?? '')).toLowerCase();
    if (
      brand.length >= 3 &&
      !GENERIC_DOMAIN_LABELS.has(brand) &&
      !PURPOSE_DOMAIN_LABELS.has(brand)
    ) {
      brands.add(brand);
    }
  }
  return brands;
}

/** True when a label contains a brand token, marking it as named after a sender rather than a purpose. */
function isSenderSpecific(label: string, brands: Set<string>): boolean {
  return significantTokens(label).some((t) => brands.has(t));
}

/**
 * Drop overlapping/twin categories by embedding similarity. Purpose-based topics are preferred over
 * sender-specific ones, so when an overlapping pair collapses the purpose label survives. A
 * candidate is dropped when it is too similar to an already-kept topic. A sender-specific candidate
 * merges at a lower similarity than two purpose-based categories.
 */
export function mergeOverlappingTopics<T extends { topic: { label: string }; vec: Float32Array }>(
  embedded: T[],
  brands: Set<string>,
): T[] {
  const ordered = embedded
    .map((e, i) => ({ e, i, sender: isSenderSpecific(e.topic.label, brands) }))
    .sort((a, b) => (a.sender === b.sender ? a.i - b.i : a.sender ? 1 : -1));

  const kept: T[] = [];
  for (const { e, sender } of ordered) {
    const overlaps = kept.some((k) => {
      const sim = cosineFromL2Distance(l2Distance(e.vec, k.vec));
      const threshold =
        sender || isSenderSpecific(k.topic.label, brands) ? OVERLAP_SENDER : OVERLAP_PURPOSE;
      return sim >= threshold;
    });
    if (!overlaps) kept.push(e);
  }
  return kept;
}

/** Drop near-duplicate labels by significant-word overlap so the taxonomy has no twins. */
export function dedupeNearLabels<T extends { label: string }>(topics: T[]): T[] {
  const kept: T[] = [];
  const keptTokens: Set<string>[] = [];
  for (const t of topics) {
    const tok = new Set(significantTokens(t.label));
    if (keptTokens.some((k) => jaccard(tok, k) >= 0.6)) continue;
    kept.push(t);
    keptTokens.push(tok);
  }
  return kept;
}

/** Keep the first occurrence of each message id, preserving order. */
function dedupeById<T extends { messageId: string }>(arr: T[]): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const e of arr) {
    if (seen.has(e.messageId)) continue;
    seen.add(e.messageId);
    out.push(e);
  }
  return out;
}

/** Sender domains across the inbox, most frequent first, as [domain, count] pairs. */
export function domainFrequency(
  senders: Array<{ fromAddr: string | null }>,
): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const s of senders) {
    const d = senderDomain(s.fromAddr);
    if (d) counts.set(d, (counts.get(d) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
}

/**
 * Sample for topic discovery: half sender-diverse for breadth, half frequency-weighted so
 * high-volume senders like GitHub and job boards appear in proportion to their share of the inbox
 * and get their own topic instead of being hidden behind one-per-domain breadth.
 */
export function mixedSampleBySender<T extends { fromAddr: string | null }>(
  arr: T[],
  n: number,
  seed: number,
): T[] {
  if (arr.length <= n) return [...arr];

  const half = Math.ceil(n / 2);
  const diverse = diverseSampleBySender(arr, half, seed);
  const chosen = new Set<T>(diverse);

  const rest = seededShuffle(
    arr.filter((e) => !chosen.has(e)),
    (seed ^ 0x9e3779b9) >>> 0,
  );

  const result = [...diverse];
  for (let i = 0; result.length < n && i < rest.length; i++) {
    result.push(rest[i]!);
  }
  return result;
}

/** Lowercase sender domain extracted from a From header. */
export function senderDomain(fromAddr: string | null): string {
  if (!fromAddr) return '';
  const at = fromAddr.lastIndexOf('@');
  const tail = at === -1 ? fromAddr : fromAddr.slice(at + 1);
  return tail
    .toLowerCase()
    .replace(/[>\s].*$/, '')
    .trim();
}

/**
 * Pick up to n emails that maximize distinct sender domains so the LLM sees the
 * inbox's breadth instead of many copies of the loudest sender, which would bias the
 * discovered topics. One email per domain is taken first in random order, then any
 * shortfall is filled from the rest. Uniform-random within each tier.
 */
export function diverseSampleBySender<T extends { fromAddr: string | null }>(
  arr: T[],
  n: number,
  seed: number,
): T[] {
  if (arr.length <= n) return [...arr];

  const shuffled = seededShuffle(arr, seed);

  const seen = new Set<string>();
  const primary: T[] = [];
  const leftovers: T[] = [];
  for (const e of shuffled) {
    const d = senderDomain(e.fromAddr);
    if (d && !seen.has(d)) {
      seen.add(d);
      primary.push(e);
    } else {
      leftovers.push(e);
    }
  }

  const result = primary.slice(0, n);
  for (let i = 0; result.length < n && i < leftovers.length; i++) {
    result.push(leftovers[i]!);
  }
  return result;
}
