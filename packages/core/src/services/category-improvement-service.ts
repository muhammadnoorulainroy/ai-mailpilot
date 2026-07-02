/**
 * Suggests and applies improvements to an account's category taxonomy by sampling the uncategorized
 * backlog, clustering it, and asking the LLM for expansions, new categories, and merges. Parsing,
 * scoring, and centroid updates stay local, only the suggestion call may use a cloud model.
 */
import type { Database } from 'better-sqlite3';
import type { Logger } from 'pino';
import type { AccountRepository } from '../repositories/account-repository.js';
import type { DiscoveryAuditRepository } from '../repositories/discovery-audit-repository.js';
import type { LlmConfig } from '../config/schema.js';
import { assertDiscoveryLocal } from './discovery-guard.js';
import type {
  ApplyImprovementsResponse,
  ImproveSuggestionsResponse,
  SuggestedCategory,
  SuggestedCategoryExpansion,
  SuggestedMerge,
} from '@ai-mailpilot/shared';
import type { LlmClient } from '../llm/client.js';
import type { CategoryRepository, CentroidEntry } from '../repositories/category-repository.js';
import type { EmailRepository, EmailSummary } from '../repositories/email-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import { parseLlmJson, stripThink } from '../util/json-llm.js';
import { cosineFromL2Distance, l2Distance, meanNormalize } from '../util/vector.js';
import { purposeSignature, purposeTextEvidence } from './categorize-strategy.js';
import { rankCategories } from './categorization-service.js';
import { clusterBySenderAndContent, type ClusterInput } from './sender-clustering.js';
import {
  dedupeNearLabels,
  isNearDuplicateLabel,
  isVagueTopicLabel,
  mixedSampleBySender,
} from './topic-discovery-service.js';

const MIN_UNCATEGORIZED = 15;
const UNCATEGORIZED_SCAN_LIMIT = 50_000;
const UNCATEGORIZED_POOL = 8000;
const SAMPLE_SIZE = 120;
const ASSIGNMENT_THRESHOLD = 1.0;
const MIN_NEW_CATEGORY_COVERAGE = 4;
const COVERAGE_SEARCH_LIMIT = 400;
const EXISTING_DUP_COSINE = 0.9;
const IMPROVE_CLUSTER_THRESHOLD = 0.86;

interface SampledBucket {
  subject: string | null;
  fromAddr: string | null;
  clusterSize: number;
  memberIds: string[];
}

const LABEL_MAX = 80;
const DESCRIPTION_MAX = 300;
const REASON_MAX = 200;
const MAX_NEW = 20;
const MAX_MERGES = 12;

/** Taxonomy changes parsed out of the model reply: expansions of existing categories, new categories, merges, and clusters to leave uncategorized. */
export interface SalvagedSuggestions {
  existingCategoryExpansions: Array<{
    category: string;
    clusterNumbers: number[];
    reason: string;
  }>;
  newCategories: Array<{ label: string; description: string; clusterNumbers: number[] }>;
  merges: Array<{ source: string; target: string; reason: string }>;
  leaveUncategorized?: {
    clusterNumbers: number[];
    reason: string;
  };
}

/** Narrow a value to a plain object, returning null for arrays, null, and non-objects. */
function asObject(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/** Return the first trimmed string value found under any of the candidate keys, or an empty string. */
function stringField(obj: Record<string, unknown>, names: string[]): string {
  for (const name of names) {
    const v = obj[name];
    if (typeof v === 'string') return v.trim();
  }
  return '';
}

/** Coerce a raw value into a deduped list of positive integer cluster numbers, accepting strings or numbers. */
function coerceClusterNumbers(v: unknown): number[] {
  if (!Array.isArray(v)) return [];
  const out: number[] = [];
  for (const item of v) {
    const n =
      typeof item === 'number'
        ? item
        : typeof item === 'string'
          ? Number.parseInt(item, 10)
          : Number.NaN;
    if (Number.isInteger(n) && n > 0 && !out.includes(n)) out.push(n);
  }
  return out;
}

/** Coerce a raw array into new-category suggestions, dropping items without a label and capping the count. */
function coerceNewCategories(v: unknown): SalvagedSuggestions['newCategories'] {
  if (!Array.isArray(v)) return [];
  const out: SalvagedSuggestions['newCategories'] = [];
  for (const item of v) {
    const obj = asObject(item);
    if (!obj) continue;
    const label = stringField(obj, ['label', 'name', 'category']).slice(0, LABEL_MAX);
    if (!label) continue;
    const desc = stringField(obj, ['description', 'reason', 'rationale']);
    const clusterNumbers = coerceClusterNumbers(
      obj.clusterNumbers ?? obj.clusters ?? obj.clusterIds ?? obj.clusterIndexes,
    );
    out.push({ label, description: desc.slice(0, DESCRIPTION_MAX), clusterNumbers });
    if (out.length >= MAX_NEW) break;
  }
  return out;
}

/** Coerce a raw array into existing-category expansions, requiring both a category label and at least one cluster. */
function coerceExistingCategoryExpansions(
  v: unknown,
): SalvagedSuggestions['existingCategoryExpansions'] {
  if (!Array.isArray(v)) return [];
  const out: SalvagedSuggestions['existingCategoryExpansions'] = [];
  for (const item of v) {
    const obj = asObject(item);
    if (!obj) continue;
    const category = stringField(obj, [
      'category',
      'categoryLabel',
      'target',
      'targetCategory',
      'existingCategory',
      'label',
    ]).slice(0, LABEL_MAX);
    const clusterNumbers = coerceClusterNumbers(
      obj.clusterNumbers ?? obj.clusters ?? obj.clusterIds ?? obj.clusterIndexes,
    );
    if (!category || clusterNumbers.length === 0) continue;
    const reason = stringField(obj, ['reason', 'rationale', 'description']).slice(0, REASON_MAX);
    out.push({ category, clusterNumbers, reason });
    if (out.length >= MAX_NEW) break;
  }
  return out;
}

/** Coerce a raw array into merge suggestions, requiring both source and target labels and capping the count. */
function coerceMerges(v: unknown): SalvagedSuggestions['merges'] {
  if (!Array.isArray(v)) return [];
  const out: SalvagedSuggestions['merges'] = [];
  for (const item of v) {
    const obj = asObject(item);
    if (!obj) continue;
    const source = typeof obj.source === 'string' ? obj.source.trim() : '';
    const target = typeof obj.target === 'string' ? obj.target.trim() : '';
    if (!source || !target) continue;
    const reason = typeof obj.reason === 'string' ? obj.reason.trim().slice(0, REASON_MAX) : '';
    out.push({ source, target, reason });
    if (out.length >= MAX_MERGES) break;
  }
  return out;
}

/**
 * Tolerantly pull suggestions out of the model reply. Trims over-long fields and drops only the
 * unusable items, so one bad field never discards the whole set. Returns null only when the reply is
 * not parseable JSON.
 */
export function salvageSuggestions(raw: string): SalvagedSuggestions | null {
  let obj: unknown;
  try {
    obj = parseLlmJson(stripThink(raw));
  } catch {
    return null;
  }
  if (Array.isArray(obj)) {
    return { existingCategoryExpansions: [], newCategories: coerceNewCategories(obj), merges: [] };
  }
  const rec = asObject(obj);
  if (!rec) return null;
  const leave = (() => {
    const obj = asObject(rec.leaveUncategorized ?? rec.unmatched ?? rec.remaining);
    if (!obj) return undefined;
    return {
      clusterNumbers: coerceClusterNumbers(
        obj.clusterNumbers ?? obj.clusters ?? obj.clusterIds ?? obj.clusterIndexes,
      ),
      reason: stringField(obj, ['reason', 'rationale', 'description']).slice(0, REASON_MAX),
    };
  })();
  const out: SalvagedSuggestions = {
    existingCategoryExpansions: coerceExistingCategoryExpansions(
      rec.existingCategoryExpansions ?? rec.expandExisting ?? rec.assignToExisting,
    ),
    newCategories: coerceNewCategories(rec.newCategories),
    merges: coerceMerges(rec.merges),
  };
  if (leave) out.leaveUncategorized = leave;
  return out;
}

const STRICTER_SUGGEST_FEEDBACK = `\n\nReturn ONLY compact JSON, no markdown or commentary. Keep each description under 200 characters.`;

const SYSTEM_PROMPT = `You improve an email user's category taxonomy. You are shown the EXISTING categories and recurring clusters of emails that currently fit NONE of them.

The emails are shown as the largest recurring patterns first, with how many similar emails each represents. Favour patterns that represent many emails.

Suggest concrete, purpose-based improvements:
- existingCategoryExpansions: clusters that clearly belong in an EXISTING category but were left uncategorized. Use exact existing category labels. This is the preferred answer when the category already exists.
- newCategories: only for large recurring clusters that do NOT fit any existing category. Each needs a concrete purpose anchor (banking, invoice, job, course, security, developer, travel, insurance, shipping, newsletter, personal, entertainment, crypto, administration, etc.). Never propose a vague catch-all like "Technical Support", "Notifications", "General Updates", "Service Announcements".
- merges: only when two EXISTING categories clearly overlap; give source and target by their exact existing labels.
- leaveUncategorized: clusters that are one-offs, mixed, too ambiguous, or not worth filing.

Scan every cluster number before answering, not only the largest sender. Include all clear existing-category expansions you find.
Do not put a cluster in leaveUncategorized if it appears in existingCategoryExpansions or newCategories.
A cluster may appear in two existing categories only when both tags are independently useful; otherwise choose the primary purpose.
Keep boundaries strict: professional network/profile/connection activity is Professional Networking; non-work posts, videos, tweets, likes, and follows are Social Media Updates; repository, pull request, CI, build, commit, deployment, and token/platform mail is Developer-related, but generic software-developer job alerts are Job Opportunities; offers, deals, discounts, and newsletters are Marketing/Newsletters unless they are concrete receipts, shipping, banking, security, travel, or course notices.

Use 1-based cluster numbers from the prompt. Suggest nothing rather than forcing a bad category. Output ONLY JSON:
{"existingCategoryExpansions": [{"category": "Job Opportunities", "clusterNumbers": [1,2], "reason": "..."}], "newCategories": [{"label": "Newsletters & Digests", "description": "...", "clusterNumbers": [3]}], "merges": [{"source": "Old Label", "target": "Keep Label", "reason": "..."}], "leaveUncategorized": {"clusterNumbers": [4], "reason": "mixed one-off mail"}}`;

/** Normalize a label to a trimmed lowercase form for case-insensitive comparison. */
function normalizeLabel(label: string): string {
  return label.trim().toLowerCase();
}

/** Deduplicate a list of ids and drop empty ones. */
function uniqueIds(ids: string[]): string[] {
  return [...new Set(ids.filter((id) => id.length > 0))];
}

/** Collect up to limit unique non-empty trimmed values, preserving order. */
function uniqueLimited(items: string[], limit: number): string[] {
  const out: string[] = [];
  for (const item of items) {
    const value = item.trim();
    if (value && !out.includes(value)) out.push(value);
    if (out.length >= limit) break;
  }
  return out;
}

/** Pick up to three representative subjects across the given clusters. */
function sampleSubjects(clusters: SampledBucket[]): string[] {
  return uniqueLimited(
    clusters.map((c) => c.subject ?? '').filter((s) => s.length > 0),
    3,
  );
}

/** Pick up to three representative sender addresses across the given clusters. */
function sampleSenders(clusters: SampledBucket[]): string[] {
  return uniqueLimited(
    clusters.map((c) => c.fromAddr ?? '').filter((s) => s.length > 0),
    3,
  );
}

/** FNV-1a hash of a string, used to order items deterministically when sampling. */
function stableHash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/** Deterministically sample up to limit items by hashing their message id, stable across runs. */
function stableMessageSample<T extends { messageId: string }>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  return [...items]
    .sort((a, b) => stableHash(a.messageId) - stableHash(b.messageId))
    .slice(0, limit);
}

/** Suggests and applies taxonomy improvements for an account using its uncategorized email backlog. */
export class CategoryImprovementService {
  /** Wire up the repositories, LLM client, database, and logger this service depends on. */
  constructor(
    private db: Database,
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
   * Propose taxonomy changes from the uncategorized backlog. Nothing is applied here. The suggestion
   * LLM call runs on `provider` ('chat' = the cloud model when the user opted in, else 'main' local),
   * exactly like Refine, embeddings/clustering always stay local. modelId is the cloud chat model when
   * provider is 'chat', else the local generation model.
   */
  async suggest(
    accountId: string,
    embeddingModelId: string,
    modelId: string,
    provider: 'main' | 'chat' = 'main',
  ): Promise<ImproveSuggestionsResponse> {
    const cfg = this.getConfig?.();
    const accountKind = this.accounts?.findById(accountId)?.kind ?? 'unknown';
    const uncategorizedCount = this.categories.countUncategorized(accountId);
    const empty: ImproveSuggestionsResponse = {
      uncategorizedCount,
      sampledCount: 0,
      existingCategoryExpansions: [],
      newCategories: [],
      merges: [],
    };

    if (this.accounts && !this.accounts.isDiscoveryEligible(accountId)) {
      this.audit?.log({
        accountId,
        flow: 'improve_categories',
        accountKind,
        provider: provider === 'main' ? 'local' : 'cloud',
        status: 'skipped',
        modelId,
      });
      return empty;
    }
    if (cfg) {
      try {
        assertDiscoveryLocal(cfg, provider);
      } catch (err) {
        this.audit?.log({
          accountId,
          flow: 'improve_categories',
          accountKind,
          provider: provider === 'main' ? 'local' : 'cloud',
          status: 'blocked',
          modelId,
          error: String(err),
        });
        throw err;
      }
    }
    if (uncategorizedCount < MIN_UNCATEGORIZED) {
      this.audit?.log({
        accountId,
        flow: 'improve_categories',
        accountKind,
        provider: provider === 'main' ? 'local' : 'cloud',
        status: 'insufficient',
        modelId,
      });
      return empty;
    }

    const uncategorized = this.listUncategorizedForImprove(accountId);
    const vectorsByMsg = new Map<string, Float32Array>();
    for (const e of this.embeddings.listForAccount(accountId, embeddingModelId)) {
      vectorsByMsg.set(e.messageId, e.vector);
    }
    const sample = this.clusterFirstSample(uncategorized, vectorsByMsg, SAMPLE_SIZE);
    const auditImprove = (status: 'ok' | 'failed', error?: string): void =>
      this.audit?.log({
        accountId,
        flow: 'improve_categories',
        accountKind,
        provider: provider === 'main' ? 'local' : 'cloud',
        status,
        modelId,
        sampleSize: sample.length,
        emailsExposed: sample.length,
        fieldsRead: ['subject', 'from_addr'],
        error,
      });
    const existing = this.categories.listActive(accountId);
    const centroids = this.categories.getCentroidEntries(accountId, embeddingModelId);

    const existingText =
      existing.length > 0
        ? existing.map((c) => `- ${c.label}: ${c.description ?? ''}`).join('\n')
        : '(none yet)';
    const sampleText = sample
      .map((e, i) => {
        const subject = (e.subject ?? '(no subject)').slice(0, 64);
        const from = (e.fromAddr ?? 'unknown').slice(0, 40);
        const similar = e.clusterSize > 1 ? ` (~${e.clusterSize} similar)` : '';
        return `${i + 1}. "${subject}" - ${from}${similar}`;
      })
      .join('\n');
    const userPrompt =
      `EXISTING CATEGORIES:\n${existingText}\n\n` +
      `UNCATEGORIZED EMAILS (${sample.length} of ${uncategorizedCount}):\n${sampleText}\n\n` +
      `Suggest new categories and merges as JSON.`;

    const emptyWith = (warning: string): ImproveSuggestionsResponse => ({
      uncategorizedCount,
      sampledCount: sample.length,
      existingCategoryExpansions: [],
      newCategories: [],
      merges: [],
      warning,
    });
    const CONNECTIVITY_WARNING = 'The model could not be reached. Check Settings and try again.';
    const PARSE_WARNING =
      'Could not read the model output. Try again, or switch to a stronger model in Settings.';

    let raw: string;
    try {
      raw = await this.requestSuggestions(modelId, userPrompt, provider);
    } catch (err) {
      this.logger.warn({ accountId, err }, 'improve categories: suggestion request failed');
      auditImprove('failed', String(err));
      return emptyWith(CONNECTIVITY_WARNING);
    }

    let parsed = salvageSuggestions(raw);
    if (!parsed) {
      try {
        parsed = salvageSuggestions(
          await this.requestSuggestions(modelId, userPrompt + STRICTER_SUGGEST_FEEDBACK, provider),
        );
      } catch (err) {
        this.logger.warn({ accountId, err }, 'improve categories: stricter retry request failed');
        auditImprove('failed', String(err));
        return emptyWith(CONNECTIVITY_WARNING);
      }
    }
    if (!parsed) {
      this.logger.warn({ accountId }, 'improve categories: could not parse model suggestions');
      auditImprove('failed', 'could not parse model suggestions');
      return emptyWith(PARSE_WARNING);
    }

    auditImprove('ok');

    const directExpansions = this.resolveExistingCategoryExpansions(
      parsed.existingCategoryExpansions,
      existing,
      sample,
      vectorsByMsg,
      centroids,
    );
    const duplicateNewExpansions = this.expansionsFromDuplicateNewCategories(
      parsed.newCategories,
      existing,
      sample,
      vectorsByMsg,
      centroids,
    );
    const existingCategoryExpansions = this.mergeExpansions([
      ...directExpansions,
      ...duplicateNewExpansions,
    ]);
    const newCategories = await this.scoreNewCategories(
      accountId,
      embeddingModelId,
      parsed.newCategories,
      existing,
      uncategorized,
      sample,
    );
    const merges = this.resolveMerges(parsed.merges, existing);

    const diagnostics =
      existingCategoryExpansions.length === 0 && newCategories.length === 0 && merges.length === 0
        ? this.diagnoseBacklog(accountId, embeddingModelId, uncategorized, vectorsByMsg)
        : undefined;

    this.logger.info(
      {
        accountId,
        expansions: existingCategoryExpansions.length,
        suggested: newCategories.length,
        merges: merges.length,
        uncategorizedCount,
      },
      'improve categories: suggestions ready',
    );
    const usedMessageIds = new Set<string>([
      ...existingCategoryExpansions.flatMap((x) => x.messageIds),
      ...newCategories.flatMap((x) => x.messageIds ?? []),
    ]);
    const leaveUncategorized = this.resolveLeaveUncategorized(
      parsed.leaveUncategorized,
      sample,
      usedMessageIds,
    );
    return {
      uncategorizedCount,
      sampledCount: sample.length,
      existingCategoryExpansions,
      newCategories,
      merges,
      leaveUncategorized,
      diagnostics,
    };
  }

  /**
   * Explain an empty suggestion result by measuring how much of the backlog already sits near an
   * existing centroid, recommending Refine when most do and a stronger model otherwise.
   */
  private diagnoseBacklog(
    accountId: string,
    embeddingModelId: string,
    uncategorized: EmailSummary[],
    vectorsByMsg: Map<string, Float32Array>,
  ): ImproveSuggestionsResponse['diagnostics'] {
    const centroids = this.categories.getCentroidEntries(accountId, embeddingModelId);
    if (centroids.length === 0) return undefined;
    let withVector = 0;
    let covered = 0;
    for (const e of uncategorized) {
      const v = vectorsByMsg.get(e.messageId);
      if (!v) continue;
      withVector += 1;
      const nearest = Math.min(...centroids.map((c) => l2Distance(v, c.vector)));
      if (nearest < ASSIGNMENT_THRESHOLD) covered += 1;
    }
    if (withVector === 0) return undefined;
    const likely = covered / withVector >= 0.5;
    return {
      existingCategoriesLikelyCoverBacklog: likely,
      recommendation: likely
        ? 'Most uncategorized emails look like they fit your existing categories. Run "Refine (accurate AI)" to file them.'
        : 'Many uncategorized emails do not clearly fit any category. Try a stronger model in Settings, or add categories manually.',
    };
  }

  /** Read a stable, capped pool of uncategorized email summaries for clustering, preferring the stable reader when available. */
  private listUncategorizedForImprove(accountId: string): EmailSummary[] {
    const stableReader = this.emails as EmailRepository & {
      listUncategorizedSummariesStable?: (accountId: string, limit: number) => EmailSummary[];
    };
    const scanned = stableReader.listUncategorizedSummariesStable
      ? stableReader.listUncategorizedSummariesStable(accountId, UNCATEGORIZED_SCAN_LIMIT)
      : this.emails.listUncategorizedSummaries(accountId, UNCATEGORIZED_POOL);
    return stableMessageSample(scanned, UNCATEGORIZED_POOL);
  }

  /**
   * Turn the model's existing-category expansions into concrete suggestions, matching labels to real
   * categories and keeping only clusters whose evidence supports the expansion above the coverage floor.
   */
  private resolveExistingCategoryExpansions(
    raw: SalvagedSuggestions['existingCategoryExpansions'],
    existing: Array<{ id: string; label: string; description: string | null; emailCount: number }>,
    sample: SampledBucket[],
    vectorsByMsg: Map<string, Float32Array>,
    centroids: CentroidEntry[],
  ): SuggestedCategoryExpansion[] {
    const out: SuggestedCategoryExpansion[] = [];
    for (const item of raw) {
      const cat = this.matchExistingCategory(item.category, existing);
      if (!cat) continue;
      const clusters = this.clustersForNumbers(item.clusterNumbers, sample).filter((cluster) =>
        this.clusterSupportsExpansion(cluster, cat, vectorsByMsg, centroids),
      );
      const messageIds = uniqueIds(clusters.flatMap((c) => c.memberIds));
      if (messageIds.length < MIN_NEW_CATEGORY_COVERAGE) continue;
      out.push({
        categoryId: cat.id,
        categoryLabel: cat.label,
        estimatedCount: messageIds.length,
        sampleSubjects: sampleSubjects(clusters),
        sampleSenders: sampleSenders(clusters),
        reason: item.reason || `These uncategorized clusters fit ${cat.label}.`,
        messageIds,
      });
    }
    return out;
  }

  /**
   * Catch proposed new categories that actually duplicate an existing one and rewrite them as
   * expansions of the matching existing category instead.
   */
  private expansionsFromDuplicateNewCategories(
    raw: SalvagedSuggestions['newCategories'],
    existing: Array<{ id: string; label: string; description: string | null; emailCount: number }>,
    sample: SampledBucket[],
    vectorsByMsg: Map<string, Float32Array>,
    centroids: CentroidEntry[],
  ): SuggestedCategoryExpansion[] {
    const out: SuggestedCategoryExpansion[] = [];
    for (const item of raw) {
      if (item.clusterNumbers.length === 0) continue;
      const cat = this.matchExistingCategory(item.label, existing);
      if (!cat) continue;
      const clusters = this.clustersForNumbers(item.clusterNumbers, sample).filter((cluster) =>
        this.clusterSupportsExpansion(cluster, cat, vectorsByMsg, centroids),
      );
      const messageIds = uniqueIds(clusters.flatMap((c) => c.memberIds));
      if (messageIds.length < MIN_NEW_CATEGORY_COVERAGE) continue;
      out.push({
        categoryId: cat.id,
        categoryLabel: cat.label,
        estimatedCount: messageIds.length,
        sampleSubjects: sampleSubjects(clusters),
        sampleSenders: sampleSenders(clusters),
        reason: `${item.label} is already covered by ${cat.label}; expand the existing category instead.`,
        messageIds,
      });
    }
    return out;
  }

  /** Combine expansions targeting the same category, unioning message ids and samples, sorted by estimated count. */
  private mergeExpansions(items: SuggestedCategoryExpansion[]): SuggestedCategoryExpansion[] {
    const byCategory = new Map<string, SuggestedCategoryExpansion>();
    for (const item of items) {
      const existing = byCategory.get(item.categoryId);
      if (!existing) {
        byCategory.set(item.categoryId, { ...item, messageIds: uniqueIds(item.messageIds) });
        continue;
      }
      const messageIds = uniqueIds([...existing.messageIds, ...item.messageIds]);
      byCategory.set(item.categoryId, {
        ...existing,
        estimatedCount: messageIds.length,
        sampleSubjects: uniqueLimited([...existing.sampleSubjects, ...item.sampleSubjects], 3),
        sampleSenders: uniqueLimited([...existing.sampleSenders, ...item.sampleSenders], 3),
        reason: existing.reason,
        messageIds,
      });
    }
    return [...byCategory.values()].sort((a, b) => b.estimatedCount - a.estimatedCount);
  }

  /**
   * Decide whether a cluster genuinely belongs in a category, accepting it on textual purpose evidence
   * or, failing that, when its vector ranks the category highly enough by centroid similarity.
   */
  private clusterSupportsExpansion(
    cluster: SampledBucket,
    category: { id: string; label: string; description: string | null },
    vectorsByMsg: Map<string, Float32Array>,
    centroids: CentroidEntry[],
  ): boolean {
    const subjectAndSender = `${cluster.subject ?? ''} ${cluster.fromAddr ?? ''}`;
    const evidence = purposeTextEvidence(subjectAndSender, category);
    if (evidence.words > 0 || evidence.phrases > 0 || evidence.labelTokens > 0) return true;

    const vector = vectorsByMsg.get(cluster.memberIds[0] ?? '');
    if (!vector || centroids.length === 0) return false;
    const ranked = rankCategories(vector, centroids);
    const rank = ranked.findIndex((m) => m.categoryId === category.id);
    if (rank < 0) return false;
    const match = ranked[rank]!;
    return (rank === 0 && match.confidence >= 0.55) || (rank <= 2 && match.confidence >= 0.7);
  }

  /**
   * Build the leave-uncategorized summary from the model's clusters, excluding any messages already
   * claimed by an expansion or new category so nothing is double counted.
   */
  private resolveLeaveUncategorized(
    raw: SalvagedSuggestions['leaveUncategorized'],
    sample: SampledBucket[],
    usedMessageIds: ReadonlySet<string> = new Set(),
  ): ImproveSuggestionsResponse['leaveUncategorized'] {
    if (!raw) return undefined;
    const clusters = this.clustersForNumbers(raw.clusterNumbers, sample).filter(
      (c) => !c.memberIds.some((id) => usedMessageIds.has(id)),
    );
    const messageIds = uniqueIds(clusters.flatMap((c) => c.memberIds));
    if (messageIds.length === 0) return undefined;
    return {
      estimatedCount: messageIds.length,
      reason: raw.reason || 'These sampled clusters are mixed, one-off, or too ambiguous.',
      sampleSubjects: sampleSubjects(clusters),
    };
  }

  /**
   * Resolve a label to an existing category, trying exact match, then near-duplicate label, then the
   * highest-volume category sharing the same purpose signature.
   */
  private matchExistingCategory(
    label: string,
    existing: Array<{ id: string; label: string; description: string | null; emailCount: number }>,
  ): { id: string; label: string; description: string | null } | null {
    const norm = normalizeLabel(label);
    const exact = existing.find((c) => normalizeLabel(c.label) === norm);
    if (exact) return exact;
    const near = existing.find((c) => isNearDuplicateLabel(label, [c.label]));
    if (near) return near;
    const sig = purposeSignature(label);
    if (sig === null) return null;
    const samePurpose = existing
      .filter((c) => purposeSignature(c.label, c.description) === sig)
      .sort((a, b) => b.emailCount - a.emailCount);
    return samePurpose[0] ?? null;
  }

  /** Map 1-based cluster numbers from the prompt back to their sampled buckets, skipping out-of-range numbers. */
  private clustersForNumbers(numbers: number[], sample: SampledBucket[]): SampledBucket[] {
    const out: SampledBucket[] = [];
    for (const n of numbers) {
      const cluster = sample[n - 1];
      if (cluster) out.push(cluster);
    }
    return out;
  }

  /**
   * Send the system and user prompts to the LLM and return the raw reply. Disables thinking and
   * prepends /no_think for the local provider to keep the JSON output clean.
   */
  private async requestSuggestions(
    model: string,
    userPrompt: string,
    provider: 'main' | 'chat',
  ): Promise<string> {
    const local = provider === 'main';
    return this.llm.chat({
      model,
      provider,
      messages: [
        { role: 'system', content: local ? `/no_think\n${SYSTEM_PROMPT}` : SYSTEM_PROMPT },
        { role: 'user', content: userPrompt },
      ],
      responseFormat: 'json_object',
      temperature: 0,
      think: local ? false : undefined,
    });
  }

  /** Apply only what the user approved: create new categories with centroids, then merge. */
  async apply(
    accountId: string,
    embeddingModelId: string,
    approved: {
      existingCategoryExpansions?: Array<{ categoryId: string; messageIds: string[] }>;
      newCategories: Array<{ label: string; description: string; messageIds?: string[] }>;
      merges: Array<{ sourceId: string; targetId: string }>;
    },
  ): Promise<ApplyImprovementsResponse> {
    const vectorsByMsg = new Map<string, Float32Array>();
    for (const e of this.embeddings.listForAccount(accountId, embeddingModelId)) {
      vectorsByMsg.set(e.messageId, e.vector);
    }

    const stagedExpansions = (approved.existingCategoryExpansions ?? [])
      .map((e) => {
        const cat = this.categories.findById(e.categoryId);
        if (!cat || cat.accountId !== accountId || cat.status !== 'active') return null;
        const messageIds = uniqueIds(e.messageIds).filter((id) => vectorsByMsg.has(id));
        if (messageIds.length === 0) return null;
        return { categoryId: cat.id, messageIds };
      })
      .filter((e): e is { categoryId: string; messageIds: string[] } => e !== null);

    // Dedupe new-category labels across ALL statuses so we never create a duplicate of an active,
    // suggested, or retired category. A collision skips creation rather than assigning into a
    // hidden category. TODO(phase 2): offer explicit reactivation of a retired label match.
    const seen = new Set(
      this.categories.listForAccount(accountId).map((c) => normalizeLabel(c.label)),
    );
    const staged: Array<{
      label: string;
      description: string;
      centroid: Float32Array;
      emailCount: number;
      messageIds: string[];
    }> = [];
    for (const cat of approved.newCategories) {
      if (isVagueTopicLabel(cat.label)) continue;
      const norm = normalizeLabel(cat.label);
      if (seen.has(norm)) continue;
      seen.add(norm);
      const messageIds = uniqueIds(cat.messageIds ?? []).filter((id) => vectorsByMsg.has(id));
      let vectors = messageIds
        .map((id) => vectorsByMsg.get(id))
        .filter((v): v is Float32Array => v !== undefined);
      let descVec: Float32Array | null = null;
      if (vectors.length === 0) {
        descVec = Float32Array.from(
          await this.llm.embed(`${cat.label}: ${cat.description}`, embeddingModelId),
        );
        vectors = this.embeddings
          .search(accountId, embeddingModelId, descVec, 50)
          .filter((h) => h.distance < ASSIGNMENT_THRESHOLD)
          .map((h) => vectorsByMsg.get(h.messageId))
          .filter((v): v is Float32Array => v !== undefined);
      }
      const centroid =
        (vectors.length > 0 ? meanNormalize(vectors) : null) ??
        descVec ??
        Float32Array.from(
          await this.llm.embed(`${cat.label}: ${cat.description}`, embeddingModelId),
        );
      staged.push({
        label: cat.label,
        description: cat.description,
        centroid,
        emailCount: vectors.length,
        messageIds,
      });
    }

    const result = this.db.transaction((): ApplyImprovementsResponse => {
      let expanded = 0;
      for (const e of stagedExpansions) {
        const assigned = this.assignStillUncategorized(
          accountId,
          e.categoryId,
          e.messageIds,
          'llm',
        );
        this.updateCentroidFromMessages(e.categoryId, embeddingModelId, assigned, vectorsByMsg);
        expanded += assigned.length;
      }

      let created = 0;
      for (const s of staged) {
        const row = this.categories.create({
          accountId,
          label: s.label,
          description: s.description,
          source: 'auto',
        });
        const assigned = this.assignStillUncategorized(accountId, row.id, s.messageIds, 'llm');
        if (assigned.length > 0) {
          this.updateCentroidFromMessages(row.id, embeddingModelId, assigned, vectorsByMsg);
        } else {
          this.categories.saveCentroid(row.id, embeddingModelId, s.centroid, s.emailCount);
        }
        expanded += assigned.length;
        created += 1;
      }
      let merged = 0;
      for (const m of approved.merges) {
        const src = this.categories.findById(m.sourceId);
        const tgt = this.categories.findById(m.targetId);
        if (!src || !tgt || src.id === tgt.id) continue;
        if (src.accountId !== accountId || tgt.accountId !== accountId) continue;
        if (src.status !== 'active' || tgt.status !== 'active') continue;
        this.categories.mergeInto(m.sourceId, m.targetId);
        merged += 1;
      }
      return { expanded, created, merged };
    })();

    this.logger.info({ accountId, ...result }, 'improve categories: applied');
    return result;
  }

  /**
   * Auto-assign the given messages to a category, skipping ones missing, user-assigned, or already in
   * that category, then clear their pending decisions. Returns the message ids actually assigned.
   */
  private assignStillUncategorized(
    accountId: string,
    categoryId: string,
    messageIds: string[],
    method: 'llm' | 'gate' | 'embed',
  ): string[] {
    const eligible: string[] = [];
    const now = Date.now();
    for (const messageId of uniqueIds(messageIds)) {
      if (!this.emails.findById(messageId, accountId)) continue;
      const existing = this.categories.getEmailCategories(messageId, accountId);
      if (existing.some((a) => a.assignedBy === 'user')) continue;
      if (existing.some((a) => a.categoryId === categoryId)) continue;
      eligible.push(messageId);
    }
    if (eligible.length === 0) return [];
    this.categories.addAutoAssignments(
      accountId,
      eligible.map((messageId) => ({
        messageId,
        accountId,
        categoryId,
        confidence: 0.95,
        assignedBy: 'auto' as const,
        assignedAt: now,
        method,
      })),
    );
    for (const messageId of eligible) this.categories.clearDecisionsForEmail(messageId, accountId);
    return eligible;
  }

  /**
   * Fold the given message vectors into a category's centroid, seeding a fresh normalized mean when
   * none exists or otherwise weighting the running centroid by its email count before renormalizing.
   */
  private updateCentroidFromMessages(
    categoryId: string,
    embeddingModelId: string,
    messageIds: string[],
    vectorsByMsg: Map<string, Float32Array>,
  ): void {
    const vectors = messageIds
      .map((id) => vectorsByMsg.get(id))
      .filter((v): v is Float32Array => v !== undefined);
    if (vectors.length === 0) return;

    const current = this.categories.getCentroid(categoryId, embeddingModelId);
    if (!current) {
      const centroid = meanNormalize(vectors);
      if (centroid)
        this.categories.saveCentroid(categoryId, embeddingModelId, centroid, vectors.length);
      return;
    }

    const sum = new Float32Array(current.vector.length);
    for (let i = 0; i < sum.length; i++) sum[i] = current.vector[i]! * current.emailCount;
    for (const vec of vectors) {
      for (let i = 0; i < sum.length; i++) sum[i] += vec[i]!;
    }
    const total = current.emailCount + vectors.length;
    let normSq = 0;
    for (let i = 0; i < sum.length; i++) {
      sum[i] /= total;
      normSq += sum[i]! * sum[i]!;
    }
    const norm = Math.sqrt(normSq);
    if (norm > 0) {
      const inv = 1 / norm;
      for (let i = 0; i < sum.length; i++) sum[i] *= inv;
    }
    this.categories.saveCentroid(categoryId, embeddingModelId, sum, total);
  }

  /**
   * Cluster the uncategorized pool by sender and content and return the largest clusters as buckets,
   * falling back to a mixed-by-sender sample when no vectors are available.
   */
  private clusterFirstSample(
    uncategorized: EmailSummary[],
    vectorsByMsg: Map<string, Float32Array>,
    size: number,
  ): SampledBucket[] {
    const inputs: ClusterInput[] = [];
    for (const e of uncategorized) {
      const v = vectorsByMsg.get(e.messageId);
      if (v) inputs.push({ messageId: e.messageId, fromAddr: e.fromAddr, vector: v });
    }
    if (inputs.length === 0) {
      return mixedSampleBySender(
        uncategorized,
        size,
        stableHash(uncategorized.map((e) => e.messageId).join('|')),
      ).map((e) => ({
        subject: e.subject,
        fromAddr: e.fromAddr,
        clusterSize: 1,
        memberIds: [e.messageId],
      }));
    }
    const byId = new Map(uncategorized.map((e) => [e.messageId, e]));
    return clusterBySenderAndContent(inputs, IMPROVE_CLUSTER_THRESHOLD)
      .sort((a, b) => b.memberIds.length - a.memberIds.length)
      .slice(0, size)
      .map((cl) => {
        const rep = byId.get(cl.representativeId);
        return {
          subject: rep?.subject ?? null,
          fromAddr: rep?.fromAddr ?? null,
          clusterSize: cl.memberIds.length,
          memberIds: cl.memberIds,
        };
      });
  }

  /**
   * Filter and score proposed new categories, dropping vague, duplicate, or already-covered labels,
   * then estimating coverage from cluster hits or an embedding search and keeping those above the floor.
   */
  private async scoreNewCategories(
    accountId: string,
    embeddingModelId: string,
    raw: Array<{ label: string; description: string; clusterNumbers: number[] }>,
    existingCategories: Array<{
      id: string;
      label: string;
      description: string | null;
      emailCount: number;
    }>,
    uncategorized: Array<{ messageId: string; subject: string | null }>,
    sample: SampledBucket[],
  ): Promise<SuggestedCategory[]> {
    const existingLabels = existingCategories.map((c) => normalizeLabel(c.label));
    const existing = new Set(existingLabels);
    const fresh = dedupeNearLabels(
      raw.filter((c) => {
        return (
          !isVagueTopicLabel(c.label) &&
          !existing.has(normalizeLabel(c.label)) &&
          !isNearDuplicateLabel(c.label, existingLabels) &&
          !this.matchExistingCategory(c.label, existingCategories)
        );
      }),
    );
    const existingCentroids = this.categories.getCentroidEntries(accountId, embeddingModelId);
    const uncatIds = new Set(uncategorized.map((e) => e.messageId));
    const subjectById = new Map(uncategorized.map((e) => [e.messageId, e.subject]));

    const out: SuggestedCategory[] = [];
    for (const c of fresh) {
      let vec: Float32Array;
      try {
        vec = Float32Array.from(
          await this.llm.embed(`${c.label}: ${c.description}`, embeddingModelId),
        );
      } catch (err) {
        this.logger.warn({ err, label: c.label }, 'improve: embed failed, suggestion skipped');
        continue;
      }
      const dupOf = existingCentroids.find(
        (ec) => cosineFromL2Distance(l2Distance(vec, ec.vector)) >= EXISTING_DUP_COSINE,
      );
      if (dupOf) {
        this.logger.info(
          { label: c.label, existing: dupOf.label },
          'improve: dropped suggestion duplicating an existing category',
        );
        continue;
      }
      const clusterHits = this.clustersForNumbers(c.clusterNumbers, sample).flatMap(
        (cluster) => cluster.memberIds,
      );
      const messageIds =
        clusterHits.length > 0
          ? uniqueIds(clusterHits).filter((id) => uncatIds.has(id))
          : this.embeddings
              .search(accountId, embeddingModelId, vec, COVERAGE_SEARCH_LIMIT)
              .filter((h) => h.distance < ASSIGNMENT_THRESHOLD && uncatIds.has(h.messageId))
              .map((h) => h.messageId);
      if (messageIds.length < MIN_NEW_CATEGORY_COVERAGE) continue;
      const samples =
        clusterHits.length > 0
          ? sampleSubjects(this.clustersForNumbers(c.clusterNumbers, sample))
          : messageIds
              .slice(0, 3)
              .map((id) => subjectById.get(id) ?? '')
              .filter((s): s is string => !!s);
      out.push({
        label: c.label,
        description: c.description,
        estimatedCount: messageIds.length,
        sampleSubjects: samples,
        messageIds,
      });
    }
    out.sort((a, b) => b.estimatedCount - a.estimatedCount);
    return out;
  }

  /** Resolve proposed merges to real category ids by label, dropping unknown labels and self-merges. */
  private resolveMerges(
    raw: Array<{ source: string; target: string; reason: string }>,
    existing: Array<{ id: string; label: string }>,
  ): SuggestedMerge[] {
    const byLabel = new Map(existing.map((c) => [normalizeLabel(c.label), c]));
    const out: SuggestedMerge[] = [];
    for (const m of raw) {
      const src = byLabel.get(normalizeLabel(m.source));
      const tgt = byLabel.get(normalizeLabel(m.target));
      if (!src || !tgt || src.id === tgt.id) continue;
      out.push({
        sourceId: src.id,
        sourceLabel: src.label,
        targetId: tgt.id,
        targetLabel: tgt.label,
        reason: m.reason,
      });
    }
    return out;
  }
}
