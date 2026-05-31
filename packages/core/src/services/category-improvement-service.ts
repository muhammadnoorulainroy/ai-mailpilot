/**
 * Suggests and applies improvements to an account's category taxonomy by sampling the uncategorized
 * backlog, clustering it, and asking the LLM for expansions, new categories, and merges. Parsing,
 * scoring, and centroid updates stay local, only the suggestion call may use a cloud model.
 */
import type { Database } from 'better-sqlite3';
import type { Logger } from 'pino';
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