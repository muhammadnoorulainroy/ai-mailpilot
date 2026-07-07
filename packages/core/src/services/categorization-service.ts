/**
 * Embedding-based email categorization. Scores email vectors against per-account
 * category centroids and decides which category labels to assign.
 */
import type { CategoryRepository, CentroidEntry } from '../repositories/category-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import { cosineFromL2Distance, l2Distance } from '../util/vector.js';
import { gateFastAssignment } from './categorize-strategy.js';

/** A category assigned to an email, with its distance and derived confidence. */
export interface CategoryMatch {
  categoryId: string;
  label: string;
  distance: number;
  confidence: number;
}

/** Tuning knobs that control which category matches are accepted. */
export interface CategorizationOptions {
  /** Hard upper bound on distance. Any category beyond this is never assigned. */
  hardThreshold?: number;
  /** Maximum number of labels per email. */
  maxLabels?: number;
  /** A category is accepted if its distance is within bestDistance plus relativeMargin. */
  relativeMargin?: number;
  /** Fast-pass mode: assign only a clear single winner, else nothing. Ignores the other options. */
  fastGate?: boolean;
}

const DEFAULT_HARD_THRESHOLD = 1.0;
const DEFAULT_MAX_LABELS = 3;
const DEFAULT_RELATIVE_MARGIN = 0.1;

export class CategorizationService {
  /**
   * Creates the service over the category and embedding repositories. `multiPrototypeEnabled` reads
   * the Phase 4 feature flag on each call; when false (the default), matching uses only the aggregate
   * centroid per category, identical to the pre-Phase-4 behavior.
   */
  constructor(
    private categories: CategoryRepository,
    private embeddings: EmbeddingRepository,
    private multiPrototypeEnabled: () => boolean = () => false,
  ) {}

  /**
   * Classify a single email against the account's current category centroids.
   * Loads centroids on every call, prefer categorizeBatch when processing many emails.
   */
  categorize(
    messageId: string,
    accountId: string,
    embeddingModelId: string,
    options: CategorizationOptions = {},
  ): CategoryMatch[] {
    const emailVec = this.embeddings.getEmbedding({
      messageId,
      accountId,
      modelId: embeddingModelId,
    });
    if (!emailVec) return [];

    const centroids = this.categories.getEffectivePrototypeEntries(
      accountId,
      embeddingModelId,
      this.multiPrototypeEnabled(),
    );
    if (centroids.length === 0) return [];

    return matchAgainstCentroids(emailVec, centroids, options);
  }

  /**
   * Classify every embedded email for an account in one pass, loading embeddings and
   * centroids once. matches is keyed by messageId, scored counts evaluated emails, and
   * emails in lockedIds are skipped so auto categorization never overrides a user correction.
   */
  categorizeBatch(
    accountId: string,
    embeddingModelId: string,
    options: CategorizationOptions = {},
    lockedIds: ReadonlySet<string> = new Set(),
  ): { matches: Map<string, CategoryMatch[]>; scored: number } {
    const matches = new Map<string, CategoryMatch[]>();

    const centroids = this.categories.getEffectivePrototypeEntries(
      accountId,
      embeddingModelId,
      this.multiPrototypeEnabled(),
    );
    if (centroids.length === 0) return { matches, scored: 0 };

    const entries = this.embeddings.listForAccount(accountId, embeddingModelId);

    let scored = 0;
    for (const entry of entries) {
      if (lockedIds.has(entry.messageId)) continue;
      scored += 1;
      const m = matchAgainstCentroids(entry.vector, centroids, options);
      if (m.length > 0) matches.set(entry.messageId, m);
    }

    return { matches, scored };
  }

  /** Replace all assignments for a single email with the given matches in one transaction. */
  replaceForEmail(
    messageId: string,
    accountId: string,
    matches: CategoryMatch[],
    assignedBy: 'user' | 'auto' = 'user',
  ): void {
    const now = Date.now();
    const assignments = matches.map((m) => ({
      messageId,
      accountId,
      categoryId: m.categoryId,
      confidence: m.confidence,
      assignedBy,
      assignedAt: now,
    }));
    this.categories.replaceEmailAssignments(messageId, accountId, assignments);
  }
}

/**
 * Rank all categories for an email vector by similarity, best first, with no acceptance cutoff.
 * Used by the LLM pass to build a shortlist of plausible categories for the model.
 */
export function rankCategories(
  emailVec: Float32Array,
  centroids: CentroidEntry[],
): CategoryMatch[] {
  // A category may be represented by several prototype vectors (Phase 4). Keep only its NEAREST
  // prototype so each category is ranked once, by its best (minimum) distance, and its confidence is
  // the winning prototype's cosine. With exactly one centroid per category this is a no-op and
  // reproduces the single-centroid ranking exactly.
  const bestByCategory = new Map<string, CategoryMatch>();
  for (const c of centroids) {
    const d = l2Distance(emailVec, c.vector);
    const existing = bestByCategory.get(c.categoryId);
    if (!existing || d < existing.distance) {
      bestByCategory.set(c.categoryId, {
        categoryId: c.categoryId,
        label: c.label,
        distance: d,
        confidence: cosineFromL2Distance(d),
      });
    }
  }
  const scored = Array.from(bestByCategory.values());
  scored.sort((a, b) => a.distance - b.distance);
  return scored;
}

/**
 * Score an email vector against centroids and pick the accepted matches per the options.
 * In fast-gate mode returns a single clear winner or nothing, otherwise accepts categories
 * within the relative margin of the best, bounded by the hard threshold and maxLabels.
 */
function matchAgainstCentroids(
  emailVec: Float32Array,
  centroids: CentroidEntry[],
  options: CategorizationOptions,
): CategoryMatch[] {
  const hardThreshold = options.hardThreshold ?? DEFAULT_HARD_THRESHOLD;
  const maxLabels = options.maxLabels ?? DEFAULT_MAX_LABELS;
  const relativeMargin = options.relativeMargin ?? DEFAULT_RELATIVE_MARGIN;

  const scored = rankCategories(emailVec, centroids);

  if (options.fastGate) {
    const gated = gateFastAssignment(scored);
    return gated ? [gated] : [];
  }

  const best = scored[0];
  if (!best || best.distance >= hardThreshold) return [];

  const cutoff = Math.min(hardThreshold, best.distance + relativeMargin);
  const accepted: CategoryMatch[] = [];
  for (const s of scored) {
    if (s.distance > cutoff) break;
    accepted.push(s);
    if (accepted.length >= maxLabels) break;
  }
  return accepted;
}
