/**
 * Generates structural (split / merge / retire) review-queue proposals from the read-only Phase 3.2
 * health metrics (Phase 3.3 detection). It only proposes obvious, safe cases and never applies
 * anything: a merge when two active auto categories are near-duplicate (high overlap / low
 * separation), a retire when an active auto category is empty, and a split when a loose active auto
 * category separates into two or more tight, well-separated subclusters.
 *
 * Everything is user-approved on apply. This service writes only category_proposals rows (via the
 * existing createStructural / createSplit paths) and never touches a category, assignment, or
 * centroid. It skips user-created categories entirely (their structural changes are applicable, so
 * they must not be auto-suggested), and it deduplicates on the deterministic suppressionKey so a
 * purpose that is already pending, applied, or dismissed is never re-proposed.
 */
import { randomUUID } from 'node:crypto';
import type { Logger } from 'pino';
import {
  canonicalKeyBase,
  type CategoryRepository,
  type CategoryWithCount,
} from '../repositories/category-repository.js';
import type { CategoryProposalRepository } from '../repositories/category-proposal-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import type { EmailRepository } from '../repositories/email-repository.js';
import type { CategoryHealthService } from './category-health-service.js';
import { purposeSignature } from './categorize-strategy.js';
import { isNearDuplicateLabel } from './topic-discovery-service.js';
import { clusterResidual, type ClusterPoint } from './discovery-clustering.js';
import { clusterKeyphrases } from './discovery-candidates.js';
import { cosineSimilarity } from '../util/vector.js';
import { normalizeForMatch } from '../util/text.js';

/** Lowest overlap (cosine of stored centroids) at which two auto categories are proposed for merge. */
export const MERGE_MIN_OVERLAP = 0.9;

/** Fewest auto members an active category needs before a split is worth considering. */
export const SPLIT_MIN_SIZE = 12;
/** A proposed split child must carry at least this many of the source's auto members to be viable. */
export const SPLIT_MIN_CHILD_SIZE = 5;
/** A category already tighter than this cohesion is left alone; splitting it would gain little. */
export const SPLIT_MAX_PARENT_COHESION = 0.82;
/** Each split child must be at least this internally coherent (mean cosine to its own centroid). */
export const SPLIT_MIN_CHILD_COHESION = 0.55;
/** The split children's centroids must differ by at least this cosine gap (pairwise minimum). */
export const SPLIT_MIN_CHILD_SEPARATION = 0.2;
/** Each child must be at least this much tighter than the parent, so the split genuinely improves it. */
export const SPLIT_MIN_COHESION_GAIN = 0.08;
/** Cap on children per split, so a loose category is not shattered into many thin fragments. */
export const SPLIT_MAX_CHILDREN = 4;
/** Fixed seed so subclustering a given category is deterministic across runs. */
export const SPLIT_CLUSTER_SEED = 1337;

/** One structural proposal produced by a generate run. */
export interface GeneratedStructural {
  id: string;
  kind: 'merge' | 'retire' | 'split';
  categoryId: string;
  sourceCategoryId: string | null;
  suppressionKey: string;
  label: string;
}

/** Outcome of a structural generate run. Counts are for debugging why proposals were or were not made. */
export interface StructuralGenerateResult {
  runId: string;
  created: GeneratedStructural[];
  mergeCandidates: number;
  retireCandidates: number;
  splitCandidates: number;
  skippedExisting: number;
}

/** A proposed split child worked out by the deterministic analyzer, ready to persist. */
interface SplitChild {
  label: string;
  canonicalKey: string;
  description: string;
  centroid: Float32Array;
  memberIds: string[];
  proposedCount: number;
  cohesion: number;
  separation: number;
  confidence: number;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Capitalizes the first letter of a single keyphrase token for use as a category label. */
function titleCase(token: string): string {
  return token.length === 0 ? token : token[0]!.toUpperCase() + token.slice(1);
}

/** Deterministic survivor of a merge pair: the larger category, breaking ties by canonical key. */
function survivorFirst(
  a: CategoryWithCount,
  b: CategoryWithCount,
): [CategoryWithCount, CategoryWithCount] {
  if (a.emailCount !== b.emailCount) {
    return a.emailCount > b.emailCount ? [a, b] : [b, a];
  }
  return a.canonicalKey <= b.canonicalKey ? [a, b] : [b, a];
}

/**
 * Conservative same-purpose gate for a merge pair (F1). High stored-centroid overlap alone is not
 * enough: transactional categories (invoices vs shipping vs security) collapse in embedding space, so
 * a raw-overlap merge can pair genuinely different purposes. Deterministic, no LLM: reject when the
 * two labels/descriptions map to DIFFERENT known purpose groups; otherwise require a positive
 * same-purpose signal (the same known purpose group, or near-duplicate labels). A wrong merge is
 * worse than a missed one, so with no positive signal we do not propose.
 */
function sameMergePurpose(a: CategoryWithCount, b: CategoryWithCount): boolean {
  const sigA = purposeSignature(a.label, a.description);
  const sigB = purposeSignature(b.label, b.description);
  if (sigA !== null && sigB !== null && sigA !== sigB) return false;
  if (sigA !== null && sigA === sigB) return true;
  return isNearDuplicateLabel(a.label, [b.label]);
}

/**
 * Decide whether a loose category's auto members separate into distinct child categories, and if so
 * describe those children. Pure and deterministic given the seeded clustering: it subclusters the
 * members, keeps only tight, viable subclusters, and requires them to be meaningfully tighter than
 * the parent and well separated from each other. Child labels come from deterministic class-TF-IDF
 * keyphrases of each subcluster's subjects; if any child cannot be given a distinct, meaningful
 * label, the whole split is abandoned (returns null) rather than proposing an unclear change.
 */
function analyzeSplit(
  parentLabel: string,
  points: ClusterPoint[],
  subjectById: Map<string, string>,
  parentCohesion: number,
): SplitChild[] | null {
  if (points.length < SPLIT_MIN_SIZE) return null;

  const minId = (ids: string[]): string => ids.reduce((m, x) => (x < m ? x : m), ids[0]!);
  const clusters = clusterResidual(points, SPLIT_CLUSTER_SEED)
    .filter((c) => c.size >= SPLIT_MIN_CHILD_SIZE && c.cohesion >= SPLIT_MIN_CHILD_COHESION)
    .sort((a, b) => b.size - a.size || minId(a.memberIds).localeCompare(minId(b.memberIds)))
    .slice(0, SPLIT_MAX_CHILDREN);
  if (clusters.length < 2) return null;

  // Every kept child must be clearly tighter than the loose parent, or the split buys nothing.
  if (clusters.some((c) => c.cohesion < parentCohesion + SPLIT_MIN_COHESION_GAIN)) return null;

  // The children must be well separated from each other (smallest pairwise centroid gap).
  let minPairSeparation = 1;
  for (let i = 0; i < clusters.length; i++) {
    for (let j = i + 1; j < clusters.length; j++) {
      const gap = 1 - cosineSimilarity(clusters[i]!.centroid, clusters[j]!.centroid);
      if (gap < minPairSeparation) minPairSeparation = gap;
    }
  }
  if (minPairSeparation < SPLIT_MIN_CHILD_SEPARATION) return null;

  // Deterministic, distinct labels from class-TF-IDF keyphrases of each subcluster's subjects.
  const subjectsPerChild = clusters.map((c) =>
    c.memberIds.map((id) => subjectById.get(id) ?? '').filter((s) => s.trim() !== ''),
  );
  const keyphrasesPerChild = clusterKeyphrases(subjectsPerChild);
  const usedLabels = new Set<string>();
  const usedKeys = new Set<string>();
  const children: SplitChild[] = [];
  for (let i = 0; i < clusters.length; i++) {
    const phrase = keyphrasesPerChild[i]!.find((p) => !usedLabels.has(normalizeForMatch(p)));
    if (!phrase) return null;
    const label = titleCase(phrase);
    const canonicalKey = canonicalKeyBase(label);
    if (canonicalKey === '' || usedKeys.has(canonicalKey)) return null;
    usedLabels.add(normalizeForMatch(phrase));
    usedKeys.add(canonicalKey);
    const c = clusters[i]!;
    children.push({
      label,
      canonicalKey,
      description: `"${label}" mail separated out of ${parentLabel}.`,
      centroid: c.centroid,
      memberIds: c.memberIds,
      proposedCount: c.size,
      cohesion: c.cohesion,
      separation: c.separation,
      confidence: clamp01(0.6 * c.cohesion + 0.4 * minPairSeparation),
    });
  }
  return children;
}

/** Detects safe merge, retire, and split proposals from health metrics. Never applies a change. */
export class StructuralProposalService {
  constructor(
    private categories: CategoryRepository,
    private proposals: CategoryProposalRepository,
    private health: CategoryHealthService,
    private embeddings: EmbeddingRepository,
    private emails: EmailRepository,
    private logger: Logger,
  ) {}

  /**
   * Scan the account's active auto categories and enqueue merge/retire/split proposals for the obvious
   * cases. Reads health metrics, embeddings, and existing proposals; writes only proposal rows.
   * Idempotent: a signature already pending, applied, or dismissed is skipped.
   */
  generate(accountId: string, embeddingModelId: string): StructuralGenerateResult {
    const runId = randomUUID();
    const active = this.categories.listActive(accountId);
    const byId = new Map(active.map((c) => [c.id, c] as const));
    const metrics = this.health.metricsForAccount(accountId, embeddingModelId);
    const cohesionById = new Map(metrics.map((m) => [m.categoryId, m.cohesion] as const));

    // Signatures already spoken for: applied/dismissed (resolved) and still-pending structural
    // proposals. Keys minted this run are added as we go so a symmetric merge pair is made once.
    const existingKeys = new Set<string>(
      this.proposals.resolvedStructuralSuppressionKeys(accountId),
    );
    for (const p of this.proposals.listPending(accountId)) {
      if (p.kind !== 'new_category' && p.suppressionKey !== '') existingKeys.add(p.suppressionKey);
    }

    const created: GeneratedStructural[] = [];
    let mergeCandidates = 0;
    let retireCandidates = 0;
    let splitCandidates = 0;
    let skippedExisting = 0;
    let rejectedDistinctPurpose = 0;

    // RETIRE: an active auto category with no assigned mail. User-created categories are never
    // proposed (their retire would be applicable), and a category with any assignment is left alone.
    for (const cat of active) {
      if (cat.source !== 'auto' || cat.emailCount !== 0) continue;
      retireCandidates += 1;
      const key = `retire:${cat.canonicalKey}`;
      if (existingKeys.has(key)) {
        skippedExisting += 1;
        this.logger.debug(
          { accountId, categoryId: cat.id, key },
          'structural: retire already proposed',
        );
        continue;
      }
      const proposal = this.proposals.createStructural({
        accountId,
        kind: 'retire',
        categoryId: cat.id,
        sourceCategoryId: null,
        runId,
        label: `Retire ${cat.label}`,
        description: `${cat.label} has no assigned mail.`,
        canonicalKey: cat.canonicalKey,
        suppressionKey: key,
        embeddingModelId,
        confidence: 1,
        evidence: ['category is empty'],
      });
      existingKeys.add(key);
      created.push({
        id: proposal.id,
        kind: 'retire',
        categoryId: cat.id,
        sourceCategoryId: null,
        suppressionKey: key,
        label: proposal.label,
      });
    }

    // MERGE: two active auto categories whose stored centroids are near-duplicate (high overlap).
    // Both sides must be auto; the larger survives as the target and absorbs the smaller source.
    for (const m of metrics) {
      if (m.overlap === null || m.nearestCategoryId === null || m.overlap < MERGE_MIN_OVERLAP) {
        continue;
      }
      const a = byId.get(m.categoryId);
      const b = byId.get(m.nearestCategoryId);
      if (!a || !b || a.source !== 'auto' || b.source !== 'auto') continue;
      // An empty category (retire candidate) may still carry a stale centroid that reports high
      // overlap. Never merge one: retire is the only structural proposal for an empty category, and
      // merging it would conflict with its own retire proposal for the same category this run.
      if (a.emailCount === 0 || b.emailCount === 0) continue;
      mergeCandidates += 1;
      // Conservative F1 gate: high centroid overlap can pair distinct transactional purposes that
      // collapse in embedding space. Require a deterministic same-purpose label signal.
      if (!sameMergePurpose(a, b)) {
        rejectedDistinctPurpose += 1;
        this.logger.debug(
          { accountId, a: a.id, b: b.id, aLabel: a.label, bLabel: b.label, overlap: m.overlap },
          'structural: merge rejected (no same-purpose signal)',
        );
        continue;
      }
      const key = `merge:${[a.canonicalKey, b.canonicalKey].sort().join('|')}`;
      if (existingKeys.has(key)) {
        skippedExisting += 1;
        this.logger.debug(
          { accountId, a: a.id, b: b.id, key, overlap: m.overlap },
          'structural: merge already proposed or seen this run',
        );
        continue;
      }
      const [target, source] = survivorFirst(a, b);
      const proposal = this.proposals.createStructural({
        accountId,
        kind: 'merge',
        categoryId: target.id,
        sourceCategoryId: source.id,
        runId,
        label: `Merge ${source.label} into ${target.label}`,
        description: `${source.label} and ${target.label} look near-duplicate (overlap ${m.overlap.toFixed(2)}).`,
        canonicalKey: target.canonicalKey,
        suppressionKey: key,
        embeddingModelId,
        confidence: m.overlap,
        evidence: [`overlap ${m.overlap.toFixed(2)} with ${target.label}`],
      });
      existingKeys.add(key);
      created.push({
        id: proposal.id,
        kind: 'merge',
        categoryId: target.id,
        sourceCategoryId: source.id,
        suppressionKey: key,
        label: proposal.label,
      });
    }

    // SPLIT: a loose active auto category whose auto members separate into distinct subclusters. Only
    // categories that are large enough and not already tight are analyzed, and embeddings are loaded
    // once, lazily, only when at least one category clears that cheap pre-filter.
    const splitPre = active.filter(
      (cat) =>
        cat.source === 'auto' &&
        cat.emailCount >= SPLIT_MIN_SIZE &&
        (cohesionById.get(cat.id) ?? 1) < SPLIT_MAX_PARENT_COHESION &&
        !existingKeys.has(`split:${cat.canonicalKey}`),
    );
    if (splitPre.length > 0) {
      const vecByMessage = new Map(
        this.embeddings
          .listForAccount(accountId, embeddingModelId)
          .map((e) => [e.messageId, e.vector] as const),
      );
      for (const cat of splitPre) {
        const parentCohesion = cohesionById.get(cat.id);
        if (parentCohesion === null || parentCohesion === undefined) continue;
        splitCandidates += 1;
        const autoIds = [
          ...this.categories.listCategoryMemberIds(accountId, cat.id, 'auto'),
        ].sort();
        const points: ClusterPoint[] = [];
        for (const id of autoIds) {
          const vec = vecByMessage.get(id);
          if (vec) points.push({ messageId: id, vector: vec });
        }
        const subjectById = new Map(
          this.emails
            .summariesByIds(accountId, autoIds)
            .map((s) => [s.messageId, s.subject ?? ''] as const),
        );
        const children = analyzeSplit(cat.label, points, subjectById, parentCohesion);
        if (!children) continue;
        // Never propose a split whose child would collide with an existing or sibling category; that
        // split could not be applied anyway. applySplit re-checks this before any write.
        const collides = children.some(
          (c) =>
            this.categories.findByCanonicalKey(accountId, c.canonicalKey) !== null ||
            this.categories.findByLabel(accountId, c.label) !== null,
        );
        if (collides) continue;
        const key = `split:${cat.canonicalKey}`;
        const proposal = this.proposals.createSplit(
          {
            accountId,
            kind: 'split',
            categoryId: cat.id,
            sourceCategoryId: null,
            runId,
            label: `Split ${cat.label} into ${children.map((c) => c.label).join(', ')}`,
            description: `${cat.label} looks like ${children.length} distinct groups (cohesion ${parentCohesion.toFixed(2)}).`,
            canonicalKey: cat.canonicalKey,
            suppressionKey: key,
            embeddingModelId,
            confidence: children.reduce((min, c) => Math.min(min, c.confidence), 1),
            evidence: [
              `cohesion ${parentCohesion.toFixed(2)}`,
              ...children.map((c) => `${c.label} (~${c.proposedCount})`),
            ],
          },
          children.map((c) => ({
            label: c.label,
            description: c.description,
            canonicalKey: c.canonicalKey,
            embeddingModelId,
            centroid: c.centroid,
            memberIds: c.memberIds,
            proposedCount: c.proposedCount,
            cohesion: c.cohesion,
            separation: c.separation,
            confidence: c.confidence,
          })),
        );
        existingKeys.add(key);
        created.push({
          id: proposal.id,
          kind: 'split',
          categoryId: cat.id,
          sourceCategoryId: null,
          suppressionKey: key,
          label: proposal.label,
        });
      }
    }

    this.logger.info(
      {
        accountId,
        runId,
        created: created.length,
        mergeCandidates,
        retireCandidates,
        splitCandidates,
        skippedExisting,
        rejectedDistinctPurpose,
      },
      'structural proposal generation',
    );
    return { runId, created, mergeCandidates, retireCandidates, splitCandidates, skippedExisting };
  }
}
