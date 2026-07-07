/**
 * Generates structural (split / merge / retire) review-queue proposals from the read-only Phase 3.2
 * health metrics (Phase 3.3 detection). It only proposes obvious, safe cases and never applies
 * anything: a merge when two active auto categories are near-duplicate (high overlap / low
 * separation), a retire when an active auto category is empty, and a split when a loose active auto
 * category separates into two or more tight, well-separated subclusters that the local model can name
 * as distinct, non-overlapping purposes.
 *
 * Split child labels are produced by the SAME local naming + deterministic validation gate that
 * discovery uses (never raw keyphrases), so a child is only kept when the model names a real purpose
 * and that name survives the gate (not vague, not a sender/brand, not overlapping the parent or an
 * existing category). A split with fewer than two survivable children is abandoned. Naming is
 * local-only unless cloud discovery is explicitly enabled.
 *
 * Everything is user-approved on apply. This service writes only category_proposals rows (via the
 * existing createStructural / createSplit paths) and never touches a category, assignment, or
 * centroid. It skips user-created categories entirely and deduplicates on the deterministic
 * suppressionKey so a purpose that is already pending, applied, or dismissed is never re-proposed.
 */
import { randomUUID } from 'node:crypto';
import type { Logger } from 'pino';
import type { LlmClient } from '../llm/client.js';
import type { LlmConfig } from '../config/schema.js';
import {
  canonicalKeyBase,
  type CategoryRepository,
  type CategoryWithCount,
} from '../repositories/category-repository.js';
import type { CategoryProposalRepository } from '../repositories/category-proposal-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import type { EmailRepository, EmailSummary } from '../repositories/email-repository.js';
import type { CategoryHealthService } from './category-health-service.js';
import { purposeSignature } from './categorize-strategy.js';
import { isNearDuplicateLabel, domainFrequency, brandTokens } from './topic-discovery-service.js';
import {
  clusterResidual,
  type ClusterPoint,
  type DiscoveredCluster,
} from './discovery-clustering.js';
import {
  clusterKeyphrases,
  validateBatch,
  type NamedCandidate,
  type ActiveCategoryRef,
} from './discovery-candidates.js';
import {
  buildNamingMessages,
  parseNamedCandidates,
  NAMING_SAMPLE_PER_CLUSTER,
  type ClusterNamingInput,
} from './discovery-naming.js';
import { assertDiscoveryLocal, discoveryProvider } from './discovery-guard.js';
import { cosineSimilarity } from '../util/vector.js';

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
/** Most split candidates named per run, to bound how many local-model naming calls a run makes. */
export const SPLIT_MAX_NAMED_CANDIDATES = 6;
/** Output-token budget for one split's child-naming call (a split names at most SPLIT_MAX_CHILDREN). */
const SPLIT_NAMING_OUTPUT_TOKENS = 1200;

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

/** A named, validated split child ready to persist. */
interface NamedSplitChild {
  label: string;
  description: string;
  canonicalKey: string;
  centroid: Float32Array;
  memberIds: string[];
  proposedCount: number;
  cohesion: number;
  separation: number;
  confidence: number;
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
 * Detect whether a loose category's auto members separate into distinct subclusters worth naming.
 * Pure and deterministic given the seeded clustering: it subclusters the members and keeps only the
 * viable ones (enough members, internally coherent, meaningfully tighter than the parent) that are
 * also well separated from each other. Returns the kept subclusters (unlabelled), or null when there
 * are not at least two. Naming and the quality gate happen afterwards, on these subclusters.
 */
function detectSplitSubclusters(
  points: ClusterPoint[],
  parentCohesion: number,
): DiscoveredCluster[] | null {
  if (points.length < SPLIT_MIN_SIZE) return null;

  const minId = (ids: string[]): string => ids.reduce((m, x) => (x < m ? x : m), ids[0]!);
  const clusters = clusterResidual(points, SPLIT_CLUSTER_SEED)
    .filter(
      (c) =>
        c.size >= SPLIT_MIN_CHILD_SIZE &&
        c.cohesion >= Math.max(SPLIT_MIN_CHILD_COHESION, parentCohesion + SPLIT_MIN_COHESION_GAIN),
    )
    .sort((a, b) => b.size - a.size || minId(a.memberIds).localeCompare(minId(b.memberIds)))
    .slice(0, SPLIT_MAX_CHILDREN);
  if (clusters.length < 2) return null;

  let minPairSeparation = 1;
  for (let i = 0; i < clusters.length; i++) {
    for (let j = i + 1; j < clusters.length; j++) {
      const gap = 1 - cosineSimilarity(clusters[i]!.centroid, clusters[j]!.centroid);
      if (gap < minPairSeparation) minPairSeparation = gap;
    }
  }
  if (minPairSeparation < SPLIT_MIN_CHILD_SEPARATION) return null;

  return clusters;
}

/** Detects safe merge, retire, and split proposals from health metrics. Never applies a change. */
export class StructuralProposalService {
  constructor(
    private categories: CategoryRepository,
    private proposals: CategoryProposalRepository,
    private health: CategoryHealthService,
    private embeddings: EmbeddingRepository,
    private emails: EmailRepository,
    private llm: LlmClient,
    private getConfig: () => LlmConfig,
    private logger: Logger,
  ) {}

  /**
   * Scan the account's active auto categories and enqueue merge/retire/split proposals for the obvious
   * cases. Reads health metrics, embeddings, and existing proposals, and (for split naming) calls the
   * local model; writes only proposal rows. Idempotent: a signature already pending, applied, or
   * dismissed is skipped.
   */
  async generate(
    accountId: string,
    embeddingModelId: string,
    generationModelId: string,
  ): Promise<StructuralGenerateResult> {
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

    // SPLIT: a loose active auto category whose auto members separate into distinct subclusters that
    // the local model can name as real, non-overlapping purposes. Only categories that are large
    // enough and not already tight are analyzed; embeddings are loaded once, lazily, and the number of
    // naming calls per run is bounded by SPLIT_MAX_NAMED_CANDIDATES.
    const splitPre = active
      .filter(
        (cat) =>
          cat.source === 'auto' &&
          cat.emailCount >= SPLIT_MIN_SIZE &&
          (cohesionById.get(cat.id) ?? 1) < SPLIT_MAX_PARENT_COHESION &&
          !existingKeys.has(`split:${cat.canonicalKey}`),
      )
      .sort((a, b) => b.emailCount - a.emailCount)
      .slice(0, SPLIT_MAX_NAMED_CANDIDATES);
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
        const subclusters = detectSplitSubclusters(points, parentCohesion);
        if (!subclusters) continue;
        const children = await this.nameSplitChildren(
          accountId,
          embeddingModelId,
          generationModelId,
          subclusters,
          purposeSignature(cat.label, cat.description),
        );
        if (!children) continue;
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

  /**
   * Name a split's subclusters with the local model and keep only those whose names survive the
   * deterministic validation gate (not vague, not a sender/brand, not overlapping the parent or an
   * existing category, distinct from each other). Returns the kept children, or null when fewer than
   * two survive or the naming call fails (in which case no split is proposed). Naming is local-only
   * unless cloud discovery is explicitly enabled; a naming failure never aborts the whole run.
   *
   * `parentSig` is the parent category's canonical purpose group (or null for a generic/mixed bucket).
   * When the parent is a recognized purpose, the split is only kept if a child introduces a genuinely
   * DIFFERENT purpose; children that are all sub-variants of the parent's own purpose (Shipping ->
   * Package / Order / Delivered) are fragmentation, not a real split, and the split is abandoned.
   */
  private async nameSplitChildren(
    accountId: string,
    embeddingModelId: string,
    generationModelId: string,
    subclusters: DiscoveredCluster[],
    parentSig: number | null,
  ): Promise<NamedSplitChild[] | null> {
    const samples = subclusters.map((c) => this.sampleSubcluster(accountId, c));
    const keyphrases = clusterKeyphrases(samples.map((s) => s.subjects));
    const namingInputs: ClusterNamingInput[] = samples.map((s, i) => ({
      index: i,
      size: s.cluster.size,
      keyphrases: keyphrases[i] ?? [],
      sampleSubjects: s.subjects,
      senderHints: s.senderTokens,
    }));

    const cfg = this.getConfig();
    const provider = discoveryProvider(cfg);
    assertDiscoveryLocal(cfg, provider);
    const local = provider === 'main';
    const model = local ? generationModelId : cfg.chatModel || generationModelId;
    let raw: string;
    try {
      raw = await this.llm.chat({
        model,
        provider,
        messages: buildNamingMessages(namingInputs, { noThink: local }),
        responseFormat: 'json_object',
        temperature: 0.2,
        maxTokens: SPLIT_NAMING_OUTPUT_TOKENS,
        think: local ? false : undefined,
      });
    } catch (err) {
      this.logger.warn(
        { accountId, err: String(err) },
        'structural: split child naming failed, skipping split',
      );
      return null;
    }

    const parsed = parseNamedCandidates(raw, subclusters.length);
    const candidates: NamedCandidate[] = parsed.map((p) => ({
      clusterIndex: p.clusterIndex,
      action: p.action,
      label: p.label,
      description: p.description,
      suggestedKey: p.suggestedKey,
      evidence: keyphrases[p.clusterIndex] ?? [],
    }));

    const activeCategories = this.activeCategoryRefs(accountId, embeddingModelId);
    const totalResidual = subclusters.reduce((n, c) => n + c.size, 0);
    const results = validateBatch(candidates, (c) => ({
      cluster: subclusters[c.clusterIndex]!,
      senderTokens: samples[c.clusterIndex]?.senderTokens ?? [],
      totalResidual,
      activeCategories,
      existingSuggestedLabels: [],
      existingSuggestedKeys: [],
    }));

    const children: NamedSplitChild[] = [];
    const usedKeys = new Set<string>();
    for (const r of results) {
      if (!r.verdict.accepted) continue;
      const cluster = subclusters[r.candidate.clusterIndex]!;
      const label = r.candidate.label.trim();
      const canonicalKey = canonicalKeyBase(label);
      // A child must have a distinct key and must not collide with an existing category (which would
      // block apply anyway, and usually means those emails should merge into it, not form a split).
      if (canonicalKey === '' || usedKeys.has(canonicalKey)) continue;
      if (
        this.categories.findByCanonicalKey(accountId, canonicalKey) !== null ||
        this.categories.findByLabel(accountId, label) !== null
      ) {
        continue;
      }
      usedKeys.add(canonicalKey);
      children.push({
        label,
        description: r.candidate.description || `${label} mail.`,
        canonicalKey,
        centroid: cluster.centroid,
        memberIds: cluster.memberIds,
        proposedCount: cluster.size,
        cohesion: cluster.cohesion,
        separation: cluster.separation,
        confidence: r.verdict.confidence,
      });
    }
    if (children.length < 2) return null;

    // Parent-purpose gate: when the parent is itself a recognized purpose (a mature category, not a
    // generic/mixed bucket), only split it if some child introduces a genuinely DIFFERENT purpose.
    // Children that all map to the parent's own purpose (or to no distinguishing purpose) are
    // sub-variants, not a real split, so the split is abandoned.
    if (parentSig !== null) {
      const introducesNewPurpose = children.some((c) => {
        const sig = purposeSignature(c.label, c.description);
        return sig !== null && sig !== parentSig;
      });
      if (!introducesNewPurpose) return null;
    }
    return children;
  }

  /**
   * A bounded, deterministic sample of one subcluster: the first NAMING_SAMPLE_PER_CLUSTER member
   * subjects (member order is deterministic) and the brand tokens of their dominant sender domains.
   */
  private sampleSubcluster(
    accountId: string,
    cluster: DiscoveredCluster,
  ): { cluster: DiscoveredCluster; subjects: string[]; senderTokens: string[] } {
    const sampleIds = cluster.memberIds.slice(0, NAMING_SAMPLE_PER_CLUSTER);
    const byId = new Map(
      this.emails.summariesByIds(accountId, sampleIds).map((s) => [s.messageId, s] as const),
    );
    const ordered = sampleIds
      .map((id) => byId.get(id))
      .filter((s): s is EmailSummary => s !== undefined);
    const subjects = ordered.map((s) => s.subject?.trim() ?? '').filter((s) => s.length > 0);
    const freq = domainFrequency(ordered.map((s) => ({ fromAddr: s.fromAddr })));
    return { cluster, subjects, senderTokens: [...brandTokens(freq)] };
  }

  /** Active categories with their stored centroids, as the gate needs them for overlap checks. */
  private activeCategoryRefs(accountId: string, embeddingModelId: string): ActiveCategoryRef[] {
    const centroids = new Map(
      this.categories
        .getCentroidEntries(accountId, embeddingModelId)
        .map((c) => [c.categoryId, c.vector] as const),
    );
    return this.categories.listActive(accountId).map((c) => ({
      label: c.label,
      description: c.description,
      centroid: centroids.get(c.id) ?? null,
      createdBy: c.source,
    }));
  }
}
