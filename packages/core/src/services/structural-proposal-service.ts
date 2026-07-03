/**
 * Generates structural (merge / retire) review-queue proposals from the read-only Phase 3.2 health
 * metrics (Phase 3.3 detection). It only proposes obvious, safe cases and never applies anything: a
 * merge when two active auto categories are near-duplicate (high overlap / low separation), and a
 * retire when an active auto category is empty. Split detection is deliberately out of scope here.
 *
 * Everything is user-approved on apply. This service writes only category_proposals rows (via the
 * existing createStructural path) and never touches a category, assignment, or centroid. It skips
 * user-created categories entirely (their structural changes are applicable, so they must not be
 * auto-suggested), and it deduplicates on the deterministic suppressionKey so a purpose that is
 * already pending, applied, or dismissed is never re-proposed.
 */
import { randomUUID } from 'node:crypto';
import type { Logger } from 'pino';
import type { CategoryRepository, CategoryWithCount } from '../repositories/category-repository.js';
import type { CategoryProposalRepository } from '../repositories/category-proposal-repository.js';
import type { CategoryHealthService } from './category-health-service.js';

/** Lowest overlap (cosine of stored centroids) at which two auto categories are proposed for merge. */
export const MERGE_MIN_OVERLAP = 0.9;

/** One structural proposal produced by a generate run. */
export interface GeneratedStructural {
  id: string;
  kind: 'merge' | 'retire';
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
  skippedExisting: number;
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

/** Detects safe merge and retire proposals from health metrics. Never applies a change. */
export class StructuralProposalService {
  constructor(
    private categories: CategoryRepository,
    private proposals: CategoryProposalRepository,
    private health: CategoryHealthService,
    private logger: Logger,
  ) {}

  /**
   * Scan the account's active auto categories and enqueue merge/retire proposals for the obvious
   * cases. Reads health metrics and existing proposals; writes only proposal rows. Idempotent: a
   * signature already pending, applied, or dismissed is skipped.
   */
  generate(accountId: string, embeddingModelId: string): StructuralGenerateResult {
    const runId = randomUUID();
    const active = this.categories.listActive(accountId);
    const byId = new Map(active.map((c) => [c.id, c] as const));
    const metrics = this.health.metricsForAccount(accountId, embeddingModelId);

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
    let skippedExisting = 0;

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
      mergeCandidates += 1;
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

    this.logger.info(
      {
        accountId,
        runId,
        created: created.length,
        mergeCandidates,
        retireCandidates,
        skippedExisting,
      },
      'structural proposal generation',
    );
    return { runId, created, mergeCandidates, retireCandidates, skippedExisting };
  }
}
