/**
 * Rebuilds a category's stored centroid from its trusted (user-confirmed) member embeddings
 * (Phase 3.1, extended for Phase 4). Deterministic and centroid-only: it never changes a category's
 * label, description, source, status, or canonical key, never mutates email_categories, and never
 * moves an assignment.
 *
 * It always recomputes the AGGREGATE centroid (prototype_index 0) - the normalized mean of all trusted
 * members, identical to the pre-Phase-4 single centroid. When the multi-prototype feature flag is on
 * and the trusted members clearly split into two or more tight, well-separated groups, it ADDITIONALLY
 * writes sub-prototypes (1..K); otherwise it removes any stale sub-prototypes and keeps only the
 * aggregate. Below the trusted floor it leaves the existing prototype set untouched.
 */
import type { Logger } from 'pino';
import { meanNormalize, cosineSimilarity } from '../util/vector.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import { clusterResidual, type ClusterPoint } from './discovery-clustering.js';

/** Fewest vectors a rebuild needs. Below this the existing centroid set is left unchanged. Also the
 * per-sub-prototype member floor: a sub-prototype must be backed by at least this many members. */
export const MIN_TRUSTED_REBUILD = 3;
/** Cap on sub-prototypes per category, so a category is never fragmented into many thin prototypes. */
export const MAX_SUB_PROTOTYPES = 4;
/** A sub-prototype cluster must be at least this internally coherent to be kept. */
export const SUB_PROTOTYPE_MIN_COHESION = 0.55;
/** Sub-prototypes must differ by at least this cosine gap (pairwise minimum) to be worth keeping. */
export const SUB_PROTOTYPE_MIN_SEPARATION = 0.2;
/** Fixed seed so partitioning a given member set is deterministic across runs. */
export const SUB_PROTOTYPE_CLUSTER_SEED = 4242;

/** Why a rebuild did or did not run. */
export type RebuildStatus = 'rebuilt' | 'insufficient_trusted_data' | 'category_not_found';

/** Outcome of a rebuild attempt. */
export interface RebuildResult {
  status: RebuildStatus;
  categoryId: string;
  vectorsUsed: number;
  usedAutoFallback: boolean;
  /** Number of sub-prototypes (>= 1) written this rebuild; 0 means aggregate-only. */
  subPrototypeCount: number;
}

/** How a rebuild may treat auto assignments. */
export interface RebuildOptions {
  /** When true, and only when the category has zero user-confirmed members, rebuild from auto members. */
  allowAutoFallback?: boolean;
}

interface MemberVector {
  messageId: string;
  vector: Float32Array;
}

/** Recomputes a category centroid (and optional sub-prototypes) from its trusted member embeddings. */
export class CategoryCentroidRebuildService {
  constructor(
    private categories: CategoryRepository,
    private embeddings: EmbeddingRepository,
    private logger: Logger,
    private multiPrototypeEnabled: () => boolean = () => false,
  ) {}

  /**
   * Rebuild one category's centroid set from its user-confirmed members for the given embedding model.
   * Trusted (assigned_by='user') vectors are used by default. Auto vectors are used only as an
   * explicit fallback and only when the category has zero user-confirmed members. When too few usable
   * vectors are available the stored centroid set is left untouched ('insufficient_trusted_data').
   * Always writes the aggregate (prototype 0); adds sub-prototypes only under the feature flag.
   */
  rebuild(
    accountId: string,
    categoryId: string,
    embeddingModelId: string,
    opts: RebuildOptions = {},
  ): RebuildResult {
    const category = this.categories.findById(categoryId);
    if (!category || category.accountId !== accountId) {
      return {
        status: 'category_not_found',
        categoryId,
        vectorsUsed: 0,
        usedAutoFallback: false,
        subPrototypeCount: 0,
      };
    }

    const userMemberIds = this.categories.listCategoryMemberIds(accountId, categoryId, 'user');
    const trusted = this.vectorsFor(accountId, embeddingModelId, userMemberIds);
    if (trusted.length >= MIN_TRUSTED_REBUILD) {
      return this.saveSet(accountId, categoryId, embeddingModelId, trusted, false);
    }

    // Auto fallback only when the caller allows it AND the category has no user-confirmed members at
    // all. Zero usable user vectors is not enough: a category with user members whose embeddings are
    // missing must not be rebuilt from auto assignments.
    if (opts.allowAutoFallback && userMemberIds.length === 0) {
      const autoIds = this.categories.listCategoryMemberIds(accountId, categoryId, 'auto');
      const auto = this.vectorsFor(accountId, embeddingModelId, autoIds);
      if (auto.length >= MIN_TRUSTED_REBUILD) {
        return this.saveSet(accountId, categoryId, embeddingModelId, auto, true);
      }
    }

    this.logger.info(
      { accountId, categoryId, userMembers: userMemberIds.length, trusted: trusted.length },
      'category centroid rebuild: insufficient trusted data, centroid unchanged',
    );
    return {
      status: 'insufficient_trusted_data',
      categoryId,
      vectorsUsed: trusted.length,
      usedAutoFallback: false,
      subPrototypeCount: 0,
    };
  }

  /** Stored embeddings for the given message ids and model, skipping any that are missing or deleted. */
  private vectorsFor(
    accountId: string,
    embeddingModelId: string,
    messageIds: string[],
  ): MemberVector[] {
    const members: MemberVector[] = [];
    for (const messageId of messageIds) {
      const vec = this.embeddings.getEmbedding({ messageId, accountId, modelId: embeddingModelId });
      if (vec) members.push({ messageId, vector: vec });
    }
    return members;
  }

  /**
   * Write the aggregate centroid (prototype 0) from all members, plus sub-prototypes when the flag is
   * on and the members clearly separate into groups. `savePrototypeSet` is atomic and always keeps the
   * aggregate, replacing any prior sub-prototypes; an empty sub set removes stale sub-prototypes.
   */
  private saveSet(
    accountId: string,
    categoryId: string,
    embeddingModelId: string,
    members: MemberVector[],
    usedAutoFallback: boolean,
  ): RebuildResult {
    const aggregate = meanNormalize(members.map((m) => m.vector));
    if (!aggregate) {
      return {
        status: 'insufficient_trusted_data',
        categoryId,
        vectorsUsed: 0,
        usedAutoFallback: false,
        subPrototypeCount: 0,
      };
    }

    const subs = this.multiPrototypeEnabled() ? this.partition(members) : [];
    this.categories.savePrototypeSet(
      categoryId,
      embeddingModelId,
      aggregate,
      members.length,
      subs.map((s) => ({ vector: s.vector, emailCount: s.count })),
    );
    this.logger.info(
      {
        accountId,
        categoryId,
        vectorsUsed: members.length,
        usedAutoFallback,
        subPrototypes: subs.length,
      },
      'category centroid rebuilt from trusted members',
    );
    return {
      status: 'rebuilt',
      categoryId,
      vectorsUsed: members.length,
      usedAutoFallback,
      subPrototypeCount: subs.length,
    };
  }

  /**
   * Deterministically partition members into sub-prototypes. Returns the kept sub-cluster centroids and
   * sizes, or an empty array (aggregate-only) when the members are unimodal, the groups are too thin,
   * or they are not well separated. "Grow only on strong evidence."
   */
  private partition(members: MemberVector[]): Array<{ vector: Float32Array; count: number }> {
    const points: ClusterPoint[] = [...members]
      .sort((a, b) => a.messageId.localeCompare(b.messageId))
      .map((m) => ({ messageId: m.messageId, vector: m.vector }));
    const minId = (ids: string[]): string => ids.reduce((m, x) => (x < m ? x : m), ids[0]!);
    const clusters = clusterResidual(points, SUB_PROTOTYPE_CLUSTER_SEED)
      .filter((c) => c.size >= MIN_TRUSTED_REBUILD && c.cohesion >= SUB_PROTOTYPE_MIN_COHESION)
      .sort((a, b) => b.size - a.size || minId(a.memberIds).localeCompare(minId(b.memberIds)))
      .slice(0, MAX_SUB_PROTOTYPES);
    if (clusters.length < 2) return [];

    let minPairSeparation = 1;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const gap = 1 - cosineSimilarity(clusters[i]!.centroid, clusters[j]!.centroid);
        if (gap < minPairSeparation) minPairSeparation = gap;
      }
    }
    if (minPairSeparation < SUB_PROTOTYPE_MIN_SEPARATION) return [];

    return clusters.map((c) => ({ vector: c.centroid, count: c.size }));
  }
}
