/**
 * Rebuilds a category's stored centroid from its trusted (user-confirmed) member embeddings
 * (Phase 3.1). Deterministic and centroid-only: it never changes a category's label, description,
 * source, status, or canonical key, never mutates email_categories, and never moves an assignment.
 * It recomputes and saves the centroid vector for one category and one embedding model, or leaves the
 * existing centroid untouched when there is too little trusted data.
 */
import type { Logger } from 'pino';
import { meanNormalize } from '../util/vector.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';

/** Fewest vectors a rebuild needs. Below this the existing centroid is left unchanged. */
export const MIN_TRUSTED_REBUILD = 3;

/** Why a rebuild did or did not run. */
export type RebuildStatus = 'rebuilt' | 'insufficient_trusted_data' | 'category_not_found';

/** Outcome of a rebuild attempt. */
export interface RebuildResult {
  status: RebuildStatus;
  categoryId: string;
  vectorsUsed: number;
  usedAutoFallback: boolean;
}

/** How a rebuild may treat auto assignments. */
export interface RebuildOptions {
  /** When true, and only when the category has zero user-confirmed members, rebuild from auto members. */
  allowAutoFallback?: boolean;
}

/** Recomputes a category centroid from its trusted member embeddings. */
export class CategoryCentroidRebuildService {
  constructor(
    private categories: CategoryRepository,
    private embeddings: EmbeddingRepository,
    private logger: Logger,
  ) {}

  /**
   * Rebuild one category's centroid from its user-confirmed members for the given embedding model.
   * Trusted (assigned_by='user') vectors are used by default. Auto vectors are used only as an
   * explicit fallback and only when the category has zero user-confirmed members, so a category the
   * user has ever filed into is never rebuilt from auto assignments (even if those user members'
   * embeddings happen to be missing for the model). When too few usable vectors are available the
   * stored centroid is left untouched and the result is 'insufficient_trusted_data'. Writes nothing
   * but the centroid.
   */
  rebuild(
    accountId: string,
    categoryId: string,
    embeddingModelId: string,
    opts: RebuildOptions = {},
  ): RebuildResult {
    const category = this.categories.findById(categoryId);
    if (!category || category.accountId !== accountId) {
      return { status: 'category_not_found', categoryId, vectorsUsed: 0, usedAutoFallback: false };
    }

    const userMemberIds = this.categories.listCategoryMemberIds(accountId, categoryId, 'user');
    const trusted = this.vectorsFor(accountId, embeddingModelId, userMemberIds);
    if (trusted.length >= MIN_TRUSTED_REBUILD) {
      return this.save(accountId, categoryId, embeddingModelId, trusted, false);
    }

    // Auto fallback only when the caller allows it AND the category has no user-confirmed members at
    // all. Zero usable user vectors is not enough: a category with user members whose embeddings are
    // missing must not be rebuilt from auto assignments.
    if (opts.allowAutoFallback && userMemberIds.length === 0) {
      const autoIds = this.categories.listCategoryMemberIds(accountId, categoryId, 'auto');
      const auto = this.vectorsFor(accountId, embeddingModelId, autoIds);
      if (auto.length >= MIN_TRUSTED_REBUILD) {
        return this.save(accountId, categoryId, embeddingModelId, auto, true);
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
    };
  }

  /** Stored embeddings for the given message ids and model, skipping any that are missing or deleted. */
  private vectorsFor(
    accountId: string,
    embeddingModelId: string,
    messageIds: string[],
  ): Float32Array[] {
    const vectors: Float32Array[] = [];
    for (const messageId of messageIds) {
      const vec = this.embeddings.getEmbedding({ messageId, accountId, modelId: embeddingModelId });
      if (vec) vectors.push(vec);
    }
    return vectors;
  }

  /** Compute the normalized mean and store it as the category centroid, with the count of vectors used. */
  private save(
    accountId: string,
    categoryId: string,
    embeddingModelId: string,
    vectors: Float32Array[],
    usedAutoFallback: boolean,
  ): RebuildResult {
    const centroid = meanNormalize(vectors);
    if (!centroid) {
      return {
        status: 'insufficient_trusted_data',
        categoryId,
        vectorsUsed: 0,
        usedAutoFallback: false,
      };
    }
    this.categories.saveCentroid(categoryId, embeddingModelId, centroid, vectors.length);
    this.logger.info(
      { accountId, categoryId, vectorsUsed: vectors.length, usedAutoFallback },
      'category centroid rebuilt from trusted members',
    );
    return { status: 'rebuilt', categoryId, vectorsUsed: vectors.length, usedAutoFallback };
  }
}
