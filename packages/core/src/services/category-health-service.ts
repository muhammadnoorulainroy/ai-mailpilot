/**
 * Read-only category health metrics (Phase 3.2): the sensor layer for self-healing. For each active
 * category it computes deterministic signals from the members' embeddings and the stored centroids,
 * with no writes and no mutation of any kind. These metrics feed the split/merge/retire detection in
 * a later phase; nothing here decides or applies a structural change.
 *
 * Definitions:
 * - size: total assigned members; userMemberCount: the user-confirmed subset.
 * - cohesion: mean cosine of each member embedding to the members' own recomputed centroid (how
 *   tight the category's content is). Null when no member embedding is available.
 * - separation / overlap: measured against the other active categories' STORED centroids in one pass;
 *   overlap is the highest cosine to any other centroid and separation is 1 - overlap, with the
 *   nearest category id. Null when this category has no stored centroid.
 * - drift: 1 - cosine(stored centroid, members' recomputed centroid), i.e. how stale the stored
 *   centroid is relative to its current members. Null when either vector is unavailable. This is the
 *   read-only proxy for centroid movement; true across-rebuild history would need persistence.
 * - coverage: this category's share of all assigned members across the account's active categories.
 */
import { cosineSimilarity, meanNormalize } from '../util/vector.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';

/** Deterministic health signals for one active category. */
export interface CategoryHealth {
  categoryId: string;
  label: string;
  size: number;
  userMemberCount: number;
  cohesion: number | null;
  separation: number | null;
  overlap: number | null;
  nearestCategoryId: string | null;
  drift: number | null;
  coverage: number;
}

/** Computes read-only health metrics for an account's active categories. */
export class CategoryHealthService {
  constructor(
    private categories: CategoryRepository,
    private embeddings: EmbeddingRepository,
  ) {}

  /**
   * Health metrics for every active category on the account, using the given embedding model. Pure
   * read: loads members, embeddings, and stored centroids, and writes nothing.
   */
  metricsForAccount(accountId: string, embeddingModelId: string): CategoryHealth[] {
    const active = this.categories.listActive(accountId);
    const storedById = new Map(
      this.categories
        .getCentroidEntries(accountId, embeddingModelId)
        .map((c) => [c.categoryId, c.vector] as const),
    );
    const vecByMessage = new Map(
      this.embeddings
        .listForAccount(accountId, embeddingModelId)
        .map((e) => [e.messageId, e.vector] as const),
    );

    // First pass: gather each category's member counts and the centroid recomputed from its members.
    const perCategory = active.map((cat) => {
      const userIds = this.categories.listCategoryMemberIds(accountId, cat.id, 'user');
      const autoIds = this.categories.listCategoryMemberIds(accountId, cat.id, 'auto');
      const memberVectors: Float32Array[] = [];
      for (const id of [...userIds, ...autoIds]) {
        const vec = vecByMessage.get(id);
        if (vec) memberVectors.push(vec);
      }
      return {
        cat,
        size: userIds.length + autoIds.length,
        userMemberCount: userIds.length,
        memberVectors,
        recomputed: memberVectors.length > 0 ? meanNormalize(memberVectors) : null,
      };
    });

    const totalMembers = perCategory.reduce((sum, p) => sum + p.size, 0);

    // Second pass: separation, overlap, cohesion, drift, coverage from the gathered data.
    return perCategory.map((p) => {
      const stored = storedById.get(p.cat.id) ?? null;

      let overlap: number | null = null;
      let separation: number | null = null;
      let nearestCategoryId: string | null = null;
      if (stored) {
        let maxCos = -Infinity;
        for (const other of active) {
          if (other.id === p.cat.id) continue;
          const otherVec = storedById.get(other.id);
          if (!otherVec || otherVec.length !== stored.length) continue;
          const cos = cosineSimilarity(stored, otherVec);
          if (cos > maxCos) {
            maxCos = cos;
            nearestCategoryId = other.id;
          }
        }
        if (maxCos === -Infinity) {
          overlap = 0;
          separation = 1;
          nearestCategoryId = null;
        } else {
          overlap = maxCos;
          separation = 1 - maxCos;
        }
      }

      const cohesion =
        p.recomputed && p.memberVectors.length > 0
          ? p.memberVectors.reduce((sum, v) => sum + cosineSimilarity(v, p.recomputed!), 0) /
            p.memberVectors.length
          : null;

      const drift =
        stored && p.recomputed && stored.length === p.recomputed.length
          ? 1 - cosineSimilarity(stored, p.recomputed)
          : null;

      return {
        categoryId: p.cat.id,
        label: p.cat.label,
        size: p.size,
        userMemberCount: p.userMemberCount,
        cohesion,
        separation,
        overlap,
        nearestCategoryId,
        drift,
        coverage: totalMembers > 0 ? p.size / totalMembers : 0,
      };
    });
  }
}
