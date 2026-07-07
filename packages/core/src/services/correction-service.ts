/**
 * Service that applies user category corrections and runs the centroid learning
 * loop, so manual filing both records the assignment and improves future
 * auto-categorization.
 */
import type { Database } from 'better-sqlite3';
import { EMBEDDING_DIM } from '../db/schema.js';
import { runningMeanUpdate, cosineSimilarity } from '../util/vector.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';

/**
 * Apply user corrections and run the learning loop. Filing an email replaces its
 * assignments with the user's choice and nudges each newly chosen category centroid
 * toward the email embedding so future similar mail auto-lands there. The whole
 * operation runs in one transaction so a failure can never leave the assignment
 * committed while centroids are only partially updated.
 */
export class CorrectionService {
  /** Creates the service with the repositories it updates when a user corrects a category. */
  constructor(
    private db: Database,
    private categories: CategoryRepository,
    private embeddings: EmbeddingRepository,
    private multiPrototypeEnabled: () => boolean = () => false,
  ) {}

  /**
   * Replace an email's category assignments with the user's choice and nudge each
   * newly chosen category centroid toward the email embedding, all in one transaction.
   */
  setUserCategories(
    accountId: string,
    messageId: string,
    categoryIds: string[],
    embeddingModelId: string,
  ): { applied: number; centroidsUpdated: number } {
    const unique = [...new Set(categoryIds)];

    // Swaps in the new user assignments, clears stale decisions, and nudges each
    // newly chosen category centroid toward the email embedding within one transaction.
    const apply = this.db.transaction((): { applied: number; centroidsUpdated: number } => {
      const priorUser = new Set(
        this.categories
          .getEmailCategories(messageId, accountId)
          .filter((a) => a.assignedBy === 'user')
          .map((a) => a.categoryId),
      );

      const now = Date.now();
      this.categories.replaceEmailAssignments(
        messageId,
        accountId,
        unique.map((categoryId) => ({
          messageId,
          accountId,
          categoryId,
          confidence: 1,
          assignedBy: 'user' as const,
          assignedAt: now,
        })),
      );
      this.categories.clearDecisionsForEmail(messageId, accountId);

      const emailVec = this.embeddings.getEmbedding({
        messageId,
        accountId,
        modelId: embeddingModelId,
      });
      if (!emailVec) {
        return { applied: unique.length, centroidsUpdated: 0 };
      }

      const useMulti = this.multiPrototypeEnabled();
      let centroidsUpdated = 0;
      for (const categoryId of unique) {
        if (priorUser.has(categoryId)) continue;
        // Always nudge the aggregate (prototype 0), exactly as before.
        const current = this.categories.getCentroid(categoryId, embeddingModelId);
        const base = current ?? { vector: new Float32Array(EMBEDDING_DIM), emailCount: 0 };
        const updated = runningMeanUpdate(base.vector, base.emailCount, emailVec);
        this.categories.saveCentroid(categoryId, embeddingModelId, updated, base.emailCount + 1);
        centroidsUpdated += 1;

        // When multi-prototype is on and the category already has sub-prototypes, also nudge the
        // NEAREST one toward this correction. We never spawn a new sub-prototype from a single
        // correction, so K cannot grow here; siblings are left unchanged.
        if (useMulti) {
          const subs = this.categories
            .getPrototypes(categoryId, embeddingModelId)
            .filter((p) => p.prototypeIndex >= 1);
          if (subs.length > 0) {
            let nearest = subs[0]!;
            let bestSim = cosineSimilarity(emailVec, nearest.vector);
            for (let i = 1; i < subs.length; i++) {
              const sim = cosineSimilarity(emailVec, subs[i]!.vector);
              if (sim > bestSim) {
                bestSim = sim;
                nearest = subs[i]!;
              }
            }
            const nudged = runningMeanUpdate(nearest.vector, nearest.emailCount, emailVec);
            this.categories.saveCentroid(
              categoryId,
              embeddingModelId,
              nudged,
              nearest.emailCount + 1,
              nearest.prototypeIndex,
            );
          }
        }
      }

      return { applied: unique.length, centroidsUpdated };
    });

    return apply();
  }
}
