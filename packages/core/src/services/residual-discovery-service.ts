/**
 * Selects residual mail (what active categories could not confidently take) and clusters it into
 * candidate clusters for discovery (Phase 2a). Pure reads only: no writes, no LLM, no category
 * creation, no assignment. Naming, validation, persistence, and apply are later Phase 2 sub-phases.
 */
import { cosineSimilarity } from '../util/vector.js';
import { stableHash } from '../util/rand.js';
import {
  clusterResidual,
  type ClusterPoint,
  type DiscoveredCluster,
} from './discovery-clustering.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';

/**
 * An auto-assigned email is residual (low-confidence) when the fresh cosine between its embedding and
 * its best assigned active-category centroid is below this floor. Computed live, never read from
 * stored assignment confidence, which can be synthetic for gate or LLM assignments.
 */
export const RESIDUAL_COSINE_FLOOR = 0.55;

/** Selects and clusters residual mail. Read-only; produces candidate clusters, persists nothing. */
export class ResidualDiscoveryService {
  constructor(
    private embeddings: EmbeddingRepository,
    private categories: CategoryRepository,
    private multiPrototypeEnabled: () => boolean = () => false,
  ) {}

  /**
   * The residual set: uncategorized emails, plus emails whose only assignments are automatic and
   * whose fresh cosine to the best assigned active centroid is below RESIDUAL_COSINE_FLOOR. Any email
   * with a user assignment is never residual, so user corrections are always respected.
   */
  selectResidual(accountId: string, embeddingModelId: string): ClusterPoint[] {
    const userAssigned = this.categories.getUserAssignedMessageIds(accountId);
    // Effective prototypes per category: sub-prototypes when the flag is on and present, else the
    // aggregate. Coverage of an email is the MAX cosine over its assigned category's prototypes, so mail
    // that a secondary sub-pattern covers is no longer wrongly declared residual. Flag off -> one
    // aggregate vector per category -> identical to the pre-Phase-4 behavior.
    const prototypesByCat = new Map<string, Float32Array[]>();
    for (const p of this.categories.getEffectivePrototypeEntries(
      accountId,
      embeddingModelId,
      this.multiPrototypeEnabled(),
    )) {
      const arr = prototypesByCat.get(p.categoryId);
      if (arr) arr.push(p.vector);
      else prototypesByCat.set(p.categoryId, [p.vector]);
    }
    const autoByMsg = new Map<string, string[]>();
    for (const a of this.categories.listAutoAssignments(accountId)) {
      const arr = autoByMsg.get(a.messageId);
      if (arr) arr.push(a.categoryId);
      else autoByMsg.set(a.messageId, [a.categoryId]);
    }

    const residual: ClusterPoint[] = [];
    for (const e of this.embeddings.listForAccount(accountId, embeddingModelId)) {
      if (userAssigned.has(e.messageId)) continue; // user-assigned mail is never residual
      const autoCats = autoByMsg.get(e.messageId);
      if (!autoCats || autoCats.length === 0) {
        residual.push({ messageId: e.messageId, vector: e.vector }); // uncategorized
        continue;
      }
      let best = -Infinity;
      for (const catId of autoCats) {
        const protos = prototypesByCat.get(catId);
        if (protos) for (const v of protos) best = Math.max(best, cosineSimilarity(e.vector, v));
      }
      if (best < RESIDUAL_COSINE_FLOOR) {
        residual.push({ messageId: e.messageId, vector: e.vector }); // low-confidence auto assignment
      }
    }
    return residual;
  }

  /**
   * Cluster the residual set into candidate clusters, deterministically per account and model. Pure
   * computation: reads embeddings and centroids, writes nothing.
   */
  discover(accountId: string, embeddingModelId: string): DiscoveredCluster[] {
    const residual = this.selectResidual(accountId, embeddingModelId);
    return clusterResidual(residual, stableHash(`${accountId}|${embeddingModelId}`));
  }
}
