/**
 * Groups emails by sender and content similarity so near-identical messages
 * share a cluster, letting one LLM judgement be reused across cluster members.
 */
import { cosineFromL2Distance, l2Distance } from '../util/vector.js';
import { senderDomain } from './topic-discovery-service.js';

/** A single email to be clustered, carrying its sender and embedding vector. */
export interface ClusterInput {
  messageId: string;
  fromAddr: string | null;
  vector: Float32Array;
}

/** A group of near-identical emails from one sender, with a chosen representative. */
export interface ContentCluster {
  representativeId: string;
  memberIds: string[];
}

const MAX_CLUSTERS_PER_SENDER = 400;

/**
 * Group emails so near-identical messages from the same sender share a cluster.
 * Two emails join only if their embeddings are at least `threshold` cosine similar,
 * so different templates from one sender land in separate clusters and get judged
 * independently. Each cluster then needs one LLM call, propagated to its members.
 */
export function clusterBySenderAndContent(
  emails: ClusterInput[],
  threshold: number,
): ContentCluster[] {
  const bySender = new Map<string, ClusterInput[]>();
  for (const e of emails) {
    const key = senderDomain(e.fromAddr);
    const arr = bySender.get(key);
    if (arr) arr.push(e);
    else bySender.set(key, [e]);
  }

  const clusters: ContentCluster[] = [];
  for (const group of bySender.values()) {
    const reps: Array<{ vec: Float32Array; cluster: ContentCluster }> = [];

    for (const e of group) {
      let target: ContentCluster | null = null;
      let nearest: { vec: Float32Array; cluster: ContentCluster } | null = null;
      let nearestCos = -1;

      for (const r of reps) {
        const cos = cosineFromL2Distance(l2Distance(e.vector, r.vec));
        if (cos >= threshold) {
          target = r.cluster;
          break;
        }
        if (cos > nearestCos) {
          nearestCos = cos;
          nearest = r;
        }
      }

      if (target) {
        target.memberIds.push(e.messageId);
      } else if (reps.length >= MAX_CLUSTERS_PER_SENDER && nearest) {
        nearest.cluster.memberIds.push(e.messageId);
      } else {
        const cluster: ContentCluster = { representativeId: e.messageId, memberIds: [e.messageId] };
        reps.push({ vec: e.vector, cluster });
        clusters.push(cluster);
      }
    }
  }

  return clusters;
}
