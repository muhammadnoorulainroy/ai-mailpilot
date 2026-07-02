/**
 * Names the on-device residual clusters with the local model and runs every proposed name through
 * the deterministic validation gate (Phase 2b). Read-only: it creates no category, assigns no email,
 * and writes nothing. The model only proposes names; the gate decides acceptance. Persistence of the
 * accepted proposals and the review UI are later Phase 2 sub-phases.
 */
import type { Logger } from 'pino';
import type { LlmClient } from '../llm/client.js';
import type { LlmConfig } from '../config/schema.js';
import type { EmailRepository, EmailSummary } from '../repositories/email-repository.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import { assertDiscoveryLocal, discoveryProvider } from './discovery-guard.js';
import { domainFrequency, brandTokens } from './topic-discovery-service.js';
import {
  clusterKeyphrases,
  validateBatch,
  rankAccepted,
  type NamedCandidate,
  type ActiveCategoryRef,
  type RejectReason,
} from './discovery-candidates.js';
import {
  buildNamingMessages,
  parseNamedCandidates,
  NAMING_SAMPLE_PER_CLUSTER,
  type ClusterNamingInput,
} from './discovery-naming.js';
import type { ResidualDiscoveryService } from './residual-discovery-service.js';
import type { DiscoveredCluster } from './discovery-clustering.js';

/** Largest clusters named per run, so the prompt and the model's answer stay bounded. */
export const NAMING_MAX_CLUSTERS = 24;
const NAMING_OUTPUT_TOKENS = 1800;

/** A rejected proposal with the deterministic reason, for the audit trail and later review UI. */
export interface RejectedProposal {
  clusterIndex: number;
  label: string;
  reason: RejectReason;
}

/** Outcome of one proposal run. Accepted candidates are ranked by deterministic confidence. */
export interface ProposalRunResult {
  clusterCount: number;
  sampledEmails: number;
  accepted: NamedCandidate[];
  rejected: RejectedProposal[];
}

interface ClusterSample {
  cluster: DiscoveredCluster;
  subjects: string[];
  senderTokens: string[];
}

/** Names residual clusters and validates the names. Produces proposals, persists nothing. */
export class DiscoveryProposalService {
  constructor(
    private residual: ResidualDiscoveryService,
    private emails: EmailRepository,
    private categories: CategoryRepository,
    private llm: LlmClient,
    private getConfig: () => LlmConfig,
    private logger: Logger,
  ) {}

  /**
   * Cluster the residual set, name the largest clusters with the local model, and validate each name.
   * Local-only unless the user opted in to cloud discovery. Returns accepted and rejected proposals;
   * writes nothing.
   */
  async propose(
    accountId: string,
    embeddingModelId: string,
    generationModelId: string,
  ): Promise<ProposalRunResult> {
    const clusters = this.residual.discover(accountId, embeddingModelId);
    if (clusters.length === 0) {
      return { clusterCount: 0, sampledEmails: 0, accepted: [], rejected: [] };
    }

    const chosen = [...clusters].sort((a, b) => b.size - a.size).slice(0, NAMING_MAX_CLUSTERS);
    const samples = chosen.map((cluster) => this.sampleCluster(accountId, cluster));
    const keyphrases = clusterKeyphrases(samples.map((s) => s.subjects));
    const sampledEmails = samples.reduce((n, s) => n + s.subjects.length, 0);

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
    // On the cloud chat provider use the configured cloud model and drop the Ollama-only controls
    // (`/no_think`, `think`), which would pollute or fail an OpenAI-style request.
    const local = provider === 'main';
    const model = local ? generationModelId : cfg.chatModel || generationModelId;
    const raw = await this.llm.chat({
      model,
      provider,
      messages: buildNamingMessages(namingInputs, { noThink: local }),
      responseFormat: 'json_object',
      temperature: 0.2,
      maxTokens: NAMING_OUTPUT_TOKENS,
      think: local ? false : undefined,
    });

    const parsed = parseNamedCandidates(raw, chosen.length);
    const candidates: NamedCandidate[] = parsed.map((p) => ({
      clusterIndex: p.clusterIndex,
      action: p.action,
      label: p.label,
      description: p.description,
      suggestedKey: p.suggestedKey,
      evidence: keyphrases[p.clusterIndex] ?? [],
    }));

    const activeCategories = this.activeCategoryRefs(accountId, embeddingModelId);
    const suggested = this.categories.listSuggested(accountId);
    const totalResidual = clusters.reduce((n, c) => n + c.size, 0);

    const results = validateBatch(candidates, (c) => ({
      cluster: chosen[c.clusterIndex]!,
      senderTokens: samples[c.clusterIndex]?.senderTokens ?? [],
      totalResidual,
      activeCategories,
      existingSuggestedLabels: suggested.map((s) => s.label),
      existingSuggestedKeys: suggested.map((s) => s.canonicalKey),
    }));

    const accepted = rankAccepted(results);
    const rejected: RejectedProposal[] = results
      .filter((r) => !r.verdict.accepted)
      .map((r) => ({
        clusterIndex: r.candidate.clusterIndex,
        label: r.candidate.label,
        reason: r.verdict.reason as RejectReason,
      }));

    this.logger.info(
      {
        accountId,
        clusters: chosen.length,
        named: candidates.length,
        accepted: accepted.length,
        rejected: rejected.length,
      },
      'discovery proposal: named residual clusters',
    );
    return { clusterCount: chosen.length, sampledEmails, accepted, rejected };
  }

  /**
   * A bounded, deterministic sample of one cluster: the first NAMING_SAMPLE_PER_CLUSTER member
   * subjects (member order is deterministic) and the brand tokens of their dominant sender domains.
   */
  private sampleCluster(accountId: string, cluster: DiscoveredCluster): ClusterSample {
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
