/**
 * Background orchestrator for LLM-assisted email categorization. Clusters emails by sender
 * and embedding similarity, runs the LLM on one representative per cluster, and tracks
 * per-account progress that the UI can poll.
 */
import type { Logger } from 'pino';
import type { CategoryRepository, CentroidEntry } from '../repositories/category-repository.js';
import type { CategorizeJobRepository } from '../repositories/categorize-job-repository.js';
import type { EmailRepository } from '../repositories/email-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import { rankCategories, type CategoryMatch } from './categorization-service.js';
import {
  shortlistFor,
  gateDecision,
  collapseLabels,
  adjudicate,
  deterministicFallback,
} from './categorize-strategy.js';
import type {
  CategoryCandidate,
  CorrectionExample,
  DecisionInput,
  EmailForCategorization,
  LlmCategorizer,
} from './llm-categorizer.js';
import {
  clusterBySenderAndContent,
  type ClusterInput,
  type ContentCluster,
} from './sender-clustering.js';
import { LlmApiError } from '../llm/client.js';
import { preprocessForEmbedding } from '../util/text.js';

const CORRECTION_EXAMPLES = 6;
const EVIDENCE_BODY_CHARS = 1000;

/** Builds the preprocessed body text used as adjudication evidence, capped to a char budget. */
function buildEvidenceText(email: EmailForCategorization): string {
  return email.body
    ? preprocessForEmbedding(email.body, {
        format: email.bodyFormat,
        maxChars: EVIDENCE_BODY_CHARS,
      })
    : '';
}

/**
 * Turns a non-retryable LLM API error into a user-facing message, tailored to whether the
 * failure came from the cloud (chat) or local (main) provider and to the kind of error.
 */
function describeFatalLlmError(
  err: LlmApiError,
  modelId: string,
  provider: 'main' | 'chat',
): string {
  const missing = err.status === 404 || /model_not_found|does not exist|not found/i.test(err.body);
  if (missing) {
    return provider === 'chat'
      ? `The cloud provider rejected model "${modelId}" as unknown. Check the model name and your account access in Settings.`
      : `Generation model "${modelId}" is not available on the local LLM server. Install it with "ollama pull ${modelId}" or choose an installed model in Settings.`;
  }
  if (err.status === 401 || err.status === 403) {
    return provider === 'chat'
      ? 'The cloud provider rejected the request. Check the API key in Settings.'
      : 'The local LLM server rejected the request.';
  }
  return err.message;
}

interface CategorizeContext {
  centroids: CentroidEntry[];
  centroidById: Map<string, Float32Array>;
  candidateById: Map<string, CategoryCandidate>;
  noCentroid: CategoryCandidate[];
  totalCategories: number;
  vectorByMessageId: Map<string, Float32Array>;
  examples: CorrectionExample[];
}

/** Progress snapshot for a categorization run, persisted per account and polled by the UI. */
export interface LlmCategorizeProgress {
  status:
    | 'idle'
    | 'running'
    | 'completed'
    | 'completed_with_failures'
    | 'error'
    | 'stopped'
    | 'interrupted';
  accountId: string | null;
  modelId: string | null;
  total: number;
  processed: number;
  assigned: number;
  uncategorized: number;
  failed: number;
  phase?: 'preparing' | 'clustering' | 'categorizing';
  clusters: number;
  clustersProcessed: number;
  gatedClusters: number;
  llmCalls: number;
  error?: string;
  startedAt?: number;
  completedAt?: number;
}

const CONCURRENCY =
  Number.parseInt(process.env.MAILPILOT_LLM_CATEGORIZE_CONCURRENCY ?? '', 10) > 0
    ? Number.parseInt(process.env.MAILPILOT_LLM_CATEGORIZE_CONCURRENCY ?? '', 10)
    : 1;
const LOCAL_BATCH_SIZE = 4;
const CLOUD_BATCH_SIZE = 8;
/** Resolves the cluster batch size, honoring an env override and defaulting per provider. */
function resolveBatchSize(provider: 'main' | 'chat'): number {
  const env = Number.parseInt(process.env.MAILPILOT_LLM_CATEGORIZE_BATCH ?? '', 10);
  if (env > 0) return env;
  return provider === 'chat' ? CLOUD_BATCH_SIZE : LOCAL_BATCH_SIZE;
}
const CLUSTER_THRESHOLD = (() => {
  const v = Number.parseFloat(process.env.MAILPILOT_LLM_CLUSTER_THRESHOLD ?? '');
  return Number.isFinite(v) && v > 0 && v <= 1 ? v : 0.95;
})();

/**
 * Content-aware categorization that scales to large inboxes. Emails are grouped by
 * sender then by embedding similarity, so near-identical messages form one cluster.
 * The LLM reads only one representative per cluster and the decision propagates to every
 * member, turning tens of thousands of emails into a few hundred LLM calls. Runs in the
 * background, and user-filed emails are skipped.
 */
export class LlmCategorizeOrchestrator {
  private progress: LlmCategorizeProgress = this.idle();
  private running = false;
  private stopRequested = false;
  private fatalError: string | null = null;
  private runProvider: 'main' | 'chat' = 'main';
  private llmAnySuccess = false;
  private llmAnyFailure = false;
  private llmReachedClusters = 0;
  private llmUnusableClusters = 0;

  /**
   * Wires up repositories and the categorizer. Marks any job left running from a prior process
   * as interrupted, then restores the most recent job as the starting progress snapshot.
   */
  constructor(
    private categorizer: LlmCategorizer,
    private emails: EmailRepository,
    private embeddings: EmbeddingRepository,
    private categories: CategoryRepository,
    private jobs: CategorizeJobRepository,
    private logger: Logger,
    private multiPrototypeEnabled: () => boolean = () => false,
  ) {
    this.jobs.markRunningInterrupted(Date.now());
    const last = this.jobs.getMostRecent();
    if (last) this.progress = last;
  }

  /**
   * Returns live progress for the running account, otherwise the persisted job for the given
   * account so one account never sees another account's state.
   */
  getProgress(accountId?: string): LlmCategorizeProgress {
    if (accountId && !(this.running && this.progress.accountId === accountId)) {
      const persisted = this.jobs.get(accountId);
      if (persisted) return persisted;
    }
    return { ...this.progress };
  }

  /** Saves the current progress snapshot for the account so the UI can poll it across restarts. */
  private persist(accountId: string): void {
    this.jobs.save(accountId, this.progress, Date.now());
  }

  /** Whether a categorization run is currently in progress. */
  isRunning(): boolean {
    return this.running;
  }

  /** Request a cooperative stop. The run halts at the next batch boundary. */
  stop(): boolean {
    if (!this.running) return false;
    this.stopRequested = true;
    return true;
  }

  /**
   * Starts a background categorization run for the account. Returns the pending email count and
   * whether a run was started, declining if one is already running or nothing is eligible.
   */
  start(
    accountId: string,
    generationModelId: string,
    embeddingModelId: string,
    opts: {
      force?: boolean;
      retryUncategorized?: boolean;
      messageIds?: string[];
      provider?: 'main' | 'chat';
    } = {},
  ): { pending: number; started: boolean } {
    if (this.running) {
      return { pending: this.progress.total - this.progress.processed, started: false };
    }

    // Active only: the LLM may assign to live categories, never to a suggested proposal awaiting
    // review or a retired category.
    const candidates: CategoryCandidate[] = this.categories
      .listActive(accountId)
      .map((c) => ({ id: c.id, label: c.label, description: c.description }));
    if (opts.force || opts.retryUncategorized) {
      this.categories.clearNoneDecisions(accountId);
    }
    const locked = opts.force
      ? this.categories.getUserAssignedMessageIds(accountId)
      : new Set<string>([
          ...this.categories.getLlmProtectedMessageIds(accountId),
          ...this.categories.getNoneDecisionIds(accountId, generationModelId),
        ]);
    const total = opts.messageIds
      ? opts.messageIds.filter((id) => !locked.has(id)).length
      : Math.max(0, this.embeddings.countForModel(accountId, embeddingModelId) - locked.size);

    if (candidates.length === 0 || total === 0) {
      this.progress = {
        ...this.idle(),
        status: 'completed',
        accountId,
        modelId: generationModelId,
        startedAt: Date.now(),
        completedAt: Date.now(),
      };
      this.persist(accountId);
      return { pending: 0, started: false };
    }

    this.progress = {
      ...this.idle(),
      status: 'running',
      phase: 'preparing',
      accountId,
      modelId: generationModelId,
      total,
      startedAt: Date.now(),
    };
    this.persist(accountId);
    this.running = true;
    this.stopRequested = false;
    this.fatalError = null;
    this.llmAnySuccess = false;
    this.llmAnyFailure = false;
    this.llmReachedClusters = 0;
    this.llmUnusableClusters = 0;
    this.runProvider = opts.provider ?? 'main';

    setImmediate(() => {
      void this.runLoop(
        accountId,
        generationModelId,
        embeddingModelId,
        candidates,
        locked,
        opts.messageIds,
      );
    });

    return { pending: total, started: true };
  }

  /**
   * Main background pass: gathers embeddings, clusters them, processes clusters in batches across
   * workers, then computes a final status that distinguishes systemic provider failure or unusable
   * model output from ordinary partial failures. The scopeIds limit the run to specific messages.
   */
  private async runLoop(
    accountId: string,
    generationModelId: string,
    embeddingModelId: string,
    candidates: CategoryCandidate[],
    locked: ReadonlySet<string>,
    scopeIds?: string[],
  ): Promise<void> {
    try {
      const scope = scopeIds ? new Set(scopeIds) : null;
      const senders = new Map(
        this.emails.listSenders(accountId).map((s) => [s.messageId, s.fromAddr]),
      );
      const inputs: ClusterInput[] = [];
      const vectorByMessageId = new Map<string, Float32Array>();
      for (const e of this.embeddings.listForAccount(accountId, embeddingModelId)) {
        if (locked.has(e.messageId)) continue;
        if (scope && !scope.has(e.messageId)) continue;
        inputs.push({
          messageId: e.messageId,
          fromAddr: senders.get(e.messageId) ?? null,
          vector: e.vector,
        });
        vectorByMessageId.set(e.messageId, e.vector);
      }

      // Shortlist ranking uses effective prototypes (nearest-prototype) when multi-prototype is on, so
      // an email near a category's sub-pattern still shortlists that category. The centroidById map used
      // for label collapsing is a category-to-category comparison, so it stays on the AGGREGATE centroid
      // (one vector per category), consistent with keeping merge/collapse on the aggregate.
      const aggregate = this.categories.getCentroidEntries(accountId, embeddingModelId);
      const centroids = this.categories.getEffectivePrototypeEntries(
        accountId,
        embeddingModelId,
        this.multiPrototypeEnabled(),
      );
      const centroidIds = new Set(aggregate.map((c) => c.categoryId));
      const ctx: CategorizeContext = {
        centroids,
        centroidById: new Map(aggregate.map((c) => [c.categoryId, c.vector])),
        candidateById: new Map(candidates.map((c) => [c.id, c])),
        noCentroid: candidates.filter((c) => !centroidIds.has(c.id)),
        totalCategories: candidates.length,
        vectorByMessageId,
        examples: this.categories.getUserCorrectionExamples(accountId, CORRECTION_EXAMPLES),
      };

      this.progress.phase = 'clustering';
      this.progress.total = inputs.length;
      this.persist(accountId);
      await new Promise<void>((resolve) => setImmediate(resolve));

      const clusters = this.stopRequested
        ? []
        : clusterBySenderAndContent(inputs, CLUSTER_THRESHOLD);
      this.progress.phase = 'categorizing';
      this.progress.clusters = clusters.length;
      this.logger.info(
        {
          accountId,
          emails: inputs.length,
          clusters: clusters.length,
          threshold: CLUSTER_THRESHOLD,
        },
        'llm categorization: clustered emails',
      );

      const batchSize = resolveBatchSize(this.runProvider);
      const batches: ContentCluster[][] = [];
      for (let i = 0; i < clusters.length; i += batchSize) {
        batches.push(clusters.slice(i, i + batchSize));
      }

      this.persist(accountId);
      let cursor = 0;
      /** Pulls batches off the shared cursor and processes them until the queue drains or a stop is requested. */
      const worker = async (): Promise<void> => {
        while (cursor < batches.length && !this.stopRequested && !this.fatalError) {
          await this.processBatch(batches[cursor++]!, accountId, generationModelId, ctx);
          this.persist(accountId);
        }
      };
      const concurrency = this.runProvider === 'chat' ? 1 : CONCURRENCY;
      await Promise.all(Array.from({ length: concurrency }, () => worker()));

      const systemicFailure = this.llmAnyFailure && !this.llmAnySuccess;
      const systemicUnusable =
        !this.fatalError &&
        !systemicFailure &&
        !this.stopRequested &&
        this.llmReachedClusters >= 5 &&
        this.llmUnusableClusters >= this.llmReachedClusters * 0.9;
      this.progress = {
        ...this.progress,
        phase: undefined,
        status:
          this.fatalError || ((systemicFailure || systemicUnusable) && !this.stopRequested)
            ? 'error'
            : this.stopRequested
              ? 'stopped'
              : this.progress.failed > 0
                ? 'completed_with_failures'
                : 'completed',
        error:
          this.fatalError ??
          (systemicFailure && !this.stopRequested
            ? this.runProvider === 'chat'
              ? 'Every request to the cloud provider failed, so no emails were categorized. Check your connection and API key in Settings, then try again.'
              : 'Every request to the local LLM server failed, so no emails were categorized. Check that it is running and reachable, then try again.'
            : systemicUnusable
              ? 'The model returned output that could not be used for almost every email. Check that the selected generation model supports JSON output, or try a different model in Settings.'
              : undefined),
        completedAt: Date.now(),
      };
      this.logger.info(
        {
          accountId,
          modelId: generationModelId,
          stopped: this.stopRequested,
          processed: this.progress.processed,
          assigned: this.progress.assigned,
          uncategorized: this.progress.uncategorized,
          failed: this.progress.failed,
          llmCalls: this.progress.llmCalls,
          gatedClusters: this.progress.gatedClusters,
          clustersProcessed: this.progress.clustersProcessed,
          clusters: this.progress.clusters,
        },
        'llm categorization finished',
      );
    } catch (err) {
      this.logger.error({ err, accountId }, 'llm categorization failed');
      this.progress = {
        ...this.progress,
        phase: undefined,
        status: 'error',
        error: err instanceof Error ? err.message : String(err),
        completedAt: Date.now(),
      };
    } finally {
      this.running = false;
      this.persist(accountId);
    }
  }

  /**
   * Categorizes one batch of clusters: gates high-confidence clusters without the LLM, sends the
   * rest as a single LLM batch with per-cluster retry, then applies fallbacks and adjudication so
   * each cluster's decision propagates to all its members.
   */
  private async processBatch(
    batch: ContentCluster[],
    accountId: string,
    modelId: string,
    ctx: CategorizeContext,
  ): Promise<void> {
    const reps = batch
      .map((cluster) => {
        const rep = this.emails.findById(cluster.representativeId, accountId);
        const vector = ctx.vectorByMessageId.get(cluster.representativeId);
        return rep && vector ? { cluster, rep, vector } : null;
      })
      .filter((r): r is NonNullable<typeof r> => r !== null);

    const batchMembers = batch.reduce((n, c) => n + c.memberIds.length, 0);
    const liveMembers = reps.reduce((n, r) => n + r.cluster.memberIds.length, 0);
    if (batchMembers > liveMembers) this.progress.total -= batchMembers - liveMembers;

    const toLlm: Array<{ cluster: ContentCluster; entry: DecisionInput; ranked: CategoryMatch[] }> =
      [];
    for (const r of reps) {
      const ranked = rankCategories(r.vector, ctx.centroids);
      const gated = gateDecision(ranked);
      if (gated) {
        this.categories.bulkReplaceForCluster(
          accountId,
          r.cluster.memberIds,
          [gated.categoryId],
          'gate',
        );
        this.categories.clearDecisions(accountId, modelId, r.cluster.memberIds);
        this.progress.processed += r.cluster.memberIds.length;
        this.progress.assigned += r.cluster.memberIds.length;
        this.progress.gatedClusters += 1;
        continue;
      }
      const shortlist = shortlistFor(ranked, ctx.totalCategories)
        .map((m) => ctx.candidateById.get(m.categoryId))
        .filter((c): c is CategoryCandidate => c !== undefined);
      toLlm.push({
        cluster: r.cluster,
        ranked,
        entry: {
          email: {
            subject: r.rep.subject,
            fromAddr: r.rep.fromAddr,
            body: r.rep.body,
            bodyFormat: r.rep.bodyFormat,
          },
          candidates: [...shortlist, ...ctx.noCentroid],
        },
      });
    }

    if (toLlm.length > 0) {
      this.progress.llmCalls += 1;
      let decisions: (string[] | null)[];
      try {
        decisions = await this.categorizer.decideBatch(
          toLlm.map((t) => t.entry),
          ctx.examples,
          modelId,
          this.runProvider,
        );
        this.llmAnySuccess = true;
      } catch (err) {
        if (err instanceof LlmApiError && err.nonRetryable) {
          this.fatalError = describeFatalLlmError(err, modelId, this.runProvider);
          this.logger.error({ err, modelId }, 'llm categorize aborted: non-retryable LLM error');
          return;
        }
        this.llmAnyFailure = true;
        const failedEmails = toLlm.reduce((n, t) => n + t.cluster.memberIds.length, 0);
        this.progress.failed += failedEmails;
        const timedOut = err instanceof Error && /timed out/i.test(err.message);
        this.logger.error(
          { err, clusters: toLlm.length, emails: failedEmails, timedOut },
          'llm categorize failed for a batch (emails left for the next run)',
        );
        this.progress.clustersProcessed += batch.length;
        return;
      }

      for (let i = 0; i < toLlm.length; i++) {
        if (decisions[i] != null) continue;
        try {
          this.progress.llmCalls += 1;
          const [one] = await this.categorizer.decideBatch(
            [toLlm[i]!.entry],
            ctx.examples,
            modelId,
            this.runProvider,
          );
          decisions[i] = one ?? null;
        } catch (err) {
          if (err instanceof LlmApiError && err.nonRetryable) {
            this.fatalError = describeFatalLlmError(err, modelId, this.runProvider);
            this.logger.error(
              { err, modelId },
              'llm categorize aborted: non-retryable LLM error on retry',
            );
            return;
          }
        }
      }

      this.llmReachedClusters += toLlm.length;
      toLlm.forEach((t, i) => {
        const raw = decisions[i];
        const evidence = {
          text: `${t.entry.email.subject ?? ''} ${buildEvidenceText(t.entry.email)}`,
          categoryById: ctx.candidateById,
        };
        if (raw == null) {
          this.llmUnusableClusters += 1;
          const fallback = deterministicFallback(t.ranked, evidence);
          if (fallback) {
            this.categories.bulkReplaceForCluster(
              accountId,
              t.cluster.memberIds,
              [fallback],
              'gate',
            );
            this.categories.clearDecisions(accountId, modelId, t.cluster.memberIds);
            this.progress.processed += t.cluster.memberIds.length;
            this.progress.assigned += t.cluster.memberIds.length;
            this.logger.info(
              {
                subject: t.entry.email.subject?.slice(0, 80) ?? null,
                fallback: ctx.candidateById.get(fallback)?.label ?? fallback,
                modelId,
                provider: this.runProvider,
              },
              'refine: deterministic fallback after unusable LLM output',
            );
          } else {
            this.progress.processed += t.cluster.memberIds.length;
            this.progress.uncategorized += t.cluster.memberIds.length;
            this.logger.warn(
              {
                rep: { subject: t.entry.email.subject, from: t.entry.email.fromAddr },
                members: t.cluster.memberIds.length,
                candidates: t.entry.candidates.map((c) => c.label),
                modelId,
                provider: this.runProvider,
              },
              'llm categorize: no usable decision and no confident fallback, left uncategorized',
            );
          }
          return;
        }
        const collapsed = collapseLabels(raw, ctx.centroidById);
        const verdict = adjudicate(collapsed, t.ranked, evidence);
        const ids = verdict.ids;
        const routine =
          verdict.reason === 'accepted_close_to_top' ||
          verdict.reason === 'accepted_new_category' ||
          verdict.reason === 'none';
        if (!routine) {
          const selectedId = collapsed[0];
          const selectedRank = selectedId
            ? t.ranked.findIndex((m) => m.categoryId === selectedId)
            : -1;
          const selectedConf = selectedId
            ? (t.ranked.find((m) => m.categoryId === selectedId)?.confidence ?? null)
            : null;
          this.logger.info(
            {
              subject: t.entry.email.subject?.slice(0, 80) ?? null,
              top: {
                label: t.ranked[0]?.label ?? null,
                confidence: t.ranked[0]?.confidence ?? null,
              },
              selected: {
                label: selectedId ? (ctx.candidateById.get(selectedId)?.label ?? selectedId) : null,
                confidence: selectedConf,
                rank: selectedRank,
              },
              margin:
                selectedConf != null && t.ranked[0]
                  ? Number((t.ranked[0].confidence - selectedConf).toFixed(3))
                  : null,
              finalAccepted: ids.map((id) => ctx.candidateById.get(id)?.label ?? id),
              reason: verdict.reason,
            },
            'refine: adjudication decision',
          );
        }
        this.categories.bulkReplaceForCluster(accountId, t.cluster.memberIds, ids, 'llm');
        if (ids.length === 0) {
          if (verdict.reason === 'none') {
            this.categories.recordNoneDecisions(accountId, modelId, t.cluster.memberIds);
          }
          this.progress.uncategorized += t.cluster.memberIds.length;
        } else {
          this.categories.clearDecisions(accountId, modelId, t.cluster.memberIds);
        }
        this.progress.processed += t.cluster.memberIds.length;
        this.progress.assigned += ids.length * t.cluster.memberIds.length;
      });
    }
    this.progress.clustersProcessed += batch.length;
  }

  /** Returns a fresh zeroed progress snapshot with idle status. */
  private idle(): LlmCategorizeProgress {
    return {
      status: 'idle',
      accountId: null,
      modelId: null,
      total: 0,
      processed: 0,
      assigned: 0,
      uncategorized: 0,
      failed: 0,
      clusters: 0,
      clustersProcessed: 0,
      gatedClusters: 0,
      llmCalls: 0,
    };
  }
}
