/**
 * Persists reviewed discovery proposals and applies or dismisses them on user approval (Phase 2c).
 * Generate runs 2a to 2b (cluster, name, validate) then materializes each accepted new-category
 * proposal as a status='suggested' category plus a proposal row, deduped by frozen canonical key so a
 * re-run never re-proposes a purpose that already exists, was already suggested, or was dismissed.
 *
 * Apply promotes the suggested category to active, seeds its centroid from the accepted cluster, and
 * assigns only the members that are still uncategorized (never user-assigned or already auto-assigned
 * mail). Dismiss is soft: the suggested category is retired and the proposal row is kept as a
 * suppression record. No mailbox moves, no auto-merge, no auto-retire, no silent rename.
 */
import type { Database } from 'better-sqlite3';
import type { Logger } from 'pino';
import { randomUUID } from 'node:crypto';
import type { LlmConfig } from '../config/schema.js';
import { normalizeForMatch } from '../util/text.js';
import { discoveryProvider } from './discovery-guard.js';
import { canonicalKeyBase } from '../repositories/category-repository.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { AccountRepository } from '../repositories/account-repository.js';
import type { EmailRepository } from '../repositories/email-repository.js';
import type { DiscoveryAuditRepository } from '../repositories/discovery-audit-repository.js';
import type { CategoryProposalRepository } from '../repositories/category-proposal-repository.js';
import type {
  AcceptedProposal,
  DiscoveryProposalService,
  RejectedProposal,
} from './discovery-proposal-service.js';

/** A newly persisted proposal, returned from a generate run. */
export interface GeneratedProposal {
  id: string;
  categoryId: string;
  label: string;
  description: string;
  canonicalKey: string;
  proposedCount: number;
  cohesion: number;
  confidence: number;
  evidence: string[];
}

/** Outcome of a generate run. Writes suggested categories and proposal rows; no active categories. */
export interface GenerateResult {
  runId: string;
  clusterCount: number;
  sampledEmails: number;
  created: GeneratedProposal[];
  skippedDuplicates: number;
  rejected: RejectedProposal[];
}

/** A pending proposal as shown in the review queue, without the heavy centroid or member list. */
export interface ProposalView {
  id: string;
  categoryId: string;
  label: string;
  description: string;
  proposedCount: number;
  cohesion: number;
  separation: number;
  confidence: number;
  evidence: string[];
  createdAt: number;
}

/** Result of applying a proposal. */
export interface ApplyResult {
  categoryId: string;
  label: string;
  assigned: number;
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** Persists, lists, applies, and dismisses discovery proposals. */
export class DiscoveryProposalOrchestrator {
  constructor(
    private db: Database,
    private proposals: CategoryProposalRepository,
    private categories: CategoryRepository,
    private emails: EmailRepository,
    private proposalService: DiscoveryProposalService,
    private accounts: AccountRepository,
    private audit: DiscoveryAuditRepository,
    private getConfig: () => LlmConfig,
    private logger: Logger,
  ) {}

  /**
   * Run discovery and persist the accepted new-category proposals as suggested categories plus
   * proposal rows, deduped by canonical key. Local-only unless cloud discovery is opted in. Writes
   * nothing active and assigns no mail; that happens only on apply.
   */
  async generate(
    accountId: string,
    embeddingModelId: string,
    generationModelId: string,
  ): Promise<GenerateResult> {
    const cfg = this.getConfig();
    const auditProvider = discoveryProvider(cfg) === 'main' ? 'local' : 'cloud';
    const accountKind = this.accounts.findById(accountId)?.kind ?? 'unknown';
    const empty: GenerateResult = {
      runId: '',
      clusterCount: 0,
      sampledEmails: 0,
      created: [],
      skippedDuplicates: 0,
      rejected: [],
    };

    if (!this.accounts.isDiscoveryEligible(accountId)) {
      this.audit.log({
        accountId,
        flow: 'discovery_proposal',
        accountKind,
        provider: auditProvider,
        status: 'skipped',
        modelId: generationModelId,
      });
      return empty;
    }

    let run;
    try {
      run = await this.proposalService.propose(accountId, embeddingModelId, generationModelId);
    } catch (err) {
      this.audit.log({
        accountId,
        flow: 'discovery_proposal',
        accountKind,
        provider: auditProvider,
        status: 'failed',
        modelId: generationModelId,
        error: String(err),
      });
      throw err;
    }

    const runId = randomUUID();
    const created: GeneratedProposal[] = [];
    let skippedDuplicates = 0;
    // Suggested keys of purposes the user already applied or dismissed, so a re-run does not
    // re-propose one even if the model relabels it and the label-derived canonical key drifts.
    const resolvedKeys = this.proposals.resolvedSuggestedKeys(accountId);

    // Persist one accepted candidate as a suggested category plus a proposal row, atomically.
    const persistOne = this.db.transaction((accepted: AcceptedProposal, canonicalKey: string) => {
      const category = this.categories.create({
        accountId,
        label: accepted.candidate.label,
        description: accepted.candidate.description,
        source: 'auto',
        status: 'suggested',
        canonicalKey,
      });
      const proposal = this.proposals.create({
        accountId,
        categoryId: category.id,
        runId,
        clusterIndex: accepted.candidate.clusterIndex,
        label: accepted.candidate.label,
        description: accepted.candidate.description,
        canonicalKey,
        suggestedKey: normalizeForMatch(accepted.candidate.suggestedKey),
        embeddingModelId,
        centroid: accepted.cluster.centroid,
        memberIds: accepted.cluster.memberIds,
        proposedCount: accepted.cluster.size,
        cohesion: accepted.cluster.cohesion,
        separation: accepted.cluster.separation,
        confidence: accepted.confidence,
        evidence: accepted.candidate.evidence,
      });
      return { category, proposal };
    });

    for (const accepted of run.accepted) {
      if (accepted.candidate.action !== 'new_category') continue;
      const label = accepted.candidate.label;
      const canonicalKey = canonicalKeyBase(label);
      const suggestedKey = normalizeForMatch(accepted.candidate.suggestedKey);
      // Dedup: an existing category with this key or label (active, suggested, or dismissed-then-
      // retired), or a resolved proposal with this suggested key. The label check also avoids the
      // UNIQUE(account_id, label) constraint when a canonical key was freed but the label survives.
      if (
        this.categories.findByCanonicalKey(accountId, canonicalKey) ||
        this.categories.findByLabel(accountId, label) ||
        (suggestedKey !== '' && resolvedKeys.has(suggestedKey))
      ) {
        skippedDuplicates += 1;
        continue;
      }
      try {
        const { category, proposal } = persistOne(accepted, canonicalKey);
        created.push({
          id: proposal.id,
          categoryId: category.id,
          label: proposal.label,
          description: proposal.description,
          canonicalKey,
          proposedCount: proposal.proposedCount,
          cohesion: proposal.cohesion,
          confidence: proposal.confidence,
          evidence: proposal.evidence,
        });
      } catch (err) {
        // A unique-constraint conflict that slipped past the checks skips just this candidate rather
        // than aborting the whole run.
        skippedDuplicates += 1;
        this.logger.warn(
          { accountId, label, err: String(err) },
          'discovery proposal: skipped candidate on constraint conflict',
        );
      }
    }

    this.audit.log({
      accountId,
      flow: 'discovery_proposal',
      accountKind,
      provider: auditProvider,
      status: created.length > 0 ? 'ok' : 'insufficient',
      modelId: generationModelId,
      poolSize: run.clusterCount,
      sampleSize: run.sampledEmails,
      emailsExposed: run.sampledEmails,
      fieldsRead: ['subject', 'from_addr'],
    });

    this.logger.info(
      {
        accountId,
        runId,
        created: created.length,
        skippedDuplicates,
        rejected: run.rejected.length,
      },
      'discovery proposal: persisted review queue',
    );
    return {
      runId,
      clusterCount: run.clusterCount,
      sampledEmails: run.sampledEmails,
      created,
      skippedDuplicates,
      rejected: run.rejected,
    };
  }

  /** Pending proposals for the review queue, strongest first. */
  listPending(accountId: string): ProposalView[] {
    return this.proposals.listPending(accountId).map((p) => ({
      id: p.id,
      categoryId: p.categoryId,
      label: p.label,
      description: p.description,
      proposedCount: p.proposedCount,
      cohesion: p.cohesion,
      separation: p.separation,
      confidence: p.confidence,
      evidence: p.evidence,
      createdAt: p.createdAt,
    }));
  }

  /**
   * Approve a proposal: promote its suggested category to active, seed the centroid from the accepted
   * cluster, and assign the members that are still uncategorized. User-assigned mail and members that
   * already carry any assignment are left untouched, so no existing label is removed.
   */
  apply(accountId: string, proposalId: string): ApplyResult {
    const proposal = this.proposals.findById(proposalId);
    if (!proposal || proposal.accountId !== accountId) {
      throw new Error('proposal not found');
    }
    if (proposal.status !== 'pending') {
      throw new Error(`proposal is ${proposal.status}, not pending`);
    }
    const category = this.categories.findById(proposal.categoryId);
    if (!category) {
      throw new Error('proposal category no longer exists');
    }
    if (category.status !== 'suggested') {
      throw new Error(`category is ${category.status}, not suggested`);
    }

    // Assign only members that (a) still exist and (b) have no assignment yet, so a member deleted
    // between generate and apply is skipped instead of failing the whole apply on a foreign key, and
    // no user-assigned or already auto-assigned mail is touched.
    const assigned = this.categories.getAssignedMessageIds(accountId);
    const existing = this.emails.existingIds(accountId, proposal.memberIds);
    const toAssign = proposal.memberIds.filter((id) => existing.has(id) && !assigned.has(id));
    const now = Date.now();

    const run = this.db.transaction(() => {
      this.categories.setStatus(proposal.categoryId, 'active');
      this.categories.saveCentroid(
        proposal.categoryId,
        proposal.embeddingModelId,
        proposal.centroid,
        proposal.proposedCount,
      );
      if (toAssign.length > 0) {
        this.categories.addAutoAssignments(
          accountId,
          toAssign.map((messageId) => ({
            messageId,
            accountId,
            categoryId: proposal.categoryId,
            confidence: clamp01(proposal.confidence),
            assignedBy: 'auto' as const,
            assignedAt: now,
            method: 'proposal' as const,
          })),
        );
      }
      this.proposals.markApplied(proposalId);
    });
    run();

    this.logger.info(
      { accountId, proposalId, categoryId: proposal.categoryId, assigned: toAssign.length },
      'discovery proposal: applied',
    );
    return { categoryId: proposal.categoryId, label: proposal.label, assigned: toAssign.length };
  }

  /**
   * Dismiss a proposal (soft): retire its suggested category and mark the proposal dismissed. The
   * proposal row and the retired category keep the canonical key, so a re-run suppresses the same
   * purpose. No active category, assignment, or user correction is touched.
   */
  dismiss(accountId: string, proposalId: string): { dismissed: true; categoryId: string } {
    const proposal = this.proposals.findById(proposalId);
    if (!proposal || proposal.accountId !== accountId) {
      throw new Error('proposal not found');
    }
    if (proposal.status !== 'pending') {
      throw new Error(`proposal is ${proposal.status}, not pending`);
    }
    const category = this.categories.findById(proposal.categoryId);

    const run = this.db.transaction(() => {
      if (category && category.status === 'suggested') {
        this.categories.retire(proposal.categoryId);
      }
      this.proposals.markDismissed(proposalId);
    });
    run();

    this.logger.info(
      { accountId, proposalId, categoryId: proposal.categoryId },
      'discovery proposal: dismissed',
    );
    return { dismissed: true, categoryId: proposal.categoryId };
  }
}
