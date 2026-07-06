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
import type { CategoryAliasRepository } from '../repositories/category-alias-repository.js';
import type {
  CategoryProposal,
  CategoryProposalRepository,
  ProposalKind,
} from '../repositories/category-proposal-repository.js';
import type {
  AcceptedProposal,
  DiscoveryProposalService,
  RejectedProposal,
} from './discovery-proposal-service.js';
import type {
  CategoryCentroidRebuildService,
  RebuildResult,
} from './category-centroid-rebuild-service.js';

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
  kind: ProposalKind;
  categoryId: string;
  /** The absorbed source for a merge; null for every other kind. Lets the queue label a merge safely. */
  sourceCategoryId: string | null;
  label: string;
  description: string;
  proposedCount: number;
  cohesion: number;
  separation: number;
  confidence: number;
  evidence: string[];
  createdAt: number;
}

/** Result of applying a proposal. `assigned` is the members assigned (new_category) or reassigned (merge), 0 for retire. */
export interface ApplyResult {
  kind: ProposalKind;
  categoryId: string;
  label: string;
  assigned: number;
}

/**
 * A precondition or block failure during apply, carrying the HTTP status the route should return.
 * A block is a client-actionable conflict (409) or a missing referent (404), never a 500.
 */
export class ProposalApplyError extends Error {
  constructor(
    message: string,
    readonly httpStatus: number,
  ) {
    super(message);
    this.name = 'ProposalApplyError';
  }
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
    private centroidRebuild: CategoryCentroidRebuildService,
    private aliases: CategoryAliasRepository,
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
      kind: p.kind,
      categoryId: p.categoryId,
      sourceCategoryId: p.sourceCategoryId,
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
   * Approve a pending proposal, branching on its kind. `new_category` promotes and assigns (2c);
   * `retire` and `merge` apply the structural change (3.3); `split` is blocked until its safe write
   * semantics ship. Every kind marks the proposal applied only after its change commits, and every
   * block check throws before any write so a blocked apply leaves the proposal pending and retry-safe.
   */
  apply(accountId: string, proposalId: string): ApplyResult {
    const proposal = this.proposals.findById(proposalId);
    if (!proposal || proposal.accountId !== accountId) {
      throw new Error('proposal not found');
    }
    if (proposal.status !== 'pending') {
      throw new Error(`proposal is ${proposal.status}, not pending`);
    }
    switch (proposal.kind) {
      case 'new_category':
        return this.applyNewCategory(accountId, proposal);
      case 'retire':
        return this.applyRetire(accountId, proposal);
      case 'merge':
        return this.applyMerge(accountId, proposal);
      case 'split':
        return this.applySplit(accountId, proposal);
      default: {
        const exhaustive: never = proposal.kind;
        throw new Error(`unsupported proposal kind: ${String(exhaustive)}`);
      }
    }
  }

  /**
   * new_category (2c): promote the suggested category to active, seed the centroid from the accepted
   * cluster, and assign the members that are still uncategorized. User-assigned mail and members that
   * already carry any assignment are left untouched, so no existing label is removed.
   */
  private applyNewCategory(accountId: string, proposal: CategoryProposal): ApplyResult {
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
      this.proposals.markApplied(proposal.id);
    });
    run();

    this.logger.info(
      {
        accountId,
        proposalId: proposal.id,
        categoryId: proposal.categoryId,
        assigned: toAssign.length,
      },
      'discovery proposal: applied',
    );
    return {
      kind: 'new_category',
      categoryId: proposal.categoryId,
      label: proposal.label,
      assigned: toAssign.length,
    };
  }

  /**
   * retire (3.3): hide the target category, keeping its rows and history. Blocks (zero writes) when
   * the target is missing or not active, or when it has any user-confirmed member. This commit has no
   * confirmation path, so a category the user has filed mail into is never silently retired; the user
   * must resolve those assignments first.
   */
  private applyRetire(accountId: string, proposal: CategoryProposal): ApplyResult {
    const category = this.categories.findById(proposal.categoryId);
    if (!category || category.accountId !== accountId) {
      throw new ProposalApplyError('retire target category not found', 404);
    }
    if (category.status !== 'active') {
      throw new ProposalApplyError(`retire target is ${category.status}, not active`, 409);
    }
    const userMembers = this.categories.listCategoryMemberIds(accountId, category.id, 'user');
    if (userMembers.length > 0) {
      throw new ProposalApplyError(
        `cannot retire "${category.label}": it has ${userMembers.length} user-confirmed assignment(s); confirmation is required`,
        409,
      );
    }
    // The proposal was generated for an empty category, but a categorize/refine run may have
    // auto-assigned mail into it since. Re-check total emptiness against live state so retiring never
    // hides mail the category has gained; the user should refresh the queue and re-run cleanup.
    const totalMembers = this.categories.countEmails(category.id);
    if (totalMembers !== 0) {
      throw new ProposalApplyError(
        `cannot retire "${category.label}": it is no longer empty (${totalMembers} assigned email(s)); refresh the queue and re-run cleanup`,
        409,
      );
    }

    const run = this.db.transaction(() => {
      this.categories.retire(category.id);
      this.proposals.markApplied(proposal.id);
    });
    run();

    this.logger.info(
      { accountId, proposalId: proposal.id, categoryId: category.id },
      'discovery proposal: retired category',
    );
    return { kind: 'retire', categoryId: category.id, label: category.label, assigned: 0 };
  }

  /**
   * merge (3.3): absorb the source category into the target with the hardened, user-dominant
   * mergeInto (rows move preserving provenance, then the source is deleted), then rebuild the target
   * centroid from its now-merged members so future categorization learns from the combined category.
   * Blocks (zero writes) when either category is missing or not active, when the proposal carries no
   * source, or when source and target are the same. The move, the centroid rebuild, and markApplied
   * run in ONE transaction, so if the rebuild throws the whole merge rolls back and the proposal stays
   * pending; the proposal is marked applied only after all three succeed.
   */
  private applyMerge(accountId: string, proposal: CategoryProposal): ApplyResult {
    const target = this.categories.findById(proposal.categoryId);
    if (!target || target.accountId !== accountId) {
      throw new ProposalApplyError('merge target category not found', 404);
    }
    if (proposal.sourceCategoryId === null) {
      throw new ProposalApplyError('merge proposal has no source category', 409);
    }
    const source = this.categories.findById(proposal.sourceCategoryId);
    if (!source || source.accountId !== accountId) {
      throw new ProposalApplyError('merge source category not found', 404);
    }
    if (source.id === target.id) {
      throw new ProposalApplyError('merge source and target must differ', 409);
    }
    if (target.status !== 'active') {
      throw new ProposalApplyError(`merge target is ${target.status}, not active`, 409);
    }
    if (source.status !== 'active') {
      throw new ProposalApplyError(`merge source is ${source.status}, not active`, 409);
    }

    const run = this.db.transaction(
      (): {
        reassigned: number;
        rebuild: RebuildResult;
        aliasesMoved: number;
        skippedAliases: string[];
      } => {
        // Keep the target answering to the absorbed source's names so a re-run does not re-propose the
        // merged-away purpose and categorization still recognizes its mail. Re-point the source's own
        // aliases (conflict-free, a normalized alias is unique per account), then add the source label
        // and canonical key as target aliases. An alias another category already owns is skipped and
        // logged, never a merge failure. This runs before mergeInto deletes the source (which would
        // otherwise cascade the source aliases away).
        const aliasesMoved = this.aliases.reassign(accountId, source.id, target.id);
        const skippedAliases: string[] = [];
        for (const text of [source.label, source.canonicalKey]) {
          if (!this.aliases.addAlias(accountId, target.id, text, 'auto')) skippedAliases.push(text);
        }
        const reassigned = this.categories.mergeInto(source.id, target.id).reassigned;
        // Rebuild the target centroid from its merged members with Phase 3.1 semantics: user-confirmed
        // members dominate, and auto members are used only when the target has zero user members. When
        // there is too little trusted data the rebuild leaves the existing target centroid untouched
        // (a safe fallback, not a failure), so the merge still applies.
        const rebuild = this.centroidRebuild.rebuild(
          accountId,
          target.id,
          proposal.embeddingModelId,
          {
            allowAutoFallback: true,
          },
        );
        this.proposals.markApplied(proposal.id);
        return { reassigned, rebuild, aliasesMoved, skippedAliases };
      },
    );
    const { reassigned, rebuild, aliasesMoved, skippedAliases } = run();

    this.logger.info(
      {
        accountId,
        proposalId: proposal.id,
        sourceId: source.id,
        targetId: target.id,
        reassigned,
        centroidRebuild: rebuild.status,
        aliasesMoved,
        skippedAliases,
      },
      'discovery proposal: merged categories',
    );
    return { kind: 'merge', categoryId: target.id, label: target.label, assigned: reassigned };
  }

  /**
   * split (3.3): turn the source category into its proposed child categories, relocating only the
   * source's AUTO members into the child that lists them. Conservative and user-safe:
   *  - Blocks (zero writes) when the source is missing/not active, fewer than two children are
   *    proposed, or any child label/key collides with an existing or sibling category.
   *  - Moves an auto member only when exactly one child lists it and its email still exists; a member
   *    listed by two children is ambiguous and stays on the source. User assignments are never moved
   *    (moveAutoAssignment only touches auto rows), so no user-confirmed label is destroyed or hidden.
   *  - Retires the source only when nothing remains on it after the moves; otherwise it stays active,
   *    so a leftover auto member or any user member keeps the source visible.
   * Child creation, centroid saves, moves, the optional retire, and markApplied run in ONE
   * transaction: any failure rolls the whole split back and the proposal stays pending.
   */
  private applySplit(accountId: string, proposal: CategoryProposal): ApplyResult {
    const source = this.categories.findById(proposal.categoryId);
    if (!source || source.accountId !== accountId) {
      throw new ProposalApplyError('split source category not found', 404);
    }
    if (source.status !== 'active') {
      throw new ProposalApplyError(`split source is ${source.status}, not active`, 409);
    }
    const children = this.proposals.listChildren(proposal.id);
    if (children.length < 2) {
      throw new ProposalApplyError('split needs at least two child categories', 409);
    }

    // Collision check BEFORE any write: a child label or canonical key must not already exist as a
    // category, and the children must not collide with each other. A collision blocks the whole split
    // (zero writes); the user must resolve it and re-propose rather than have it partially apply.
    const seenKeys = new Set<string>();
    const seenLabels = new Set<string>();
    for (const child of children) {
      if (
        this.categories.findByCanonicalKey(accountId, child.canonicalKey) ||
        this.categories.findByLabel(accountId, child.label) ||
        seenKeys.has(child.canonicalKey) ||
        seenLabels.has(child.label)
      ) {
        throw new ProposalApplyError(
          `split child "${child.label}" collides with an existing or duplicate category; resolve it and re-propose`,
          409,
        );
      }
      seenKeys.add(child.canonicalKey);
      seenLabels.add(child.label);
    }

    // Eligible move set per child: an auto member of the source whose email still exists, listed by
    // exactly one child. A message listed by two children is ambiguous and is left on the source.
    const autoOnSource = new Set(
      this.categories.listCategoryMemberIds(accountId, source.id, 'auto'),
    );
    const allChildMemberIds = children.flatMap((c) => c.memberIds);
    const existing = this.emails.existingIds(accountId, allChildMemberIds);
    const childCount = new Map<string, number>();
    for (const id of allChildMemberIds) {
      childCount.set(id, (childCount.get(id) ?? 0) + 1);
    }
    const moveSets = children.map((child) =>
      child.memberIds.filter(
        (id) => autoOnSource.has(id) && existing.has(id) && childCount.get(id) === 1,
      ),
    );

    const run = this.db.transaction(() => {
      const createdIds: string[] = [];
      let moved = 0;
      children.forEach((child, i) => {
        const created = this.categories.create({
          accountId,
          label: child.label,
          description: child.description,
          source: 'auto',
          status: 'active',
          canonicalKey: child.canonicalKey,
        });
        createdIds.push(created.id);
        this.categories.saveCentroid(
          created.id,
          child.embeddingModelId,
          child.centroid,
          child.proposedCount,
        );
        for (const messageId of moveSets[i]!) {
          if (this.categories.moveAutoAssignment(messageId, accountId, source.id, created.id)) {
            moved += 1;
          }
        }
      });
      // Retire the source only when nothing remains on it (every member moved, no user member left).
      let sourceRetired = false;
      if (this.categories.countEmails(source.id) === 0) {
        this.categories.retire(source.id);
        sourceRetired = true;
      }
      this.proposals.markApplied(proposal.id);
      return { createdIds, moved, sourceRetired };
    });
    const { createdIds, moved, sourceRetired } = run();

    this.logger.info(
      {
        accountId,
        proposalId: proposal.id,
        sourceId: source.id,
        childIds: createdIds,
        moved,
        sourceRetired,
      },
      'discovery proposal: split applied',
    );
    return { kind: 'split', categoryId: source.id, label: source.label, assigned: moved };
  }

  /**
   * Dismiss a pending proposal (soft), branching on kind. For `new_category` the linked SUGGESTED
   * category is retired (it exists only for this proposal). For a structural proposal (split / merge /
   * retire) the target is a LIVE active category the user still uses, so dismiss touches no category or
   * assignment at all; it only records the dismissal. Every kind keeps the proposal row and its
   * suppression key so a re-run does not re-propose the same purpose.
   */
  dismiss(accountId: string, proposalId: string): { dismissed: true; categoryId: string } {
    const proposal = this.proposals.findById(proposalId);
    if (!proposal || proposal.accountId !== accountId) {
      throw new Error('proposal not found');
    }
    if (proposal.status !== 'pending') {
      throw new Error(`proposal is ${proposal.status}, not pending`);
    }

    const run = this.db.transaction(() => {
      if (proposal.kind === 'new_category') {
        const category = this.categories.findById(proposal.categoryId);
        if (category && category.status === 'suggested') {
          this.categories.retire(proposal.categoryId);
        }
      }
      this.proposals.markDismissed(proposalId);
    });
    run();

    this.logger.info(
      { accountId, proposalId, kind: proposal.kind, categoryId: proposal.categoryId },
      'discovery proposal: dismissed',
    );
    return { dismissed: true, categoryId: proposal.categoryId };
  }
}
