/**
 * Deterministic candidate generation support for discovery (Phase 2b): cluster keyphrases for the
 * naming prompt, and the deterministic validation gate that accepts or rejects a named candidate.
 *
 * The gate never trusts the model's self-reported confidence. Every decision is a function of the
 * candidate label/description and the cluster's deterministic metrics (size, cohesion, separation)
 * plus the account's existing taxonomy. Pure functions: no LLM, no I/O, no persistence.
 */
import { cosineSimilarity } from '../util/vector.js';
import { normalizeForMatch } from '../util/text.js';
import {
  isVagueTopicLabel,
  isNearDuplicateLabel,
  significantTokens,
  VAGUE_TOKENS,
} from './topic-discovery-service.js';
import type { DiscoveredCluster } from './discovery-clustering.js';

/** Minimum cluster cohesion for any candidate to be accepted. A high-value purpose relaxes the size
 * floor but never this one: every accepted category must be internally coherent. */
export const MIN_COHESION = 0.55;
/**
 * Candidate cluster centroid this close (cosine) to an active category centroid is treated as the
 * same category. Set below the observed same-purpose twin band (e.g. Billing vs Invoices sit around
 * 0.83) so semantic twins with no shared label words are still caught. A single scalar cannot tell a
 * true twin from an adjacent sibling; true synonym resolution is deferred to the review-queue phase,
 * and this gate biases toward not polluting the taxonomy since suggestions still pass human review.
 */
export const OVERLAP_CENTROID_COSINE = 0.8;
/** Clusters smaller than this are too thin to form a normal category. */
export const THIN_CLUSTER_SIZE = 5;
/** A high-value purpose relaxes the size floor to this (only the size floor, never cohesion). */
export const HIGH_RISK_MIN_SIZE = 2;
/**
 * High-value purposes that may form a small category despite thin support. Matched as exact label
 * tokens (never description or sampled evidence), so a curated stem set is used rather than prefix
 * matching, which would over-match ('law' in 'lawn') and wrongly relax the floor.
 */
export const HIGH_RISK_PURPOSES: readonly string[] = [
  // Tax and government financial documents.
  'tax',
  'taxes',
  'irs',
  '1099',
  // Medical and health.
  'medical',
  'health',
  'healthcare',
  'doctor',
  'doctors',
  'physician',
  'dental',
  'dentist',
  'clinic',
  'hospital',
  'pharmacy',
  'prescription',
  'prescriptions',
  'patient',
  'diagnosis',
  // Insurance.
  'insurance',
  // Legal.
  'legal',
  'lawyer',
  'lawyers',
  'attorney',
  'attorneys',
  'lawsuit',
  'litigation',
  // Immigration.
  'visa',
  'immigration',
  'passport',
  'citizenship',
  'naturalization',
  'uscis',
  'asylum',
  // High-value financial commitments.
  'mortgage',
  'mortgages',
];

/**
 * Low-value promotional buckets. A label made up entirely of these (plus generic decorator words) is
 * a marketing catch-all, not a purpose, so it is rejected rather than turned into a category.
 */
const MARKETING_TOKENS = new Set([
  'deal',
  'deals',
  'discount',
  'discounts',
  'promotion',
  'promotions',
  'promo',
  'promos',
  'offer',
  'offers',
  'sale',
  'sales',
  'coupon',
  'coupons',
  'newsletter',
  'newsletters',
  'marketing',
  'shopping',
  'retail',
]);

/** Over-broad labels that name no concrete purpose. Complements the shared VAGUE_TOKENS set. */
const OVERBROAD_TOKENS = new Set(['work', 'personal', 'todo', 'todos']);

/** What the model proposed for a cluster. Identity (canonical key) is assigned later, not by the model. */
export type CandidateAction =
  | 'new_category'
  | 'expand_existing'
  | 'merge_suggestion'
  | 'leave_uncategorized';

/** A model-named candidate tied to a source cluster. */
export interface NamedCandidate {
  clusterIndex: number;
  action: CandidateAction;
  label: string;
  description: string;
  suggestedKey: string;
  evidence: string[];
}

/** An active category the candidate is validated against. */
export interface ActiveCategoryRef {
  label: string;
  description: string | null;
  centroid: Float32Array | null;
  createdBy: 'auto' | 'user' | 'imported';
}

/** Everything the gate needs to judge one candidate, none of it from the model's confidence. */
export interface CandidateContext {
  cluster: DiscoveredCluster;
  /** Brand tokens of the cluster's dominant sender domains. A sampling signal only, never cluster logic. */
  senderTokens: string[];
  totalResidual: number;
  activeCategories: ActiveCategoryRef[];
  existingSuggestedLabels: string[];
  /** Canonical keys of already-suggested proposals, to catch a same-key collision when labels differ. */
  existingSuggestedKeys: string[];
  otherCandidateLabels: string[];
}

/** Why a candidate was rejected, or 'accepted'. */
export type Verdict =
  | { accepted: true; reason: 'accepted'; confidence: number }
  | { accepted: false; reason: RejectReason; confidence: number };

/** The deterministic rejection reasons. */
export type RejectReason =
  | 'model_left_uncategorized'
  | 'vague_label'
  | 'low_value_label'
  | 'sender_name_only'
  | 'conflicts_user_category'
  | 'overlaps_active_label'
  | 'overlaps_active_content'
  | 'duplicate_of_suggestion'
  | 'duplicate_of_existing_proposal'
  | 'low_cohesion'
  | 'thin_support';

/**
 * Whether a candidate names a high-value purpose that earns relaxed (not skipped) size/cohesion
 * floors. Keyed only on the model-chosen LABEL, never the free-text description or sampled evidence:
 * a full sentence of prose can incidentally contain a high-risk word ('...the local basketball
 * court...') and unlock the floors for a cluster that is not a high-value purpose at all.
 */
export function isHighRiskPurpose(candidate: NamedCandidate): boolean {
  const words = new Set(significantTokens(candidate.label));
  return HIGH_RISK_PURPOSES.some((p) => words.has(p));
}

/** Deterministic confidence in [0, 1] from cluster metrics and coverage. Never the model's number. */
export function candidateConfidence(ctx: CandidateContext): number {
  const { cohesion, separation, size } = ctx.cluster;
  const coverage = ctx.totalResidual > 0 ? Math.min(1, size / ctx.totalResidual) : 0;
  const c = 0.5 * cohesion + 0.3 * separation + 0.2 * coverage;
  return c < 0 ? 0 : c > 1 ? 1 : c;
}

/**
 * The deterministic validation gate for one candidate. Rejection rules run before the quality
 * floors, and the high-risk allowance only relaxes the floors, never a rejection rule (a high-risk
 * label that duplicates or overlaps an existing category is still rejected).
 */
export function validateCandidate(candidate: NamedCandidate, ctx: CandidateContext): Verdict {
  const confidence = candidateConfidence(ctx);
  const reject = (reason: RejectReason): Verdict => ({ accepted: false, reason, confidence });

  if (candidate.action === 'leave_uncategorized') return reject('model_left_uncategorized');

  const label = candidate.label.trim();

  // A raw email address is a sender, not a purpose, and leaks PII into a category name.
  if (label.includes('@')) return reject('sender_name_only');

  if (isVagueTopicLabel(label)) return reject('vague_label');

  const tokens = significantTokens(label);
  // Numeric-only ('2024') or over-broad ('Work', 'Personal') labels name no concrete purpose. A
  // high-value numeric token (e.g. the '1099' tax form) is exempt, so it reaches the relaxed floor.
  if (
    tokens.length > 0 &&
    tokens.every((t) => /^\d+$/.test(t)) &&
    !tokens.some((t) => HIGH_RISK_PURPOSES.includes(t))
  ) {
    return reject('vague_label');
  }
  if (tokens.length > 0 && tokens.every((t) => OVERBROAD_TOKENS.has(t) || VAGUE_TOKENS.has(t))) {
    return reject('vague_label');
  }
  // A label that is all promotional/decorator words is a marketing catch-all, not a purpose.
  if (
    tokens.length > 0 &&
    tokens.some((t) => MARKETING_TOKENS.has(t)) &&
    tokens.every((t) => MARKETING_TOKENS.has(t) || VAGUE_TOKENS.has(t))
  ) {
    return reject('low_value_label');
  }

  // Sender-name-only: once generic decorator words are stripped, the label is nothing but the
  // dominant sender's brand tokens ('Coinbase Updates', 'LinkedIn Notifications'), naming no purpose.
  // A high-value purpose word that happens to coincide with a sender brand ('Visa' the immigration
  // purpose vs visa.com the card network) is exempt: dropping a potential medical/legal/immigration
  // category is worse than surfacing a possible false positive to human review.
  const senderBrands = new Set(
    ctx.senderTokens.flatMap((s) => normalizeForMatch(s).split(' ')).filter(Boolean),
  );
  const contentTokens = tokens.filter((t) => !VAGUE_TOKENS.has(t));
  if (
    senderBrands.size > 0 &&
    contentTokens.length > 0 &&
    contentTokens.every((t) => senderBrands.has(t)) &&
    !contentTokens.some((t) => HIGH_RISK_PURPOSES.includes(t))
  ) {
    return reject('sender_name_only');
  }

  // Conflict with a user-created category, checked before the generic active overlap for a clearer reason.
  const userLabels = ctx.activeCategories.filter((c) => c.createdBy === 'user').map((c) => c.label);
  if (isNearDuplicateLabel(label, userLabels)) return reject('conflicts_user_category');

  // Overlap with an active category, by label or by centroid content. The centroid comparison is
  // skipped when dimensions differ (e.g. a legacy centroid from another embedding model), which would
  // otherwise produce a NaN cosine that silently passes the check.
  const activeLabels = ctx.activeCategories.map((c) => c.label);
  if (isNearDuplicateLabel(label, activeLabels)) return reject('overlaps_active_label');
  const clusterCentroid = ctx.cluster.centroid;
  const overlapsContent = ctx.activeCategories.some(
    (c) =>
      c.centroid != null &&
      c.centroid.length === clusterCentroid.length &&
      cosineSimilarity(clusterCentroid, c.centroid) >= OVERLAP_CENTROID_COSINE,
  );
  if (overlapsContent) return reject('overlaps_active_content');

  // Duplicate of another accepted candidate this run, or of an already-suggested proposal.
  if (isNearDuplicateLabel(label, ctx.otherCandidateLabels))
    return reject('duplicate_of_suggestion');
  if (isNearDuplicateLabel(label, ctx.existingSuggestedLabels)) {
    return reject('duplicate_of_existing_proposal');
  }

  // Quality floors. Cohesion is a universal bar: every accepted category must be internally coherent,
  // so a mislabeled promotional cluster that merely contains a high-value word cannot buy its way in
  // with low cohesion. A high-value purpose only relaxes the SIZE floor, since a rare medical/legal/
  // immigration category may be small; a normal small cluster is still rejected as thin_support.
  if (ctx.cluster.cohesion < MIN_COHESION) return reject('low_cohesion');
  const minSize = isHighRiskPurpose(candidate) ? HIGH_RISK_MIN_SIZE : THIN_CLUSTER_SIZE;
  if (ctx.cluster.size < minSize) return reject('thin_support');

  return { accepted: true, reason: 'accepted', confidence };
}

/**
 * Validate a batch in order. Deduplication against other candidates is order-stable: the first
 * accepted candidate wins and a later duplicate is rejected. A later candidate is a duplicate when
 * its label near-matches an accepted one (handled inside validateCandidate via otherCandidateLabels)
 * or when it carries the same normalized suggestedKey as another candidate this run or an existing
 * proposal, which would collide on the same canonical category downstream even though the labels
 * differ. A key collision against an existing proposal is reported as duplicate_of_existing_proposal;
 * against another candidate this run, as duplicate_of_suggestion. Returns a verdict per candidate.
 */
export function validateBatch(
  candidates: NamedCandidate[],
  baseContext: (c: NamedCandidate) => Omit<CandidateContext, 'otherCandidateLabels'>,
): Array<{ candidate: NamedCandidate; verdict: Verdict }> {
  const acceptedLabels: string[] = [];
  const acceptedKeys = new Set<string>();
  let existingKeys: Set<string> | null = null;
  const results: Array<{ candidate: NamedCandidate; verdict: Verdict }> = [];
  for (const candidate of candidates) {
    const ctx: CandidateContext = {
      ...baseContext(candidate),
      otherCandidateLabels: [...acceptedLabels],
    };
    if (existingKeys === null) {
      existingKeys = new Set(
        ctx.existingSuggestedKeys.map((k) => normalizeForMatch(k)).filter(Boolean),
      );
    }
    let verdict = validateCandidate(candidate, ctx);
    if (verdict.accepted) {
      const key = normalizeForMatch(candidate.suggestedKey);
      if (key.length > 0 && existingKeys.has(key)) {
        verdict = {
          accepted: false,
          reason: 'duplicate_of_existing_proposal',
          confidence: verdict.confidence,
        };
      } else if (key.length > 0 && acceptedKeys.has(key)) {
        verdict = {
          accepted: false,
          reason: 'duplicate_of_suggestion',
          confidence: verdict.confidence,
        };
      } else {
        if (key.length > 0) acceptedKeys.add(key);
        acceptedLabels.push(candidate.label);
      }
    }
    results.push({ candidate, verdict });
  }
  return results;
}

/** Accepted candidates ranked by deterministic confidence, highest first. */
export function rankAccepted(
  results: Array<{ candidate: NamedCandidate; verdict: Verdict }>,
): NamedCandidate[] {
  return results
    .filter((r) => r.verdict.accepted)
    .sort((a, b) => b.verdict.confidence - a.verdict.confidence)
    .map((r) => r.candidate);
}

const KEYPHRASE_STOPWORDS = new Set([
  're',
  'fwd',
  'the',
  'a',
  'an',
  'of',
  'to',
  'for',
  'and',
  'or',
  'your',
  'you',
  'is',
  'are',
  'on',
  'in',
  'at',
  'with',
  'this',
  'that',
  'from',
  'new',
]);

/**
 * Word tokens with stopwords and very short tokens removed. Accents are stripped first (via
 * normalizeForMatch) so French subjects tokenize as whole words ('électricité' -> 'electricite')
 * instead of being split on the accented letters by an ASCII-only match.
 */
function keyTokens(text: string): string[] {
  return (normalizeForMatch(text).match(/[a-z0-9]{3,}/g) ?? []).filter(
    (t) => !KEYPHRASE_STOPWORDS.has(t),
  );
}

/**
 * Class-based TF-IDF keyphrases per cluster: terms frequent within a cluster but not spread across
 * many clusters. Grounds the naming prompt in real words without letting a term shared by everything
 * dominate. Deterministic. `clusterSubjects[i]` is the list of subjects for cluster i.
 *
 * Uses smoothed IDF `1 + ln(n / df)` rather than `ln((n+1)/(df+1))`: the latter collapses to 0 for
 * every term when there is a single cluster (df === n), which would drop the ordering back to
 * alphabetical and lose the frequency signal. The smoothed form keeps frequency ordering within one
 * cluster while still down-weighting terms that appear in many clusters.
 */
export function clusterKeyphrases(clusterSubjects: string[][], topK = 8): string[][] {
  const n = clusterSubjects.length;
  const tfPerCluster = clusterSubjects.map((subjects) => {
    const tf = new Map<string, number>();
    for (const s of subjects) for (const t of keyTokens(s)) tf.set(t, (tf.get(t) ?? 0) + 1);
    return tf;
  });
  const clusterFreq = new Map<string, number>();
  for (const tf of tfPerCluster)
    for (const t of tf.keys()) clusterFreq.set(t, (clusterFreq.get(t) ?? 0) + 1);

  return tfPerCluster.map((tf) =>
    [...tf.entries()]
      .map(([term, freq]) => ({
        term,
        score: freq * (1 + Math.log(n / (clusterFreq.get(term) ?? n))),
      }))
      .sort((a, b) => b.score - a.score || a.term.localeCompare(b.term))
      .slice(0, topK)
      .map((e) => e.term),
  );
}
