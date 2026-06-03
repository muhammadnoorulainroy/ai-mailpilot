/**
 * Decision logic for email categorization: collapses overlapping labels, adjudicates the LLM's
 * category choice against embedding ranking and lexical text evidence, and gates fast or
 * deterministic auto-assignment when confidence is high enough to skip the LLM.
 */

import type { CategoryMatch } from './categorization-service.js';
import { cosineFromL2Distance, l2Distance } from '../util/vector.js';

const LABEL_OVERLAP = 0.92;
const MAX_LABELS_PER_EMAIL = 2;

/**
 * Reduce the LLM's chosen category ids to a clean set: keep best-first order, drop a later label
 * whose centroid overlaps an already-kept one, and cap at two. Categories without a centroid (new,
 * empty) are kept as-is since they cannot be compared.
 */
export function collapseLabels(ids: string[], centroidById: Map<string, Float32Array>): string[] {
  const kept: string[] = [];
  for (const id of ids) {
    if (kept.length >= MAX_LABELS_PER_EMAIL) break;
    if (kept.includes(id)) continue;
    const vec = centroidById.get(id);
    const overlaps =
      vec != null &&
      kept.some((k) => {
        const kvec = centroidById.get(k);
        return kvec != null && cosineFromL2Distance(l2Distance(vec, kvec)) >= LABEL_OVERLAP;
      });
    if (!overlaps) kept.push(id);
  }
  return kept;
}

const ADJ_CLOSE_TO_TOP_WINDOW = 0.1;
const ADJ_STRONG_TOP_CONFIDENCE = 0.7;
const ADJ_TOP_OVERRIDE_MARGIN = 0.2;
const ADJ_VERY_STRONG_TOP = 0.8;
const ADJ_WEAK_ALL_AROUND = 0.45;
const ADJ_STRONG_TEXT_EVIDENCE = 3;
const ADJ_TEXT_UNIQUE_MARGIN = 2;
const ADJ_SECONDARY_STRONG_CONF = 0.6;
const ADJ_MIN_TEXT_SUPPORT = 2;
const PHRASE_WEIGHT = 3;
const SHORT_TEXT_MAX_TOKENS = 8;

const PURPOSE_GROUPS: string[][] = [
  [
    'job',
    'jobs',
    'career',
    'careers',
    'hiring',
    'recruiter',
    'recruiting',
    'recruitment',
    'vacancy',
    'position',
    'opening',
    'openings',
    'candidate',
    'apply',
    'application',
    'employer',
    'employment',
    'interview',
    'emploi',
    'emplois',
    'poste',
    'candidature',
    'recrutement',
    'embauche',
    'recruteur',
    'developpeur',
  ],
  [
    'grade',
    'grades',
    'grading',
    'mark',
    'marks',
    'result',
    'results',
    'exam',
    'exams',
    'score',
    'scores',
    'transcript',
    'assessment',
    'resultat',
    'resultats',
    'examen',
    'bulletin',
    'evaluation',
  ],
  [
    'bank',
    'banking',
    'transaction',
    'transactions',
    'statement',
    'statements',
    'card',
    'payment',
    'payments',
    'balance',
    'transfer',
    'deposit',
    'withdrawal',
    'overdraft',
    'compte',
    'carte',
    'virement',
    'releve',
    'paiement',
    'banque',
  ],
  [
    'security',
    'login',
    'signin',
    'password',
    'passcode',
    'verification',
    'verify',
    'verified',
    'authentication',
    'authenticate',
    'otp',
    'breach',
    'unauthorized',
    'connexion',
    'identifiant',
    'verification',
    'securite',
    'authentification',
  ],
  [
    'travel',
    'trip',
    'hotel',
    'hotels',
    'flight',
    'flights',
    'booking',
    'bookings',
    'reservation',
    'accommodation',
    'itinerary',
    'airfare',
    'fare',
    'voyage',
    'vol',
    'vols',
    'hotel',
    'sejour',
    'hebergement',
    'billet',
  ],
  [
    'social',
    'post',
    'posted',
    'posts',
    'comment',
    'comments',
    'video',
    'videos',
    'follow',
    'follower',
    'followers',
    'profile',
    'mention',
    'mentions',
    'like',
    'likes',
    'tweet',
    'story',
    'publication',
    'abonne',
    'video',
    'commentaire',
  ],
  [
    'invoice',
    'invoices',
    'receipt',
    'receipts',
    'bill',
    'billing',
    'purchase',
    'subscription',
    'renewal',
    'charged',
    'facture',
    'factures',
    'recu',
    'achat',
    'abonnement',
    'renouvellement',
  ],
  [
    'delivery',
    'deliveries',
    'delivered',
    'shipping',
    'shipped',
    'shipment',
    'tracking',
    'dispatch',
    'parcel',
    'package',
    'courier',
    'livraison',
    'expedition',
    'suivi',
    'colis',
    'commande',
    'expedie',
  ],
  [
    'marketing',
    'promo',
    'promotion',
    'promotions',
    'discount',
    'offer',
    'offers',
    'deal',
    'deals',
    'sale',
    'coupon',
    'newsletter',
    'savings',
    'remise',
    'reduction',
    'promotionnel',
  ],
  [
    'developer',
    'code',
    'review',
    'reviews',
    'pull',
    'request',
    'commit',
    'commits',
    'repository',
    'repo',
    'merge',
    'branch',
    'build',
    'builds',
    'deploy',
    'deployment',
    'depot',
    'fusion',
  ],
  [
    'api',
    'apikey',
    'endpoint',
    'quota',
    'webhook',
    'sdk',
    'platform',
    'integration',
    'credential',
    'credentials',
    'sandbox',
  ],
  [
    'course',
    'courses',
    'assignment',
    'assignments',
    'lecture',
    'lectures',
    'syllabus',
    'homework',
    'class',
    'classes',
    'reading',
    'material',
    'materials',
    'module',
    'cours',
    'devoir',
    'devoirs',
  ],
  [
    'health',
    'medical',
    'insurance',
    'pharmacy',
    'prescription',
    'claim',
    'claims',
    'policy',
    'appointment',
    'doctor',
    'clinic',
    'sante',
    'assurance',
    'medical',
    'pharmacie',
    'remboursement',
    'ordonnance',
    'mutuelle',
  ],
  [
    'network',
    'networking',
    'connection',
    'connections',
    'connect',
    'invitation',
    'endorsement',
    'recommendation',
    'colleague',
    'reseau',
    'relation',
    'recommandation',
  ],
];

/** Split text into a set of lowercased, accent-stripped word tokens of at least three characters. */
function tokenizeWords(text: string): Set<string> {
  const normalized = text.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
  return new Set((normalized.match(/[\p{L}\p{N}]+/gu) ?? []).filter((t) => t.length >= 3));
}

const evidenceWordsCache = new Map<string, Set<string>>();
/**
 * Build the set of words that count as evidence for a category: its own label/description tokens
 * plus the best-matching canonical purpose group. Results are cached by label and description.
 */
function categoryEvidenceWords(label: string, description: string | null): Set<string> {
  const key = JSON.stringify([label, description ?? '']);
  const cached = evidenceWordsCache.get(key);
  if (cached) return cached;
  const labelTokens = tokenizeWords(label);
  let best: string[] | null = null;
  let bestScore = 0;
  for (const group of PURPOSE_GROUPS) {
    const score = group.reduce((n, g) => n + (labelTokens.has(g) ? 1 : 0), 0);
    if (score > bestScore) {
      bestScore = score;
      best = group;
    }
  }
  const words = tokenizeWords(`${label} ${description ?? ''}`);
  if (best) for (const w of best) words.add(w);
  evidenceWordsCache.set(key, words);
  return words;
}

/** Count how many of a category's evidence words appear in the given token set. */
function evidenceCount(
  words: Set<string>,
  category: { label: string; description: string | null } | undefined,
): number {
  if (!category) return 0;
  const evidence = categoryEvidenceWords(category.label, category.description);
  let count = 0;
  for (const w of evidence) if (words.has(w)) count += 1;
  return count;
}

const PURPOSE_PHRASES: string[][] = [
  ['job opportunity', 'job opening', 'we are hiring', 'apply now', 'fit for', 'offre d emploi'],
  ['your grades', 'exam results', 'final grade', 'grade report'],
  ['transaction alert', 'payment received', 'account statement', 'low balance'],
  [
    'password reset',
    'security alert',
    'verification code',
    'sign in attempt',
    'verify your account',
    'suspicious activity',
  ],
  ['booking confirmation', 'your reservation', 'your itinerary', 'price drop', 'travel deal'],
  ['posted a', 'new follower', 'tagged you', 'mentioned you', 'shared a'],
  [
    'payment confirmation',
    'invoice attached',
    'your receipt',
    'order confirmation',
    'your invoice',
  ],
  ['out for delivery', 'order shipped', 'tracking number', 'has been delivered', 'on its way'],
  ['special offer', 'limited time', 'off your', 'exclusive deal', 'save up to'],
  ['pull request', 'merge request', 'code review', 'build failed', 'review requested'],
  ['api key', 'access token', 'token expir', 'personal access'],
  ['assignment due', 'course material', 'lecture notes', 'due date', 'submit your assignment'],
  ['appointment reminder', 'insurance claim', 'prescription ready', 'test results'],
  [
    'professional network',
    'connection request',
    'connect with you',
    'join your network',
    'invitation to connect',
    'wants to connect',
  ],
];

const GENERIC_LABEL_TOKENS = new Set([
  'updates',
  'update',
  'notification',
  'notifications',
  'alert',
  'alerts',
  'email',
  'emails',
  'message',
  'messages',
  'account',
  'accounts',
  'new',
  'info',
  'general',
  'inbox',
  'mail',
]);

/** Lowercase, strip accents, and collapse non-alphanumeric runs to single spaces for phrase matching. */
function normalizeText(text: string): string {
  return text
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, ' ')
    .trim();
}

/** Count how many of a category's purpose phrases occur in the normalized text. */
function phraseEvidenceStrength(
  normalized: string,
  category: { label: string; description: string | null } | undefined,
): number {
  if (!category) return 0;
  const sig = purposeSignature(category.label, category.description);
  if (sig === null) return 0;
  const padded = ` ${normalized} `;
  return (PURPOSE_PHRASES[sig] ?? []).reduce((n, p) => n + (padded.includes(` ${p} `) ? 1 : 0), 0);
}

/** Crude singularizer that drops a trailing s from words longer than three characters. */
function singular(word: string): string {
  return word.length > 3 && word.endsWith('s') ? word.slice(0, -1) : word;
}

/** Count distinct non-generic label tokens whose stem appears in the email text. */
function labelTokenSupport(words: Set<string>, label: string): number {
  const stems = new Set([...words].map(singular));
  let count = 0;
  for (const t of tokenizeWords(label)) {
    const stem = singular(t);
    if (!GENERIC_LABEL_TOKENS.has(t) && !GENERIC_LABEL_TOKENS.has(stem) && stems.has(stem)) {
      count += 1;
    }
  }
  return count;
}

/** Count how many words from the category's matched purpose group appear in the token set. */
function groupEvidenceCount(
  words: Set<string>,
  category: { label: string; description: string | null } | undefined,
): number {
  if (!category) return 0;
  const sig = purposeSignature(category.label, category.description);
  if (sig === null) return 0;
  return PURPOSE_GROUPS[sig]!.reduce((n, g) => n + (words.has(g) ? 1 : 0), 0);
}

/** Breakdown of how strongly an email's text supports a category: purpose words, phrases, and label tokens. */
export interface PurposeTextEvidence {
  words: number;
  phrases: number;
  labelTokens: number;
}

/** Compute the purpose words, phrases, and label tokens a category has in the given email text. */
export function purposeTextEvidence(
  text: string,
  category: { label: string; description: string | null },
): PurposeTextEvidence {
  const words = tokenizeWords(text);
  const normalized = normalizeText(text);
  return {
    words: groupEvidenceCount(words, category),
    phrases: phraseEvidenceStrength(normalized, category),
    labelTokens: labelTokenSupport(words, category.label),
  };
}

/** Whether the normalized text is short enough to apply the relaxed short-text evidence rules. */
function isShortText(normalized: string): boolean {
  return normalized.split(' ').filter(Boolean).length <= SHORT_TEXT_MAX_TOKENS;
}

interface EvidenceParts {
  w: number;
  p: number;
  labeled: boolean;
}
/**
 * Compute the per-category evidence parts for one email: purpose-word count, phrase count, and a
 * short-text label-match flag that only applies when the text is short.
 */
function evidenceParts(
  words: Set<string>,
  normalized: string,
  category: { label: string; description: string | null } | undefined,
  short: boolean,
): EvidenceParts {
  return {
    w: groupEvidenceCount(words, category),
    p: phraseEvidenceStrength(normalized, category),
    labeled:
      short &&
      !!category &&
      labelTokenSupport(words, category.label) >= 1 &&
      groupEvidenceCount(words, category) >= 1,
  };
}

/** Whether the evidence parts amount to strong text support for a category. */
function isStrongText(parts: EvidenceParts): boolean {
  return parts.w >= ADJ_STRONG_TEXT_EVIDENCE || (parts.w >= 1 && parts.p >= 1);
}
/** Whether the evidence parts meet the minimum bar to count as any text support. */
function isSupported(parts: EvidenceParts): boolean {
  return parts.w >= ADJ_MIN_TEXT_SUPPORT || (parts.w >= 1 && parts.p >= 1) || parts.labeled;
}
/** Numeric strength of text support, weighting phrases more heavily, or zero when unsupported. */
function supportMagnitude(parts: EvidenceParts): number {
  if (!isSupported(parts)) return 0;
  return parts.w + PHRASE_WEIGHT * parts.p + (parts.labeled ? ADJ_MIN_TEXT_SUPPORT : 0);
}

/** Count distinct purpose groups among the high-confidence embedding matches. */
function distinctStrongPurposes(
  ranked: CategoryMatch[],
  evidence: AdjudicationEvidence | undefined,
): number {
  return new Set(
    ranked
      .filter((m) => m.confidence >= ADJ_STRONG_TOP_CONFIDENCE)
      .map((m) => signatureOf(m.categoryId, evidence))
      .filter((s): s is number => s !== null),
  ).size;
}

/** Count of DISTINCT purpose words for the category found in the email text. */
export function textEvidenceStrength(
  text: string,
  category: { label: string; description: string | null } | undefined,
): number {
  return evidenceCount(tokenizeWords(text), category);
}

/** Whether the email text lexically supports the category at all (at least one purpose word). */
export function hasTextEvidence(
  text: string,
  category: { label: string; description: string | null } | undefined,
): boolean {
  return textEvidenceStrength(text, category) >= 1;
}

/**
 * Index of the canonical purpose group a category's label/description best matches, or null when it
 * matches none. Two categories sharing a signature are candidate twins (same broad purpose), used as
 * a coarse filter that callers confirm with embedding/label similarity before merging.
 */
export function purposeSignature(label: string, description?: string | null): number | null {
  const tokens = tokenizeWords(`${label} ${description ?? ''}`);
  let best = -1;
  let bestScore = 0;
  for (let i = 0; i < PURPOSE_GROUPS.length; i++) {
    const score = PURPOSE_GROUPS[i]!.reduce((n, g) => n + (tokens.has(g) ? 1 : 0), 0);
    if (score > bestScore) {
      bestScore = score;
      best = i;
    }
  }
  return bestScore > 0 ? best : null;
}

/** Text and category lookup the adjudicator uses to score candidates against an email's content. */
export interface AdjudicationEvidence {
  text: string;
  categoryById: Map<string, { label: string; description: string | null }>;
}

/** Result of adjudication: the chosen category ids and the reason code for the decision. */
export interface Adjudication {
  ids: string[];
  reason:
    | 'accepted_close_to_top'
    | 'accepted_text_evidence'
    | 'accepted_new_category'
    | 'overridden_strong_embedding'
    | 'overridden_by_text'
    | 'rejected_low_rank'
    | 'rejected_weak_all_around'
    | 'none';
}

/**
 * Adjudicate the LLM's category choice against embedding ranking and generic text evidence, so the
 * correct category can win even when the LLM picked a different weak one. A second label is only added
 * when it independently clears a high embedding confidence or has strong text evidence of its own.
 */
export function adjudicate(
  chosenIds: string[],
  ranked: CategoryMatch[],
  evidence?: AdjudicationEvidence,
): Adjudication {
  if (chosenIds.length === 0) return { ids: [], reason: 'none' };
  if (ranked.length === 0) {
    return withSecondary(chosenIds[0]!, chosenIds, ranked, evidence, 'accepted_new_category');
  }
  const top = ranked[0]!;
  const second = ranked[1];
  const confById = new Map(ranked.map((m) => [m.categoryId, m.confidence]));

  const selected = chosenIds[0]!;
  const selectedConf = confById.get(selected);
  const margin = top.confidence - (selectedConf ?? 0);

  if (selectedConf === undefined) {
    return withSecondary(selected, chosenIds, ranked, evidence, 'accepted_new_category');
  }

  const words = evidence ? tokenizeWords(evidence.text) : new Set<string>();
  const normalized = evidence ? normalizeText(evidence.text) : '';
  const short = isShortText(normalized);
  const partsOf = (id: string): EvidenceParts =>
    evidenceParts(words, normalized, evidence?.categoryById.get(id), short);
  const openPartsOf = (id: string): EvidenceParts =>
    evidenceParts(words, normalized, evidence?.categoryById.get(id), false);
  const textSupport = (id: string): boolean => isSupported(partsOf(id));

  if (selected === top.categoryId && top.confidence >= ADJ_WEAK_ALL_AROUND) {
    return withSecondary(selected, chosenIds, ranked, evidence, 'accepted_close_to_top');
  }

  const winner = uniqueTextWinner(chosenIds, ranked, evidence);
  if (winner) {
    const reason = winner === selected ? 'accepted_text_evidence' : 'overridden_by_text';
    return withSecondary(winner, chosenIds, ranked, evidence, reason);
  }

  if (top.confidence < ADJ_WEAK_ALL_AROUND) {
    return { ids: [], reason: 'rejected_weak_all_around' };
  }

  if (isSupportedTopTwoPick(selected, ranked, evidence)) {
    return withSecondary(selected, chosenIds, ranked, evidence, 'accepted_close_to_top');
  }

  if (top.confidence >= ADJ_STRONG_TOP_CONFIDENCE && margin >= ADJ_TOP_OVERRIDE_MARGIN) {
    if (isSupported(openPartsOf(selected)) && openPartsOf(top.categoryId).w === 0) {
      return withSecondary(selected, chosenIds, ranked, evidence, 'accepted_text_evidence');
    }
    if (distinctStrongPurposes(ranked, evidence) >= 2) {
      return { ids: [], reason: 'rejected_low_rank' };
    }
    const confirmed =
      isSupported(partsOf(top.categoryId)) ||
      (!!second &&
        second.confidence >= ADJ_STRONG_TOP_CONFIDENCE &&
        samePurposeIds(top.categoryId, second.categoryId, evidence)) ||
      top.confidence >= ADJ_VERY_STRONG_TOP;
    return confirmed
      ? withSecondary(top.categoryId, chosenIds, ranked, evidence, 'overridden_strong_embedding')
      : { ids: [], reason: 'rejected_low_rank' };
  }

  if (margin <= ADJ_CLOSE_TO_TOP_WINDOW) {
    if (!knownDifferentPurpose(selected, top.categoryId, evidence) || textSupport(selected)) {
      return withSecondary(selected, chosenIds, ranked, evidence, 'accepted_close_to_top');
    }
    if (textSupport(top.categoryId)) {
      return withSecondary(top.categoryId, chosenIds, ranked, evidence, 'overridden_by_text');
    }
    return { ids: [], reason: 'rejected_low_rank' };
  }

  return { ids: [], reason: 'rejected_low_rank' };
}

/** Purpose-group signature for a category id, or null when the category or signature is unknown. */
function signatureOf(id: string, evidence: AdjudicationEvidence | undefined): number | null {
  const cat = evidence?.categoryById.get(id);
  return cat ? purposeSignature(cat.label, cat.description) : null;
}

/** Whether two category ids share the same known purpose signature. */
function samePurposeIds(
  idA: string,
  idB: string,
  evidence: AdjudicationEvidence | undefined,
): boolean {
  const sigA = signatureOf(idA, evidence);
  return sigA !== null && sigA === signatureOf(idB, evidence);
}

/** Whether both category ids have known signatures that differ, marking distinct purposes. */
function knownDifferentPurpose(
  idA: string,
  idB: string,
  evidence: AdjudicationEvidence | undefined,
): boolean {
  const sigA = signatureOf(idA, evidence);
  const sigB = signatureOf(idB, evidence);
  return sigA !== null && sigB !== null && sigA !== sigB;
}

/**
 * Conservative deterministic assignment used ONLY when the LLM output for a cluster is unusable after
 * retry: assign the embedding top when it is STRONG, text-supported, and not contested by a different
 * strong purpose. Returns the category id, or null to leave the email uncategorized (never a guess).
 */
export function deterministicFallback(
  ranked: CategoryMatch[],
  evidence?: AdjudicationEvidence,
): string | null {
  const top = ranked[0];
  if (!top || top.confidence < ADJ_STRONG_TOP_CONFIDENCE) return null;
  const words = evidence ? tokenizeWords(evidence.text) : new Set<string>();
  const normalized = evidence ? normalizeText(evidence.text) : '';
  const short = isShortText(normalized);
  if (
    !isSupported(
      evidenceParts(words, normalized, evidence?.categoryById.get(top.categoryId), short),
    )
  ) {
    return null;
  }
  if (isFallbackBlockedByCompetingPurpose(ranked, evidence)) return null;
  return top.categoryId;
}

/**
 * Whether the deterministic fallback should be blocked because a near-equal strong runner-up of a
 * different purpose contests the top match without the top having a clear text-support margin.
 */
function isFallbackBlockedByCompetingPurpose(
  ranked: CategoryMatch[],
  evidence: AdjudicationEvidence | undefined,
): boolean {
  const top = ranked[0];
  const second = ranked[1];
  if (!top || !second || second.confidence < ADJ_STRONG_TOP_CONFIDENCE) return false;
  if (samePurposeIds(top.categoryId, second.categoryId, evidence)) return false;
  if (!evidence) return true;
  const words = tokenizeWords(evidence.text);
  const normalized = normalizeText(evidence.text);
  const short = isShortText(normalized);
  const topSupport = supportMagnitude(
    evidenceParts(words, normalized, evidence.categoryById.get(top.categoryId), short),
  );
  const secondSupport = supportMagnitude(
    evidenceParts(words, normalized, evidence.categoryById.get(second.categoryId), short),
  );
  return topSupport < secondSupport + ADJ_TEXT_UNIQUE_MARGIN;
}

/**
 * Whether the selected category sits at rank one or two, has text support, is strong on embedding,
 * and is not clearly out-supported by the top match, making it safe to accept.
 */
function isSupportedTopTwoPick(
  selected: string,
  ranked: CategoryMatch[],
  evidence: AdjudicationEvidence | undefined,
): boolean {
  if (!evidence) return false;
  const rank = ranked.findIndex((m) => m.categoryId === selected);
  if (rank !== 0 && rank !== 1) return false;
  const top = ranked[0]!;
  const sel = ranked[rank]!;
  const words = tokenizeWords(evidence.text);
  const normalized = normalizeText(evidence.text);
  const short = isShortText(normalized);
  const selParts = evidenceParts(words, normalized, evidence.categoryById.get(selected), short);
  if (!isSupported(selParts) || distinctStrongPurposes(ranked, evidence) >= 3) return false;
  const strong =
    sel.confidence >= ADJ_STRONG_TOP_CONFIDENCE ||
    top.confidence - sel.confidence <= ADJ_CLOSE_TO_TOP_WINDOW;
  if (!strong) return false;
  const topParts = evidenceParts(
    words,
    normalized,
    evidence.categoryById.get(top.categoryId),
    short,
  );
  return supportMagnitude(topParts) <= supportMagnitude(selParts) + ADJ_TEXT_UNIQUE_MARGIN;
}

/**
 * Pick the single category whose text evidence is strong and uncontested by a different purpose, or
 * null when none stands out. Prefers the LLM's pick when it ties for the top purpose.
 */
function uniqueTextWinner(
  chosenIds: string[],
  ranked: CategoryMatch[],
  evidence?: AdjudicationEvidence,
): string | null {
  if (!evidence) return null;
  const words = tokenizeWords(evidence.text);
  const normalized = normalizeText(evidence.text);
  const confById = new Map(ranked.map((m) => [m.categoryId, m.confidence]));
  const ids = new Set<string>([...ranked.map((m) => m.categoryId), ...chosenIds]);
  const scored = [...ids]
    .map((id) => {
      const cat = evidence.categoryById.get(id);
      const parts = evidenceParts(words, normalized, cat, false);
      return {
        id,
        parts,
        strength: supportMagnitude(parts),
        sig: cat ? purposeSignature(cat.label, cat.description) : null,
        conf: confById.get(id) ?? -1,
      };
    })
    .filter((c) => isStrongText(c.parts))
    .sort((a, b) => b.strength - a.strength);
  const best = scored[0];
  if (!best) return null;
  const conflict = scored.some(
    (c) =>
      c.sig !== null &&
      best.sig !== null &&
      c.sig !== best.sig &&
      best.strength - c.strength <= ADJ_TEXT_UNIQUE_MARGIN - 1,
  );
  if (conflict) return null;
  const sameTop = scored.filter((c) => c.sig === best.sig);
  const llmPick = sameTop.find((c) => c.id === chosenIds[0]);
  return (llmPick ?? sameTop.sort((a, b) => b.conf - a.conf)[0]!).id;
}

/**
 * Build the final adjudication from a primary id, optionally adding one second label from the LLM's
 * remaining picks when it has a different purpose and clears strong text or grounded embedding support.
 */
function withSecondary(
  primary: string,
  chosenIds: string[],
  ranked: CategoryMatch[],
  evidence: AdjudicationEvidence | undefined,
  reason: Adjudication['reason'],
): Adjudication {
  const confById = new Map(ranked.map((m) => [m.categoryId, m.confidence]));
  const words = evidence ? tokenizeWords(evidence.text) : new Set<string>();
  const normalized = evidence ? normalizeText(evidence.text) : '';
  const short = isShortText(normalized);
  const primaryCat = evidence?.categoryById.get(primary);
  const primarySig = evidence
    ? purposeSignature(primaryCat?.label ?? '', primaryCat?.description)
    : null;
  const ids = [primary];
  for (const id of chosenIds) {
    if (ids.length >= MAX_LABELS_PER_EMAIL) break;
    if (ids.includes(id)) continue;
    const cat = evidence?.categoryById.get(id);
    if (primarySig !== null && cat && purposeSignature(cat.label, cat.description) === primarySig) {
      continue;
    }
    const conf = confById.get(id);
    const strongEmbed = conf !== undefined && conf >= ADJ_SECONDARY_STRONG_CONF;
    if (!evidence) {
      if (strongEmbed) ids.push(id);
      continue;
    }
    const parts = evidenceParts(words, normalized, cat, short);
    const groundedEmbed = strongEmbed && groupEvidenceCount(words, cat) >= 1;
    if (isStrongText(parts) || groundedEmbed) ids.push(id);
  }
  return { ids, reason };
}

const SHORTLIST_MAX = 12;
const SHORTLIST_WINDOW = 0.08;

const GATE_MIN_CONFIDENCE = 0.8;
const GATE_SECONDARY_MAX = 0.5;

/**
 * Categories to offer the LLM for one email, ranked best-first. Sends all when there are few,
 * otherwise the top MAX. Severe centroid collapse also falls back to all categories.
 */
export function shortlistFor(ranked: CategoryMatch[], totalCategories: number): CategoryMatch[] {
  if (ranked.length === 0) return [];
  if (totalCategories <= SHORTLIST_MAX) return ranked;
  const best = ranked[0]!.confidence;
  const within = ranked.filter((c) => best - c.confidence <= SHORTLIST_WINDOW).length;
  if (within > SHORTLIST_MAX) return ranked;
  return ranked.slice(0, SHORTLIST_MAX);
}

/**
 * Return the category to auto-assign when the embedding is confident enough to judge alone,
 * otherwise null to send to the LLM. Requires high absolute confidence and a clearly inapplicable
 * runner-up.
 */
export function gateDecision(ranked: CategoryMatch[]): CategoryMatch | null {
  const top = ranked[0];
  if (!top || top.confidence < GATE_MIN_CONFIDENCE) return null;
  const second = ranked[1];
  if (second && second.confidence > GATE_SECONDARY_MAX) return null;
  return top;
}

const FAST_GATE_MIN_CONFIDENCE = 0.78;
const FAST_GATE_MIN_MARGIN = 0.1;

/**
 * Category the fast pass may auto-assign, or null to defer to the LLM. Requires the top match to
 * clear an absolute confidence floor and beat the second by a clear margin.
 */
export function gateFastAssignment(ranked: CategoryMatch[]): CategoryMatch | null {
  const top = ranked[0];
  if (!top || top.confidence < FAST_GATE_MIN_CONFIDENCE) return null;
  const second = ranked[1];
  if (second && top.confidence - second.confidence < FAST_GATE_MIN_MARGIN) return null;
  return top;
}
