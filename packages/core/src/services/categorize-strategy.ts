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