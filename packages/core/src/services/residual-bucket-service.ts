/**
 * Covers the uncategorized long tail with a small curated set of broad "residual" buckets
 * (Newsletters, Promotions, Notifications) whose membership is decided by high-precision SIGNAL
 * matchers over sender + subject, not by centroid distance. A single blurry centroid cannot represent
 * such diverse mail, which is why this mail is left uncategorized today. Precision-first: an email is
 * only bucketed on a clear signal, otherwise it stays uncategorized rather than being misfiled.
 *
 * When a bucket's purpose already exists as an active category (e.g. "Marketing and Promotions"), the
 * matching mail is assigned to THAT category (widening it) instead of creating a near-duplicate.
 * Nothing here reads embeddings or calls the LLM.
 */
import type { Logger } from 'pino';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmailRepository, EmailSummary } from '../repositories/email-repository.js';

/** Fewest uncategorized matches a bucket needs before it is worth creating/assigning. */
const MIN_BUCKET_COVERAGE = 25;
/** Cap on the uncategorized pool scanned per run, so a huge backlog stays bounded. */
const SCAN_LIMIT = 50_000;
const ASSIGN_CONFIDENCE = 0.9;

/** A curated broad bucket for the long tail, matched deterministically from sender + subject. */
interface ResidualBucketDef {
  key: string;
  label: string;
  description: string;
  /** True when this bucket clearly owns the email, from its lowercased subject and sender. */
  match(subject: string, sender: string): boolean;
  /** An existing active category whose label matches this is widened instead of creating a duplicate. */
  widens: RegExp;
}

const has = (text: string, re: RegExp): boolean => re.test(text);

/** Security / transactional signals that must never be swept into a broad bucket. */
const TRANSACTIONAL =
  /\b(verify|verification|password|passcode|one[- ]?time|sign[- ]?in|log[- ]?in|2fa|otp|receipt|invoice|payment|refund|order (confirmed|placed|number)|shipped|tracking|delivered|statement|balance|transaction)\b/;

const BUCKETS: ResidualBucketDef[] = [
  {
    key: 'newsletters_digests',
    label: 'Newsletters & Digests',
    description: 'Recurring newsletters, digests, and content roundups you subscribed to.',
    widens: /newsletter|digest/i,
    match: (s, f) =>
      !has(s, TRANSACTIONAL) &&
      (has(
        s,
        /\b(digest|newsletter|round[- ]?up|weekly|this week|top stories|new post|latest (posts|stories|articles)|read of the (day|week)|edition|bulletin)\b/,
      ) ||
        has(f, /(newsletter|digest|substack|@.*\bnews\b|noreply@.*(medium|wordpress|substack))/)),
  },
  {
    key: 'promotions_deals',
    label: 'Promotions & Deals',
    description: 'Marketing offers, discounts, sales, and promotional deals from brands.',
    widens: /marketing|promotion|promo|\bdeals?\b/i,
    match: (s) =>
      !has(s, TRANSACTIONAL) &&
      has(
        s,
        /(\d+\s?%\s?off|\bpercent off\b|\bsale\b|\bdeal(s)?\b|\bdiscount\b|\bcoupon\b|\bpromo\b|\boffer(s)?\b|\bsave\b|\blimited time\b|\bshop now\b|\bclearance\b|\bflash sale\b|\bvoucher\b|\bcashback\b|\bblack friday\b|\bcyber monday\b|price drop)/,
      ),
  },
  {
    key: 'notifications_alerts',
    label: 'Notifications & Alerts',
    description:
      'Automated activity notifications and reminders that are not security or receipts.',
    widens: /notification|alert/i,
    match: (s, f) =>
      !has(s, TRANSACTIONAL) &&
      has(
        f,
        /(no[- ]?reply|noreply|donotreply|do[- ]?not[- ]?reply|notifications?@|mailer|automated)/,
      ) &&
      has(
        s,
        /\b(notification|alert|reminder|activity|new (message|comment|reply|follower|connection)|mentioned you|liked|invited you|weekly summary|recap)\b/,
      ),
  },
];

/** One bucket's coverage: how much of the backlog it matches and where it would be filed. */
export interface ResidualBucketPlan {
  key: string;
  label: string;
  description: string;
  matched: number;
  /** An existing active category this bucket's mail should widen, or null to create the bucket. */
  targetCategoryId: string | null;
  targetLabel: string;
}

/** Outcome of covering the residual: what was created and how much was filed. */
export interface CoverResidualResult {
  scanned: number;
  buckets: Array<{ label: string; created: boolean; assigned: number }>;
  totalAssigned: number;
}

/** Detects and fills broad residual buckets from the uncategorized backlog by deterministic signal. */
export class ResidualBucketService {
  constructor(
    private categories: CategoryRepository,
    private emails: EmailRepository,
    private logger: Logger,
    private enabled: () => boolean = () => false,
  ) {}

  /** The uncategorized pool to scan, preferring the stable reader when available. */
  private uncategorized(accountId: string): EmailSummary[] {
    const reader = this.emails as EmailRepository & {
      listUncategorizedSummariesStable?: (accountId: string, limit: number) => EmailSummary[];
    };
    return reader.listUncategorizedSummariesStable
      ? reader.listUncategorizedSummariesStable(accountId, SCAN_LIMIT)
      : this.emails.listUncategorizedSummaries(accountId, SCAN_LIMIT);
  }

  /**
   * Plan (read-only) which buckets clear the coverage floor, how many emails each matches, and
   * whether they would widen an existing category or create a new one.
   */
  detect(accountId: string): ResidualBucketPlan[] {
    return this.detectFrom(accountId, this.uncategorized(accountId));
  }

  /**
   * Create the qualifying buckets (or reuse an existing same-purpose category) and file every matching
   * uncategorized email into them by signal. No-op unless enabled. Returns what was created/assigned.
   */
  coverResidual(accountId: string): CoverResidualResult {
    if (!this.enabled()) return { scanned: 0, buckets: [], totalAssigned: 0 };

    const pool = this.uncategorized(accountId);
    const plans = this.detectFrom(accountId, pool);
    const now = Date.now();
    const out: CoverResidualResult['buckets'] = [];
    let totalAssigned = 0;

    for (const plan of plans) {
      const def = BUCKETS.find((b) => b.key === plan.key)!;
      const matched = pool.filter((e) =>
        def.match((e.subject ?? '').toLowerCase(), (e.fromAddr ?? '').toLowerCase()),
      );
      if (matched.length < MIN_BUCKET_COVERAGE) continue;

      let categoryId = plan.targetCategoryId;
      let created = false;
      if (!categoryId) {
        const row = this.categories.create({
          accountId,
          label: plan.label,
          description: plan.description,
          source: 'auto',
        });
        categoryId = row.id;
        created = true;
      }
      this.categories.addAutoAssignments(
        accountId,
        matched.map((e) => ({
          messageId: e.messageId,
          accountId,
          categoryId: categoryId!,
          confidence: ASSIGN_CONFIDENCE,
          assignedBy: 'auto' as const,
          assignedAt: now,
          method: 'gate' as const,
        })),
      );
      totalAssigned += matched.length;
      out.push({ label: plan.targetLabel, created, assigned: matched.length });
    }

    this.logger.info(
      { accountId, buckets: out.length, totalAssigned },
      'residual buckets: covered uncategorized long tail',
    );
    return { scanned: pool.length, buckets: out, totalAssigned };
  }

  /** detect() over an already-loaded pool, so coverResidual scans the backlog once. */
  private detectFrom(accountId: string, pool: EmailSummary[]): ResidualBucketPlan[] {
    const active = this.categories.listActive(accountId);
    const plans: ResidualBucketPlan[] = [];
    for (const bucket of BUCKETS) {
      const matched = pool.filter((e) =>
        bucket.match((e.subject ?? '').toLowerCase(), (e.fromAddr ?? '').toLowerCase()),
      );
      if (matched.length < MIN_BUCKET_COVERAGE) continue;
      // Widen an existing category whose label already covers this purpose, else create the bucket.
      const target = active.find((c) => bucket.widens.test(c.label)) ?? null;
      plans.push({
        key: bucket.key,
        label: bucket.label,
        description: bucket.description,
        matched: matched.length,
        targetCategoryId: target?.id ?? null,
        targetLabel: target?.label ?? bucket.label,
      });
    }
    return plans;
  }
}
