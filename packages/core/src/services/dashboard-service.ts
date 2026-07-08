/**
 * Builds read-only dashboard snapshots aggregating email, triage, and category
 * state for one account using in-memory and single SQL queries only.
 */
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmailRepository } from '../repositories/email-repository.js';
import type {
  EmailUserLabelRepository,
  UserLabelStat,
} from '../repositories/email-user-label-repository.js';
import type { TriageBucket, TriageRepository } from '../repositories/triage-repository.js';

/** Minimal email projection shown in dashboard lists. */
export interface DashboardEmail {
  messageId: string;
  folder: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  hasAttachments: boolean;
}

/** A triaged email with its classification reasoning and timestamp. */
export interface DashboardUrgentEmail extends DashboardEmail {
  reasoning: string | null;
  classifiedAt: number;
}

/** A category with its label, description, and count of assigned emails. */
export interface DashboardCategorySummary {
  id: string;
  label: string;
  description: string | null;
  emailCount: number;
}

/** The user's own Thunderbird organization detected during sync, surfaced read-only. */
export interface UserOrganization {
  tagCount: number;
  folderCount: number;
  topTags: UserLabelStat[];
  topFolders: UserLabelStat[];
}

/** Aggregated snapshot of email, triage, and category state for one account. */
export interface Dashboard {
  accountId: string;
  generatedAt: number;
  emails: {
    total: number;
    unclassified: number;
    uncategorized: number;
  };
  triage: {
    buckets: Record<TriageBucket, number>;
    urgent: DashboardUrgentEmail[];
    summarize: DashboardUrgentEmail[];
  };
  recent: DashboardEmail[];
  categoryCount: number;
  categories: DashboardCategorySummary[];
  userOrganization: UserOrganization;
}

/** Limits controlling how many rows each dashboard section includes. */
export interface DashboardOptions {
  urgentLimit?: number;
  summarizeLimit?: number;
  recentLimit?: number;
  topCategoriesLimit?: number;
  topLabelsLimit?: number;
}

const DEFAULTS: Required<DashboardOptions> = {
  urgentLimit: 20,
  summarizeLimit: 20,
  recentLimit: 30,
  topCategoriesLimit: 24,
  topLabelsLimit: 12,
};

/** Builds dashboard snapshots from email, triage, and category repositories. */
export class DashboardService {
  /** Wires the service to the email, triage, and category repositories it reads from. */
  constructor(
    private emails: EmailRepository,
    private triage: TriageRepository,
    private categories: CategoryRepository,
    private userLabels: EmailUserLabelRepository,
  ) {}

  /**
   * Aggregate a single snapshot for the dashboard view.
   * In-memory or single SQL queries only, no LLM or embedding work.
   */
  build(accountId: string, options: DashboardOptions = {}): Dashboard {
    const opts = { ...DEFAULTS, ...options };

    const total = this.emails.count(accountId);
    const unclassified = this.triage.countUnclassified(accountId);
    const uncategorized = this.categories.countUncategorized(accountId);
    const bucketCounts = this.triage.countByBucket(accountId);
    const buckets: Record<TriageBucket, number> = {
      urgent: 0,
      summarize: 0,
      spam: 0,
      personal: 0,
    };
    for (const c of bucketCounts) buckets[c.bucket] = c.count;

    const urgentRows = this.triage.listByBucket(accountId, 'urgent', opts.urgentLimit);
    const summarizeRows = this.triage.listByBucket(accountId, 'summarize', opts.summarizeLimit);

    const recentSummaries = this.emails.listSummaries({ accountId, limit: opts.recentLimit });

    // Active only: suggested and retired categories never appear on the dashboard.
    const activeCategories = this.categories.listActive(accountId);
    const topCategories = [...activeCategories]
      .sort((a, b) => b.emailCount - a.emailCount)
      .slice(0, opts.topCategoriesLimit)
      .map((c) => ({
        id: c.id,
        label: c.label,
        description: c.description,
        emailCount: c.emailCount,
      }));

    return {
      accountId,
      generatedAt: Date.now(),
      emails: { total, unclassified, uncategorized },
      triage: {
        buckets,
        urgent: urgentRows.map(toDashboardClassified),
        summarize: summarizeRows.map(toDashboardClassified),
      },
      recent: recentSummaries.map((e) => ({
        messageId: e.messageId,
        folder: e.folder,
        subject: e.subject,
        fromAddr: e.fromAddr,
        date: e.date,
        hasAttachments: e.hasAttachments,
      })),
      categoryCount: activeCategories.length,
      categories: topCategories,
      userOrganization: {
        tagCount: this.userLabels.countDistinct(accountId, 'thunderbird_tag'),
        folderCount: this.userLabels.countDistinct(accountId, 'folder'),
        topTags: this.userLabels.topLabels(accountId, 'thunderbird_tag', opts.topLabelsLimit),
        topFolders: this.userLabels.topLabels(accountId, 'folder', opts.topLabelsLimit),
      },
    };
  }
}

/** Maps a triage repository row to the dashboard's urgent email projection. */
function toDashboardClassified(row: {
  messageId: string;
  folder: string;
  subject: string | null;
  fromAddr: string | null;
  date: number | null;
  hasAttachments: boolean;
  reasoning: string | null;
  classifiedAt: number;
}): DashboardUrgentEmail {
  return {
    messageId: row.messageId,
    folder: row.folder,
    subject: row.subject,
    fromAddr: row.fromAddr,
    date: row.date,
    hasAttachments: row.hasAttachments,
    reasoning: row.reasoning,
    classifiedAt: row.classifiedAt,
  };
}
