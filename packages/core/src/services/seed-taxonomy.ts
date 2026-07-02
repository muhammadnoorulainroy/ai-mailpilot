/**
 * Built-in seed taxonomy for cold start. Seeds are dormant: they exist only as this constant
 * and are surfaced as suggestions. Nothing materializes a seed as a category automatically, and
 * Phase 1 exposes no route to accept one, so a seed can never become active or receive mail on
 * its own. Acceptance is a Phase 2 concern (the review queue).
 */

/** A dormant seed category with a frozen canonical key. */
export interface SeedCategory {
  label: string;
  description: string;
  canonicalKey: string;
}

/** The common life categories offered at cold start. */
export const SEED_TAXONOMY: readonly SeedCategory[] = [
  {
    label: 'Banking Transactions',
    description: 'Bank statements, transfers, and account activity.',
    canonicalKey: 'finance.banking',
  },
  {
    label: 'Security Alerts',
    description: 'Sign-in alerts, password resets, and account security notices.',
    canonicalKey: 'security.alerts',
  },
  {
    label: 'Receipts & Invoices',
    description: 'Purchase receipts, invoices, and payment confirmations.',
    canonicalKey: 'finance.receipts',
  },
  {
    label: 'Shipping & Deliveries',
    description: 'Order confirmations, shipping updates, and delivery notices.',
    canonicalKey: 'commerce.shipping',
  },
  {
    label: 'Job Opportunities',
    description: 'Job postings, recruiter outreach, and application updates.',
    canonicalKey: 'career.jobs',
  },
  {
    label: 'Professional Networking',
    description: 'Connection requests and professional network activity.',
    canonicalKey: 'career.networking',
  },
  {
    label: 'Travel & Accommodation',
    description: 'Flight, hotel, and travel booking confirmations and itineraries.',
    canonicalKey: 'travel.accommodation',
  },
  {
    label: 'Course Materials',
    description: 'Lecture notes, assignments, and course announcements.',
    canonicalKey: 'education.materials',
  },
  {
    label: 'Course Grades',
    description: 'Grade reports and assessment results.',
    canonicalKey: 'education.grades',
  },
  {
    label: 'Developer Code Reviews',
    description: 'Pull request reviews, code review requests, and CI notifications.',
    canonicalKey: 'dev.code_reviews',
  },
  {
    label: 'Newsletters & Digests',
    description: 'Newsletters, digests, and subscription content.',
    canonicalKey: 'content.newsletters',
  },
  {
    label: 'Health & Insurance',
    description: 'Medical, health, and insurance correspondence.',
    canonicalKey: 'health.insurance',
  },
];

/**
 * Seeds whose canonical key is not already represented among an account's categories. Used to
 * offer cold-start suggestions without proposing a category the account already has.
 */
export function getSeedSuggestions(existingKeys: ReadonlySet<string>): SeedCategory[] {
  return SEED_TAXONOMY.filter((s) => !existingKeys.has(s.canonicalKey));
}
