/**
 * Topic discovery for an account's inbox. Samples email subjects, asks the LLM for a purpose-based
 * taxonomy, then embeds, merges, and reconciles the result into auto categories with centroids.
 */
import { z } from 'zod';
import type { Logger } from 'pino';
import { LlmApiError, type LlmClient } from '../llm/client.js';
import type { CategoryRepository } from '../repositories/category-repository.js';
import type { EmailRepository, EmailSummary } from '../repositories/email-repository.js';
import type { EmbeddingRepository } from '../repositories/embedding-repository.js';
import { parseLlmJson, stripCodeFence } from '../util/json-llm.js';
import { cosineFromL2Distance, l2Distance, meanNormalize } from '../util/vector.js';
import { purposeSignature } from './categorize-strategy.js';

const TWIN_COSINE_STRONG = 0.93;
const TWIN_COSINE_SAME_PURPOSE = 0.9;

/**
 * True when two category labels name the same purpose, by label-token overlap or label-embedding
 * cosine. Categories sharing a known purpose signature merge at a lower cosine, distinct known
 * purposes never merge.
 */
export function areTwins(
  labelA: string,
  vecA: Float32Array,
  labelB: string,
  vecB: Float32Array,
): boolean {
  if (isNearDuplicateLabel(labelA, [labelB])) return true;
  const sigA = purposeSignature(labelA);
  const sigB = purposeSignature(labelB);
  if (sigA !== null && sigB !== null && sigA !== sigB) return false;
  const cos = cosineFromL2Distance(l2Distance(vecA, vecB));
  const threshold = sigA !== null && sigA === sigB ? TWIN_COSINE_SAME_PURPOSE : TWIN_COSINE_STRONG;
  return cos >= threshold;
}

const TopicListSchema = z.object({
  topics: z
    .array(
      z.object({
        label: z.string().min(1).max(80),
        description: z.string().min(1).max(300),
      }),
    )
    .min(2)
    .max(20),
});

/** A single discovered topic with its label and one-sentence purpose description. */
export type DiscoveredTopic = z.infer<typeof TopicListSchema>['topics'][number];

const SingleTopicSchema = z.object({
  label: z.string().min(1).max(80),
  description: z.string().min(1).max(300),
});

/** Remove any reasoning block the model emitted, including an unterminated one at the end. */
function stripThink(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<think>[\s\S]*$/i, '');
}

/**
 * Pull complete topic objects out of a possibly truncated JSON answer. Scans the topics array,
 * balancing braces while respecting strings, and keeps each fully-closed object that validates so a
 * cut-off answer still yields usable topics instead of failing the whole run.
 */
export function salvageTopics(raw: string): DiscoveredTopic[] {
  const text = stripThink(stripCodeFence(raw));
  const openers = Array.from(text.matchAll(/"topics"\s*:\s*\[/g));
  const last = openers.length > 0 ? openers[openers.length - 1] : undefined;
  const start =
    last && last.index !== undefined ? last.index + last[0].length - 1 : text.indexOf('[');
  if (start === -1) return [];

  const out: DiscoveredTopic[] = [];
  let depth = 0;
  let objStart = -1;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length && out.length < 20; i++) {
    const ch = text[i]!;
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') {
      inStr = true;
    } else if (ch === '{') {
      if (depth === 0) objStart = i;
      depth += 1;
    } else if (ch === '}') {
      if (depth > 0) {
        depth -= 1;
        if (depth === 0 && objStart !== -1) {
          try {
            const parsed = SingleTopicSchema.safeParse(JSON.parse(text.slice(objStart, i + 1)));
            if (parsed.success) out.push(parsed.data);
          } catch {}
          objStart = -1;
        }
      }
    } else if (ch === ']' && depth === 0) {
      break;
    }
  }
  return out;
}

/**
 * Outcome of a discovery run. A status of 'insufficient_categories' means too few concrete topics
 * were produced to form a safe taxonomy, so the existing categories were kept.
 */
export interface DiscoveryResult {
  status: 'ok' | 'insufficient_categories';
  topicsCreated: number;
  emailsSampled: number;
  centroidsComputed: number;
}

const SAMPLE_SIZE = 90;
const TOPIC_OUTPUT_TOKENS = 1500;
const RECENT_POOL = 600;
const HISTORICAL_POOL = 1000;
const UNCATEGORIZED_POOL = 400;
const TOP_DOMAIN_COUNT = 10;
const PER_DOMAIN_EXAMPLES = 20;
const SMALL_INBOX_MAX = 200;
const MODEST_INBOX_MAX = 2000;
const MIN_CATEGORIES_LARGE = 8;
const TARGET_TOPIC_COUNT = '10 to 16';
const ASSIGNMENT_THRESHOLD = 1.0;

/** Minimum number of concrete categories required to accept a taxonomy, scaled to inbox size. */
function minCategoriesFor(inboxSize: number): number {
  if (inboxSize < SMALL_INBOX_MAX) return 2;
  if (inboxSize < MODEST_INBOX_MAX) return 4;
  return MIN_CATEGORIES_LARGE;
}

const SYSTEM_PROMPT = `You analyze a sample of email subjects and identify recurring TOPICS in the user's inbox.

A topic is a recurring theme of communication defined by its PURPOSE, NOT a single email's content.

Each topic is a BROAD purpose that groups many senders. Any email clearly belongs to one topic, not several.

COMMON PURPOSES to draw from. Include any that the sample clearly shows, named in your own words; do NOT invent a category with no evidence in the sample, and do NOT force one that the inbox does not contain:
- Developer code reviews - GitHub, GitLab, Bitbucket pull requests, CI runs, commits, repository notifications
- Developer platforms and tokens - developer tool accounts, API keys, token expiry, deploy and platform notices
- Travel and accommodation - flights, hotels, bookings, travel deals from any travel site
- Banking and transactions - bank transfers, balances, statements, card activity
- Security and sign-in - login alerts, password resets, verification codes, account security
- Job opportunities - job listings and openings from any board or recruiter
- Professional networking - connection requests, profile views, network updates
- Social media updates - posts, mentions, comments, follows, and activity from social networks
- Receipts and invoices - payment confirmations, invoices, billing statements for things bought
- Shipping and deliveries - order shipment, tracking, delivery updates
- Course grades - grades, marks, exam results
- Course materials - assignments, lectures, schedules from courses
- Health and insurance - medical, pharmacy, insurance claims and policies
- Marketing and promotions - marketing, digests, opt-in newsletters and offers

GUIDELINES:
- Identify ${TARGET_TOPIC_COUNT} topics that together cover most of the inbox. Aim for at least 8 distinct concrete topics for a large inbox.
- Cover the high-volume senders listed below: each should have a fitting topic from the common purposes above or one you name.
- Every topic must be PURPOSE-based. NEVER name a topic after a single sender, brand, or service. Use one broad "Developer Code Reviews" for all GitHub/GitLab/CI mail, never "GitHub Notifications".
- Do NOT split one purpose into small variants (e.g. one "Travel & Accommodation", not separate "Flight Bookings" + "Hotel Bookings" + "Travel Deals").
- Make topics MUTUALLY EXCLUSIVE: no two may overlap or be near-synonyms. If two ideas overlap, merge them into one broader topic.
- BANNED: vague catch-all topics where every word is generic, such as "Technical Support", "Service Announcements", "Account Notifications", "General Updates", "Notifications", "Miscellaneous". Each topic needs a concrete anchor word naming a real purpose (banking, invoice, job, course, security, developer, travel, insurance, shipping, etc.).
- Topic descriptions: ONE short sentence (under 20 words) naming the concrete PURPOSE only. Do NOT name specific senders, brands, apps, or services in the description.
- Topic labels: 2-4 words, Title Case, no quotes.
- Output ONLY valid JSON, no extra text, no code fences, no markdown.

OUTPUT FORMAT:
{"topics": [{"label": "Receipts & Invoices", "description": "Payment confirmations and invoices for purchases and subscriptions."}, ...]}`;

const STRICTER_FEEDBACK = `\n\nReturn ONLY a compact JSON object with at most 12 topics. Keep each description under 10 words. No markdown, no code fences, no extra text.`;

interface StagedTopic {
  label: string;
  description: string;
  centroid: Float32Array;
  emailCount: number;
}

/**
 * Discovers recurring topics in an account's inbox by sampling email subjects, asking the LLM for a
 * purpose-based taxonomy, then computing embedding centroids and reconciling with existing
 * categories.
 */