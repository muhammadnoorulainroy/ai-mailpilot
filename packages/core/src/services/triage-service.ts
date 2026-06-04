/**
 * Triage service that classifies emails into focus-view buckets using the LLM,
 * with JSON salvage and rule-based fallbacks plus normalization of code-platform notifications.
 */
import { z } from 'zod';
import type { Logger } from 'pino';
import type { TriageMetadata } from '@ai-mailpilot/shared';
import type { LlmClient } from '../llm/client.js';
import type { TriageBucket } from '../repositories/triage-repository.js';
import { preprocessForEmbedding } from '../util/text.js';
import { LlmJsonParseError, parseLlmJson } from '../util/json-llm.js';

const BucketSchema = z.enum(['urgent', 'summarize', 'spam', 'personal']);

/** Outcome of triaging an email: its bucket, a short reasoning line, and derived metadata. */
export interface TriageResult {
  bucket: TriageBucket;
  reasoning: string;
  metadata: TriageMetadata;
}

const REASONING_MAX = 200;
const ACTION_MAX = 140;
const SUMMARY_MAX = 200;
const SINGLE_BODY_CHARS = 2000;
const BATCH_BODY_CHARS = 900;
const BATCH_TOKENS_PER_EMAIL = 180;
const MAX_DEADLINE_HOURS = 24 * 30;

const DEFAULT_IMPORTANCE: Record<TriageBucket, number> = {
  urgent: 85,
  personal: 55,
  summarize: 35,
  spam: 5,
};

/** Extracts a valid bucket from a raw LLM string when JSON parsing failed. */
function salvageBucket(raw: string): TriageBucket | null {
  const m = /"bucket"\s*:\s*"(urgent|summarize|spam|personal)"/i.exec(raw);
  return m ? (m[1]!.toLowerCase() as TriageBucket) : null;
}

/** Coerces an unknown LLM value into a boolean, accepting truthy strings and numbers. */
function asBool(v: unknown): boolean {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'number') return v !== 0;
  if (typeof v === 'string') return /^\s*(true|yes|1)\s*$/i.test(v);
  return false;
}
/** Coerces an unknown LLM value into a finite number, or null when not parseable. */
function asNum(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string') {
    const n = Number(v.trim());
    return Number.isFinite(n) ? n : null;
  }
  return null;
}
/** Coerces an unknown value into a trimmed string capped at max characters, or null if empty. */
function asText(v: unknown, max: number): string | null {
  if (typeof v !== 'string') return null;
  const t = v.trim();
  return t.length > 0 ? t.slice(0, max) : null;
}

/**
 * Builds normalized triage metadata from a raw LLM object, clamping scores and
 * converting a stated deadline in hours into an absolute timestamp relative to now.
 */
function toMetadata(
  obj: Record<string, unknown>,
  bucket: TriageBucket,
  now: number,
): TriageMetadata {
  const rawImportance = asNum(obj.importanceScore);
  const importanceScore =
    rawImportance === null
      ? DEFAULT_IMPORTANCE[bucket]
      : Math.max(0, Math.min(100, Math.round(rawImportance)));

  const hours = asNum(obj.deadlineHours);
  const deadlineAt =
    hours !== null && hours > 0 && hours <= MAX_DEADLINE_HOURS
      ? now + Math.round(hours) * 3_600_000
      : null;

  const confidence = asNum(obj.confidence);

  return {
    actionRequired: asBool(obj.actionRequired),
    needsReply: asBool(obj.needsReply),
    deadlineAt,
    importanceScore,
    suggestedAction: asText(obj.suggestedAction, ACTION_MAX),
    shortSummary: asText(obj.shortSummary, SUMMARY_MAX),
    confidence: confidence === null ? null : Math.max(0, Math.min(1, confidence)),
  };
}

/** Produces fallback metadata for a bucket when the LLM provided no usable fields. */
function defaultMetadata(bucket: TriageBucket): TriageMetadata {
  return {
    actionRequired: bucket === 'urgent',
    needsReply: false,
    deadlineAt: null,
    importanceScore: DEFAULT_IMPORTANCE[bucket],
    suggestedAction: null,
    shortSummary: null,
    confidence: null,
  };
}

/** Minimal email fields required to triage a single message. */
export interface EmailToTriage {
  messageId: string;
  subject: string | null;
  fromAddr: string | null;
  date?: number | null;
  body: string | null;
  bodyFormat: string | null;
}

/** Which LLM provider to route triage requests through, the local main model or the chat model. */
export type TriageProvider = 'main' | 'chat';

const SYSTEM_PROMPT = `You triage ONE email for a daily focus view. Output ONLY a JSON object, nothing else.

Pick exactly ONE bucket:
- urgent: time-sensitive AND the user must act soon. A person is waiting on the user's reply/decision within ~a day, OR a stated deadline is within 24 hours, OR a security event needs immediate verification.
- summarize: informational. Newsletters, digests, automated notifications, job alerts, social and GitHub/code-platform notifications, bank transaction alerts, receipts, course announcements without a near deadline, app updates, opted-in marketing.
- personal: a message from a friend, family member, or private contact about non-work matters.
- spam: unsolicited bulk marketing, phishing, scams, unknown senders pushing a product.

Then set these fields (they are INDEPENDENT of the bucket):
- actionRequired (boolean): the user likely must reply, submit, approve, verify, pay, attend, decide, or review. Can be true even when not urgent.
- needsReply (boolean): the user is expected to write a reply.
- deadlineHours (number or null): whole hours from now until an EXPLICITLY STATED deadline. If no deadline is clearly stated, it MUST be null. Never invent one.
- importanceScore (0-100): how much this deserves attention. urgent/action 70-100, personal 50-70, routine notifications 20-40, spam 0-10.
- suggestedAction (short string or null): e.g. "Reply to confirm", "Verify the login", "Pay the invoice". Null if none.
- shortSummary (short string or null): one terse line of what it is.
- reasoning (short string, under 200 chars): one terse sentence.

Rules:
- A no-reply / notification / mailer address is almost never urgent (exception: a security alert about the user's OWN account can be urgent).
- A GitHub/GitLab/code-platform notification is summarize by default. If the user is directly requested as reviewer/assignee, set summarize + actionRequired=true. Do NOT mark it urgent unless it states an explicit near deadline (within 24 hours) or reports account/security compromise.
- A bank transaction confirmation, payment receipt, or statement alert is summarize, actionRequired=false (unless it explicitly asks the user to verify a suspicious charge).
- A course/class announcement is summarize unless it states a deadline within 24 hours or explicitly requires the user to submit/reply.
- A suspicious-login or account-compromise alert is urgent, actionRequired=true.
- personal is its own bucket; never label a friend's message spam or summarize.
- Interpret relative deadlines using the email sent time and the current time below. If an old email says "tomorrow" but that date is already past, deadlineHours MUST be null.

OUTPUT (only this JSON, no prose, no code fences):
{"bucket":"summarize","actionRequired":false,"needsReply":false,"deadlineHours":null,"importanceScore":35,"suggestedAction":null,"shortSummary":"...","reasoning":"..."}

Examples:
"Due tomorrow: submit project report" from no-reply+classroom
-> {"bucket":"urgent","actionRequired":true,"needsReply":false,"deadlineHours":24,"importanceScore":90,"suggestedAction":"Submit the report","shortSummary":"Project report due in ~24h.","reasoning":"Stated deadline within 24 hours."}
"Funds Transfer Alert" from your bank
-> {"bucket":"summarize","actionRequired":false,"needsReply":false,"deadlineHours":null,"importanceScore":30,"suggestedAction":null,"shortSummary":"Automated bank transfer notice.","reasoning":"Routine bank alert, no action."}
"@user requested your review on PR #42" from notifications@github
-> {"bucket":"summarize","actionRequired":true,"needsReply":false,"deadlineHours":null,"importanceScore":65,"suggestedAction":"Review the pull request","shortSummary":"Review requested on PR #42.","reasoning":"Direct code review request, no near deadline."}
"Security alert: new sign-in on Windows" from Google
-> {"bucket":"urgent","actionRequired":true,"needsReply":false,"deadlineHours":null,"importanceScore":85,"suggestedAction":"Verify the sign-in","shortSummary":"New sign-in to verify.","reasoning":"Account security event."}
"Are you free for lunch Saturday?" from a friend
-> {"bucket":"personal","actionRequired":true,"needsReply":true,"deadlineHours":null,"importanceScore":60,"suggestedAction":"Reply about Saturday","shortSummary":"Friend asking about lunch.","reasoning":"Personal message expecting a reply."}`;

const BATCH_SYSTEM_PROMPT = `${SYSTEM_PROMPT.replace(
  'You triage ONE email for a daily focus view. Output ONLY a JSON object, nothing else.',
  'You triage emails for a daily focus view. Output ONLY a JSON object, nothing else.',
)}

Batch mode:
- You will receive a JSON array of emails.
- Return ONLY this JSON shape: {"results":[{"messageId":"...","bucket":"summarize","actionRequired":false,"needsReply":false,"deadlineHours":null,"importanceScore":35,"suggestedAction":null,"shortSummary":"...","reasoning":"..."}]}
- Include exactly one result for every input messageId.
- Do not add markdown, comments, or fields outside the JSON object.`;

interface NormalizedTriage {
  bucket: TriageBucket;
  reasoning: string;
  metadata: TriageMetadata;
}

/** Formats a timestamp for inclusion in the prompt, returning a placeholder when absent or invalid. */
function formatPromptTime(ms: number | null | undefined): string {
  if (!ms) return '(unknown)';
  const d = new Date(ms);
  return Number.isNaN(d.getTime()) ? '(unknown)' : `${d.toISOString()} (${d.toString()})`;
}

/** Strips diacritics and lowercases text so the rule regexes match consistently. */
function normalizeTextForRules(s: string): string {
  return s
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase();
}

/** Detects whether an email looks like an automated notification from a code-hosting platform. */
function isCodePlatformNotification(email: EmailToTriage, text: string): boolean {
  const from = (email.fromAddr ?? '').toLowerCase();
  return (
    /(?:^|@|\.)(github\.com|gitlab\.com|bitbucket\.org|dev\.azure\.com)\b/.test(from) ||
    /\b(github|gitlab|bitbucket|pull request|merge request|repository|personal access token)\b/.test(
      text,
    )
  );
}

/** Detects an account-security event that warrants treating the email as immediately urgent. */
function hasImmediateCodeSecurity(text: string): boolean {
  return /\b(security alert|suspicious|unauthorized|unauthorised|new sign-?in|new login|account compromise|password reset|third-party application has been added|oauth app|oauth application)\b/.test(
    text,
  );
}

/** Detects a security or token related notice that implies the user should review it. */
function hasCodeSecurityOrTokenAction(text: string): boolean {
  return /\b(security alert|suspicious|unauthorized|unauthorised|sign-?in|login|password|2fa|two-factor|oauth|third-party application|personal access token|access token|token will expire|token expires)\b/.test(
    text,
  );
}

/** Detects a direct request such as a review or assignment that asks the user to act. */
function hasDirectCodeAction(text: string): boolean {
  return /\b(requested your review|review requested|requested review|assigned you|assigned to you|mentioned you|please review|requires your review|changes requested|requested changes|needs your reply|needs your response)\b/.test(
    text,
  );
}

/** Reports whether the email implies a deadline within 24 hours, by timestamp or phrasing. */
function hasNearDeadlineSignal(text: string, metadata: TriageMetadata, now: number): boolean {
  if (metadata.deadlineAt !== null && metadata.deadlineAt > now) {
    return metadata.deadlineAt - now <= 24 * 3_600_000;
  }
  return /\b(today|tonight|tomorrow|within 24 hours|within twenty-four hours|due in 24 hours|by eod|asap|urgent|immediate|immediately|aujourd'hui|demain|ce soir|dans les 24 heures)\b/.test(
    text,
  );
}

/** Clamps the importance score down for a demoted code-platform notification. */
function downgradedImportance(metadata: TriageMetadata, actionRequired: boolean): number {
  if (actionRequired) return Math.min(Math.max(metadata.importanceScore, 65), 75);
  return Math.min(metadata.importanceScore, 35);
}

/** Returns metadata with importance lowered for a code-platform notification being demoted to summarize. */
function downgradedCodePlatformMetadata(
  metadata: TriageMetadata,
  actionRequired: boolean,
): TriageMetadata {
  return {
    ...metadata,
    importanceScore: downgradedImportance(metadata, actionRequired),
    deadlineAt: metadata.deadlineAt,
  };
}

/**
 * Applies code-platform rules to an LLM result, keeping genuine security or near-deadline
 * urgents but demoting other code notifications to summarize with adjusted metadata.
 */
function normalizeTriage(
  email: EmailToTriage,
  body: string,
  result: NormalizedTriage,
  now: number,
): NormalizedTriage {
  const text = normalizeTextForRules(`${email.subject ?? ''}\n${email.fromAddr ?? ''}\n${body}`);
  if (!isCodePlatformNotification(email, text) || result.bucket !== 'urgent') return result;

  const immediateSecurity = hasImmediateCodeSecurity(text);
  const nearDeadline = hasNearDeadlineSignal(text, result.metadata, now);
  if (immediateSecurity || nearDeadline) {
    return {
      ...result,
      metadata: {
        ...result.metadata,
        actionRequired: true,
        importanceScore: Math.max(result.metadata.importanceScore, 80),
      },
    };
  }

  const actionRequired =
    result.metadata.actionRequired ||
    hasDirectCodeAction(text) ||
    hasCodeSecurityOrTokenAction(text);
  const suggestedAction =
    result.metadata.suggestedAction ??
    (hasDirectCodeAction(text)
      ? 'Review the code notification'
      : hasCodeSecurityOrTokenAction(text)
        ? 'Review the account/security notice'
        : null);

  return {
    bucket: 'summarize',
    reasoning:
      'Automated code-platform notification without an explicit near deadline or account compromise.',
    metadata: {
      ...downgradedCodePlatformMetadata(result.metadata, actionRequired),
      actionRequired,
      needsReply: result.metadata.needsReply && actionRequired,
      deadlineAt: null,
      suggestedAction,
    },
  };
}

/** Builds a cleaned text snippet of the email body for the prompt, honoring its format. */
function buildSnippet(email: EmailToTriage, maxChars: number): string {
  return email.body
    ? preprocessForEmbedding(email.body, {
        format: email.bodyFormat === 'html' ? 'html' : 'text',
        maxChars,
      })
    : '';
}

/** Narrows an unknown value to a plain object record, rejecting arrays and non-objects. */
function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : null;
}

/**
 * Validates a single triage object, salvaging the bucket from raw text when missing,
 * then returns the normalized triage result.
 */
function parseTriageObject(
  obj: Record<string, unknown>,
  raw: string,
  email: EmailToTriage,
  body: string,
  now: number,
): TriageResult {
  const bucketResult = BucketSchema.safeParse(obj.bucket);
  if (!bucketResult.success) {
    const salvaged = salvageBucket(raw);
    if (salvaged) {
      return normalizeTriage(
        email,
        body,
        {
          bucket: salvaged,
          reasoning: asText(obj.reasoning, REASONING_MAX) ?? '',
          metadata: toMetadata(obj, salvaged, now),
        },
        now,
      );
    }
    throw new Error('triage response did not include a valid bucket');
  }

  const bucket = bucketResult.data;
  return normalizeTriage(
    email,
    body,
    {
      bucket,
      reasoning: asText(obj.reasoning, REASONING_MAX) ?? '',
      metadata: toMetadata(obj, bucket, now),
    },
    now,
  );
}

/** Extracts per-message triage objects from a batch LLM response keyed by message id, tolerating array or map shapes. */
function extractBatchObjects(parsed: unknown): Map<string, Record<string, unknown>> {
  const out = new Map<string, Record<string, unknown>>();
  const wrapper = asRecord(parsed);
  const container = wrapper?.results ?? wrapper?.emails ?? parsed;

  if (Array.isArray(container)) {
    for (const item of container) {
      const rec = asRecord(item);
      if (!rec || typeof rec.messageId !== 'string') continue;
      if (!out.has(rec.messageId)) out.set(rec.messageId, rec);
    }
    return out;
  }

  const byId = asRecord(container);
  if (byId) {
    for (const [messageId, value] of Object.entries(byId)) {
      const rec = asRecord(value);
      if (!rec) continue;
      out.set(messageId, { ...rec, messageId });
    }
  }
  return out;
}

/** Classifies emails into triage buckets via the LLM, with salvage and rule-based fallbacks. */
export class TriageService {
  /** Creates the service with the LLM client used for classification and a logger for fallbacks. */
  constructor(
    private llm: LlmClient,
    private logger: Logger,
  ) {}

  /** Triages a single email and returns its bucket, reasoning, and metadata. */
  async classify(
    email: EmailToTriage,
    model?: string,
    provider: TriageProvider = 'main',
  ): Promise<TriageResult> {
    const body = buildSnippet(email, SINGLE_BODY_CHARS);
    const now = Date.now();
    const userPrompt = [
      `Current time: ${formatPromptTime(now)}`,
      `Email sent time: ${formatPromptTime(email.date)}`,
      `Subject: ${email.subject ?? '(no subject)'}`,
      `From: ${email.fromAddr ?? '(unknown)'}`,
      '',
      body || '(no body)',
    ].join('\n');

    const system = provider === 'main' ? `/no_think\n${SYSTEM_PROMPT}` : SYSTEM_PROMPT;
    const raw = await this.llm.chat({
      model,
      provider,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: userPrompt },
      ],
      responseFormat: 'json_object',
      temperature: 0.1,
      maxTokens: 400,
      think: provider === 'main' ? false : undefined,
    });

    let obj: Record<string, unknown> | null = null;
    try {
      const parsed = parseLlmJson(raw);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
        obj = parsed as Record<string, unknown>;
      }
    } catch (err) {
      const bucket = salvageBucket(raw);
      if (bucket) {
        return normalizeTriage(
          email,
          body,
          { bucket, reasoning: '', metadata: defaultMetadata(bucket) },
          now,
        );
      }
      if (err instanceof LlmJsonParseError) {
        this.logger.warn({ raw: err.raw, err }, 'triage response not valid JSON');
      }
      throw err;
    }

    if (!obj) {
      const bucket = salvageBucket(raw);
      if (bucket) {
        return normalizeTriage(
          email,
          body,
          { bucket, reasoning: '', metadata: defaultMetadata(bucket) },
          now,
        );
      }
      this.logger.warn({ raw }, 'triage response was not a JSON object');
      throw new Error('triage response was not a JSON object');
    }

    const bucketResult = BucketSchema.safeParse(obj.bucket);
    if (!bucketResult.success) {
      const salvaged = salvageBucket(raw);
      if (salvaged) return parseTriageObject({ ...obj, bucket: salvaged }, raw, email, body, now);
      this.logger.warn(
        { raw, issues: bucketResult.error.issues },
        'triage response missing a bucket',
      );
      throw new Error('triage response did not include a valid bucket');
    }

    return parseTriageObject(obj, raw, email, body, now);
  }

  /** Triages multiple emails in one request, keyed by message id, falling back to single classify for one email. */
  async classifyBatch(
    emails: EmailToTriage[],
    model?: string,
    provider: TriageProvider = 'main',
  ): Promise<Map<string, TriageResult>> {
    const out = new Map<string, TriageResult>();
    if (emails.length === 0) return out;
    if (emails.length === 1) {
      const email = emails[0]!;
      out.set(email.messageId, await this.classify(email, model, provider));
      return out;
    }

    const now = Date.now();
    const snippets = new Map<string, string>();
    const payload = emails.map((email) => {
      const body = buildSnippet(email, BATCH_BODY_CHARS);
      snippets.set(email.messageId, body);
      return {
        messageId: email.messageId,
        currentTime: formatPromptTime(now),
        sentTime: formatPromptTime(email.date),
        subject: email.subject ?? '(no subject)',
        from: email.fromAddr ?? '(unknown)',
        body: body || '(no body)',
      };
    });

    const system = provider === 'main' ? `/no_think\n${BATCH_SYSTEM_PROMPT}` : BATCH_SYSTEM_PROMPT;
    const raw = await this.llm.chat({
      model,
      provider,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: JSON.stringify({ emails: payload }) },
      ],
      responseFormat: 'json_object',
      temperature: 0.1,
      maxTokens: Math.min(4096, 120 + BATCH_TOKENS_PER_EMAIL * emails.length),
      think: provider === 'main' ? false : undefined,
    });

    let parsed: unknown;
    try {
      parsed = parseLlmJson(raw);
    } catch (err) {
      if (err instanceof LlmJsonParseError) {
        this.logger.warn({ raw: err.raw, err }, 'batched triage response not valid JSON');
      }
      throw err;
    }

    const byId = extractBatchObjects(parsed);
    for (const email of emails) {
      const obj = byId.get(email.messageId);
      if (!obj) throw new Error(`batched triage response missing ${email.messageId}`);
      out.set(
        email.messageId,
        parseTriageObject(
          obj,
          JSON.stringify(obj),
          email,
          snippets.get(email.messageId) ?? '',
          now,
        ),
      );
    }
    return out;
  }
}

/** Returns a short single-character marker representing the given triage bucket. */
export function bucketEmoji(bucket: TriageBucket): string {
  switch (bucket) {
    case 'urgent':
      return '!';
    case 'summarize':
      return 'i';
    case 'spam':
      return 'x';
    case 'personal':
      return '~';
  }
}
