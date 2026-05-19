import { z } from 'zod';
import type { Logger } from 'pino';
import type { LlmClient } from '../llm/client.js';
import type { TriageBucket } from '../repositories/triage-repository.js';
import { preprocessForEmbedding } from '../util/text.js';

const TriageResultSchema = z.object({
  bucket: z.enum(['urgent', 'summarize', 'spam', 'personal']),
  reasoning: z.string().max(200),
});

export type TriageResult = z.infer<typeof TriageResultSchema>;

export interface EmailToTriage {
  messageId: string;
  subject: string | null;
  fromAddr: string | null;
  body: string | null;
  bodyFormat: string | null;
}

const SYSTEM_PROMPT = `You are an email triage classifier. Classify each email into exactly ONE bucket.

BUCKETS:
- urgent: A specific person directly addressed the user and is waiting for THEIR reply or decision within hours or a day. OR a hard deadline is within 24 hours that the user must meet. OR a security event needs immediate verification (suspicious login, account compromise).
- summarize: Informational content. Newsletters, digests, automated notifications, mailing lists, course/group announcements, marketing the user opted into, app updates, social network alerts, calendar invites to non-mandatory events, automated welcome emails.
- spam: Unsolicited bulk marketing, phishing attempts, content from unknown senders pushing a product or service, scams.
- personal: A message from a friend, family member, or other private contact about non-work matters. Personal conversation, social plans, family news.

STRICT RULES (apply in order):
1. If the sender is a no-reply, no-reply+, notification, mailer, mail-noreply, noreplies@, classroom-noreply, or similar AUTOMATED address: NEVER urgent. Almost always summarize. (Exception: security alerts about the user's own account compromise can be urgent.)
2. A class/course announcement with a link to slides, materials, a quiz date, or homework is SUMMARIZE. It is not urgent unless the user is explicitly being asked to reply or the deadline is within 24 hours stated clearly.
3. A bank transaction confirmation, payment receipt, or statement notification is SUMMARIZE.
4. A welcome email, verification email older than a day, or "your account was created" is SUMMARIZE.
5. urgent requires BOTH: (a) addressed to this user personally and (b) a clear request or deadline. Generic "click here to view" is NOT urgent.
6. personal trumps all other buckets if the sender is clearly a friend/family.

EXAMPLES (study these carefully):

Email: "New announcement: Quiz of Calculus will be taken on Monday" from no-reply+classroom
-> {"bucket": "summarize", "reasoning": "Course announcement from no-reply about a future event. Informational."}

Email: "Due tomorrow: Submit project report" from no-reply+classroom
-> {"bucket": "urgent", "reasoning": "Hard deadline within 24 hours stated in subject."}

Email: "Funds Transfer Alert" from bank
-> {"bucket": "summarize", "reasoning": "Automated bank transaction notification. No action required."}

Email: "Hey, are you free for lunch on Saturday?" from a friend@gmail.com
-> {"bucket": "personal", "reasoning": "Personal invitation from a friend."}

Email: "Security alert: new sign-in on Windows" from Google
-> {"bucket": "urgent", "reasoning": "Account security event requires verification."}

Email: "Welcome to Internet Archive" from noreply@archive.org
-> {"bucket": "summarize", "reasoning": "Automated welcome email, no action required."}

Email: "[7372] Email address verification" from no-reply@nintendo, sent moments ago
-> {"bucket": "urgent", "reasoning": "Verification code likely needed within hours."}

OUTPUT FORMAT (only valid JSON, no extra text, no code fences):
{"bucket": "urgent" | "summarize" | "spam" | "personal", "reasoning": "under 200 characters"}`;

export class TriageService {
  constructor(
    private llm: LlmClient,
    private logger: Logger,
  ) {}

  async classify(email: EmailToTriage, model?: string): Promise<TriageResult> {
    const body = email.body
      ? preprocessForEmbedding(email.body, {
          format: email.bodyFormat === 'html' ? 'html' : 'text',
          maxChars: 2000,
        })
      : '';

    const userPrompt = [
      `Subject: ${email.subject ?? '(no subject)'}`,
      `From: ${email.fromAddr ?? '(unknown)'}`,
      '',
      body || '(no body)',
    ].join('\n');

    const raw = await this.llm.chat({
      model,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userPrompt },
      ],
      responseFormat: 'json_object',
      temperature: 0.1,
    });

    const cleaned = stripCodeFence(raw);
    let parsed: unknown;
    try {
      parsed = JSON.parse(cleaned);
    } catch (err) {
      this.logger.warn({ raw, err }, 'triage response not valid JSON');
      throw new Error(`triage parse error: ${err instanceof Error ? err.message : String(err)}`);
    }

    const result = TriageResultSchema.safeParse(parsed);
    if (!result.success) {
      this.logger.warn({ raw, issues: result.error.issues }, 'triage response schema invalid');
      throw new Error('triage response did not match schema');
    }

    return result.data;
  }
}

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

function stripCodeFence(text: string): string {
  const trimmed = text.trim();
  if (trimmed.startsWith('```')) {
    const without = trimmed.replace(/^```(?:json)?\s*/i, '').replace(/```\s*$/i, '');
    return without.trim();
  }
  return trimmed;
}
