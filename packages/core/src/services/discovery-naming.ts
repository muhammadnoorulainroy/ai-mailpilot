/**
 * Deterministic prompt building and parsing for discovery candidate naming (Phase 2b). The local
 * model only names clusters an on-device pass already formed; it never decides identity, quality, or
 * acceptance, which are the deterministic gate's job. Pure functions: no I/O and no LLM call here.
 */
import { z } from 'zod';
import type { ChatMessage } from '../llm/client.js';
import { stripCodeFence } from '../util/json-llm.js';
import type { CandidateAction } from './discovery-candidates.js';

/** How many example subjects per cluster are shown to the model. */
export const NAMING_SAMPLE_PER_CLUSTER = 15;

/** What the naming prompt shows the model for one cluster. */
export interface ClusterNamingInput {
  index: number;
  size: number;
  keyphrases: string[];
  sampleSubjects: string[];
  senderHints: string[];
}

/** The model's proposed name for one cluster, before the gate judges it. */
export interface ParsedNaming {
  clusterIndex: number;
  action: CandidateAction;
  label: string;
  description: string;
  suggestedKey: string;
}

const SYSTEM_PROMPT = `You name clusters of emails that an on-device model has already grouped. Each cluster is a set of similar emails. Give each cluster a purpose-based name, or mark it uncategorized when it is mixed or noise.

For each cluster you get its index, how many emails it holds, distinctive keywords, a few example subjects, and the dominant sender brands.

RULES:
- Name the PURPOSE, never a sender. Never name a cluster after a brand or service. Use "Developer Code Reviews" for GitHub or GitLab mail, never "GitHub Notifications".
- A label is 2 to 4 words, Title Case, with a concrete anchor word naming a real purpose (banking, invoice, job, course, security, travel, insurance, shipping, tax, medical, and so on).
- BANNED vague labels where every word is generic: "Notifications", "Updates", "General", "Account Notices", "Service Messages", "Miscellaneous". When a cluster is only generic mail, set action to "leave_uncategorized".
- A description is ONE short sentence naming the purpose only. Do NOT name senders, brands, or apps.
- suggestedKey is a short lowercase dotted slug of the purpose, like "finance.invoices" or "travel.bookings".
- Set action to "new_category" for a nameable purpose, or "leave_uncategorized" for a mixed or noise cluster. Do not invent a purpose the keywords and subjects do not support.
- Output ONLY a JSON object, no extra text, no code fences, no markdown.

OUTPUT FORMAT:
{"clusters": [{"clusterIndex": 0, "action": "new_category", "label": "Receipts & Invoices", "description": "Payment confirmations and invoices for purchases.", "suggestedKey": "finance.invoices"}]}`;

/**
 * Build the system and user messages that ask the model to name every given cluster. The `/no_think`
 * control line is Ollama-specific, so it is added only for the local model (`noThink`); a cloud model
 * must not receive it.
 */
export function buildNamingMessages(
  clusters: ClusterNamingInput[],
  opts: { noThink?: boolean } = {},
): ChatMessage[] {
  const body = clusters
    .map((c) => {
      const keys = c.keyphrases.slice(0, 8).join(', ') || '(none)';
      const senders = c.senderHints.slice(0, 5).join(', ') || '(various)';
      const subjects = c.sampleSubjects
        .slice(0, NAMING_SAMPLE_PER_CLUSTER)
        .map((s) => `  - ${s.slice(0, 80)}`)
        .join('\n');
      return `Cluster ${c.index} (${c.size} emails)\nKeywords: ${keys}\nSenders: ${senders}\nExample subjects:\n${subjects}`;
    })
    .join('\n\n');
  const user = `Name these ${clusters.length} clusters. Return one object per cluster in the "clusters" array.\n\n${body}`;
  const system = opts.noThink ? `/no_think\n${SYSTEM_PROMPT}` : SYSTEM_PROMPT;
  return [
    { role: 'system', content: system },
    { role: 'user', content: user },
  ];
}

const NamedSchema = z.object({
  clusterIndex: z.coerce.number().int(),
  action: z
    .enum(['new_category', 'expand_existing', 'merge_suggestion', 'leave_uncategorized'])
    .catch('new_category'),
  label: z.string().catch(''),
  description: z.string().catch(''),
  suggestedKey: z.string().catch(''),
});

/** Remove a reasoning block the model emitted, including an unterminated one at the end. */
function stripThink(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/<think>[\s\S]*$/i, '');
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/**
 * Pull complete objects out of a possibly truncated JSON array or object, balancing braces while
 * respecting strings, so a cut-off answer still yields the objects that did arrive intact.
 */
function salvageObjects(text: string): unknown[] {
  const start = text.indexOf('[') === -1 ? text.indexOf('{') : text.indexOf('[');
  if (start === -1) return [];
  const out: unknown[] = [];
  let depth = 0;
  let objStart = -1;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i]!;
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === '{') {
      if (depth === 0) objStart = i;
      depth += 1;
    } else if (ch === '}') {
      if (depth > 0) {
        depth -= 1;
        if (depth === 0 && objStart !== -1) {
          try {
            out.push(JSON.parse(text.slice(objStart, i + 1)));
          } catch {}
          objStart = -1;
        }
      }
    }
  }
  return out;
}

/**
 * Parse the model's answer into per-cluster names. Tolerates a bare array, an object wrapping a
 * "clusters"/"candidates" array, code fences, a reasoning block, and truncation. Keeps only the
 * first entry per in-range cluster index; malformed entries are dropped, never accepted blindly.
 */
export function parseNamedCandidates(raw: string, clusterCount: number): ParsedNaming[] {
  const text = stripThink(stripCodeFence(raw)).trim();
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    data = salvageObjects(text);
  }
  const arr = Array.isArray(data)
    ? data
    : isRecord(data)
      ? ((data.clusters ?? data.candidates ?? data.topics ?? []) as unknown)
      : [];
  const items = Array.isArray(arr) ? arr : [];

  const seen = new Set<number>();
  const out: ParsedNaming[] = [];
  for (const item of items) {
    const parsed = NamedSchema.safeParse(item);
    if (!parsed.success) continue;
    const idx = parsed.data.clusterIndex;
    if (!Number.isInteger(idx) || idx < 0 || idx >= clusterCount || seen.has(idx)) continue;
    seen.add(idx);
    out.push({
      clusterIndex: idx,
      action: parsed.data.action,
      label: parsed.data.label.trim(),
      description: parsed.data.description.trim(),
      suggestedKey: parsed.data.suggestedKey.trim(),
    });
  }
  return out;
}
