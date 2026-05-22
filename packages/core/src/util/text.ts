const HTML_TAG = /<[^>]+>/g;
const HTML_ENTITY = /&(amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);/g;
const WHITESPACE = /\s+/g;
const QUOTED_REPLY = /^>.*$/gm;
const SIGNATURE_MARKER = /^-- ?$\n[\s\S]*$/m;

const PRE_TRUNCATE_FACTOR = 4;

export function stripHtml(html: string): string {
  return html
    .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, ' ')
    .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, ' ')
    .replace(HTML_TAG, ' ')
    .replace(HTML_ENTITY, (m) => decodeEntity(m));
}

export function normalizeWhitespace(text: string): string {
  return text.replace(WHITESPACE, ' ').trim();
}

function safeSlice(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  let end = maxChars;
  const last = text.charCodeAt(end - 1);
  if (last >= 0xd800 && last <= 0xdbff) end -= 1;
  return text.slice(0, end);
}

export function stripQuotesAndSignature(text: string): string {
  return text.replace(/\r\n?/g, '\n').replace(SIGNATURE_MARKER, '').replace(QUOTED_REPLY, '');
}

export interface PreprocessOptions {
  format?: 'text' | 'html';
  maxChars?: number;
  keepQuotes?: boolean;
}

export function preprocessForEmbedding(raw: string, opts: PreprocessOptions = {}): string {
  const format = opts.format ?? 'text';
  const maxChars = opts.maxChars ?? 8000;
  const keepQuotes = opts.keepQuotes ?? false;

  let text =
    raw.length > maxChars * PRE_TRUNCATE_FACTOR ? safeSlice(raw, maxChars * PRE_TRUNCATE_FACTOR) : raw;

  if (format === 'html') text = stripHtml(text);
  if (!keepQuotes) text = stripQuotesAndSignature(text);
  text = normalizeWhitespace(text);

  return safeSlice(text, maxChars);
}

const EMBEDDING_BODY_CHARS = 8000;

export function buildEmbeddingInput(
  parts: {
    subject?: string | null;
    fromAddr?: string | null;
    body?: string | null;
    bodyFormat?: 'text' | 'html';
  },
  maxChars: number = EMBEDDING_BODY_CHARS,
): string {
  const subject = parts.subject?.trim() ?? '';
  const from = parts.fromAddr?.trim() ?? '';
  const body = parts.body
    ? preprocessForEmbedding(parts.body, { format: parts.bodyFormat, maxChars })
    : '';

  const lines: string[] = [];
  if (subject) lines.push(`Subject: ${subject}`);
  if (from) lines.push(`From: ${from}`);
  if (body) lines.push('', body);

  return lines.join('\n');
}

export function sanitizeFtsQuery(raw: string): string {
  const tokens = raw.toLowerCase().match(/[\p{L}\p{N}]{2,}/gu);
  if (!tokens) return '';
  const unique = [...new Set(tokens)].slice(0, 16);
  return unique.map((t) => `"${t.replace(/"/g, '""')}"`).join(' OR ');
}

export function normalizeForMatch(s: string): string {
  return s
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

export function normalizeFilename(s: string): string {
  return normalizeForMatch(
    s
      .normalize('NFD')
      .replace(/\p{M}/gu, '')
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2'),
  );
}

function decodeEntity(entity: string): string {
  if (entity === '&amp;') return '&';
  if (entity === '&lt;') return '<';
  if (entity === '&gt;') return '>';
  if (entity === '&quot;') return '"';
  if (entity === '&apos;') return "'";
  if (entity === '&nbsp;') return ' ';
  if (entity.startsWith('&#x')) {
    return String.fromCodePoint(parseInt(entity.slice(3, -1), 16));
  }
  if (entity.startsWith('&#')) {
    return String.fromCodePoint(parseInt(entity.slice(2, -1), 10));
  }
  return entity;
}
