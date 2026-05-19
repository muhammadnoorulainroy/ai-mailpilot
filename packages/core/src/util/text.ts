const HTML_TAG = /<[^>]+>/g;
const HTML_ENTITY = /&(amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);/g;
const WHITESPACE = /\s+/g;
const QUOTED_REPLY = /^>.*$/gm;
const SIGNATURE_MARKER = /^-- ?$\n[\s\S]*$/m;

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

export function stripQuotesAndSignature(text: string): string {
  return text.replace(SIGNATURE_MARKER, '').replace(QUOTED_REPLY, '');
}

export interface PreprocessOptions {
  format?: 'text' | 'html';
  maxChars?: number;
  keepQuotes?: boolean;
}

export function preprocessForEmbedding(
  raw: string,
  opts: PreprocessOptions = {},
): string {
  const format = opts.format ?? 'text';
  const maxChars = opts.maxChars ?? 8000;
  const keepQuotes = opts.keepQuotes ?? false;

  let text = format === 'html' ? stripHtml(raw) : raw;
  if (!keepQuotes) text = stripQuotesAndSignature(text);
  text = normalizeWhitespace(text);

  if (text.length > maxChars) text = text.slice(0, maxChars);
  return text;
}

export function buildEmbeddingInput(parts: {
  subject?: string | null;
  fromAddr?: string | null;
  body?: string | null;
  bodyFormat?: 'text' | 'html';
}): string {
  const subject = parts.subject?.trim() ?? '';
  const from = parts.fromAddr?.trim() ?? '';
  const body = parts.body
    ? preprocessForEmbedding(parts.body, { format: parts.bodyFormat })
    : '';

  const lines: string[] = [];
  if (subject) lines.push(`Subject: ${subject}`);
  if (from) lines.push(`From: ${from}`);
  if (body) lines.push('', body);

  return lines.join('\n');
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
