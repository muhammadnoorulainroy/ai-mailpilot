/**
 * Text utilities for email content: HTML stripping, quote and signature removal,
 * embedding input assembly, FTS query sanitization, and fuzzy match normalization.
 */
const HTML_STYLE = /<style[^>]*>[\s\S]*?<\/style>/gi;
const HTML_SCRIPT = /<script[^>]*>[\s\S]*?<\/script>/gi;
const HTML_TAG = /<[^>]+>/g;
const HTML_ENTITY = /&(amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);/g;
const WHITESPACE = /\s+/g;
const QUOTED_REPLY = /^>.*$/gm;
const SIGNATURE_MARKER = /^-- ?$\n[\s\S]*$/m;

const PRE_TRUNCATE_FACTOR = 4;

/**
 * Strip style, script, and tag markup from HTML and decode common entities to plain text.
 */
export function stripHtml(html: string): string {
  return html
    .replace(HTML_STYLE, ' ')
    .replace(HTML_SCRIPT, ' ')
    .replace(HTML_TAG, ' ')
    .replace(HTML_ENTITY, (m) => decodeEntity(m));
}

/**
 * Collapse runs of whitespace to single spaces and trim the result.
 */
export function normalizeWhitespace(text: string): string {
  return text.replace(WHITESPACE, ' ').trim();
}

/**
 * Slice text to maxChars without splitting a surrogate pair, backing off by one
 * when the cut would land on a high surrogate.
 */
function safeSlice(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  let end = maxChars;
  const last = text.charCodeAt(end - 1);
  if (last >= 0xd800 && last <= 0xdbff) end -= 1;
  return text.slice(0, end);
}

/**
 * Remove quoted reply lines and a trailing signature block from email text.
 */
export function stripQuotesAndSignature(text: string): string {
  return text.replace(/\r\n?/g, '\n').replace(SIGNATURE_MARKER, '').replace(QUOTED_REPLY, '');
}

/**
 * Options controlling how raw email content is preprocessed before embedding.
 */
export interface PreprocessOptions {
  format?: 'text' | 'html';
  maxChars?: number;
  keepQuotes?: boolean;
}

/**
 * Preprocess raw email content for embedding: optional HTML stripping, quote and
 * signature removal, whitespace normalization, and length capping.
 */
export function preprocessForEmbedding(raw: string, opts: PreprocessOptions = {}): string {
  const format = opts.format ?? 'text';
  const maxChars = opts.maxChars ?? 8000;
  const keepQuotes = opts.keepQuotes ?? false;

  let text =
    raw.length > maxChars * PRE_TRUNCATE_FACTOR
      ? safeSlice(raw, maxChars * PRE_TRUNCATE_FACTOR)
      : raw;

  if (format === 'html') text = stripHtml(text);
  if (!keepQuotes) text = stripQuotesAndSignature(text);
  text = normalizeWhitespace(text);

  return safeSlice(text, maxChars);
}

const EMBEDDING_BODY_CHARS = 8000;

/**
 * Assemble subject, sender, and preprocessed body into the text fed to the embedding model.
 */
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

/**
 * Turn free text into a safe FTS5 MATCH string for keyword retrieval. Extracts word
 * tokens so French words survive, drops punctuation and FTS5 operators, quotes each
 * token as a literal, and ORs them for recall. Returns '' when nothing usable remains.
 */
export function sanitizeFtsQuery(raw: string): string {
  const tokens = raw.toLowerCase().match(/[\p{L}\p{N}]{2,}/gu);
  if (!tokens) return '';
  const unique = [...new Set(tokens)].slice(0, 16);
  return unique.map((t) => `"${t.replace(/"/g, '""')}"`).join(' OR ');
}

/**
 * Normalize a string for fuzzy filename and name matching: lowercase, strip accents,
 * and collapse runs of non-alphanumeric characters to one space. Lets run-together and
 * spaced wordings reduce to comparable token streams, and accented French names match
 * unaccented queries.
 */
export function normalizeForMatch(s: string): string {
  return s
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

/**
 * Like normalizeForMatch but also splits camelCase so a run-together filename matches
 * spaced wording. Only for filenames, since chunk and prose scoring keeps normalizeForMatch
 * so a camelCased word in body text is not split out of a query match.
 */
export function normalizeFilename(s: string): string {
  return normalizeForMatch(
    s
      .normalize('NFD')
      .replace(/\p{M}/gu, '')
      .replace(/([a-z0-9])([A-Z])/g, '$1 $2'),
  );
}

/**
 * Decode a single named or numeric HTML entity to its character, returning the
 * input unchanged when it is not recognized.
 */
function decodeEntity(entity: string): string {
  switch (entity) {
    case '&amp;':
      return '&';
    case '&lt;':
      return '<';
    case '&gt;':
      return '>';
    case '&quot;':
      return '"';
    case '&apos;':
      return "'";
    case '&nbsp;':
      return ' ';
  }
  if (entity.startsWith('&#x')) {
    return String.fromCodePoint(parseInt(entity.slice(3, -1), 16));
  }
  if (entity.startsWith('&#')) {
    return String.fromCodePoint(parseInt(entity.slice(2, -1), 10));
  }
  return entity;
}
