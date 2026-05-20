/**
 * Splits long text into overlapping chunks at natural boundaries for embedding and retrieval.
 */

/**
 * Options controlling chunk size and overlap for {@link chunkText}.
 */
export interface ChunkOptions {
  maxChars?: number;
  overlapChars?: number;
}

const DEFAULT_MAX = 1200;
const DEFAULT_OVERLAP = 150;

/**
 * Splits text into overlapping chunks for embedding and retrieval. Each chunk targets
 * roughly maxChars and breaks at a paragraph, sentence, or word boundary near the limit,
 * with consecutive chunks overlapping so a fact spanning a boundary stays retrievable.
 */
export function chunkText(text: string, opts: ChunkOptions = {}): string[] {
  const maxChars = Math.max(200, opts.maxChars ?? DEFAULT_MAX);
  const overlap = Math.max(
    0,
    Math.min(opts.overlapChars ?? DEFAULT_OVERLAP, Math.floor(maxChars / 2)),
  );

  const clean = text
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+/g, ' ')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
  if (clean.length === 0) return [];
  if (clean.length <= maxChars) return [clean];

  const chunks: string[] = [];
  let start = 0;
  while (start < clean.length) {
    let end = Math.min(start + maxChars, clean.length);
    if (end < clean.length) {
      const breakAt = lastBreak(clean.slice(start, end));
      if (breakAt > maxChars * 0.5) end = start + breakAt;
    }
    const chunk = clean.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end >= clean.length) break;
    start = Math.max(end - overlap, start + 1);
  }
  return chunks;
}

/**
 * Returns the index just after the best break point in a window, preferring a paragraph,
 * then a sentence end, then a word boundary, falling back to the window length.
 */
function lastBreak(window: string): number {
  const para = window.lastIndexOf('\n\n');
  if (para !== -1) return para + 2;
  const sentence = Math.max(
    window.lastIndexOf('. '),
    window.lastIndexOf('? '),
    window.lastIndexOf('! '),
    window.lastIndexOf('.\n'),
  );
  if (sentence !== -1) return sentence + 1;
  const space = window.lastIndexOf(' ');
  return space !== -1 ? space + 1 : window.length;
}
