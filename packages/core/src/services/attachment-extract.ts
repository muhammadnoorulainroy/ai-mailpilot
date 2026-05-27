/**
 * Extracts plain text from email attachment bytes for RAG indexing, supporting PDF, DOCX,
 * and text-like formats while reporting why an attachment yielded no usable text.
 */
import { stripHtml, normalizeWhitespace } from '../util/text.js';

/** Outcome of an extraction attempt, letting the caller record why an attachment produced no chunks. */
export type ExtractStatus = 'extracted' | 'empty' | 'unsupported';

/** Extracted plain text paired with its extraction status. */
export interface ExtractResult {
  text: string;
  status: ExtractStatus;
}

const MAX_PDF_PAGES = 1000;
const PDF_RE = /pdf/i;
const DOCX_TYPE_RE = /officedocument\.wordprocessingml/i;
const DOCX_EXT_RE = /\.docx$/i;
const TEXT_TYPE_RE = /^(text\/|application\/(json|xml|csv|x-ndjson|xhtml\+xml))/i;
const TEXT_EXT_RE = /\.(txt|text|csv|tsv|md|markdown|log|json|xml|html?|xhtml)$/i;
const HTML_RE = /html/i;

/**
 * Extract plain text from an attachment's bytes for RAG, supporting PDF, DOCX, and text-like
 * formats. Unsupported types return cleanly, throwing only on an unexpected parse failure.
 */
export async function extractAttachmentText(
  bytes: Uint8Array,
  filename: string,
  contentType?: string,
): Promise<ExtractResult> {
  const ct = contentType ?? '';
  const name = filename ?? '';

  if (PDF_RE.test(ct) || /\.pdf$/i.test(name)) {
    try {
      const text = await extractPdf(bytes);
      return { text, status: text.trim() ? 'extracted' : 'empty' };
    } catch (err) {
      if (isEncryptedPdfError(err)) return { text: '', status: 'unsupported' };
      throw err;
    }
  }

  if (DOCX_TYPE_RE.test(ct) || DOCX_EXT_RE.test(name)) {
    const mammoth = (await import('mammoth')).default;
    const { value } = await mammoth.extractRawText({ buffer: Buffer.from(bytes) });
    return { text: value, status: value.trim() ? 'extracted' : 'empty' };
  }

  if (TEXT_TYPE_RE.test(ct) || TEXT_EXT_RE.test(name)) {
    const raw = new TextDecoder('utf-8', { fatal: false }).decode(bytes);
    const isHtml = HTML_RE.test(ct) || /\.x?html?$/i.test(name);
    const text = isHtml ? normalizeWhitespace(stripHtml(raw)) : raw;
    return { text, status: text.trim() ? 'extracted' : 'empty' };
  }

  return { text: '', status: 'unsupported' };
}

/** Detect the PasswordException pdfjs throws for encrypted PDFs so the caller can mark the attachment terminally unsupported. */
export function isEncryptedPdfError(err: unknown): boolean {
  const e = err as { name?: string; message?: string } | null;
  if (e?.name === 'PasswordException') return true;
  return /\b(no|incorrect|missing|wrong) password\b|password[- ]?(protected|required|given|needed)/i.test(
    e?.message ?? '',
  );
}

/** Parse a PDF with pdfjs and join the text of each page, capping at MAX_PDF_PAGES to bound work on huge files. */
async function extractPdf(bytes: Uint8Array): Promise<string> {
  const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs');
  const task = pdfjs.getDocument({
    data: bytes,
    useSystemFonts: true,
    isEvalSupported: false,
  } as Parameters<typeof pdfjs.getDocument>[0]);
  const doc = await task.promise;
  try {
    const pages: string[] = [];
    const pageCount = Math.min(doc.numPages, MAX_PDF_PAGES);
    for (let i = 1; i <= pageCount; i++) {
      const page = await doc.getPage(i);
      const content = await page.getTextContent();
      pages.push(
        content.items
          .map((it) => ('str' in it ? it.str : ''))
          .join(' ')
          .replace(/\s+/g, ' ')
          .trim(),
      );
    }
    return pages.filter(Boolean).join('\n\n');
  } finally {
    await task.destroy();
  }
}
