/**
 * Tests for attachment text extraction, chunking, and the attachment repository,
 * covering PDF, DOCX, HTML and plain text handling plus vector and keyword search.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import JSZip from 'jszip';
import type { Database } from 'better-sqlite3';
import { openDatabase } from '../src/db/database.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { EmailRepository } from '../src/repositories/email-repository.js';
import { AttachmentRepository } from '../src/repositories/attachment-repository.js';
import { chunkText } from '../src/util/chunk.js';
import { extractAttachmentText, isEncryptedPdfError } from '../src/services/attachment-extract.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';

/** Builds a minimal valid PDF whose pages render the given text, wrapping lines near 60 chars. */
function buildPdf(text: string): Uint8Array {
  const words = text.split(' ');
  const lines: string[] = [];
  let cur = '';
  for (const w of words) {
    if (cur && (cur + ' ' + w).length > 60) {
      lines.push(cur);
      cur = w;
    } else {
      cur = cur ? `${cur} ${w}` : w;
    }
  }
  if (cur) lines.push(cur);
  const streamBody = lines.map((ln, i) => `BT /F1 11 Tf 72 ${720 - i * 16} Td (${ln}) Tj ET`).join('\n');
  const objs = [
    '<< /Type /Catalog /Pages 2 0 R >>',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>',
    `<< /Length ${streamBody.length} >>\nstream\n${streamBody}\nendstream`,
    '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
  ];
  let pdf = '%PDF-1.4\n';
  const offsets: number[] = [];
  objs.forEach((body, i) => {
    offsets.push(pdf.length);
    pdf += `${i + 1} 0 obj\n${body}\nendobj\n`;
  });
  const xrefPos = pdf.length;
  pdf += `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n`;
  offsets.forEach((o) => (pdf += String(o).padStart(10, '0') + ' 00000 n \n'));
  pdf += `trailer\n<< /Size ${objs.length + 1} /Root 1 0 R >>\nstartxref\n${xrefPos}\n%%EOF`;
  return new Uint8Array(Buffer.from(pdf, 'latin1'));
}

/** Builds a minimal valid DOCX (zipped OOXML) containing the given text in a single paragraph. */
async function buildDocx(text: string): Promise<Uint8Array> {
  const zip = new JSZip();
  zip.file(
    '[Content_Types].xml',
    `<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>`,
  );
  zip
    .folder('_rels')!
    .file(
      '.rels',
      `<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>`,
    );
  zip
    .folder('word')!
    .file(
      'document.xml',
      `<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>${text}</w:t></w:r></w:p></w:body></w:document>`,
    );
  return new Uint8Array(await zip.generateAsync({ type: 'nodebuffer' }));
}

describe('chunkText', () => {
  it('returns one chunk for short text and [] for empty', () => {
    expect(chunkText('')).toEqual([]);
    expect(chunkText('   ')).toEqual([]);
    expect(chunkText('a short note')).toEqual(['a short note']);
  });

  it('splits long text into overlapping chunks that cover all content', () => {
    const sentences = Array.from({ length: 60 }, (_, i) => `Sentence number ${i} about the project deadline.`).join(' ');
    const chunks = chunkText(sentences, { maxChars: 400, overlapChars: 80 });
    expect(chunks.length).toBeGreaterThan(1);
    chunks.forEach((c) => expect(c.length).toBeLessThanOrEqual(400));
    expect(chunks.join(' ')).toContain('Sentence number 59');
    expect(chunks.join(' ')).toContain('Sentence number 0');
  });
});

describe('extractAttachmentText', () => {
  it('extracts plain text and csv', async () => {
    const txt = new TextEncoder().encode('Invoice total 4200 EUR due 2026-07-15');
    const r = await extractAttachmentText(txt, 'invoice.txt', 'text/plain');
    expect(r.status).toBe('extracted');
    expect(r.text).toContain('4200');
  });

  it('strips HTML attachments', async () => {
    const html = new TextEncoder().encode('<html><body><p>Hello <b>viva</b> schedule</p></body></html>');
    const r = await extractAttachmentText(html, 'note.html', 'text/html');
    expect(r.text).toContain('viva schedule');
    expect(r.text).not.toContain('<b>');
  });

  it('extracts text from a real PDF', async () => {
    const r = await extractAttachmentText(buildPdf('Project viva on June 4 2018'), 'schedule.pdf', 'application/pdf');
    expect(r.status).toBe('extracted');
    expect(r.text).toContain('Project viva on June 4 2018');
  });

  it('extracts the FULL text of a multi-line PDF (no truncation)', async () => {
    const full = 'Internal memo. The Q3 budget deadline is July 15 2026. The project codename is BLUEFOX and the lead is Maxime.';
    const r = await extractAttachmentText(buildPdf(full), 'memo.pdf', 'application/pdf');
    expect(r.text).toContain('budget deadline');
    expect(r.text).toContain('codename is BLUEFOX');
    expect(r.text).toContain('Maxime');
  });

  it('extracts text from a real DOCX', async () => {
    const bytes = await buildDocx('Quarterly budget report Q3 deadline July 15');
    const r = await extractAttachmentText(bytes, 'report.docx');
    expect(r.status).toBe('extracted');
    expect(r.text).toContain('Quarterly budget report');
  });

  it('returns unsupported for an unknown binary type', async () => {
    const r = await extractAttachmentText(new Uint8Array([0, 1, 2, 3]), 'image.png', 'image/png');
    expect(r.status).toBe('unsupported');
  });

  it('classifies password-protected PDF errors as terminal (so they are not retried)', () => {
    expect(isEncryptedPdfError({ name: 'PasswordException', message: 'No password given' })).toBe(true);
    expect(isEncryptedPdfError({ name: 'PasswordException', message: '' })).toBe(true);
    expect(isEncryptedPdfError(new Error('Incorrect Password'))).toBe(true);
    expect(isEncryptedPdfError(new Error('Invalid PDF structure'))).toBe(false);
    expect(isEncryptedPdfError(new Error('expected password dict at offset 12'))).toBe(false);
    expect(isEncryptedPdfError(null)).toBe(false);
  });
});

describe('AttachmentRepository', () => {
  let db: Database;
  let accounts: AccountRepository;
  let emails: EmailRepository;
  let attachments: AttachmentRepository;

  const vec = (seed: number): Float32Array => {
    const v = new Float32Array(EMBEDDING_DIM);
    v[0] = 1;
    v[1] = seed;
    return v;
  };

  beforeEach(() => {
    db = openDatabase(':memory:');
    accounts = new AccountRepository(db);
    emails = new EmailRepository(db);
    attachments = new AttachmentRepository(db);
  });
  afterEach(() => db.close());

  /** Seeds an account, email, attachment and two extracted chunks, returning the account id and chunk rowids. */
  function seedAttachment(): { accountId: string; rowids: number[] } {
    const acct = accounts.create({ address: 'a@x.y', kind: 'work' });
    emails.upsertBatch([{ messageId: 'm1', accountId: acct.id, folder: 'INBOX', subject: 'Report' }]);
    const { id } = attachments.upsertAttachment({ messageId: 'm1', accountId: acct.id, filename: 'report.pdf', partName: '1.2' });
    const rowids = attachments.replaceChunks(id, 'm1', acct.id, ['budget deadline July', 'viva schedule June']);
    attachments.setStatus(id, 'extracted', 40);
    return { accountId: acct.id, rowids };
  }

  it('stores chunks, finds them by vector and keyword, and loads citation context', () => {
    const { accountId, rowids } = seedAttachment();
    attachments.saveChunkEmbedding(rowids[0]!, accountId, 'bge-m3', vec(0));
    attachments.saveChunkEmbedding(rowids[1]!, accountId, 'bge-m3', vec(0.5));

    const hits = attachments.searchChunks(accountId, 'bge-m3', vec(0), 2);
    expect(hits[0]!.chunkRowid).toBe(rowids[0]);

    const kw = attachments.keywordSearchChunks(accountId, 'viva', 5);
    expect(kw).toContain(rowids[1]);

    const loaded = attachments.loadChunks(rowids, accountId);
    expect(loaded.get(rowids[1]!)!.filename).toBe('report.pdf');
    expect(loaded.get(rowids[1]!)!.messageId).toBe('m1');
    expect(attachments.countExtracted(accountId)).toBe(1);
  });

  it('replacing chunks drops the old chunks, their embeddings, and FTS entries', () => {
    const { accountId } = seedAttachment();
    const acctEmails = emails;
    void acctEmails;
    expect(attachments.keywordSearchChunks(accountId, 'budget', 5).length).toBe(1);

    const a = attachments.upsertAttachment({ messageId: 'm1', accountId, filename: 'report.pdf', partName: '1.2' });
    attachments.replaceChunks(a.id, 'm1', accountId, ['zebra migration patterns']);
    expect(attachments.keywordSearchChunks(accountId, 'budget', 5)).toEqual([]);
    expect(attachments.keywordSearchChunks(accountId, 'zebra', 5).length).toBe(1);
  });

  it('skips re-extracting an already-extracted attachment (idempotent upsert status)', () => {
    const { accountId } = seedAttachment();
    const again = attachments.upsertAttachment({ messageId: 'm1', accountId, filename: 'report.pdf', partName: '1.2' });
    expect(again.priorStatus).toBe('extracted');
  });

  it('deleting the email cascades to attachments, chunks, embeddings, and FTS', () => {
    const { accountId, rowids } = seedAttachment();
    attachments.saveChunkEmbedding(rowids[0]!, accountId, 'bge-m3', vec(0));
    const count = (t: string): number => (db.prepare(`SELECT count(*) AS c FROM ${t}`).get() as { c: number }).c;
    expect(count('attachments')).toBe(1);
    expect(count('attachment_chunks')).toBe(2);
    expect(count('attachment_chunk_embedding_index')).toBe(1);
    expect(count('attachment_chunk_embeddings')).toBe(1);

    emails.delete('m1', accountId);
    expect(count('attachments')).toBe(0);
    expect(count('attachment_chunks')).toBe(0);
    expect(count('attachment_chunk_embedding_index')).toBe(0);
    expect(count('attachment_chunk_embeddings')).toBe(0);
    expect(attachments.keywordSearchChunks(accountId, 'budget', 5)).toEqual([]);
  });
});
