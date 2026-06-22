/**
 * Tests for detectAttachments, verifying it distinguishes real attachments
 * from inline content such as cid images and inline-disposition parts.
 */
import { describe, it, expect } from 'vitest';
import { detectAttachments } from '../src/thunderbird/mailbox.js';

/** Casts a plain object into a TbMessagePart for use as test fixture data. */
const part = (p: Record<string, unknown>): TbMessagePart => p as unknown as TbMessagePart;
const textPlain = part({ contentType: 'text/plain', body: 'hi', headers: {} });
const textHtml = part({ contentType: 'text/html', body: '<p>hi</p>', headers: {} });

describe('detectAttachments (real attachment vs inline content)', () => {
  it('plain-text and html-only emails have no attachment', () => {
    expect(detectAttachments(part({ contentType: 'multipart/alternative', parts: [textPlain, textHtml] }))).toBe(false);
    expect(detectAttachments(textPlain)).toBe(false);
  });

  it('does NOT flag a cid: inline image (Content-ID referenced from HTML)', () => {
    const inlineImg = part({ contentType: 'image/png', name: 'logo.png', headers: { 'content-id': ['<abc@host>'] } });
    expect(detectAttachments(part({ contentType: 'multipart/related', parts: [textHtml, inlineImg] }))).toBe(false);
  });

  it('does NOT flag an inline-disposition image', () => {
    const img = part({ contentType: 'image/png', name: 'sig.png', headers: { 'content-disposition': ['inline; filename="sig.png"'] } });
    expect(detectAttachments(part({ contentType: 'multipart/mixed', parts: [textHtml, img] }))).toBe(false);
  });

  it('DOES flag a real PDF attachment (Content-Disposition: attachment)', () => {
    const pdf = part({ contentType: 'application/pdf', name: 'report.pdf', headers: { 'content-disposition': ['attachment; filename="report.pdf"'] } });
    expect(detectAttachments(part({ contentType: 'multipart/mixed', parts: [textHtml, pdf] }))).toBe(true);
  });

  it('DOES flag an attachment named via RFC 2231 (filename*= / name*=)', () => {
    const pdf = part({
      contentType: 'application/pdf',
      headers: { 'content-type': ["application/pdf; name*=UTF-8''r%C3%A9sum%C3%A9.pdf"] },
    });
    expect(detectAttachments(part({ contentType: 'multipart/mixed', parts: [textHtml, pdf] }))).toBe(true);
  });

  it('DOES flag a named, non-inline document part even without an explicit disposition', () => {
    const docx = part({
      contentType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      name: 'notes.docx',
      headers: {},
    });
    expect(detectAttachments(part({ contentType: 'multipart/mixed', parts: [textPlain, docx] }))).toBe(true);
  });
});
