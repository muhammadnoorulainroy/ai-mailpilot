/**
 * Model-aware embedding instructions. Instruction-tuned models (Qwen3-Embedding) get a purpose
 * instruction for stored content and a retrieval instruction for search queries; non-instruction
 * models (bge-m3) pass through unchanged so switching the embedding model is safe. The instruction
 * must name only generic email functions, never domain categories, so it generalizes across mailboxes.
 */
import { describe, it, expect } from 'vitest';
import {
  withEmbeddingInstruction,
  isInstructionTunedEmbedding,
  buildChunkContext,
} from '../src/util/text.js';

describe('withEmbeddingInstruction', () => {
  it('passes text through unchanged for non-instruction models', () => {
    const t = 'Subject: hi\nFrom: a@b.com';
    expect(withEmbeddingInstruction(t, 'bge-m3')).toBe(t);
    expect(withEmbeddingInstruction(t, 'bge-m3', 'query')).toBe(t);
    expect(withEmbeddingInstruction(t, undefined)).toBe(t);
  });

  it('wraps qwen content with the purpose (document) instruction by default', () => {
    const out = withEmbeddingInstruction('hello', 'qwen3-embedding:0.6b');
    expect(out).toContain('Instruct:');
    expect(out).toContain('purpose and type');
    expect(out.endsWith('\nQuery: hello')).toBe(true);
  });

  it('wraps qwen search queries with the retrieval instruction', () => {
    const out = withEmbeddingInstruction('bank receipts', 'qwen3-embedding:0.6b', 'query');
    expect(out).toContain('retrieve emails relevant');
    expect(out.endsWith('\nQuery: bank receipts')).toBe(true);
  });

  it('recognizes the qwen3-embedding family case-insensitively, and nothing else', () => {
    expect(isInstructionTunedEmbedding('qwen3-embedding:4b')).toBe(true);
    expect(isInstructionTunedEmbedding('Qwen3-Embedding:0.6B')).toBe(true);
    expect(isInstructionTunedEmbedding('bge-m3')).toBe(false);
    expect(isInstructionTunedEmbedding('nomic-embed-text')).toBe(false);
    expect(isInstructionTunedEmbedding(undefined)).toBe(false);
  });

  it('names no domain-specific category, so it generalizes to any mailbox (professor/federated-safe)', () => {
    const doc = withEmbeddingInstruction('x', 'qwen3-embedding:0.6b').toLowerCase();
    for (const banned of ['job', 'linkedin', 'github', 'bank', 'developer', 'course', 'student']) {
      expect(doc).not.toContain(banned);
    }
  });

  it('gives attachment excerpts their own instruction, not the email-purpose taxonomy', () => {
    const out = withEmbeddingInstruction('clause 4', 'qwen3-embedding:0.6b', 'attachment');
    expect(out).toContain('document');
    expect(out).not.toContain('purpose and type');
    expect(out.endsWith('\nQuery: clause 4')).toBe(true);
  });

  it('names no domain-specific category in the attachment instruction either', () => {
    const att = withEmbeddingInstruction('x', 'qwen3-embedding:0.6b', 'attachment').toLowerCase();
    for (const banned of ['insurance', 'contract', 'invoice', 'grade', 'bank', 'course']) {
      expect(att).not.toContain(banned);
    }
  });
});

describe('buildChunkContext (contextual retrieval)', () => {
  const parts = {
    filename: 'Contrat Habitation ADH 2024.pdf',
    subject: 'Votre contrat',
    fromAddr: 'contact@adh-assurances.fr',
    date: Date.UTC(2024, 2, 15),
  };

  it('situates a chunk with the document name, sender, date, and subject', () => {
    const ctx = buildChunkContext(parts);
    expect(ctx).toBe(
      'Document: Contrat Habitation ADH 2024.pdf | From: contact@adh-assurances.fr | ' +
        'Date: 2024-03-15 | Subject: Votre contrat',
    );
  });

  it('omits fields it does not know rather than emitting empty labels', () => {
    expect(buildChunkContext({ filename: 'a.pdf' })).toBe('Document: a.pdf');
    expect(buildChunkContext({ filename: 'a.pdf', subject: '  ' })).toBe('Document: a.pdf');
    expect(buildChunkContext({})).toBe('');
  });

  it('drops an unparseable date instead of writing "Invalid Date"', () => {
    expect(buildChunkContext({ filename: 'a.pdf', date: Number.NaN })).toBe('Document: a.pdf');
  });
});
