/**
 * Model-aware embedding instructions. Instruction-tuned models (Qwen3-Embedding) get a purpose
 * instruction for stored content and a retrieval instruction for search queries; non-instruction
 * models (bge-m3) pass through unchanged so switching the embedding model is safe. The instruction
 * must name only generic email functions, never domain categories, so it generalizes across mailboxes.
 */
import { describe, it, expect } from 'vitest';
import { withEmbeddingInstruction, isInstructionTunedEmbedding } from '../src/util/text.js';

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
});
