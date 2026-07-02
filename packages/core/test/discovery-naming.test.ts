/**
 * Phase 2b tests: deterministic naming-prompt assembly and tolerant parsing of the model's answer.
 * No LLM and no I/O. The parser must never accept malformed output blindly; the gate judges the rest.
 */
import { describe, it, expect } from 'vitest';
import {
  buildNamingMessages,
  parseNamedCandidates,
  NAMING_SAMPLE_PER_CLUSTER,
  type ClusterNamingInput,
} from '../src/services/discovery-naming.js';

function input(over: Partial<ClusterNamingInput> = {}): ClusterNamingInput {
  return {
    index: 0,
    size: 12,
    keyphrases: ['invoice', 'payment', 'receipt'],
    sampleSubjects: ['Invoice #100 paid', 'Your receipt', 'Payment received'],
    senderHints: ['acme'],
    ...over,
  };
}

describe('buildNamingMessages', () => {
  it('produces a system and a user message grounded in each cluster', () => {
    const messages = buildNamingMessages([input({ index: 0 }), input({ index: 1, size: 7 })]);
    expect(messages).toHaveLength(2);
    expect(messages[0]!.role).toBe('system');
    expect(messages[0]!.content.toLowerCase()).toContain('purpose');
    expect(messages[1]!.role).toBe('user');
    expect(messages[1]!.content).toContain('Cluster 0 (12 emails)');
    expect(messages[1]!.content).toContain('Cluster 1 (7 emails)');
    expect(messages[1]!.content).toContain('Invoice #100 paid');
  });

  it('caps the example subjects shown per cluster', () => {
    const many = Array.from({ length: 40 }, (_, i) => `Subject line ${i}`);
    const [, user] = buildNamingMessages([input({ sampleSubjects: many })]);
    const shown = many.filter((s) => user!.content.includes(s));
    expect(shown.length).toBe(NAMING_SAMPLE_PER_CLUSTER);
  });
});

describe('parseNamedCandidates', () => {
  it('parses a clusters-wrapped object', () => {
    const raw = JSON.stringify({
      clusters: [
        {
          clusterIndex: 0,
          action: 'new_category',
          label: 'Receipts',
          description: 'x',
          suggestedKey: 'finance.receipts',
        },
        {
          clusterIndex: 1,
          action: 'leave_uncategorized',
          label: '',
          description: '',
          suggestedKey: '',
        },
      ],
    });
    const parsed = parseNamedCandidates(raw, 2);
    expect(parsed).toHaveLength(2);
    expect(parsed[0]).toMatchObject({ clusterIndex: 0, label: 'Receipts', action: 'new_category' });
    expect(parsed[1]!.action).toBe('leave_uncategorized');
  });

  it('parses a bare array and tolerates a code fence and a reasoning block', () => {
    const raw =
      '<think>let me think</think>\n```json\n[{"clusterIndex":0,"label":"Travel Bookings","action":"new_category"}]\n```';
    const parsed = parseNamedCandidates(raw, 1);
    expect(parsed).toHaveLength(1);
    expect(parsed[0]!.label).toBe('Travel Bookings');
    // Missing description/suggestedKey default to empty, not dropped.
    expect(parsed[0]!.description).toBe('');
    expect(parsed[0]!.suggestedKey).toBe('');
  });

  it('drops out-of-range and duplicate cluster indices, first wins', () => {
    const raw = JSON.stringify({
      clusters: [
        { clusterIndex: 0, label: 'First', action: 'new_category' },
        { clusterIndex: 0, label: 'Duplicate', action: 'new_category' },
        { clusterIndex: 5, label: 'OutOfRange', action: 'new_category' },
        { clusterIndex: -1, label: 'Negative', action: 'new_category' },
      ],
    });
    const parsed = parseNamedCandidates(raw, 2);
    expect(parsed.map((p) => p.label)).toEqual(['First']);
  });

  it('coerces an unknown action to new_category rather than dropping the entry', () => {
    const raw = JSON.stringify({
      clusters: [{ clusterIndex: 0, label: 'Banking', action: 'wat' }],
    });
    expect(parseNamedCandidates(raw, 1)[0]!.action).toBe('new_category');
  });

  it('salvages complete objects from a truncated answer', () => {
    const raw =
      '{"clusters":[{"clusterIndex":0,"label":"Invoices","action":"new_category"},{"clusterIndex":1,"label":"Cut';
    const parsed = parseNamedCandidates(raw, 2);
    expect(parsed.map((p) => p.label)).toEqual(['Invoices']);
  });

  it('returns nothing for unusable input', () => {
    expect(parseNamedCandidates('not json at all', 3)).toEqual([]);
    expect(parseNamedCandidates('', 3)).toEqual([]);
  });
});
