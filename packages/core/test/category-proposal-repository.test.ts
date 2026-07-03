/**
 * Phase 2c tests: the category_proposals repository. Round-trips the centroid and member list,
 * orders the pending queue by confidence, and moves a proposal through its lifecycle.
 */
import { describe, it, expect } from 'vitest';
import { openDatabase } from '../src/db/database.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import { AccountRepository } from '../src/repositories/account-repository.js';
import { CategoryRepository } from '../src/repositories/category-repository.js';
import {
  CategoryProposalRepository,
  type CreateProposalInput,
} from '../src/repositories/category-proposal-repository.js';

function centroid(seed: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[0] = seed;
  v[1] = seed / 2;
  return v;
}

function setup() {
  const db = openDatabase(':memory:');
  const accounts = new AccountRepository(db);
  const categories = new CategoryRepository(db);
  const proposals = new CategoryProposalRepository(db);
  const acc = accounts.create({ address: 'w@x.com', kind: 'work' });
  const makeInput = (over: Partial<CreateProposalInput> = {}): CreateProposalInput => {
    const cat = categories.create({
      accountId: acc.id,
      label: over.label ?? 'Receipts & Invoices',
      source: 'auto',
      status: 'suggested',
      canonicalKey: over.canonicalKey ?? `key_${Math.round((over.confidence ?? 0.7) * 100)}`,
    });
    return {
      accountId: acc.id,
      categoryId: cat.id,
      runId: 'run-1',
      clusterIndex: 0,
      label: 'Receipts & Invoices',
      description: 'Payment confirmations and invoices.',
      canonicalKey: 'receipts_invoices',
      suggestedKey: 'finance.invoices',
      embeddingModelId: 'bge-m3',
      centroid: centroid(1),
      memberIds: ['m1', 'm2', 'm3'],
      proposedCount: 3,
      cohesion: 0.8,
      separation: 0.7,
      confidence: 0.7,
      evidence: ['invoice', 'payment'],
      ...over,
    };
  };
  return { db, accounts, categories, proposals, accountId: acc.id, makeInput };
}

describe('CategoryProposalRepository', () => {
  it('round-trips the centroid, members, and evidence', () => {
    const { proposals, makeInput } = setup();
    const created = proposals.create(makeInput());
    const loaded = proposals.findById(created.id);
    expect(loaded).not.toBeNull();
    expect(loaded!.memberIds).toEqual(['m1', 'm2', 'm3']);
    expect(loaded!.evidence).toEqual(['invoice', 'payment']);
    expect(Array.from(loaded!.centroid.slice(0, 2))).toEqual([1, 0.5]);
    expect(loaded!.status).toBe('pending');
    expect(loaded!.appliedAt).toBeNull();
    expect(loaded!.dismissedAt).toBeNull();
  });

  it('lists pending proposals by confidence descending', () => {
    const { proposals, accountId, makeInput } = setup();
    proposals.create(makeInput({ label: 'A', canonicalKey: 'a', confidence: 0.6 }));
    proposals.create(makeInput({ label: 'B', canonicalKey: 'b', confidence: 0.9 }));
    proposals.create(makeInput({ label: 'C', canonicalKey: 'c', confidence: 0.75 }));
    expect(proposals.listPending(accountId).map((p) => p.confidence)).toEqual([0.9, 0.75, 0.6]);
  });

  it('moves a proposal to applied and drops it from pending', () => {
    const { proposals, accountId, makeInput } = setup();
    const p = proposals.create(makeInput({ canonicalKey: 'x' }));
    proposals.markApplied(p.id);
    expect(proposals.findById(p.id)!.status).toBe('applied');
    expect(proposals.findById(p.id)!.appliedAt).toBeGreaterThan(0);
    expect(proposals.listPending(accountId)).toHaveLength(0);
  });

  it('moves a proposal to dismissed but keeps the record', () => {
    const { proposals, accountId, makeInput } = setup();
    const p = proposals.create(makeInput({ canonicalKey: 'y' }));
    proposals.markDismissed(p.id);
    expect(proposals.findById(p.id)!.status).toBe('dismissed');
    expect(proposals.findById(p.id)!.dismissedAt).toBeGreaterThan(0);
    expect(proposals.listPending(accountId)).toHaveLength(0);
    expect(proposals.listForAccount(accountId)).toHaveLength(1);
  });
});
