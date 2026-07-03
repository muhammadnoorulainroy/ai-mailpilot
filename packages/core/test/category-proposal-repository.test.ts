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

  it('defaults a new_category proposal to kind new_category with no source and no suppression key', () => {
    const { proposals, makeInput } = setup();
    const created = proposals.create(makeInput());
    expect(created.kind).toBe('new_category');
    expect(created.sourceCategoryId).toBeNull();
    expect(created.suppressionKey).toBe('');
    const loaded = proposals.findById(created.id)!;
    expect(loaded.kind).toBe('new_category');
    expect(loaded.sourceCategoryId).toBeNull();
    expect(loaded.suppressionKey).toBe('');
  });

  it('stores a merge proposal on the surviving target so a source delete does not cascade it away', () => {
    const { proposals, categories, accountId } = setup();
    const target = categories.create({
      accountId,
      label: 'Target',
      source: 'auto',
      canonicalKey: 'target',
    });
    const source = categories.create({
      accountId,
      label: 'Source',
      source: 'auto',
      canonicalKey: 'source',
    });
    const p = proposals.createStructural({
      accountId,
      kind: 'merge',
      categoryId: target.id,
      sourceCategoryId: source.id,
      runId: 'run-1',
      label: 'Merge Source into Target',
      description: 'd',
      canonicalKey: 'target',
      suppressionKey: 'merge:source|target',
      embeddingModelId: 'bge-m3',
      confidence: 0.6,
      evidence: [],
    });
    expect(p.kind).toBe('merge');
    expect(p.categoryId).toBe(target.id);
    expect(p.sourceCategoryId).toBe(source.id);
    expect(p.centroid.length).toBe(0); // structural parent has placeholder cluster fields
    expect(p.memberIds).toEqual([]);

    // Deleting the source must NOT cascade the proposal away (its category_id points at the target).
    expect(categories.delete(source.id)).toBe(true);
    const loaded = proposals.findById(p.id);
    expect(loaded).not.toBeNull();
    expect(loaded!.categoryId).toBe(target.id);
    expect(loaded!.sourceCategoryId).toBe(source.id); // non-cascading column keeps the (now dangling) id
  });

  it('resolves structural suppression keys only for applied or dismissed structural proposals', () => {
    const { proposals, categories, accountId, makeInput } = setup();
    const structural = (label: string, key: string, kind: 'split' | 'merge' | 'retire') => {
      const cat = categories.create({ accountId, label, source: 'auto', canonicalKey: key });
      return proposals.createStructural({
        accountId,
        kind,
        categoryId: cat.id,
        sourceCategoryId: null,
        runId: 'run-1',
        label,
        description: 'd',
        canonicalKey: key,
        suppressionKey: `${kind}:${key}`,
        embeddingModelId: 'bge-m3',
        confidence: 0.5,
        evidence: [],
      });
    };
    structural('Pending', 'kp', 'retire'); // still pending -> excluded
    proposals.markApplied(structural('Applied', 'ka', 'split').id);
    proposals.markDismissed(structural('Dismissed', 'kd', 'merge').id);
    // A resolved new_category proposal must not leak into the structural set.
    proposals.markApplied(proposals.create(makeInput({ canonicalKey: 'nc' })).id);

    const resolved = proposals.resolvedStructuralSuppressionKeys(accountId);
    expect([...resolved].sort()).toEqual(['merge:kd', 'split:ka']);
    expect(resolved.has('retire:kp')).toBe(false);
  });

  it('round-trips split children and cascades them when the proposal is deleted', () => {
    const { proposals, categories, accountId, db } = setup();
    const source = categories.create({
      accountId,
      label: 'BigCat',
      source: 'auto',
      canonicalKey: 'big',
    });
    const p = proposals.createStructural({
      accountId,
      kind: 'split',
      categoryId: source.id,
      sourceCategoryId: null,
      runId: 'run-1',
      label: 'Split BigCat',
      description: 'd',
      canonicalKey: 'big',
      suppressionKey: 'split:big',
      embeddingModelId: 'bge-m3',
      confidence: 0.6,
      evidence: [],
    });
    const c1 = proposals.createChild({
      proposalId: p.id,
      label: 'Child A',
      description: 'da',
      canonicalKey: 'child_a',
      embeddingModelId: 'bge-m3',
      centroid: centroid(2),
      memberIds: ['m1', 'm2'],
      proposedCount: 2,
      cohesion: 0.9,
      separation: 0.8,
      confidence: 0.7,
    });
    proposals.createChild({
      proposalId: p.id,
      label: 'Child B',
      description: 'db',
      canonicalKey: 'child_b',
      embeddingModelId: 'bge-m3',
      centroid: centroid(3),
      memberIds: ['m3'],
      proposedCount: 1,
      cohesion: 0.85,
      separation: 0.75,
      confidence: 0.65,
    });

    const children = proposals.listChildren(p.id);
    expect(children.map((c) => c.label).sort()).toEqual(['Child A', 'Child B']);
    const loaded = children.find((c) => c.id === c1.id)!;
    expect(loaded.memberIds).toEqual(['m1', 'm2']);
    expect(Array.from(loaded.centroid.slice(0, 2))).toEqual([2, 1]);
    expect(loaded.proposedCount).toBe(2);

    // Deleting the proposal cascades its children away.
    db.prepare('DELETE FROM category_proposals WHERE id = ?').run(p.id);
    expect(proposals.listChildren(p.id)).toEqual([]);
  });

  it('scopes structural suppression keys and proposals to the account', () => {
    const { proposals, categories, accounts, accountId } = setup();
    const other = accounts.create({ address: 'o@x.com', kind: 'work' });
    const retire = (acc: string, label: string, key: string) => {
      const cat = categories.create({ accountId: acc, label, source: 'auto', canonicalKey: key });
      const p = proposals.createStructural({
        accountId: acc,
        kind: 'retire',
        categoryId: cat.id,
        sourceCategoryId: null,
        runId: 'r',
        label,
        description: 'd',
        canonicalKey: key,
        suppressionKey: `retire:${key}`,
        embeddingModelId: 'bge-m3',
        confidence: 0.5,
        evidence: [],
      });
      proposals.markDismissed(p.id);
      return p;
    };
    const pA = retire(accountId, 'A', 'a');
    const pB = retire(other.id, 'B', 'b');

    expect([...proposals.resolvedStructuralSuppressionKeys(accountId)]).toEqual(['retire:a']);
    expect([...proposals.resolvedStructuralSuppressionKeys(other.id)]).toEqual(['retire:b']);
    expect(proposals.listForAccount(accountId).map((p) => p.id)).toEqual([pA.id]);
    expect(proposals.listForAccount(other.id).map((p) => p.id)).toEqual([pB.id]);
  });
});
