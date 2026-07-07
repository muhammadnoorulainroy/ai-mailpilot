/**
 * Phase 2b tests: the deterministic candidate validation gate. Every rejection rule and the
 * high-risk allowance, plus order-stable batch dedup, ranking, deterministic confidence, and
 * class-based TF-IDF keyphrases. No LLM, no I/O.
 */
import { describe, it, expect } from 'vitest';
import { EMBEDDING_DIM } from '../src/db/schema.js';
import type { DiscoveredCluster } from '../src/services/discovery-clustering.js';
import {
  validateCandidate,
  validateBatch,
  rankAccepted,
  candidateConfidence,
  isHighRiskPurpose,
  clusterKeyphrases,
  MIN_COHESION,
  THIN_CLUSTER_SIZE,
  OVERLAP_CENTROID_COSINE,
  type NamedCandidate,
  type CandidateContext,
  type ActiveCategoryRef,
} from '../src/services/discovery-candidates.js';

function axis(dim: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[dim] = 1;
  return v;
}

/** Unit vector whose cosine to axis(0) is exactly `target`, by mixing axes 0 and 1. */
function unitAtCosineToAxis0(target: number): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  v[0] = target;
  v[1] = Math.sqrt(1 - target * target);
  return v;
}

function cluster(over: Partial<DiscoveredCluster> = {}): DiscoveredCluster {
  return {
    memberIds: ['m1', 'm2', 'm3', 'm4', 'm5', 'm6'],
    size: 6,
    centroid: axis(0),
    cohesion: 0.85,
    separation: 0.8,
    ...over,
  };
}

function candidate(over: Partial<NamedCandidate> = {}): NamedCandidate {
  return {
    clusterIndex: 0,
    action: 'new_category',
    label: 'Crypto Trading',
    description: 'Trading platform alerts and wallet activity.',
    suggestedKey: 'finance.crypto_trading',
    evidence: ['coinbase', 'wallet'],
    ...over,
  };
}

function ctx(over: Partial<CandidateContext> = {}): CandidateContext {
  return {
    cluster: cluster(),
    senderTokens: [],
    totalResidual: 100,
    activeCategories: [],
    existingSuggestedLabels: [],
    existingSuggestedKeys: [],
    otherCandidateLabels: [],
    ...over,
  };
}

const active = (label: string, over: Partial<ActiveCategoryRef> = {}): ActiveCategoryRef => ({
  label,
  description: null,
  prototypes: [],
  createdBy: 'auto',
  ...over,
});

describe('validateCandidate', () => {
  it('accepts a solid, distinct, cohesive new category', () => {
    const v = validateCandidate(candidate(), ctx());
    expect(v).toMatchObject({ accepted: true, reason: 'accepted' });
    expect(v.confidence).toBeGreaterThan(0);
  });

  it('rejects a leave_uncategorized action', () => {
    expect(validateCandidate(candidate({ action: 'leave_uncategorized' }), ctx()).reason).toBe(
      'model_left_uncategorized',
    );
  });

  it('rejects a vague label', () => {
    expect(validateCandidate(candidate({ label: 'Report Summary' }), ctx()).reason).toBe(
      'vague_label',
    );
  });

  it('rejects a sender-name-only label but keeps a purpose label from the same sender', () => {
    expect(
      validateCandidate(candidate({ label: 'Coinbase' }), ctx({ senderTokens: ['coinbase'] }))
        .reason,
    ).toBe('sender_name_only');
    // A real purpose label mentioning the same sender is fine.
    expect(
      validateCandidate(candidate({ label: 'Crypto Trading' }), ctx({ senderTokens: ['coinbase'] }))
        .accepted,
    ).toBe(true);
  });

  it('rejects a conflict with a user-created category, distinctly from an auto overlap', () => {
    expect(
      validateCandidate(
        candidate({ label: 'Crypto Trading' }),
        ctx({ activeCategories: [active('Crypto Trading', { createdBy: 'user' })] }),
      ).reason,
    ).toBe('conflicts_user_category');
  });

  it('rejects a label overlap with an active auto category', () => {
    expect(
      validateCandidate(
        candidate({ label: 'Crypto Trading' }),
        ctx({ activeCategories: [active('Crypto Trading')] }),
      ).reason,
    ).toBe('overlaps_active_label');
  });

  it('rejects a content overlap when the cluster centroid matches an active centroid', () => {
    expect(
      validateCandidate(
        candidate({ label: 'Totally Different Name' }),
        ctx({ activeCategories: [active('Something', { prototypes: [axis(0)] })] }),
      ).reason,
    ).toBe('overlaps_active_content');
  });

  it('rejects a candidate overlapping ANY effective prototype of an active category (Phase 4)', () => {
    // The category's aggregate is on axis 0, but it also carries a sub-prototype on axis 5. A candidate
    // cluster on axis 5 overlaps the SUB-prototype (not the aggregate) and must still be rejected.
    expect(
      validateCandidate(
        candidate({ label: 'Totally Different Name', description: 'unrelated to axis zero.' }),
        ctx({
          cluster: cluster({ centroid: axis(5) }),
          activeCategories: [active('Broad', { prototypes: [axis(0), axis(5)] })],
        }),
      ).reason,
    ).toBe('overlaps_active_content');
  });

  it('rejects a duplicate of another accepted candidate and of an existing proposal', () => {
    expect(
      validateCandidate(candidate(), ctx({ otherCandidateLabels: ['Crypto Trading'] })).reason,
    ).toBe('duplicate_of_suggestion');
    expect(
      validateCandidate(candidate(), ctx({ existingSuggestedLabels: ['Crypto Trading'] })).reason,
    ).toBe('duplicate_of_existing_proposal');
  });

  it('rejects a low-cohesion cluster and a thin cluster (non high-risk)', () => {
    expect(
      validateCandidate(candidate(), ctx({ cluster: cluster({ cohesion: 0.3 }) })).reason,
    ).toBe('low_cohesion');
    expect(
      validateCandidate(candidate(), ctx({ cluster: cluster({ size: 3, cohesion: 0.6 }) })).reason,
    ).toBe('thin_support');
  });

  it('lets a coherent high-risk purpose survive thin support (relaxed size, same cohesion bar)', () => {
    const taxCluster = cluster({ size: 3, cohesion: 0.6 });
    const tax = candidate({ label: 'Tax Documents', evidence: ['irs', 'w-2'] });
    expect(validateCandidate(tax, ctx({ cluster: taxCluster })).accepted).toBe(true);
    // But a high-risk label on an incoherent cluster is still rejected: cohesion is universal.
    expect(
      validateCandidate(tax, ctx({ cluster: cluster({ size: 3, cohesion: 0.5 }) })).reason,
    ).toBe('low_cohesion');
  });

  it('still rejects a high-risk label that duplicates an existing category', () => {
    expect(
      validateCandidate(
        candidate({ label: 'Tax Documents', evidence: ['irs'] }),
        ctx({ activeCategories: [active('Tax Documents')] }),
      ).reason,
    ).toBe('overlaps_active_label');
  });
});

describe('validateCandidate hardening (adversarial findings)', () => {
  // HIGH-1: a high-risk word appearing only in sampled evidence must not unlock the floors.
  it('does not let evidence text unlock the high-risk floor relaxation', () => {
    const junk = candidate({
      label: 'Weekly Deals',
      description: 'promotional roundup',
      evidence: ['Save on health & beauty', '20% off your prescription refill'],
    });
    expect(
      validateCandidate(junk, ctx({ cluster: cluster({ size: 3, cohesion: 0.4 }) })).reason,
    ).toBe('low_value_label');
  });

  // HIGH-2: a genuinely high-risk label still needs a minimum floor; it relaxes, never skips.
  it('rejects a high-risk cluster that is below even the relaxed floor', () => {
    expect(
      validateCandidate(
        candidate({ label: 'Tax Documents', evidence: [] }),
        ctx({ cluster: cluster({ size: 3, cohesion: 0.3 }) }),
      ).reason,
    ).toBe('low_cohesion');
    expect(
      validateCandidate(
        candidate({ label: 'Tax Documents', evidence: [] }),
        ctx({ cluster: cluster({ size: 1, cohesion: 0.9 }) }),
      ).reason,
    ).toBe('thin_support');
  });

  // HIGH-3: a small non-high-value cluster is rejected regardless of how cohesive it is.
  it('rejects a small non-high-value cluster even with high cohesion', () => {
    expect(
      validateCandidate(
        candidate({ label: 'Sourdough Baking' }),
        ctx({ cluster: cluster({ size: 4, cohesion: 0.72, separation: 0.6 }) }),
      ).reason,
    ).toBe('thin_support');
  });

  // HIGH-4: a coherent marketing/promotions bucket is not turned into a category.
  it('rejects an all-marketing label as low value', () => {
    expect(
      validateCandidate(
        candidate({ label: 'Deals and Discounts' }),
        ctx({ cluster: cluster({ size: 40, cohesion: 0.62 }) }),
      ).reason,
    ).toBe('low_value_label');
    expect(validateCandidate(candidate({ label: 'Weekly Newsletter' }), ctx()).reason).toBe(
      'low_value_label',
    );
    // A concrete anchor plus a marketing word is still a real purpose.
    expect(validateCandidate(candidate({ label: 'Grocery Deals' }), ctx()).accepted).toBe(true);
  });

  // HIGH-5: brand plus a generic decorator ('Coinbase Updates') is still sender-name-only.
  it('rejects a brand-plus-decorator label as sender-name-only', () => {
    for (const [label, brand] of [
      ['Coinbase Updates', 'coinbase'],
      ['LinkedIn Notifications', 'linkedin'],
      ['GitHub Notifications', 'github'],
      ['From LinkedIn', 'linkedin'],
    ] as const) {
      expect(validateCandidate(candidate({ label }), ctx({ senderTokens: [brand] })).reason).toBe(
        'sender_name_only',
      );
    }
    // A real purpose label from the same sender still passes.
    expect(
      validateCandidate(candidate({ label: 'Crypto Trading' }), ctx({ senderTokens: ['coinbase'] }))
        .accepted,
    ).toBe(true);
  });

  // HIGH-6: a semantic twin of an active category with no shared words is caught by centroid content.
  it('rejects a semantic twin whose centroid is within the overlap band but shares no label words', () => {
    const twinCentroid = unitAtCosineToAxis0(OVERLAP_CENTROID_COSINE + 0.02);
    expect(
      validateCandidate(
        candidate({ label: 'Billing' }),
        ctx({
          cluster: cluster({ centroid: axis(0) }),
          activeCategories: [active('Invoices', { prototypes: [twinCentroid] })],
        }),
      ).reason,
    ).toBe('overlaps_active_content');
    // Just below the band is a distinct region and is allowed through.
    const distantCentroid = unitAtCosineToAxis0(OVERLAP_CENTROID_COSINE - 0.05);
    expect(
      validateCandidate(
        candidate({ label: 'Billing' }),
        ctx({
          cluster: cluster({ centroid: axis(0) }),
          activeCategories: [active('Invoices', { prototypes: [distantCentroid] })],
        }),
      ).accepted,
    ).toBe(true);
  });

  // HIGH-8: common medical/legal/immigration wordings count as high-value and survive thin support.
  it('accepts genuine rare high-value clusters named in common wordings', () => {
    for (const label of [
      'Doctor Appointments',
      'Dental Records',
      'Attorney Correspondence',
      'Mortgage Documents',
    ]) {
      expect(
        validateCandidate(
          candidate({ label }),
          ctx({ cluster: cluster({ size: 4, cohesion: 0.6 }) }),
        ).accepted,
      ).toBe(true);
    }
  });

  // MED-1: a raw email address is a sender, not a purpose, and leaks PII.
  it('rejects a raw email-address label', () => {
    expect(validateCandidate(candidate({ label: 'billing@stripe.com' }), ctx()).reason).toBe(
      'sender_name_only',
    );
  });

  // MED-2: numeric-only and over-broad labels name no concrete purpose.
  it('rejects numeric-only and over-broad labels', () => {
    expect(validateCandidate(candidate({ label: '2024' }), ctx()).reason).toBe('vague_label');
    expect(validateCandidate(candidate({ label: 'Work' }), ctx()).reason).toBe('vague_label');
    expect(validateCandidate(candidate({ label: 'Personal' }), ctx()).reason).toBe('vague_label');
  });

  // D1 (final pass): a high-value numeric token must not be pre-empted by the numeric-only rule.
  it('exempts a high-value numeric label from the numeric-only rule', () => {
    expect(
      validateCandidate(
        candidate({ label: '1099' }),
        ctx({ cluster: cluster({ size: 2, cohesion: 0.7 }) }),
      ).accepted,
    ).toBe(true);
    // A plain year is still rejected.
    expect(validateCandidate(candidate({ label: '2024' }), ctx()).reason).toBe('vague_label');
  });

  // D1: a high-risk word in the free-text description must not relax the floors for a non-high-value cluster.
  it('does not let a high-risk word in the description relax the floors', () => {
    expect(
      validateCandidate(
        candidate({
          label: 'Basketball Court Bookings',
          description: 'Reservations and schedules for the local basketball court.',
        }),
        ctx({ cluster: cluster({ size: 3, cohesion: 0.6 }) }),
      ).reason,
    ).toBe('thin_support');
  });

  // D2: a high-value purpose label is not discarded just because a sender brand shares the token.
  it('keeps a high-value purpose label that coincides with a sender brand token', () => {
    const v = validateCandidate(
      candidate({ label: 'Visa', description: 'Immigration and visa application updates.' }),
      ctx({ senderTokens: ['visa'] }),
    );
    expect(v.accepted).toBe(true);
    // A non-high-value brand-only label is still rejected.
    expect(
      validateCandidate(candidate({ label: 'Coinbase' }), ctx({ senderTokens: ['coinbase'] }))
        .reason,
    ).toBe('sender_name_only');
  });

  // D3: 'product' is a neutral domain word, not promotional, so it is not a marketing catch-all.
  it('does not reject a neutral product label as marketing', () => {
    expect(validateCandidate(candidate({ label: 'Product Recalls' }), ctx()).accepted).toBe(true);
  });

  // MED-3: a mismatched-dimension active centroid must not produce a NaN cosine that silently passes.
  it('skips content overlap against a mismatched-dimension centroid instead of a NaN compare', () => {
    const legacy = new Float32Array(768);
    legacy[0] = 1;
    const v = validateCandidate(
      candidate({ label: 'Totally Distinct Purpose' }),
      ctx({
        cluster: cluster({ centroid: axis(0) }),
        activeCategories: [active('Legacy', { prototypes: [legacy] })],
      }),
    );
    expect(v.accepted).toBe(true);
    expect(v.reason).toBe('accepted');
  });
});

describe('validateBatch and ranking', () => {
  it('dedupes duplicates order-stably: the first wins, the second is rejected', () => {
    const a = candidate({ clusterIndex: 0, label: 'Crypto Trading' });
    const b = candidate({ clusterIndex: 1, label: 'Crypto Trading' });
    const results = validateBatch([a, b], () => ctx());
    expect(results[0]!.verdict.accepted).toBe(true);
    expect(results[1]!.verdict).toMatchObject({
      accepted: false,
      reason: 'duplicate_of_suggestion',
    });
  });

  // HIGH-7: two candidates with different labels but the same suggestedKey collide downstream.
  it('rejects a second candidate that reuses an accepted suggestedKey despite a distinct label', () => {
    const a = candidate({ clusterIndex: 0, label: 'Tax Documents', suggestedKey: 'finance.tax' });
    const b = candidate({ clusterIndex: 1, label: 'IRS Filings', suggestedKey: 'finance.tax' });
    const results = validateBatch([a, b], () => ctx());
    expect(results[0]!.verdict.accepted).toBe(true);
    expect(results[1]!.verdict).toMatchObject({
      accepted: false,
      reason: 'duplicate_of_suggestion',
    });
  });

  // D2 (final pass): a key collision against an EXISTING proposal is caught even when labels differ.
  it('rejects a candidate whose key collides with an existing proposal despite a distinct label', () => {
    const c = candidate({ label: 'Fiscal Filings', suggestedKey: 'tax' });
    const results = validateBatch([c], () => ctx({ existingSuggestedKeys: ['tax'] }));
    expect(results[0]!.verdict).toMatchObject({
      accepted: false,
      reason: 'duplicate_of_existing_proposal',
    });
  });

  it('ranks accepted candidates by deterministic confidence', () => {
    const strong = candidate({ clusterIndex: 0, label: 'Strong One', suggestedKey: 'a.strong' });
    const weak = candidate({ clusterIndex: 1, label: 'Weaker Two', suggestedKey: 'b.weak' });
    const results = validateBatch([weak, strong], (c) =>
      ctx({
        cluster:
          c.clusterIndex === 0
            ? cluster({ cohesion: 0.95, separation: 0.9 })
            : cluster({ cohesion: 0.6, separation: 0.6 }),
      }),
    );
    expect(rankAccepted(results).map((c) => c.label)).toEqual(['Strong One', 'Weaker Two']);
  });
});

describe('confidence, high-risk detection, keyphrases', () => {
  it('confidence rises with cohesion, separation, and coverage, never using a model number', () => {
    const hi = candidateConfidence(
      ctx({ cluster: cluster({ cohesion: 0.95, separation: 0.95, size: 40 }) }),
    );
    const lo = candidateConfidence(
      ctx({ cluster: cluster({ cohesion: 0.5, separation: 0.4, size: 3 }) }),
    );
    expect(hi).toBeGreaterThan(lo);
    expect(hi).toBeLessThanOrEqual(1);
  });

  it('detects high-risk purposes from the label only, never description or sampled evidence', () => {
    expect(isHighRiskPurpose(candidate({ label: 'Medical Bills', evidence: [] }))).toBe(true);
    expect(
      isHighRiskPurpose(candidate({ label: 'Newsletters', description: '', evidence: [] })),
    ).toBe(false);
    // A high-risk word in the free-text description must NOT unlock the floors (D1).
    expect(
      isHighRiskPurpose(
        candidate({ label: 'Records', description: 'insurance claims', evidence: [] }),
      ),
    ).toBe(false);
    // A high-risk word in sampled evidence must NOT unlock the floors either.
    expect(
      isHighRiskPurpose(
        candidate({
          label: 'Weekly Deals',
          description: 'promos',
          evidence: ['prescription refill 20% off'],
        }),
      ),
    ).toBe(false);
  });

  it('extracts distinctive keyphrases and downweights terms shared across clusters', () => {
    const phrases = clusterKeyphrases([
      ['Coinbase wallet alert', 'Coinbase trade filled', 'wallet security'],
      ['Amazon order shipped', 'Amazon delivery update', 'order refund'],
    ]);
    expect(phrases[0]).toContain('coinbase');
    expect(phrases[1]).toContain('amazon');
    // A term present in only one cluster ranks; nothing shared dominates both.
    expect(phrases[0]).not.toContain('amazon');
  });

  it('orders a single cluster by frequency, not alphabetically (smoothed IDF)', () => {
    const [phrases] = clusterKeyphrases([
      ['wallet balance', 'wallet transfer', 'wallet alert', 'coinbase trade'],
    ]);
    // 'wallet' (freq 3) must outrank the alphabetically-earlier 'coinbase' (freq 1).
    expect(phrases[0]).toBe('wallet');
  });

  it('tokenizes accented French subjects as whole words', () => {
    const [phrases] = clusterKeyphrases([
      ['Facture électricité échéance', 'Facture électricité mensuelle', 'impôts échéance'],
    ]);
    expect(phrases).toContain('electricite');
    expect(phrases).toContain('facture');
    expect(phrases).toContain('impots');
    // No mangled fragments from splitting on accented letters.
    expect(phrases).not.toContain('lectricit');
    expect(phrases).not.toContain('ance');
  });

  it('exposes tunable thresholds as named constants', () => {
    expect(MIN_COHESION).toBeGreaterThan(0);
    expect(THIN_CLUSTER_SIZE).toBeGreaterThanOrEqual(3);
  });
});
