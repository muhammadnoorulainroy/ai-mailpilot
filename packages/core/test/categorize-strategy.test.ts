/**
 * Tests for the embedding plus LLM categorization strategy: the fast-pass gate,
 * adjudication of LLM picks against ranked centroids and text evidence, lexical
 * evidence matching, deterministic fallback, label collapsing, shortlisting, and
 * per-email batch label resolution.
 */
import { describe, it, expect } from 'vitest';
import {
  shortlistFor,
  gateDecision,
  gateFastAssignment,
  collapseLabels,
  adjudicate,
  hasTextEvidence,
  deterministicFallback,
} from '../src/services/categorize-strategy.js';
import { resolveBatchLabels } from '../src/services/llm-categorizer.js';
import type { CategoryMatch } from '../src/services/categorization-service.js';

/** Builds a CategoryMatch with distance derived from the given confidence. */
const m = (id: string, confidence: number): CategoryMatch => ({
  categoryId: id,
  label: id,
  distance: 1 - confidence,
  confidence,
});
/** Builds a ranked list of matches with generated ids from the given confidences. */
const ranked = (...confs: number[]): CategoryMatch[] => confs.map((c, i) => m(`c${i}`, c));

describe('gateFastAssignment (fast pass assigns only a clear single winner)', () => {
  it('assigns a confident winner with a clear gap to second', () => {
    expect(gateFastAssignment(ranked(0.85, 0.6))?.categoryId).toBe('c0');
  });

  it('assigns a lone confident category', () => {
    expect(gateFastAssignment(ranked(0.82))?.categoryId).toBe('c0');
  });

  it('defers a weak top to the LLM', () => {
    expect(gateFastAssignment(ranked(0.559, 0.55))).toBeNull();
  });

  it('defers a bunched top where second is within the margin', () => {
    expect(gateFastAssignment(ranked(0.8, 0.75))).toBeNull();
  });

  it('defers an empty ranking', () => {
    expect(gateFastAssignment([])).toBeNull();
  });
});

describe('adjudicate (rank/margin + text evidence)', () => {
  /** Builds a category entry with a label and no description. */
  const cat = (label: string) => ({ label, description: null });
  /** Builds an evidence object pairing email text with a label lookup by id. */
  const evidence = (text: string, ...labels: Array<[string, string]>) => ({
    text,
    categoryById: new Map(labels.map(([id, label]) => [id, cat(label)])),
  });

  it('overrides a wrong lower pick when the top is strong and clearly ahead (travel deal -> not jobs)', () => {
    const out = adjudicate(['jobs'], [m('travel', 0.82), m('jobs', 0.52)]);
    expect(out.reason).toBe('overridden_strong_embedding');
    expect(out.ids).toEqual(['travel']);
  });

  it('rejects a weak unrelated pick well below a medium top (food promo -> not banking)', () => {
    const out = adjudicate(['banking'], [m('shipping', 0.65), m('banking', 0.37)]);
    expect(out.reason).toBe('rejected_low_rank');
    expect(out.ids).toEqual([]);
  });

  it('leaves an email uncategorized when the whole ranking is weak (promo -> not jobs)', () => {
    const out = adjudicate(['jobs'], [m('news', 0.42), m('jobs', 0.35)]);
    expect(out.reason).toBe('rejected_weak_all_around');
    expect(out.ids).toEqual([]);
  });

  it('rescues a low-embedding pick when the content lexically supports it (French job)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('marketing', 0.4), m('jobs', 0.29)],
      evidence('Nouvelle offre: Développeur. Postulez à cette candidature emploi.', [
        'jobs',
        'Job Opportunities',
      ]),
    );
    expect(out.reason).toBe('accepted_text_evidence');
    expect(out.ids).toEqual(['jobs']);
  });

  it('still rescues when a sibling same-purpose category exists (issue #527 regression)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('marketing', 0.4), m('jobs', 0.29), m('recruiting', 0.27)],
      evidence(
        'Nouvelle offre: Développeur. Postulez à cette candidature emploi.',
        ['jobs', 'Job Opportunities'],
        ['recruiting', 'Recruiting and Careers'],
      ),
    );
    expect(out.reason).toBe('accepted_text_evidence');
    expect(out.ids).toEqual(['jobs']);
  });

  it('drops a same-purpose second label (Shipping + Delivery -> one, issue #552)', () => {
    const out = adjudicate(
      ['delivery', 'shipping'],
      [m('delivery', 0.7), m('shipping', 0.62)],
      evidence(
        'Your parcel shipment is out for delivery. Tracking: package dispatched by courier.',
        ['delivery', 'Shipping and Deliveries'],
        ['shipping', 'Delivery Tracking'],
      ),
    );
    expect(out.ids).toEqual(['delivery']);
  });

  it('keeps a valid second label when overriding to a strong embedding top (issue #477)', () => {
    const out = adjudicate(
      ['jobs', 'receipts'],
      [m('travel', 0.85), m('receipts', 0.65), m('jobs', 0.4)],
      evidence(
        'Your hotel booking is confirmed. Invoice and receipt for payment attached.',
        ['jobs', 'Job Opportunities'],
        ['receipts', 'Receipts and Invoices'],
        ['travel', 'Travel and Accommodation'],
      ),
    );
    expect(['overridden_strong_embedding', 'overridden_by_text']).toContain(out.reason);
    expect(out.ids).toEqual(['travel', 'receipts']);
  });

  it('overrides to a uniquely text-supported candidate the LLM did NOT pick (issue #2)', () => {
    const out = adjudicate(
      ['banking'],
      [m('shipping', 0.5), m('banking', 0.46), m('jobs', 0.3)],
      evidence(
        'We are hiring a Software Engineer. Apply now for this developer career position.',
        ['banking', 'Banking Transactions'],
        ['jobs', 'Job Opportunities'],
      ),
    );
    expect(out.reason).toBe('overridden_by_text');
    expect(out.ids).toEqual(['jobs']);
  });

  it('does not override the primary when it IS the embedding top, even if a secondary has more text', () => {
    const out = adjudicate(
      ['receipts', 'course'],
      [m('receipts', 0.7), m('course', 0.45)],
      evidence(
        'Invoice for your course: lecture materials and assignment access. Payment received.',
        ['receipts', 'Receipts and Invoices'],
        ['course', 'Course Materials'],
      ),
    );
    expect(out.ids).toEqual(['receipts', 'course']);
  });

  it('keeps only the primary when a second label is weak with no text support (YouTube case, issue #3)', () => {
    const out = adjudicate(
      ['social', 'course'],
      [m('social', 0.7), m('course', 0.5)],
      evidence(
        'Ammi Jan posted a new video. Watch and comment now.',
        ['social', 'Social Media Updates'],
        ['course', 'Course Materials'],
      ),
    );
    expect(out.ids).toEqual(['social']);
  });

  it('drops a low-evidence secondary but keeps the primary', () => {
    const out = adjudicate(
      ['a', 'c'],
      [m('a', 0.7), m('c', 0.5)],
      evidence('plain content', ['a', 'Alpha'], ['c', 'Charlie']),
    );
    expect(out.ids).toEqual(['a']);
  });

  it('does NOT rescue on a coincidental keyword when a strong isolated top dominates', () => {
    const out = adjudicate(
      ['jobs'],
      [m('travel', 0.82), m('jobs', 0.4)],
      evidence(
        'Great position by the beach. Apply your booking now.',
        ['jobs', 'Job Opportunities'],
        ['travel', 'Travel and Accommodation'],
      ),
    );
    expect(out.reason).toBe('overridden_strong_embedding');
    expect(out.ids).toEqual(['travel']);
  });

  it('does NOT rescue on a single coincidental keyword (needs >= 2 distinct purpose words)', () => {
    const out = adjudicate(
      ['banking'],
      [m('shipping', 0.6), m('banking', 0.47)],
      evidence('Your payment was received.', ['banking', 'Banking Transactions']),
    );
    expect(out.reason).toBe('rejected_low_rank');
    expect(out.ids).toEqual([]);
  });

  it('still rejects a weak-all-around email even with one matching keyword', () => {
    const out = adjudicate(
      ['banking'],
      [m('news', 0.42), m('banking', 0.35)],
      evidence('Your payment is due.', ['banking', 'Banking Transactions']),
    );
    expect(out.reason).toBe('rejected_weak_all_around');
    expect(out.ids).toEqual([]);
  });

  it('accepts the LLM when it agrees with a confident top', () => {
    const out = adjudicate(['a'], [m('a', 0.7), m('b', 0.4)]);
    expect(out.reason).toBe('accepted_close_to_top');
    expect(out.ids).toEqual(['a']);
  });

  it('accepts a close pick of the SAME purpose as the top', () => {
    const out = adjudicate(
      ['b'],
      [m('a', 0.62), m('b', 0.55)],
      evidence('weekly inbox digest', ['a', 'Marketing Promotions'], ['b', 'Marketing Offers']),
    );
    expect(out.reason).toBe('accepted_close_to_top');
    expect(out.ids).toEqual(['b']);
  });

  it('does NOT accept a cross-purpose close pick with no text support (issue 1: promo -> not networking)', () => {
    const out = adjudicate(
      ['networking'],
      [m('marketing', 0.632), m('travel', 0.627), m('networking', 0.588)],
      evidence(
        'Enjoy 30 percent off your rides this weekend',
        ['marketing', 'Marketing Promotions'],
        ['travel', 'Travel and Accommodation'],
        ['networking', 'Professional Networking'],
      ),
    );
    expect(out.ids).not.toContain('networking');
  });

  it('keeps a content-supported LLM pick over a sender-biased strong top (GitHub PR, regression #504)', () => {
    const out = adjudicate(
      ['developer'],
      [m('marketing', 0.78), m('developer', 0.5)],
      evidence(
        'Your build is ready. Merge when green.',
        ['marketing', 'Marketing Promotions'],
        ['developer', 'Developer Code Reviews'],
      ),
    );
    expect(out.ids).toEqual(['developer']);
  });

  it('uncategorizes when a THIRD strong candidate is a different purpose (issue #505)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('mktgA', 0.95), m('mktgB', 0.93), m('travel', 0.9), m('jobs', 0.5)],
      evidence(
        'some neutral announcement',
        ['mktgA', 'Marketing Promotions'],
        ['mktgB', 'Marketing Offers'],
        ['travel', 'Travel and Accommodation'],
        ['jobs', 'Job Opportunities'],
      ),
    );
    expect(out.ids).toEqual([]);
  });

  it('recovers a very strong top even for categories with no canonical purpose words (issue #508)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('family', 0.85), m('friends', 0.82), m('jobs', 0.5)],
      evidence('hi there', ['family', 'Family'], ['friends', 'Friends'], [
        'jobs',
        'Job Opportunities',
      ]),
    );
    expect(out.ids).toEqual(['family']);
  });

  it('accepts a close pick when purposes are unknown (no false cross-purpose rejection, issue #518)', () => {
    const out = adjudicate(
      ['updates'],
      [m('feed', 0.62), m('updates', 0.55)],
      evidence('hello there, please take a look when you can', ['feed', 'Activity Feed'], [
        'updates',
        'Updates Feed',
      ]),
    );
    expect(out.ids).toEqual(['updates']);
  });

  it('does not keep a bad low-rank pick below two strong tops (issue 2: Agoda)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('travel', 0.983), m('marketing', 0.964), m('jobs', 0.59)],
      evidence(
        'Special weekend promo just for you',
        ['travel', 'Travel and Accommodation'],
        ['marketing', 'Marketing Promotions'],
        ['jobs', 'Job Opportunities'],
      ),
    );
    expect(out.ids).not.toContain('jobs');
  });

  it('does not file a connection request as Job Opportunities without job evidence (regression #4)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('networking', 0.7), m('jobs', 0.45)],
      evidence(
        'You have a new connection request and a profile view from a colleague',
        ['networking', 'Professional Networking'],
        ['jobs', 'Job Opportunities'],
      ),
    );
    expect(out.ids).not.toContain('jobs');
  });

  it('files a job posting as Job Opportunities via generic text evidence (regression #5)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('networking', 0.6), m('jobs', 0.45)],
      evidence(
        'Actively recruiting: apply now for this position. Hiring a candidate for a career opening.',
        ['networking', 'Professional Networking'],
        ['jobs', 'Job Opportunities'],
      ),
    );
    expect(out.ids).toContain('jobs');
  });

  it('recovers the top when two strong tops share a purpose and the LLM pick is far below (issue 2)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('mktgA', 0.95), m('mktgB', 0.93), m('jobs', 0.5)],
      evidence(
        'some announcement',
        ['mktgA', 'Marketing Promotions'],
        ['mktgB', 'Marketing Offers'],
        ['jobs', 'Job Opportunities'],
      ),
    );
    expect(out.reason).toBe('overridden_strong_embedding');
    expect(out.ids).toEqual(['mktgA']);
  });

  it('accepts a supported rank-1 pick between two strong tops (Agoda deal -> Marketing, issue 1)', () => {
    const out = adjudicate(
      ['marketing'],
      [m('travel', 0.983), m('marketing', 0.964)],
      evidence(
        'travel deal: special offer on your booking',
        ['travel', 'Travel and Accommodation'],
        ['marketing', 'Marketing Promotions'],
      ),
    );
    expect(out.ids).toEqual(['marketing']);
  });

  it('still rejects a low-rank pick below two strong tops (Agoda -> not Job, issue 1)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('travel', 0.983), m('marketing', 0.964), m('jobs', 0.5)],
      evidence(
        'travel deal: special offer on your booking',
        ['travel', 'Travel and Accommodation'],
        ['marketing', 'Marketing Promotions'],
        ['jobs', 'Job Opportunities'],
      ),
    );
    expect(out.ids).toEqual([]);
  });

  it('does not blindly accept a rank-1 pick with no text support (issue 1)', () => {
    const out = adjudicate(
      ['b'],
      [m('a', 0.98), m('b', 0.96)],
      evidence('special offer limited time deal', ['a', 'Marketing Promotions'], [
        'b',
        'Travel and Accommodation',
      ]),
    );
    expect(out.ids).toEqual(['a']);
  });

  it('assigns a short subject-only email by its exact label word ([cps2-m1] Grades, issue 2)', () => {
    const out = adjudicate(
      ['grades'],
      [m('materials', 0.687), m('grades', 0.653)],
      evidence('[cps2-m1] Grades', ['materials', 'Course Materials'], ['grades', 'Course Grades']),
    );
    expect(out.ids).toEqual(['grades']);
  });

  it('accepts a short Invoice subject for Receipts and Invoices (singular/plural, issue 2)', () => {
    const out = adjudicate(
      ['receipts'],
      [m('billing', 0.66), m('receipts', 0.64)],
      evidence('Invoice', ['billing', 'Billing Center'], ['receipts', 'Receipts and Invoices']),
    );
    expect(out.ids).toEqual(['receipts']);
  });

  it('does not force a category from a generic label word alone (issue 2)', () => {
    const out = adjudicate(
      ['updates'],
      [m('travel', 0.9), m('updates', 0.75)],
      evidence('Update', ['travel', 'Travel and Accommodation'], ['updates', 'Account Updates']),
    );
    expect(out.ids).toEqual([]);
  });

  it('does not let a low-rank exact-label match win when embeddings strongly disagree (issue 2)', () => {
    const out = adjudicate(
      ['grades'],
      [m('marketing', 0.9), m('a', 0.6), m('b', 0.55), m('grades', 0.3)],
      evidence('Grades', ['marketing', 'Marketing Promotions'], ['a', 'Alpha'], ['b', 'Beta'], [
        'grades',
        'Course Grades',
      ]),
    );
    expect(out.ids).not.toContain('grades');
  });

  it('rescues a weak-embedding pick on a generic purpose PHRASE (LinkedIn -> Networking, issue 3)', () => {
    const out = adjudicate(
      ['network'],
      [m('network', 0.407), m('jobs', 0.36)],
      evidence("I'd like to join your professional network", ['network', 'Professional Networking'], [
        'jobs',
        'Job Opportunities',
      ]),
    );
    expect(out.ids).toEqual(['network']);
  });

  it('leaves a bare connection request uncategorized without phrase/word evidence (issue 3)', () => {
    const out = adjudicate(
      ['network'],
      [m('network', 0.4), m('jobs', 0.36)],
      evidence('I want to connect', ['network', 'Professional Networking'], [
        'jobs',
        'Job Opportunities',
      ]),
    );
    expect(out.ids).toEqual([]);
  });

  it('prefers the stronger job phrase over networking inside an invitation (issue 3)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('network', 0.42), m('jobs', 0.4)],
      evidence(
        "Let's connect. We are hiring and this is a great job opportunity for you.",
        ['network', 'Professional Networking'],
        ['jobs', 'Job Opportunities'],
      ),
    );
    expect(out.ids).toEqual(['jobs']);
  });

  it('does not override a weak correct pick to Marketing on a lone "off your" phrase (review #1)', () => {
    const out = adjudicate(
      ['travel'],
      [m('travel', 0.42), m('marketing', 0.4)],
      evidence('Time to head off your long-awaited journey. Your trip starts soon.', [
        'travel',
        'Travel and Accommodation',
      ], ['marketing', 'Marketing Promotions']),
    );
    expect(out.ids).not.toContain('marketing');
  });

  it('files a banking transfer correctly despite a coincidental "off your" (review #6)', () => {
    const out = adjudicate(
      ['marketing'],
      [m('bank', 0.5), m('marketing', 0.4)],
      evidence('Pay off your loan balance with this transfer', ['bank', 'Banking Transactions'], [
        'marketing',
        'Marketing Promotions',
      ]),
    );
    expect(out.ids).toEqual(['bank']);
  });

  it('does not match the "fit for" phrase inside "benefit form" (word boundary, review #5)', () => {
    const out = adjudicate(
      ['jobs'],
      [m('health', 0.5), m('jobs', 0.4)],
      evidence('Download your benefit form for the new health plan', ['health', 'Health Insurance'], [
        'jobs',
        'Job Opportunities',
      ]),
    );
    expect(out.ids).not.toContain('jobs');
  });

  it('does not file a sale newsletter as Social on a lone "shared a" phrase (review #9)', () => {
    const out = adjudicate(
      ['social'],
      [m('mkt', 0.5), m('social', 0.4)],
      evidence('We shared a few updates about our spring sale newsletter', [
        'mkt',
        'Marketing Promotions',
      ], ['social', 'Social Media']),
    );
    expect(out.ids).not.toContain('social');
  });

  it('does not file an invoice as Course Materials on a lone "due date" phrase (review #8)', () => {
    const out = adjudicate(
      ['receipts'],
      [m('course', 0.6), m('receipts', 0.4)],
      evidence(
        'Invoice 8842. Amount owed shown. Due date listed below. Thank you for your purchase.',
        ['course', 'Course Materials'],
        ['receipts', 'Receipts and Invoices'],
      ),
    );
    expect(out.ids).not.toContain('course');
  });

  it('does not let an email word "new" support a "...News" label via stemming (review #3)', () => {
    const out = adjudicate(
      ['healthnews'],
      [m('mkt', 0.66), m('healthnews', 0.6)],
      evidence('Brand new! Big sale today', ['mkt', 'Marketing Promotions'], [
        'healthnews',
        'Health News',
      ]),
    );
    expect(out.ids).not.toContain('healthnews');
  });

  it('does not override a clearly stronger top with a bare label-word echo (review #4)', () => {
    const out = adjudicate(
      ['a'],
      [m('travel', 0.9), m('a', 0.72)],
      evidence('Alpha', ['travel', 'Travel and Accommodation'], ['a', 'Alpha']),
    );
    expect(out.ids).not.toContain('a');
  });

  it('keeps a genuinely content-supported second label (review #7)', () => {
    const out = adjudicate(
      ['travel', 'marketing'],
      [m('travel', 0.8), m('marketing', 0.61)],
      evidence(
        'Booking confirmation for your hotel stay. Plus an exclusive discount: limited time offer, save up to 30.',
        ['travel', 'Travel and Accommodation'],
        ['marketing', 'Marketing Promotions'],
      ),
    );
    expect(out.ids).toEqual(['travel', 'marketing']);
  });

  it('drops a same-purpose second whose shared purpose is only in its description (review #12)', () => {
    const ev2 = {
      text: 'Your parcel shipment is out for delivery. Tracking shipped dispatched courier package.',
      categoryById: new Map([
        ['delivery', { label: 'Shipping and Deliveries', description: null }],
        [
          'tracking',
          { label: 'Order Center', description: 'shipment tracking parcel courier delivery shipped dispatch' },
        ],
      ]),
    };
    const out = adjudicate(['delivery', 'tracking'], [m('delivery', 0.7), m('tracking', 0.62)], ev2);
    expect(out.ids).toEqual(['delivery']);
  });

  it('does not override on TWO lone generic phrases with no purpose word (review2 #1)', () => {
    const out = adjudicate(
      ['social'],
      [m('travel', 0.7), m('social', 0.2)],
      evidence('we tagged you and shared a moment', ['travel', 'Travel and Accommodation'], [
        'social',
        'Social Media',
      ]),
    );
    expect(out.ids).not.toContain('social');
  });

  it('does not add a second label grounded only on a lone phrase (review2 #2)', () => {
    const out = adjudicate(
      ['bank', 'mkt'],
      [m('bank', 0.8), m('mkt', 0.62)],
      evidence('Transaction alert: your card payment was processed. Pay off your loan balance.', [
        'bank',
        'Banking Transactions',
      ], ['mkt', 'Marketing Promotions']),
    );
    expect(out.ids).toEqual(['bank']);
  });

  it('does not add a second label grounded only on a bare own-label echo (review2 #4)', () => {
    const out = adjudicate(
      ['travel', 'projectx'],
      [m('travel', 0.8), m('projectx', 0.62)],
      evidence('Your hotel booking is confirmed for the ProjectX retreat trip.', [
        'travel',
        'Travel and Accommodation',
      ], ['projectx', 'ProjectX']),
    );
    expect(out.ids).toEqual(['travel']);
  });

  it('does not add a second label from two contentless phrases on a weak embedding (review2 #7)', () => {
    const out = adjudicate(
      ['marketing', 'social'],
      [m('marketing', 0.8), m('social', 0.3)],
      evidence('Spring sale! Also we tagged you and shared a note.', ['marketing', 'Marketing Promotions'], [
        'social',
        'Social Media',
      ]),
    );
    expect(out.ids).toEqual(['marketing']);
  });

  it('does not let a multi-word own-label echo override via the text rescue (review3 #1)', () => {
    const out = adjudicate(
      ['projx'],
      [m('travel', 0.72), m('projx', 0.3)],
      evidence('Falcon Mercury Saturn', ['travel', 'Travel and Accommodation'], [
        'projx',
        'Falcon Mercury Saturn',
      ]),
    );
    expect(out.ids).not.toContain('projx');
  });

  it('does not add a multi-word own-label echo as a second label (review3 #2)', () => {
    const out = adjudicate(
      ['travel', 'projx'],
      [m('travel', 0.8), m('projx', 0.3)],
      evidence('Your hotel booking is confirmed for the trip. Falcon Mercury Saturn.', [
        'travel',
        'Travel and Accommodation',
      ], ['projx', 'Falcon Mercury Saturn']),
    );
    expect(out.ids).toEqual(['travel']);
  });

  it('keeps a content-supported pick when the sender-biased top has only a coincidental phrase (review3 #5)', () => {
    const out = adjudicate(
      ['developer'],
      [m('marketing', 0.78), m('developer', 0.5)],
      evidence('Your build is ready. Merge when green. Limited time only.', [
        'marketing',
        'Marketing Promotions',
      ], ['developer', 'Developer Code Reviews']),
    );
    expect(out.ids).toEqual(['developer']);
  });

  it('does not let a lone-phrase competitor block a content-supported fallback (review3 #6)', () => {
    const out = deterministicFallback(
      [m('banking', 0.8), m('social', 0.72)],
      evidence('Account statement balance. We tagged you and shared a note.', [
        'banking',
        'Banking Transactions',
      ], ['social', 'Social Media']),
    );
    expect(out).toBe('banking');
  });

  it('rejects an unsupported cross-purpose pick when the top purpose is in its description (review4)', () => {
    const ev2 = {
      text: 'Your parcel is here today.',
      categoryById: new Map([
        ['logi', { label: 'Logistics', description: 'package tracking delivery shipping courier parcel' }],
        ['promos', { label: 'Promotions', description: null }],
      ]),
    };
    const out = adjudicate(['promos'], [m('logi', 0.55), m('promos', 0.47)], ev2);
    expect(out.ids).not.toContain('promos');
  });

  it('uncategorizes when a competing strong purpose is carried by its description (review4)', () => {
    const ev2 = {
      text: 'some neutral announcement with no purpose words at all here',
      categoryById: new Map([
        ['bank', { label: 'Banking Transactions', description: null }],
        [
          'shipdesc',
          { label: 'MyLogistics', description: 'package tracking delivery shipping courier parcel shipment' },
        ],
        ['weak', { label: 'Newsletters', description: null }],
      ]),
    };
    const out = adjudicate(['weak'], [m('bank', 0.95), m('shipdesc', 0.92), m('weak', 0.5)], ev2);
    expect(out.ids).toEqual([]);
  });

  it('trusts the LLM for a new category with no centroid', () => {
    const out = adjudicate(['brand-new'], [m('a', 0.7)]);
    expect(out.reason).toBe('accepted_new_category');
    expect(out.ids).toEqual(['brand-new']);
  });

  it('drops an implausible second label but keeps the accepted primary', () => {
    const out = adjudicate(['a', 'c'], [m('a', 0.7), m('b', 0.5), m('c', 0.2)]);
    expect(out.reason).toBe('accepted_close_to_top');
    expect(out.ids).toEqual(['a']);
  });

  it('passes an empty LLM answer through as none', () => {
    expect(adjudicate([], [m('a', 0.9)])).toEqual({ ids: [], reason: 'none' });
  });

  it('trusts the LLM when there are no centroids to rank against', () => {
    expect(adjudicate(['x'], [])).toEqual({ ids: ['x'], reason: 'accepted_new_category' });
  });
});

describe('hasTextEvidence (generic lexical support, EN + FR)', () => {
  it('matches a category purpose by its label vocabulary across languages', () => {
    const job = { label: 'Job Opportunities', description: null };
    expect(hasTextEvidence('Your application for the developer position', job)).toBe(true);
    expect(hasTextEvidence('Nouvelle offre de poste, candidature', job)).toBe(true);
    expect(hasTextEvidence('Your hotel booking is confirmed', job)).toBe(false);
  });

  it('strips accents so accented French matches the vocabulary', () => {
    const grades = { label: 'Course Grades', description: null };
    expect(hasTextEvidence('Vos résultats et notes d examen', grades)).toBe(true);
  });
});

describe('deterministicFallback (graceful degradation on unusable LLM output, issue 3)', () => {
  /** Builds an evidence object pairing email text with a label lookup by id. */
  const ev = (text: string, ...labels: Array<[string, string]>) => ({
    text,
    categoryById: new Map(labels.map(([id, label]) => [id, { label, description: null }])),
  });

  it('assigns a strong top that dominates a STRONG competing purpose by evidence (Standard Chartered)', () => {
    const out = deterministicFallback(
      [m('banking', 0.823), m('receipts', 0.705)],
      ev(
        'Transaction alert: your card payment was processed. Account statement and balance available.',
        ['banking', 'Banking Transactions'],
        ['receipts', 'Receipts and Invoices'],
      ),
    );
    expect(out).toBe('banking');
  });

  it('returns null when the strong top has no text support (avoids sender-biased guess)', () => {
    const out = deterministicFallback(
      [m('banking', 0.85), m('receipts', 0.6)],
      ev('Special weekend promo just for you', ['banking', 'Banking Transactions']),
    );
    expect(out).toBeNull();
  });

  it('returns null when a competing strong purpose is as well-supported (ambiguous)', () => {
    const out = deterministicFallback(
      [m('travel', 0.95), m('marketing', 0.93)],
      ev(
        'flight hotel booking. exclusive deal offer discount sale today',
        ['travel', 'Travel and Accommodation'],
        ['marketing', 'Marketing Promotions'],
      ),
    );
    expect(out).toBeNull();
  });

  it('returns null for a weak top', () => {
    expect(deterministicFallback([m('a', 0.5)], ev('anything', ['a', 'Alpha']))).toBeNull();
  });
});

describe('collapseLabels (drop overlapping labels, cap at two)', () => {
  /** Builds a Float32Array centroid vector from the given numbers. */
  const v = (...nums: number[]) => Float32Array.from(nums);

  it('drops a later label whose centroid overlaps a kept one', () => {
    const centroids = new Map([
      ['a', v(1, 0)],
      ['b', v(0.99, 0.141)],
      ['c', v(0, 1)],
    ]);
    expect(collapseLabels(['a', 'b', 'c'], centroids)).toEqual(['a', 'c']);
  });

  it('caps at two distinct labels', () => {
    const centroids = new Map([
      ['a', v(1, 0, 0)],
      ['b', v(0, 1, 0)],
      ['c', v(0, 0, 1)],
    ]);
    expect(collapseLabels(['a', 'b', 'c'], centroids)).toEqual(['a', 'b']);
  });

  it('keeps labels without centroids as-is, still capped', () => {
    expect(collapseLabels(['x', 'y', 'z'], new Map())).toEqual(['x', 'y']);
  });

  it('passes a lone label through', () => {
    expect(collapseLabels(['a'], new Map([['a', v(1, 0)]]))).toEqual(['a']);
  });

  it('drops exact duplicate ids even when they have no centroid', () => {
    expect(collapseLabels(['x', 'x', 'y'], new Map())).toEqual(['x', 'y']);
  });
});

describe('shortlistFor (embeddings rank, LLM judges from the shortlist)', () => {
  it('sends all categories when there are not many (<= MAX)', () => {
    const r = ranked(0.9, 0.6, 0.5);
    expect(shortlistFor(r, 3)).toHaveLength(3);
    expect(shortlistFor(ranked(...Array.from({ length: 12 }, (_, i) => 0.9 - i * 0.05)), 12)).toHaveLength(12);
  });

  it('offers the generous top MAX (12) when there are many categories', () => {
    const r = ranked(0.92, ...Array.from({ length: 19 }, (_, i) => 0.5 - i * 0.01));
    const s = shortlistFor(r, 20);
    expect(s.length).toBe(12);
    expect(s[0]!.categoryId).toBe('c0');
  });

  it('falls back to all categories on severe centroid collapse (many bunched at the top)', () => {
    const r = ranked(...Array.from({ length: 20 }, () => 0.9));
    expect(shortlistFor(r, 20)).toHaveLength(20);
  });

  it('returns nothing for an empty ranking', () => {
    expect(shortlistFor([], 10)).toEqual([]);
  });
});

describe('gateDecision (skip the LLM only when clearly single-category)', () => {
  it('gates a confident match whose runner-up is clearly inapplicable', () => {
    expect(gateDecision(ranked(0.86, 0.4))?.categoryId).toBe('c0');
  });

  it('does NOT gate when a plausible second category exists (avoids losing it)', () => {
    expect(gateDecision(ranked(0.88, 0.62))).toBeNull();
  });

  it('does NOT gate collapsed/ambiguous mail (high runner-up)', () => {
    expect(gateDecision(ranked(0.96, 0.95))).toBeNull();
  });

  it('does NOT gate a low-confidence top match', () => {
    expect(gateDecision(ranked(0.7, 0.3))).toBeNull();
  });

  it('gates a lone, very confident match', () => {
    expect(gateDecision(ranked(0.95))?.categoryId).toBe('c0');
  });
});

describe('resolveBatchLabels per-email shortlists', () => {
  it('validates each email against ITS OWN candidate list', () => {
    const perEmail = [
      [{ id: 'a', label: 'Alpha', description: null }],
      [{ id: 'b', label: 'Beta', description: null }],
    ];
    const raw = JSON.stringify({ '1': ['Alpha'], '2': ['Beta'] });
    expect(resolveBatchLabels(raw, 2, perEmail)).toEqual([['a'], ['b']]);
  });

  it("rejects a label that belongs to another email's shortlist (null = retry, not a 'none' decision)", () => {
    const perEmail = [
      [{ id: 'a', label: 'Alpha', description: null }],
      [{ id: 'b', label: 'Beta', description: null }],
    ];
    const raw = JSON.stringify({ '1': ['Beta'], '2': ['Beta'] });
    expect(resolveBatchLabels(raw, 2, perEmail)).toEqual([null, ['b']]);
  });

  it('strips a leaked <think> wrapper before parsing (qwen3 under /no_think)', () => {
    const perEmail = [[{ id: 'a', label: 'Alpha', description: null }]];
    const raw = '<think></think>\n' + JSON.stringify({ '1': ['Alpha'] });
    expect(resolveBatchLabels(raw, 1, perEmail)).toEqual([['a']]);
  });

  it('tolerates an {emails: {...}} wrapper (issue #6)', () => {
    const perEmail = [[{ id: 'a', label: 'Alpha', description: null }]];
    expect(resolveBatchLabels(JSON.stringify({ emails: { '1': ['Alpha'] } }), 1, perEmail)).toEqual([
      ['a'],
    ]);
  });

  it('tolerates a {results: [...]} positional array (issue #6)', () => {
    const perEmail = [
      [{ id: 'a', label: 'Alpha', description: null }],
      [{ id: 'b', label: 'Beta', description: null }],
    ];
    const raw = JSON.stringify({ results: [['Alpha'], ['Beta']] });
    expect(resolveBatchLabels(raw, 2, perEmail)).toEqual([['a'], ['b']]);
  });

  it('tolerates an array of {index, categories} objects (issue #6)', () => {
    const perEmail = [
      [{ id: 'a', label: 'Alpha', description: null }],
      [{ id: 'b', label: 'Beta', description: null }],
    ];
    const raw = JSON.stringify([
      { index: 1, categories: ['Alpha'] },
      { index: 2, categories: ['Beta'] },
    ]);
    expect(resolveBatchLabels(raw, 2, perEmail)).toEqual([['a'], ['b']]);
  });

  it('coerces a single-string answer to one label (issue #6)', () => {
    const perEmail = [[{ id: 'a', label: 'Alpha', description: null }]];
    expect(resolveBatchLabels(JSON.stringify({ '1': 'Alpha' }), 1, perEmail)).toEqual([['a']]);
  });

  it('tolerates a positional array of plain strings (issue #102)', () => {
    const perEmail = [
      [{ id: 'a', label: 'Alpha', description: null }],
      [{ id: 'b', label: 'Beta', description: null }],
    ];
    expect(resolveBatchLabels(JSON.stringify(['Alpha', 'Beta']), 2, perEmail)).toEqual([
      ['a'],
      ['b'],
    ]);
  });

  it('handles 0-based object indices without dropping email 1 (issue #104)', () => {
    const perEmail = [
      [{ id: 'a', label: 'Alpha', description: null }],
      [{ id: 'b', label: 'Beta', description: null }],
    ];
    const raw = JSON.stringify([
      { index: 0, categories: ['Alpha'] },
      { index: 1, categories: ['Beta'] },
    ]);
    expect(resolveBatchLabels(raw, 2, perEmail)).toEqual([['a'], ['b']]);
  });

  it('keeps a valid empty array distinct from an all-invalid one', () => {
    const perEmail = [
      [{ id: 'a', label: 'Alpha', description: null }],
      [{ id: 'b', label: 'Beta', description: null }],
    ];
    const raw = JSON.stringify({ '1': [], '2': ['Bet'] });
    expect(resolveBatchLabels(raw, 2, perEmail)).toEqual([[], null]);
  });
});
