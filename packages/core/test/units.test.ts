/**
 * Pure-unit test suite for core utilities and services covering model-id
 * canonicalization, vector math, text and time-scope parsing, chat RAG prompt
 * building, topic discovery, categorization, triage, and LLM client backoff.
 */
import { describe, it, expect, vi } from 'vitest';
import type { Logger } from 'pino';
import { canonicalizeModelId } from '../src/util/model-id.js';
import { runningMeanUpdate } from '../src/util/vector.js';
import {
  preprocessForEmbedding,
  sanitizeFtsQuery,
  normalizeForMatch,
  normalizeFilename,
} from '../src/util/text.js';
import { parseTimeScope, stripTimeScope, hasTopicTerms } from '../src/util/time-scope.js';
import { resolveActiveChatModel } from '../src/util/chat-model.js';
import { resolveChatRetrievalCaps } from '../src/routes/chat.js';
import { redactConfig } from '../src/config/config.js';
import { AppConfigSchema, LlmConfigSchema } from '../src/config/schema.js';
import { CategorizationService } from '../src/services/categorization-service.js';
import { TriageOrchestrator } from '../src/services/triage-orchestrator.js';
import { TriageService } from '../src/services/triage-service.js';
import {
  LlmCategorizeOrchestrator,
  type LlmCategorizeProgress,
} from '../src/services/llm-categorize-orchestrator.js';
import { LlmApiError, rateLimitDelayMs, createLlmClient } from '../src/llm/client.js';
import {
  diverseSampleBySender,
  senderDomain,
  mixedSampleBySender,
  isVagueTopicLabel,
  dedupeNearLabels,
  domainFrequency,
  salvageTopics,
  brandTokens,
  mergeOverlappingTopics,
  areTwins,
  TopicDiscoveryService,
} from '../src/services/topic-discovery-service.js';
import {
  CategoryImprovementService,
  salvageSuggestions,
} from '../src/services/category-improvement-service.js';
import type { EmailSummary } from '../src/repositories/email-repository.js';
import { resolveBatchLabels } from '../src/services/llm-categorizer.js';
import { stripThink } from '../src/util/json-llm.js';
import { clusterBySenderAndContent } from '../src/services/sender-clustering.js';
import {
  buildChatMessages,
  buildCondensePrompt,
  buildRerankPrompt,
  buildSummaryPrompt,
  classifyIntent,
  isFollowUp,
  isCorrection,
  isAggregateQuery,
  matchNamedDocument,
  filenameMatchScore,
  dedupeDocumentVersions,
  expandQueryBilingual,
  rankChunksByLexicalOverlap,
  makeThinkSplitter,
  parseRerankOrder,
  rrfMerge,
  stripThinking,
  hasUnclosedThink,
  type RetrievedEmail,
} from '../src/services/chat-service.js';
import { EMBEDDING_DIM } from '../src/db/schema.js';

/** Builds a normalized embedding vector with the given index-to-value entries. */
function unit(spec: Record<number, number>): Float32Array {
  const v = new Float32Array(EMBEDDING_DIM);
  for (const [i, val] of Object.entries(spec)) v[Number(i)] = val;
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < v.length; i++) v[i] /= norm;
  return v;
}

const silentLogger = { info() {}, warn() {}, error() {}, debug() {} } as unknown as Logger;

describe('canonicalizeModelId (L2)', () => {
  it('strips :latest case-insensitively and trims', () => {
    expect(canonicalizeModelId('bge-m3:latest')).toBe('bge-m3');
    expect(canonicalizeModelId('bge-m3:LATEST')).toBe('bge-m3');
    expect(canonicalizeModelId('  bge-m3:Latest  ')).toBe('bge-m3');
    expect(canonicalizeModelId('bge-m3')).toBe('bge-m3');
    expect(canonicalizeModelId('bge-m3:latest-v2')).toBe('bge-m3:latest-v2');
  });
});

describe('diverseSampleBySender (topic discovery coverage)', () => {
  it('senderDomain extracts the domain from common From formats', () => {
    expect(senderDomain('jobs-noreply@linkedin.com')).toBe('linkedin.com');
    expect(senderDomain('LinkedIn <jobs@linkedin.com>')).toBe('linkedin.com');
    expect(senderDomain('A B <x@Sub.Example.COM>')).toBe('sub.example.com');
    expect(senderDomain(null)).toBe('');
  });

  it('prefers distinct sender domains over the loudest sender', () => {
    const arr = [
      ...Array.from({ length: 100 }, (_, i) => ({ fromAddr: `n${i}@linkedin.com` })),
      { fromAddr: 'a@indeed.com' },
      { fromAddr: 'b@github.com' },
      { fromAddr: 'c@bank.com' },
      { fromAddr: 'd@emse.fr' },
      { fromAddr: 'e@news.com' },
    ];
    const picked = diverseSampleBySender(arr, 6);
    const domains = new Set(picked.map((e) => senderDomain(e.fromAddr)));
    expect(domains.size).toBe(6);
    expect(domains).toContain('indeed.com');
    expect(domains).toContain('github.com');
  });

  it('fills the shortfall when distinct domains are fewer than n', () => {
    const arr = [{ fromAddr: 'a@x.com' }, { fromAddr: 'b@x.com' }, { fromAddr: 'c@y.com' }];
    const picked = diverseSampleBySender(arr, 3);
    expect(picked).toHaveLength(3);
  });

  it('returns the whole array when it is smaller than n', () => {
    const arr = [{ fromAddr: 'a@x.com' }, { fromAddr: 'b@y.com' }];
    expect(diverseSampleBySender(arr, 90)).toHaveLength(2);
  });
});

describe('clusterBySenderAndContent (per-sender content clustering)', () => {
  const a = unit({ 0: 1 });
  const aDup = unit({ 0: 1 });
  const aNear = unit({ 0: 1, 1: 0.1 });
  const orth = unit({ 1: 1 });

  it('merges near-identical emails from the same sender into one cluster', () => {
    const clusters = clusterBySenderAndContent(
      [
        { messageId: 'm1', fromAddr: 'noreply@bank.com', vector: a },
        { messageId: 'm2', fromAddr: 'alerts@bank.com', vector: aDup },
        { messageId: 'm3', fromAddr: 'bank.com', vector: aNear },
      ],
      0.95,
    );
    expect(clusters).toHaveLength(1);
    expect(clusters[0]!.memberIds.sort()).toEqual(['m1', 'm2', 'm3']);
  });

  it('splits different email types from the same sender into separate clusters', () => {
    const clusters = clusterBySenderAndContent(
      [
        { messageId: 'stmt', fromAddr: 'x@bank.com', vector: a },
        { messageId: 'alert', fromAddr: 'x@bank.com', vector: orth },
      ],
      0.95,
    );
    expect(clusters).toHaveLength(2);
  });

  it('never merges across senders even with identical content', () => {
    const clusters = clusterBySenderAndContent(
      [
        { messageId: 'm1', fromAddr: 'x@a.com', vector: a },
        { messageId: 'm2', fromAddr: 'y@b.com', vector: aDup },
      ],
      0.95,
    );
    expect(clusters).toHaveLength(2);
  });

  it('a higher threshold yields finer clusters', () => {
    const emails = [
      { messageId: 'm1', fromAddr: 'x@a.com', vector: a },
      { messageId: 'm2', fromAddr: 'x@a.com', vector: aNear },
    ];
    expect(clusterBySenderAndContent(emails, 0.95)).toHaveLength(1);
    expect(clusterBySenderAndContent(emails, 0.999)).toHaveLength(2);
  });
});

describe('buildChatMessages (chat RAG prompt)', () => {
  /** Builds a numbered retrieved-email fixture for the prompt under test. */
  const email = (i: number): RetrievedEmail => ({
    messageId: `m${i}`,
    subject: `Subject ${i}`,
    fromAddr: `s${i}@x.y`,
    date: Date.UTC(2026, 0, i + 1),
    body: `Body text ${i}`,
    bodyFormat: 'text',
    distance: 0.5,
  });

  it('numbers retrieved emails and ends with the question', () => {
    const msgs = buildChatMessages('When is the deadline?', [email(1), email(2)], []);
    expect(msgs[0]!.role).toBe('system');
    const user = msgs[msgs.length - 1]!;
    expect(user.role).toBe('user');
    expect(user.content).toContain('[1]');
    expect(user.content).toContain('[2]');
    expect(user.content).toContain('Subject 1');
    expect(user.content).toContain('Question: When is the deadline?');
  });

  it('signals when no emails were retrieved', () => {
    const msgs = buildChatMessages('anything?', [], []);
    expect(msgs[msgs.length - 1]!.content).toContain('no relevant emails');
  });

  it('includes recent turns but caps history by count', () => {
    const history = Array.from({ length: 12 }, (_, i) => ({
      role: (i % 2 === 0 ? 'user' : 'assistant') as 'user' | 'assistant',
      content: `turn ${i}`,
    }));
    const msgs = buildChatMessages('q', [email(1)], history);
    expect(msgs).toHaveLength(10);
    expect(msgs.some((m) => m.content === 'turn 3')).toBe(false);
    expect(msgs.some((m) => m.content === 'turn 11')).toBe(true);
  });

  it('drops old turns that exceed the character budget', () => {
    const big = 'x'.repeat(3000);
    const history = [
      { role: 'user' as const, content: `OLD ${big}` },
      { role: 'assistant' as const, content: `MID ${big}` },
      { role: 'user' as const, content: 'NEW short turn' },
    ];
    const msgs = buildChatMessages('q', [], history);
    const text = msgs.map((m) => m.content).join('\n');
    expect(text).toContain('NEW short turn');
    expect(text).not.toContain('OLD');
  });

  it('uses the compose prompt and a Task label for drafting', () => {
    const msgs = buildChatMessages('write a reply to Maxime', [email(1)], [], 'compose');
    expect(msgs[0]!.content.toLowerCase()).toContain('email-writing assistant');
    const user = msgs[msgs.length - 1]!;
    expect(user.content).toContain('Task: write a reply to Maxime');
    expect(user.content).not.toContain('Question:');
  });

  it('uses the summarize prompt for digests', () => {
    const msgs = buildChatMessages('summarize these', [email(1)], [], 'summarize');
    expect(msgs[0]!.content.toLowerCase()).toContain('summarize');
    expect(msgs[msgs.length - 1]!.content).toContain('Task: summarize these');
  });

  it('injects the rolling summary as a system message when present', () => {
    const msgs = buildChatMessages(
      'make it shorter',
      [email(1)],
      [],
      'compose',
      700,
      'We are drafting a reply to Maxime about a meeting.',
    );
    const sys = msgs.filter((m) => m.role === 'system');
    expect(sys).toHaveLength(2);
    expect(sys[1]!.content).toContain('Summary of earlier conversation');
    expect(sys[1]!.content).toContain('drafting a reply to Maxime');
  });

  it('omits the summary system message when there is no summary', () => {
    const msgs = buildChatMessages('make it shorter', [email(1)], [], 'compose', 700, '   ');
    expect(msgs.filter((m) => m.role === 'system')).toHaveLength(1);
  });
});

describe('buildCondensePrompt (history-aware query rewriting)', () => {
  it('includes the conversation and the follow-up so the LLM can resolve references', () => {
    const history = [
      { role: 'user' as const, content: 'When is the CPS2 internship defense?' },
      { role: 'assistant' as const, content: 'July 15-16 [1].' },
    ];
    const p = buildCondensePrompt(history, 'and what should I bring?');
    expect(p).toContain('CPS2 internship defense');
    expect(p).toContain('and what should I bring?');
    expect(p.toLowerCase()).toContain('standalone');
  });
});

describe('buildSummaryPrompt (rolling summary-buffer memory)', () => {
  it('includes the prior summary and the new exchanges to fold in', () => {
    const turns = [
      { role: 'user' as const, content: 'When is the defense?' },
      { role: 'assistant' as const, content: 'July 15-16 [1].' },
    ];
    const p = buildSummaryPrompt('User is preparing for the CPS2 defense.', turns);
    expect(p).toContain('Summary so far:');
    expect(p).toContain('User is preparing for the CPS2 defense.');
    expect(p).toContain('When is the defense?');
    expect(p.toLowerCase()).toContain('updated summary');
  });

  it('omits the prior-summary block on the first fold', () => {
    const p = buildSummaryPrompt('', [{ role: 'user' as const, content: 'hello' }]);
    expect(p).not.toContain('Summary so far:');
    expect(p).toContain('hello');
  });
});

describe('isFollowUp (self-contained vs follow-up classification)', () => {
  it('treats terse and referential questions as follow-ups (need history)', () => {
    expect(isFollowUp('yes the second one')).toBe(true);
    expect(isFollowUp('what about the deadline?')).toBe(true);
    expect(isFollowUp('and the time?')).toBe(true);
    expect(isFollowUp('that one')).toBe(true);
  });

  it('treats self-contained questions as standalone (ignore unrelated history)', () => {
    expect(
      isFollowUp('at what date, introduction to ICT final project evaluation/viva was scheduled?'),
    ).toBe(false);
    expect(isFollowUp('summarize my latest bank statements from this year')).toBe(false);
  });

  it('catches a referential phrase anywhere, not only at the start', () => {
    expect(isFollowUp('What did ABS mean in that email?')).toBe(true);
    expect(isFollowUp('Can you explain the figure in this document for me?')).toBe(true);
    expect(isFollowUp('Who prepared the report in that attachment exactly?')).toBe(true);
    expect(isFollowUp('What does it mean when the professor wrote that?')).toBe(true);
    expect(isFollowUp('Could you tell me what the email actually says here?')).toBe(true);
    expect(isFollowUp('Who is the supervisor for the Student Van Management project?')).toBe(false);
  });

  it('does not treat a first-mention "the email from X" as a follow-up', () => {
    expect(isFollowUp('summarize the email from the bank about my March statement')).toBe(false);
    expect(isFollowUp('Reply to the message from HR confirming the onboarding session')).toBe(
      false,
    );
    expect(isFollowUp('Find the attachment named invoice and tell me its total amount')).toBe(
      false,
    );
  });

  it('detects French follow-up and referential cues', () => {
    expect(isFollowUp('Et à qui dois-je envoyer la notification conditionnelle ?')).toBe(true);
    expect(isFollowUp('Puis quelles sont les autres pièces jointes à fournir au dossier ?')).toBe(
      true,
    );
    expect(isFollowUp('Que veut dire ce document exactement pour mon inscription ?')).toBe(true);
    expect(isFollowUp('Et le montant total de la bourse alors finalement ?')).toBe(true);
    expect(isFollowUp('quand commence la formation de français au CILEC cette année')).toBe(false);
  });

  it('does not treat the frozen interrogative "puis-je" as a follow-up', () => {
    expect(
      isFollowUp('Puis-je obtenir une attestation de scolarite pour mon logement etudiant ?'),
    ).toBe(false);
    expect(
      isFollowUp('Puis quelles autres pieces dois-je encore fournir pour completer le dossier ?'),
    ).toBe(true);
  });

  it('matches French cues whether the question is accented or de-accented', () => {
    expect(
      isFollowUp('Ça change quoi pour mon inscription au logement etudiant cette annee ?'),
    ).toBe(true);
    expect(
      isFollowUp('Ca change quoi pour mon inscription au logement etudiant cette annee ?'),
    ).toBe(true);
  });
});

describe('isAggregateQuery (prefer an evaluation/summary email)', () => {
  it('flags questions asking for an aggregate figure', () => {
    expect(isAggregateQuery('What was my average MCQ?')).toBe(true);
    expect(isAggregateQuery('What was my TODO average?')).toBe(true);
    expect(isAggregateQuery('What was my final TFSD grade?')).toBe(true);
    expect(isAggregateQuery('What was my EFCL grade?')).toBe(true);
    expect(isAggregateQuery('what is my overall result this term?')).toBe(true);
  });

  it('does not flag a question about one specific item', () => {
    expect(isAggregateQuery('In TFSD MCQ Test 2, how much marks did I score?')).toBe(false);
    expect(isAggregateQuery('when is the CPS2 internship defense?')).toBe(false);
    expect(isAggregateQuery('who supervised the Student Van project?')).toBe(false);
  });
});

describe('normalizeForMatch / normalizeFilename', () => {
  it('normalizeForMatch lowercases, strips accents, collapses punctuation (no camelCase split)', () => {
    expect(normalizeForMatch('Attestation de Droits - Santé.PDF')).toBe(
      'attestation de droits sante pdf',
    );
    expect(normalizeForMatch('  multiple   spaces _ and-dashes ')).toBe(
      'multiple spaces and dashes',
    );
    expect(normalizeForMatch('iCloud')).toBe('icloud');
  });

  it('normalizeFilename additionally splits camelCase for run-together filenames', () => {
    expect(normalizeFilename('StagesEtranger-1.pdf')).toBe('stages etranger 1 pdf');
    expect(normalizeFilename('SujetStageM2_CAP-ART.pdf')).toBe('sujet stage m2 cap art pdf');
  });
});

describe('matchNamedDocument (filename-targeted retrieval)', () => {
  const files = [
    'StagesEtranger-1.pdf',
    'Attestation de Droits.pdf',
    'CILEC livret formation.docx',
  ];

  it('matches an exact filename mentioned with a doc-reference word', () => {
    const m = matchNamedDocument('Dans le fichier StagesEtranger-1.pdf, quel est l ordre ?', files);
    expect(m?.index).toBe(0);
  });

  it('matches a partial filename (no extension, accent-insensitive)', () => {
    expect(matchNamedDocument('Dans le document StagesEtranger, quel ordre ?', files)?.index).toBe(
      0,
    );
    expect(matchNamedDocument("dans l'attestation de droits, quel site ?", files)?.index).toBe(1);
  });

  it('does not match without a document-reference word (no hijacking of normal questions)', () => {
    expect(matchNamedDocument('what was my average MCQ score?', files)).toBeNull();
    expect(matchNamedDocument('quand est la soutenance CPS2 ?', files)).toBeNull();
  });

  it('does not match on a common short token alone', () => {
    expect(matchNamedDocument('dans le document de 2026', files)).toBeNull();
  });

  it('ignores generic doc/category stopwords so a common word does not cause a false match', () => {
    expect(
      filenameMatchScore('dans le fichier StagesEtranger-1.pdf', 'StagesEtranger-1.pdf'),
    ).toBeGreaterThanOrEqual(5);
    expect(
      filenameMatchScore('quelle est ma formation cette annee', 'LIVRET DE FORMATION FST.pdf'),
    ).toBe(0);
    expect(
      filenameMatchScore('montre moi le contrat de travail', 'Contrat Habitation ADHE-2026.pdf'),
    ).toBe(0);
    expect(
      filenameMatchScore("l'attestation de droits", 'Attestation de Droits.pdf'),
    ).toBeGreaterThanOrEqual(5);
  });

  it('matches a camelCase filename to spaced natural wording', () => {
    expect(
      filenameMatchScore('le document stages etranger', 'StagesEtranger-1.pdf'),
    ).toBeGreaterThanOrEqual(5);
    expect(
      matchNamedDocument('quel est le document stages etranger pour mon dossier', files)?.index,
    ).toBe(0);
  });
});

describe('expandQueryBilingual (cross-language retrieval expansion)', () => {
  it('appends the English equivalent of a French document/concept term', () => {
    expect(expandQueryBilingual('Dans le resume d evaluation TFSD, que signifie ABS ?')).toContain(
      'evaluation summary',
    );
    expect(expandQueryBilingual('quelle etait ma moyenne MCQ').toLowerCase()).toContain('average');
  });

  it('appends the French equivalent of an English term (and vice versa)', () => {
    expect(expandQueryBilingual('ADH home insurance contract').toLowerCase()).toMatch(
      /assurance|habitation|contrat/,
    );
  });

  it('leaves a query with no mapped terms unchanged', () => {
    expect(expandQueryBilingual('what time is the viva')).toBe('what time is the viva');
  });
});

describe('isCorrection (a follow-up that rejects the prior answer)', () => {
  it('detects English and French corrections', () => {
    expect(isCorrection('no, i was talking about ADH home insurance')).toBe(true);
    expect(isCorrection('non, je voulais dire le contrat habitation')).toBe(true);
    expect(isCorrection('actually I meant the other document')).toBe(true);
  });

  it('does not flag a normal question or a normal follow-up', () => {
    expect(isCorrection('what was my average MCQ?')).toBe(false);
    expect(isCorrection('and the ABS one?')).toBe(false);
  });
});

describe('rankChunksByLexicalOverlap date-range boost', () => {
  it('ranks a chunk with the FULL date range above one with a single/truncated date', () => {
    const chunks = [
      { text: 'valable pour la periode comprise entre le 03/06/2026 et le 31/08' },
      {
        text: 'DUREE DU CONTRAT les garanties sont accordees pour la periode du 03/06/2026 au 31/08/2027',
      },
      { text: 'coordonnees du service et autres informations generales' },
    ];
    const ranked = rankChunksByLexicalOverlap(
      chunks,
      'what are the start and end dates of the contract?',
    );
    expect(ranked[0]!.text).toContain('31/08/2027');
  });
});

describe('dedupeDocumentVersions (collapse old/new document versions)', () => {
  /** Builds an attachment-bearing retrieved-email fixture keyed by filename. */
  const item = (attachmentName: string, date: number): RetrievedEmail => ({
    messageId: attachmentName,
    subject: null,
    fromAddr: null,
    date,
    body: 'chunk text',
    bodyFormat: 'text',
    distance: 0,
    attachmentName,
  });
  const adh2025 = item('Contrat Habitation ADHE-20250808-2731.pdf', 1000);
  const adh2026 = item('Contrat Habitation ADHE-20260603-bba7c844.pdf', 9000);

  it('keeps the newest version by default (current / latest / "my X")', () => {
    const out = dedupeDocumentVersions(
      [adh2025, adh2026],
      'what are my ADH habitation insurance dates?',
    );
    expect(out).toContain(adh2026);
    expect(out).not.toContain(adh2025);
  });

  it('keeps the version whose filename matches a year named in the query', () => {
    const out = dedupeDocumentVersions(
      [adh2025, adh2026],
      'the 2025 ADH habitation contract end date',
    );
    expect(out).toContain(adh2025);
    expect(out).not.toContain(adh2026);
  });

  it('keeps the oldest when the query asks for a previous one', () => {
    const out = dedupeDocumentVersions(
      [adh2025, adh2026],
      'my previous ADH habitation contract dates',
    );
    expect(out).toContain(adh2025);
    expect(out).not.toContain(adh2026);
  });

  it('leaves DISTINCT documents untouched (different signatures)', () => {
    const a = item('Attestation de Droits.pdf', 1);
    const b = item('CILEC livret formation.docx', 2);
    expect(dedupeDocumentVersions([a, b], 'current docs')).toEqual([a, b]);
  });

  it('does NOT merge different document TYPES that share a subject word', () => {
    const comm = item('Communication Ecole pour CROUS 2026-2027.pdf', 5000);
    const conv = item('Convention de stage CROUS.pdf', 1000);
    const out = dedupeDocumentVersions([comm, conv], 'what does the CROUS document say');
    expect(out).toContain(comm);
    expect(out).toContain(conv);
  });

  it('does NOT merge on a single shared content token (e.g. two different invoices)', () => {
    const i1 = item('Facture EDF.pdf', 1000);
    const i2 = item('Facture SFR.pdf', 5000);
    const out = dedupeDocumentVersions([i1, i2], 'my latest facture');
    expect(out).toContain(i1);
    expect(out).toContain(i2);
  });

  it('matches a query year only as a real date token, not a coincidental id substring', () => {
    const a = item('Bail Logement id2026x.pdf', 1000);
    const b = item('Bail Logement ref99.pdf', 9000);
    const out = dedupeDocumentVersions([a, b], 'the 2026 bail logement');
    expect(out).toContain(b);
    expect(out).not.toContain(a);
  });
});

describe('rankChunksByLexicalOverlap (exact-fact chunk ordering)', () => {
  it('puts the chunk containing the query terms first', () => {
    const chunks = [
      { text: 'Bienvenue dans le livret de formation. Informations generales.' },
      {
        text: 'La presente attestation est valable pour la periode comprise entre le 03/06/2026 et le 31/08/2027.',
      },
      { text: 'Coordonnees du service et autres details.' },
    ];
    const ranked = rankChunksByLexicalOverlap(
      chunks,
      'quelle est la periode de validite valable du contrat ?',
    );
    expect(ranked[0]!.text).toContain('valable pour la periode');
  });
});

describe('sanitizeFtsQuery (FTS5 MATCH safety)', () => {
  it('quotes word tokens and ORs them, dropping punctuation and short tokens', () => {
    expect(sanitizeFtsQuery('when was my viva?')).toBe('"when" OR "was" OR "my" OR "viva"');
    expect(sanitizeFtsQuery('a CSC101 viva')).toBe('"csc101" OR "viva"');
  });

  it('neutralizes FTS5 operators so a free-text query cannot be syntax', () => {
    expect(sanitizeFtsQuery('hello AND "world" OR (x*')).toBe(
      '"hello" OR "and" OR "world" OR "or"',
    );
  });

  it('returns empty when nothing usable remains', () => {
    expect(sanitizeFtsQuery('??? !! ()')).toBe('');
    expect(sanitizeFtsQuery('')).toBe('');
  });

  it('keeps Unicode (French) letters', () => {
    expect(sanitizeFtsQuery('résumé du stage')).toBe('"résumé" OR "du" OR "stage"');
  });
});

describe('rrfMerge (hybrid retrieval fusion)', () => {
  it('ranks an id appearing high in both lists above single-list ids', () => {
    const vector = ['a', 'b', 'c'];
    const keyword = ['b', 'd', 'a'];
    expect(rrfMerge([vector, keyword], 2)).toEqual(['b', 'a']);
  });

  it('falls back gracefully when one list is empty', () => {
    expect(rrfMerge([['a', 'b', 'c'], []], 2)).toEqual(['a', 'b']);
  });

  it('dedups ids and respects the k limit', () => {
    const out = rrfMerge(
      [
        ['a', 'b'],
        ['a', 'b'],
      ],
      5,
    );
    expect(out).toEqual(['a', 'b']);
  });
});

describe('parseTimeScope (time-scoped retrieval)', () => {
  const now = new Date(2026, 5, 17, 14, 30, 0).getTime();
  /** Returns whether a timestamp falls within a parsed time scope window. */
  const inRange = (s: { from: number; to: number } | null, ms: number): boolean =>
    !!s && ms >= s.from && ms <= s.to;

  it('returns null when there is no time expression', () => {
    expect(parseTimeScope('what did Maxime say about the project?', now)).toBeNull();
    expect(parseTimeScope('summarize the bank emails', now)).toBeNull();
  });

  it('scopes "today" and "yesterday" to the right single day', () => {
    const today = parseTimeScope('what did I get today?', now)!;
    expect(today.label).toBe('today');
    expect(inRange(today, now)).toBe(true);
    expect(inRange(today, new Date(2026, 5, 16, 23, 59).getTime())).toBe(false);

    const y = parseTimeScope('anything urgent from yesterday', now)!;
    expect(inRange(y, new Date(2026, 5, 16, 9, 0).getTime())).toBe(true);
    expect(inRange(y, now)).toBe(false);
  });

  it('scopes this/last week and month', () => {
    const week = parseTimeScope('summarize emails from this week', now)!;
    expect(inRange(week, new Date(2026, 5, 15, 8, 0).getTime())).toBe(true);
    expect(week.matched).toMatch(/this week/i);

    const lastMonth = parseTimeScope('what did I receive last month', now)!;
    expect(inRange(lastMonth, new Date(2026, 4, 20).getTime())).toBe(true);
    expect(inRange(lastMonth, new Date(2026, 5, 1).getTime())).toBe(false);
  });

  it('scopes an explicit "month year" (EN + FR)', () => {
    const may = parseTimeScope('check my may 2018 emails about the viva', now)!;
    expect(new Date(may.from).getFullYear()).toBe(2018);
    expect(new Date(may.from).getMonth()).toBe(4);
    expect(inRange(may, new Date(2018, 4, 15).getTime())).toBe(true);

    const mai = parseTimeScope('emails de mai 2018', now)!;
    expect(new Date(mai.from).getMonth()).toBe(4);
  });

  it('scopes "last N days" and vague recency', () => {
    const lastDays = parseTimeScope('show me the last 7 days', now)!;
    expect(inRange(lastDays, new Date(2026, 5, 12, 10).getTime())).toBe(true);
    expect(inRange(lastDays, new Date(2026, 5, 1).getTime())).toBe(false);

    const recent = parseTimeScope('any recent emails from the bank?', now)!;
    expect(recent.label).toBe('recent');
    expect(inRange(recent, new Date(2026, 5, 1).getTime())).toBe(true);
  });

  it('requires a preposition for a bare year so codes are not misread', () => {
    expect(parseTimeScope('what happened in 2018?', now)!.label).toBe('2018');
    expect(parseTimeScope('error code 2018 reference', now)).toBeNull();
  });

  it('stripTimeScope removes the time phrase but keeps the semantic query', () => {
    const scope = parseTimeScope('what did Maxime send last month', now)!;
    expect(stripTimeScope('what did Maxime send last month', scope)).toBe('what did Maxime send');
    const pure = parseTimeScope('this week', now)!;
    expect(stripTimeScope('this week', pure)).toBe('this week');
  });
});

describe('hasTopicTerms (time scope: rank by recency vs filter to window)', () => {
  it('detects a topic in a time-stripped query', () => {
    expect(hasTopicTerms('summarize my emails about cps2 defenses')).toBe(true);
    expect(hasTopicTerms('emails about invoices')).toBe(true);
    expect(hasTopicTerms('what did Maxime say')).toBe(true);
  });

  it('treats a pure time/command query as having no topic', () => {
    expect(hasTopicTerms('show my emails')).toBe(false);
    expect(hasTopicTerms('summarize my mail')).toBe(false);
    expect(hasTopicTerms('mes courriels')).toBe(false);
    expect(hasTopicTerms('')).toBe(false);
  });
});

describe('resolveChatRetrievalCaps (local clamp vs cloud budget)', () => {
  it('clamps a cloud-sized stored value down for local', () => {
    expect(resolveChatRetrievalCaps(false, 30, 1500)).toEqual({ topK: 10, snippetChars: 900 });
  });

  it('leaves local unset when the config is empty so service defaults apply', () => {
    expect(resolveChatRetrievalCaps(false, undefined, undefined)).toEqual({
      topK: undefined,
      snippetChars: undefined,
    });
    expect(resolveChatRetrievalCaps(false, null, null)).toEqual({
      topK: undefined,
      snippetChars: undefined,
    });
  });

  it('lets a local user lower below the cap but never raise above it', () => {
    expect(resolveChatRetrievalCaps(false, 5, 500)).toEqual({ topK: 5, snippetChars: 500 });
  });

  it('keeps the large budget for cloud (stored value passes through, else the cloud default)', () => {
    expect(resolveChatRetrievalCaps(true, 30, 1500)).toEqual({ topK: 30, snippetChars: 1500 });
    expect(resolveChatRetrievalCaps(true, undefined, undefined)).toEqual({
      topK: 30,
      snippetChars: 2000,
    });
  });
});

describe('resolveActiveChatModel', () => {
  it('uses a configured local chat model when the local provider exposes it', async () => {
    const model = await resolveActiveChatModel(
      LlmConfigSchema.parse({
        baseUrl: 'http://localhost:11434/v1',
        generationModel: 'qwen3:8b',
        chatModel: 'qwen3:14b',
      }),
      { health: async () => ({ ok: true, models: ['qwen3:14b', 'qwen3:8b'] }) },
      silentLogger,
    );

    expect(model).toEqual({ modelId: 'qwen3:14b', provider: 'local' });
  });

  it('falls back to the configured generation model when local mode has a stale cloud chat model', async () => {
    const model = await resolveActiveChatModel(
      LlmConfigSchema.parse({
        baseUrl: 'http://localhost:11434/v1',
        generationModel: 'qwen3:8b',
        chatModel: 'gpt-4o-mini',
        chatBaseUrl: null,
      }),
      { health: async () => ({ ok: true, models: ['qwen3:8b'] }) },
      silentLogger,
    );

    expect(model).toEqual({ modelId: 'qwen3:8b', provider: 'local' });
  });

  it('does not validate cloud chat models against the local model list', async () => {
    let healthCalls = 0;
    const model = await resolveActiveChatModel(
      LlmConfigSchema.parse({
        baseUrl: 'http://localhost:11434/v1',
        generationModel: 'qwen3:8b',
        chatModel: 'cloud-model-from-config',
        chatBaseUrl: 'https://api.example.com/v1',
      }),
      {
        health: async () => {
          healthCalls += 1;
          return { ok: true, models: ['qwen3:8b'] };
        },
      },
      silentLogger,
    );

    expect(model).toEqual({ modelId: 'cloud-model-from-config', provider: 'cloud' });
    expect(healthCalls).toBe(0);
  });
});

describe('reranker (buildRerankPrompt + parseRerankOrder)', () => {
  /** Builds a rerank candidate fixture, optionally tagged as an attachment. */
  const item = (i: number, attachment?: string): RetrievedEmail => ({
    messageId: `m${i}`,
    subject: `Subject ${i}`,
    fromAddr: `s${i}@x.y`,
    date: null,
    body: `Body content for item ${i}`,
    bodyFormat: 'text',
    distance: 0.5,
    ...(attachment ? { attachmentName: attachment } : {}),
  });

  it('numbers candidates from 0 and labels attachments distinctly', () => {
    const p = buildRerankPrompt('what is the deadline?', [item(0), item(1, 'plan.pdf')]);
    expect(p).toContain('[0]');
    expect(p).toContain('[1] Attachment "plan.pdf"');
    expect(p).toContain('what is the deadline?');
    expect(p.toLowerCase()).toContain('ranked item numbers');
  });

  it('parses a comma-separated order, dropping out-of-range and duplicate indices', () => {
    expect(parseRerankOrder('3, 0, 5', 6)).toEqual([3, 0, 5]);
    expect(parseRerankOrder('[2], [0], [2]', 4)).toEqual([2, 0]);
    expect(parseRerankOrder('1, 9, 2', 3)).toEqual([1, 2]);
    expect(parseRerankOrder('none of these', 3)).toEqual([]);
  });
});

describe('classifyIntent (ask vs compose vs summarize)', () => {
  it('routes drafting/reply requests to compose (EN + FR)', () => {
    expect(classifyIntent('write me an email to prof maxime in french')).toBe('compose');
    expect(classifyIntent('write an email for prof maxime asking for a meeting tomorrow')).toBe(
      'compose',
    );
    expect(classifyIntent('draft a reply to the bank')).toBe('compose');
    expect(classifyIntent('reply to this')).toBe('compose');
    expect(classifyIntent('écris un mail à Maxime')).toBe('compose');
    expect(classifyIntent('rédige une réponse à Pierre')).toBe('compose');
    expect(classifyIntent('réponds à ce message')).toBe('compose');
  });

  it('routes digest requests to summarize (EN + FR)', () => {
    expect(classifyIntent('summarize my recent emails from the bank')).toBe('summarize');
    expect(classifyIntent('give me a recap of the CPS2 thread')).toBe('summarize');
    expect(classifyIntent("what's pending?")).toBe('summarize');
    expect(classifyIntent("what's urgent this week?")).toBe('summarize');
    expect(classifyIntent('résume mes derniers emails')).toBe('summarize');
  });

  it('keeps factual questions as ask, including ones that mention writing', () => {
    expect(classifyIntent('when is the CPS2 internship defense?')).toBe('ask');
    expect(classifyIntent('what did professor maxime write about the personal project?')).toBe(
      'ask',
    );
    expect(classifyIntent('did the bank send me a statement?')).toBe('ask');
    expect(classifyIntent('what did he write in his last email?')).toBe('ask');
  });
});

describe('stripThinking (reasoning-model output cleanup)', () => {
  it('removes a complete think block, keeping the answer', () => {
    expect(stripThinking('<think>let me reason</think>The deadline is Friday.')).toBe(
      'The deadline is Friday.',
    );
  });

  it('keeps content after a stray closing tag (truncated think)', () => {
    expect(stripThinking('reasoning without open tag</think>Final answer.')).toBe('Final answer.');
  });

  it('slices at the FIRST </think>, preserving an answer that quotes the tag', () => {
    expect(stripThinking('reason</think>The order is </think> done.')).toContain('The order is');
  });

  it('leaves a normal answer untouched', () => {
    expect(stripThinking('Just a plain answer with [1] citation.')).toBe(
      'Just a plain answer with [1] citation.',
    );
  });
});

describe('hasUnclosedThink (reasoning-leak detector for auxiliary calls)', () => {
  it('flags a truncated (never-closed) think block', () => {
    expect(hasUnclosedThink('<think>reasoning that ran out of tokens')).toBe(true);
  });

  it('flags a second think reopened after a closed one (the residue stripThinking misses)', () => {
    expect(hasUnclosedThink('<think>a</think>real question? <think>leak')).toBe(true);
  });

  it('passes a balanced block and tagless output', () => {
    expect(hasUnclosedThink('<think>brief</think>Rolling summary.')).toBe(false);
    expect(hasUnclosedThink('<think>\n\n</think>\n\nWhat was my ABS average?')).toBe(false);
    expect(hasUnclosedThink('plain cloud answer, no tags')).toBe(false);
  });
});

describe('makeThinkSplitter (live thinking vs answer split)', () => {
  /** Feeds chunks through a fresh think splitter and accumulates the split output. */
  function run(chunks: string[]): { think: string; answer: string; promoted: boolean } {
    const s = makeThinkSplitter();
    let think = '';
    let answer = '';
    let promoted = false;
    /** Routes each emitted event into the think, answer, or promoted accumulator. */
    const apply = (events: ReturnType<typeof s.push>): void => {
      for (const e of events) {
        if (e.kind === 'think') think += e.text;
        else if (e.kind === 'answer') answer += e.text;
        else promoted = true;
      }
    };
    for (const c of chunks) apply(s.push(c));
    apply(s.flush());
    return { think, answer, promoted };
  }

  it('separates reasoning (before </think>) from the answer (after)', () => {
    const r = run(['some reasoning', '</think>', 'The answer.']);
    expect(r.think).toBe('some reasoning');
    expect(r.answer).toBe('The answer.');
    expect(r.promoted).toBe(false);
  });

  it('strips a leading <think> opening tag from the reasoning', () => {
    const r = run(['<think>', ' reasoning text', '</think>', 'Answer']);
    expect(r.think).toBe('reasoning text');
    expect(r.answer).toBe('Answer');
  });

  it('strips <think> even when the opening tag is split across chunks', () => {
    const r = run(['<thi', 'nk>actual reasoning</think>Done']);
    expect(r.think).toBe('actual reasoning');
    expect(r.answer).toBe('Done');
  });

  it('handles the </think> tag split across chunks without losing characters', () => {
    const r = run(['reasoning here', '</thi', 'nk>Final ', 'answer']);
    expect(r.think).toBe('reasoning here');
    expect(r.answer).toBe('Final answer');
  });

  it('promotes reasoning to the answer when no </think> appears (non-reasoning model)', () => {
    const r = run(['Direct ', 'answer ', 'here']);
    expect(r.answer).toBe('');
    expect(r.think).toBe('Direct answer here');
    expect(r.promoted).toBe(true);
  });

  it('splits a mixed-case close tag the same way (case-insensitive)', () => {
    const r = run(['reasoning</THINK>Answer']);
    expect(r.think).toBe('reasoning');
    expect(r.answer).toBe('Answer');
  });

  it('drops a dangling partial close tag at the end of truncated reasoning', () => {
    const r = run(['some reasoning</think']);
    expect(r.think).toBe('some reasoning');
    expect(r.promoted).toBe(true);
  });

  it('reaches the answer phase for a whitespace-only answer (no false promote)', () => {
    const r = run(['reasoning</think>   ']);
    expect(r.think).toBe('reasoning');
    expect(r.answer).toBe('');
    expect(r.promoted).toBe(false);
  });
});

describe('resolveBatchLabels (batched LLM categorize output)', () => {
  const cands = [
    { id: 'c1', label: 'Receipts & Invoices', description: null },
    { id: 'c2', label: 'Job Alerts', description: null },
    { id: 'c3', label: 'Security & Sign-in', description: null },
  ];

  it('maps each numbered entry to its email, in order', () => {
    const raw = '{"1": ["Job Alerts"], "2": ["Receipts & Invoices", "Security & Sign-in"]}';
    expect(resolveBatchLabels(raw, 2, cands)).toEqual([['c2'], ['c1', 'c3']]);
  });

  it('returns null for a missing entry (retry), not [] (a real "none fits" decision)', () => {
    const raw = '{"1": ["Job Alerts"]}';
    expect(resolveBatchLabels(raw, 3, cands)).toEqual([['c2'], null, null]);
  });

  it('drops hallucinated labels ([] is valid), and nulls a non-array entry', () => {
    const raw = '{"1": ["Made Up", "Job Alerts"], "2": "not-an-array"}';
    expect(resolveBatchLabels(raw, 2, cands)).toEqual([['c2'], null]);
  });

  it('returns all-null on malformed output (so the batch is retried, not marked uncategorizable)', () => {
    expect(resolveBatchLabels('garbage', 2, cands)).toEqual([null, null]);
    expect(resolveBatchLabels('[1,2,3]', 2, cands)).toEqual([null, null]);
  });

  it('nulls a non-empty array whose labels are ALL invalid (retry), but keeps a valid empty []', () => {
    const raw = '{"1": ["Invoice"], "2": []}';
    expect(resolveBatchLabels(raw, 2, cands)).toEqual([null, []]);
  });

  it('tolerates code fences around the JSON', () => {
    const raw = '```json\n{"1": ["Security & Sign-in"]}\n```';
    expect(resolveBatchLabels(raw, 1, cands)).toEqual([['c3']]);
  });
});

describe('preprocessForEmbedding surrogate safety (L3)', () => {
  it('never ends in a lone high surrogate at the cap', () => {
    const head = 'a'.repeat(9);
    const text = head + '😀' + 'b'.repeat(50);
    const out = preprocessForEmbedding(text, { maxChars: 10 });
    expect(out.length).toBeLessThanOrEqual(10);
    const lastCode = out.charCodeAt(out.length - 1);
    const isLoneHighSurrogate = lastCode >= 0xd800 && lastCode <= 0xdbff;
    expect(isLoneHighSurrogate).toBe(false);
    expect(JSON.parse(JSON.stringify(out))).toBe(out);
  });

  it('keeps a full astral char when it fits', () => {
    const out = preprocessForEmbedding('hi😀', { maxChars: 100 });
    expect(out).toBe('hi😀');
  });

  it('strips the signature and quoted reply even with CRLF line endings', () => {
    const lf = 'Real body line.\n-- \nSent from my phone\n> quoted reply line';
    const crlf = lf.replace(/\n/g, '\r\n');
    const fromLf = preprocessForEmbedding(lf, { maxChars: 200 });
    const fromCrlf = preprocessForEmbedding(crlf, { maxChars: 200 });
    expect(fromCrlf).toBe(fromLf);
    expect(fromCrlf).toBe('Real body line.');
    expect(fromCrlf).not.toContain('Sent from my phone');
    expect(fromCrlf).not.toContain('quoted reply');
  });
});

describe('runningMeanUpdate learning math', () => {
  it('seeds with the new vector (normalized) when count is 0', () => {
    const v = new Float32Array(EMBEDDING_DIM);
    v[0] = 3;
    v[1] = 4;
    const out = runningMeanUpdate(new Float32Array(EMBEDDING_DIM), 0, v);
    expect(out[0]).toBeCloseTo(0.6, 5);
    expect(out[1]).toBeCloseTo(0.8, 5);
  });

  it('drifts the centroid toward the new vector', () => {
    const centroid = new Float32Array(EMBEDDING_DIM);
    centroid[0] = 1;
    const v = new Float32Array(EMBEDDING_DIM);
    v[0] = 1;
    v[1] = 0.5;
    const out = runningMeanUpdate(centroid, 3, v);
    expect(out[1]).toBeGreaterThan(0);
    const norm = Math.hypot(...Array.from(out.subarray(0, 4)));
    expect(norm).toBeCloseTo(1, 5);
  });
});

describe('topic discovery sampling and label hygiene', () => {
  /** Builds a minimal sender fixture carrying only a From address. */
  const e = (from: string) => ({ fromAddr: from });

  it('mixedSampleBySender returns n and surfaces high-volume senders, unlike pure diversity', () => {
    const arr = [
      e('a@x.com'),
      e('b@y.com'),
      e('c@z.com'),
      ...Array.from({ length: 200 }, () => e('bot@github.com')),
    ];
    const mixed = mixedSampleBySender(arr, 20);
    expect(mixed).toHaveLength(20);
    const githubCount = mixed.filter((m) => senderDomain(m.fromAddr) === 'github.com').length;
    expect(githubCount).toBeGreaterThan(1);
  });

  it('isVagueTopicLabel catches catch-all families, not just exact strings', () => {
    for (const vague of [
      'Technical Support',
      'Technical Assistance',
      'Support Requests',
      'Account Notifications',
      'Service Alerts',
      'General Updates',
      'Important Updates',
      'Service Announcements',
      'Notifications',
      'System Messages',
      'Miscellaneous',
      'Customer Support',
      'Tech Support',
      'Daily Updates',
      'Company Announcements',
      'Status Reports',
    ]) {
      expect(isVagueTopicLabel(vague)).toBe(true);
    }
    for (const concrete of [
      'Developer & Code Reviews',
      'Job Alerts',
      'Banking Activity',
      'Security & Sign-in',
      'Course Grades & Results',
      'Travel & Accommodation',
      'Health & Insurance',
      'Receipts & Invoices',
    ]) {
      expect(isVagueTopicLabel(concrete)).toBe(false);
    }
  });

  it('dedupeNearLabels drops twins by significant-word overlap', () => {
    const out = dedupeNearLabels([
      { label: 'Receipts & Invoices' },
      { label: 'Invoices and Receipts' },
      { label: 'Course Grades' },
      { label: 'Course Grades & Results' },
      { label: 'Job Alerts' },
    ]);
    expect(out.map((t) => t.label)).toEqual(['Receipts & Invoices', 'Course Grades', 'Job Alerts']);
  });

  it('stripThink removes a leading wrapper but preserves a literal <think> inside JSON', () => {
    expect(stripThink('<think></think>\n{"1":["A"]}')).toBe('{"1":["A"]}');
    expect(stripThink('<think>reasoning</think>{"x":1}')).toBe('{"x":1}');
    const withLiteral = '{"label":"a <think> tag"}';
    expect(stripThink(withLiteral)).toBe(withLiteral);
  });

  it('salvageTopics recovers complete topics from a truncated answer', () => {
    const truncated =
      '{"topics": [\n' +
      '{"label": "Job Alerts", "description": "listings from job boards"},\n' +
      '{"label": "Receipts & Invoices", "description": "payment confirmations"},\n' +
      '{"label": "Banking Activity", "descr';
    expect(salvageTopics(truncated).map((t) => t.label)).toEqual([
      'Job Alerts',
      'Receipts & Invoices',
    ]);
  });

  it('salvageTopics strips think blocks and fences, and ignores braces inside strings', () => {
    const fenced =
      '<think>let me reason</think>\n```json\n' +
      '{"topics": [{"label": "Code Reviews", "description": "PRs like {x} and {y}"}]}\n```';
    expect(salvageTopics(fenced).map((t) => t.label)).toEqual(['Code Reviews']);
    expect(salvageTopics('no json here at all')).toEqual([]);
  });

  it('salvageTopics clamps a stray brace and caps the count at 20', () => {
    const stray =
      '{"topics":[ } {"label":"A","description":"d1"},{"label":"B","description":"d2"}]}';
    expect(salvageTopics(stray).map((t) => t.label)).toEqual(['A', 'B']);

    const many =
      '{"topics":[' +
      Array.from({ length: 25 }, (_, i) => `{"label":"L${i}","description":"d"}`).join(',') +
      ']}';
    expect(salvageTopics(many)).toHaveLength(20);
  });

  it('salvageTopics anchors on the last topics array', () => {
    const raw =
      '{"topics":[{"label":"Stray","description":"x"}],"more":1,' +
      '"topics":[{"label":"Real One","description":"r1"},{"label":"Real Two","description":"r2"}]}';
    expect(salvageTopics(raw).map((t) => t.label)).toEqual(['Real One', 'Real Two']);
  });

  it('areTwins merges same-purpose categories but not related-distinct or different-purpose ones', () => {
    const v = (...n: number[]) => Float32Array.from(n);
    expect(areTwins('Security Alerts', v(1, 0), 'Security and Sign-in', v(0.95, 0.312))).toBe(true);
    expect(areTwins('Developer Code Reviews', v(1, 0), 'Developer Platforms', v(0.8, 0.6))).toBe(
      false,
    );
    expect(areTwins('Course Grades', v(1, 0), 'Course Materials', v(0.95, 0.312))).toBe(false);
  });

  it('brandTokens extracts sender brands and skips generic labels', () => {
    const brands = brandTokens([
      ['mail.github.com', 100],
      ['glassdoor.com', 80],
      ['notifications.com', 5],
    ]);
    expect(brands.has('github')).toBe(true);
    expect(brands.has('glassdoor')).toBe(true);
    expect(brands.has('mail')).toBe(false);
    expect(brands.has('notifications')).toBe(false);
  });

  it('brandTokens does not treat a purpose word as a sender brand', () => {
    const brands = brandTokens([
      ['banking.com', 100],
      ['JOBS.com', 80],
    ]);
    expect(brands.has('banking')).toBe(false);
    expect(brands.has('jobs')).toBe(false);
  });

  it('mergeOverlappingTopics drops a sender-specific twin but keeps distinct purposes', () => {
    const v = (...nums: number[]) => Float32Array.from(nums);
    const t = (label: string, vec: Float32Array) => ({ topic: { label }, vec });
    const brands = brandTokens([['github.com', 100]]);

    const kept = mergeOverlappingTopics(
      [
        t('Developer Code Reviews', v(1, 0, 0)),
        t('GitHub Notifications', v(0.97, 0.2431, 0)),
        t('Banking Transactions', v(0, 1, 0)),
        t('Receipts & Invoices', v(0, 0.85, 0.5268)),
      ],
      brands,
    );

    expect(kept.map((k) => k.topic.label)).toEqual([
      'Developer Code Reviews',
      'Banking Transactions',
      'Receipts & Invoices',
    ]);
  });

  it('domainFrequency ranks sender domains by volume', () => {
    const freq = domainFrequency([
      { fromAddr: 'a@github.com' },
      { fromAddr: 'b@github.com' },
      { fromAddr: 'c@glassdoor.com' },
      { fromAddr: null },
    ]);
    expect(freq[0]).toEqual(['github.com', 2]);
    expect(freq[1]).toEqual(['glassdoor.com', 1]);
  });

  /** Builds an EmailSummary fixture keyed by message id and sender. */
  const summary = (id: string, from: string): EmailSummary => ({
    messageId: id,
    accountId: 'acct',
    folder: 'INBOX',
    subject: 's',
    fromAddr: from,
    date: 1,
    hasAttachments: false,
    indexedAt: 1,
  });

  it('discovery pool draws from newest, historical, top-domain, and uncategorized strata, deduped', () => {
    const calls: string[] = [];
    const emails = {
      listSummaries: () => {
        calls.push('recent');
        return [summary('m1', 'r@a.com'), summary('dup', 'z@a.com')];
      },
      listSummariesRandom: () => {
        calls.push('historical');
        return [summary('m2', 'h@b.com')];
      },
      listUncategorizedSummaries: () => {
        calls.push('uncat');
        return [summary('m3', 'u@c.com')];
      },
      listSummariesByDomain: () => {
        calls.push('domain');
        return [summary('m4', 'd@github.com'), summary('dup', 'z@a.com')];
      },
    };
    const svc = new TopicDiscoveryService(
      {} as never,
      emails as never,
      {} as never,
      {} as never,
      silentLogger,
    );
    const pool = (
      svc as unknown as { buildDiscoveryPool(a: string, d: string[]): EmailSummary[] }
    ).buildDiscoveryPool('acct', ['github.com']);

    expect(calls).toEqual(expect.arrayContaining(['recent', 'historical', 'uncat', 'domain']));
    expect(pool.map((e) => e.messageId).sort()).toEqual(['dup', 'm1', 'm2', 'm3', 'm4']);
  });

  it('discovery keeps the existing taxonomy when the model yields too few concrete categories', async () => {
    const vagueTopics = JSON.stringify({
      topics: [
        { label: 'Notifications', description: 'x' },
        { label: 'Service Updates', description: 'y' },
        { label: 'Support Requests', description: 'z' },
      ],
    });
    const llm = { chat: vi.fn(async () => vagueTopics), embed: vi.fn() };
    const emails = {
      listSummaries: () =>
        Array.from({ length: 200 }, (_, i) => summary(`m${i}`, `u${i % 5}@x.com`)),
      listSummariesRandom: () => [],
      listUncategorizedSummaries: () => [],
      listSummariesByDomain: () => [],
      listSenders: () => Array.from({ length: 250 }, (_, i) => ({ fromAddr: `u${i % 5}@x.com` })),
    };
    const categories = { reconcileAutoCategories: vi.fn(() => 0) };
    const svc = new TopicDiscoveryService(
      llm as never,
      emails as never,
      { listForAccount: () => [], search: () => [] } as never,
      categories as never,
      silentLogger,
    );

    const result = await svc.discover('acct', 'bge-m3', 'gen');

    expect(result.status).toBe('insufficient_categories');
    expect(result.topicsCreated).toBe(0);
    expect(categories.reconcileAutoCategories).not.toHaveBeenCalled();
    expect(llm.embed).not.toHaveBeenCalled();
  });

  it('discovery surfaces a non-retryable LLM error instead of masking it as insufficient', async () => {
    const llm = {
      chat: vi.fn(async () => {
        throw new LlmApiError(404, '/chat/completions', 'model "qwen3:4b" does not exist');
      }),
      embed: vi.fn(),
    };
    const emails = {
      listSummaries: () =>
        Array.from({ length: 200 }, (_, i) => summary(`m${i}`, `u${i % 5}@x.com`)),
      listSummariesRandom: () => [],
      listUncategorizedSummaries: () => [],
      listSummariesByDomain: () => [],
      listSenders: () => Array.from({ length: 250 }, (_, i) => ({ fromAddr: `u${i % 5}@x.com` })),
    };
    const categories = { reconcileAutoCategories: vi.fn() };
    const svc = new TopicDiscoveryService(
      llm as never,
      emails as never,
      { listForAccount: () => [], search: () => [] } as never,
      categories as never,
      silentLogger,
    );

    await expect(svc.discover('a', 'bge-m3', 'gen')).rejects.toThrow(/does not exist/);
    expect(categories.reconcileAutoCategories).not.toHaveBeenCalled();
  });
});

describe('TriageService.classify (tolerant reasoning)', () => {
  const email = {
    messageId: 'm1',
    subject: 'Are you still interested?',
    fromAddr: 'jobs@alerts.jobot.com',
    body: 'Apply now to these roles.',
    bodyFormat: 'text',
  };

  it('truncates an over-long reasoning instead of failing the classification', async () => {
    const llm = {
      chat: vi.fn(async () => JSON.stringify({ bucket: 'spam', reasoning: 'x'.repeat(400) })),
    };
    const svc = new TriageService(llm as never, silentLogger);
    const out = await svc.classify(email);
    expect(out.bucket).toBe('spam');
    expect(out.reasoning.length).toBeLessThanOrEqual(200);
  });

  it('tolerates a missing reasoning field, keeping the bucket', async () => {
    const llm = { chat: vi.fn(async () => JSON.stringify({ bucket: 'personal' })) };
    const svc = new TriageService(llm as never, silentLogger);
    const out = await svc.classify({ ...email, fromAddr: 'friend@gmail.com' });
    expect(out.bucket).toBe('personal');
    expect(out.reasoning).toBe('');
  });

  it('still rejects an invalid bucket (retryable)', async () => {
    const llm = {
      chat: vi.fn(async () => JSON.stringify({ bucket: 'important', reasoning: 'x' })),
    };
    const svc = new TriageService(llm as never, silentLogger);
    await expect(svc.classify(email)).rejects.toThrow();
  });

  it('salvages the bucket from a truncated, rambling reply that never closes the JSON', async () => {
    const truncated =
      '{"bucket": "summarize", "reasoning": "LinkedIn connection request, then rambling that never closes';
    const llm = { chat: vi.fn(async () => truncated) };
    const svc = new TriageService(llm as never, silentLogger);
    const out = await svc.classify(email);
    expect(out.bucket).toBe('summarize');
    expect(out.reasoning).toBe('');
  });

  /** Runs TriageService.classify against a stub LLM that returns the given payload. */
  const classifyWith = async (payload: Record<string, unknown>, input = email) => {
    const llm = { chat: vi.fn(async () => JSON.stringify(payload)) };
    return new TriageService(llm as never, silentLogger).classify(input);
  };

  it('parses full structured metadata, converting deadlineHours to a future timestamp', async () => {
    const before = Date.now();
    const out = await classifyWith({
      bucket: 'urgent',
      actionRequired: true,
      needsReply: true,
      deadlineHours: 24,
      importanceScore: 90,
      suggestedAction: 'Submit the report',
      shortSummary: 'Report due soon.',
      reasoning: 'Deadline within a day.',
    });
    expect(out.bucket).toBe('urgent');
    expect(out.metadata.actionRequired).toBe(true);
    expect(out.metadata.needsReply).toBe(true);
    expect(out.metadata.importanceScore).toBe(90);
    expect(out.metadata.suggestedAction).toBe('Submit the report');
    expect(out.metadata.deadlineAt).toBeGreaterThan(before + 23 * 3_600_000);
  });

  it('fills safe defaults when metadata is missing (only a bucket)', async () => {
    const out = await classifyWith({ bucket: 'summarize', reasoning: 'x' });
    expect(out.metadata.actionRequired).toBe(false);
    expect(out.metadata.needsReply).toBe(false);
    expect(out.metadata.deadlineAt).toBeNull();
    expect(out.metadata.importanceScore).toBe(35);
    expect(out.metadata.suggestedAction).toBeNull();
  });

  it('keeps deadlineAt null when no explicit deadline is given', async () => {
    const out = await classifyWith({ bucket: 'summarize', deadlineHours: null, reasoning: 'x' });
    expect(out.metadata.deadlineAt).toBeNull();
  });

  it('ignores an absurd deadlineHours rather than inventing a deadline', async () => {
    const out = await classifyWith({ bucket: 'summarize', deadlineHours: 100000, reasoning: 'x' });
    expect(out.metadata.deadlineAt).toBeNull();
  });

  it('truncates an over-long suggestedAction and shortSummary', async () => {
    const out = await classifyWith({
      bucket: 'personal',
      suggestedAction: 'a'.repeat(300),
      shortSummary: 'b'.repeat(400),
      reasoning: 'x',
    });
    expect(out.metadata.suggestedAction!.length).toBeLessThanOrEqual(140);
    expect(out.metadata.shortSummary!.length).toBeLessThanOrEqual(200);
  });

  it('coerces loosely-typed metadata (strings) without failing', async () => {
    const out = await classifyWith({
      bucket: 'urgent',
      actionRequired: 'true',
      importanceScore: '77',
      deadlineHours: '5',
      reasoning: 'x',
    });
    expect(out.metadata.actionRequired).toBe(true);
    expect(out.metadata.importanceScore).toBe(77);
    expect(out.metadata.deadlineAt).not.toBeNull();
  });

  it('passes through a GitHub notification with no direct action (summarize, no action)', async () => {
    const out = await classifyWith({
      bucket: 'summarize',
      actionRequired: false,
      reasoning: 'CI notification.',
    });
    expect(out.bucket).toBe('summarize');
    expect(out.metadata.actionRequired).toBe(false);
  });

  it('passes through a direct review request (action required)', async () => {
    const out = await classifyWith({
      bucket: 'summarize',
      actionRequired: true,
      importanceScore: 65,
      reasoning: 'Review requested.',
    });
    expect(out.metadata.actionRequired).toBe(true);
  });

  it('passes through a suspicious-login alert (urgent, action required)', async () => {
    const out = await classifyWith({
      bucket: 'urgent',
      actionRequired: true,
      reasoning: 'Account security event.',
    });
    expect(out.bucket).toBe('urgent');
    expect(out.metadata.actionRequired).toBe(true);
  });

  it('downgrades GitHub review notifications from urgent to action-required summarize', async () => {
    const out = await classifyWith(
      {
        bucket: 'urgent',
        actionRequired: true,
        importanceScore: 95,
        reasoning: 'Requires immediate action within hours.',
      },
      {
        ...email,
        subject: '[org/repo] User requested your review on PR #42',
        fromAddr: 'GitHub <notifications@github.com>',
        body: '@user requested your review on this pull request.',
      },
    );
    expect(out.bucket).toBe('summarize');
    expect(out.metadata.actionRequired).toBe(true);
    expect(out.metadata.importanceScore).toBe(75);
    expect(out.metadata.deadlineAt).toBeNull();
  });

  it('keeps real code-platform account security alerts urgent', async () => {
    const out = await classifyWith(
      {
        bucket: 'urgent',
        actionRequired: true,
        importanceScore: 85,
        reasoning: 'Security alert.',
      },
      {
        ...email,
        subject: '[GitHub] A third-party GitHub Application has been added to your account',
        fromAddr: 'GitHub <noreply@github.com>',
        body: 'A third-party application has been added to your account. Review this access.',
      },
    );
    expect(out.bucket).toBe('urgent');
    expect(out.metadata.actionRequired).toBe(true);
    expect(out.metadata.importanceScore).toBeGreaterThanOrEqual(85);
  });

  it('does not make a seven-day GitLab token expiry urgent', async () => {
    const out = await classifyWith(
      {
        bucket: 'urgent',
        actionRequired: true,
        deadlineHours: 24 * 7,
        importanceScore: 90,
        reasoning: 'Token expires soon.',
      },
      {
        ...email,
        subject: 'Your personal access tokens will expire in 7 days or less',
        fromAddr: 'GitLab <gitlab@emse.fr>',
        body: 'Your personal access tokens will expire in 7 days or less.',
      },
    );
    expect(out.bucket).toBe('summarize');
    expect(out.metadata.actionRequired).toBe(true);
    expect(out.metadata.deadlineAt).toBeNull();
    expect(out.metadata.importanceScore).toBe(75);
  });

  it('sends current time and email sent time to the triage model', async () => {
    const llm = { chat: vi.fn(async () => JSON.stringify({ bucket: 'summarize' })) };
    const dated = { ...email, date: Date.UTC(2026, 5, 25, 8, 30) };
    await new TriageService(llm as never, silentLogger).classify(dated);
    const req = llm.chat.mock.calls[0]![0];
    const user = req.messages.find((m: { role: string }) => m.role === 'user')!;
    expect(user.content).toContain('Current time:');
    expect(user.content).toContain('Email sent time: 2026-06-25T08:30:00.000Z');
  });

  it('uses the cloud chat provider without Ollama-only thinking fields when requested', async () => {
    const llm = { chat: vi.fn(async () => JSON.stringify({ bucket: 'summarize' })) };
    await new TriageService(llm as never, silentLogger).classify(email, 'gpt-4o-mini', 'chat');
    const req = llm.chat.mock.calls[0]![0];
    expect(req.provider).toBe('chat');
    expect(req.model).toBe('gpt-4o-mini');
    expect(req.think).toBeUndefined();
    expect(req.messages[0]!.content).not.toContain('/no_think');
  });

  it('classifies priority emails in a batch and keeps results aligned by message id', async () => {
    const llm = {
      chat: vi.fn(async () =>
        JSON.stringify({
          results: [
            {
              messageId: 'm1',
              bucket: 'summarize',
              actionRequired: true,
              importanceScore: 65,
              reasoning: 'Review requested.',
            },
            {
              messageId: 'm2',
              bucket: 'spam',
              actionRequired: false,
              importanceScore: 5,
              reasoning: 'Unsolicited promo.',
            },
          ],
        }),
      ),
    };
    const out = await new TriageService(llm as never, silentLogger).classifyBatch(
      [
        { ...email, messageId: 'm1', subject: 'Review requested' },
        { ...email, messageId: 'm2', subject: 'Buy now' },
      ],
      'gpt-4o-mini',
      'chat',
    );
    expect(out.get('m1')?.bucket).toBe('summarize');
    expect(out.get('m1')?.metadata.actionRequired).toBe(true);
    expect(out.get('m2')?.bucket).toBe('spam');
    const req = llm.chat.mock.calls[0]![0];
    expect(req.provider).toBe('chat');
    expect(req.think).toBeUndefined();
  });
});

describe('CategoryImprovementService', () => {
  /** Builds an uncategorized email summary fixture keyed by id and sender. */
  const uncat = (id: string, from: string) => ({ messageId: id, subject: 's', fromAddr: from });
  const dbMock = { transaction: (fn: () => unknown) => fn };

  it('suggest filters vague and duplicate categories and applies nothing', async () => {
    const llm = {
      chat: vi.fn(async () =>
        JSON.stringify({
          newCategories: [
            { label: 'Receipts & Invoices', description: 'payment confirmations' },
            { label: 'Notifications', description: 'vague catch-all' },
            { label: 'Job Alerts', description: 'duplicate of existing' },
          ],
          merges: [],
        }),
      ),
      embed: vi.fn(async () => [0.1, 0.2]),
    };
    const uncats = ['m1', 'm2', 'm3', 'm4', 'm5'].map((id) => uncat(id, 'a@x.com'));
    const emails = { listUncategorizedSummaries: () => uncats };
    const embeddings = {
      listForAccount: () => [] as Array<{ messageId: string; vector: Float32Array }>,
      search: () => ['m1', 'm2', 'm3', 'm4'].map((messageId) => ({ messageId, distance: 0.3 })),
    };
    const categories = {
      countUncategorized: () => 50,
      listActive() {
        return this.listForAccount();
      },
      listForAccount: () => [{ id: 'c1', label: 'Job Opportunities', description: 'jobs' }],
      getCentroidEntries: () =>
        [] as Array<{
          categoryId: string;
          label: string;
          vector: Float32Array;
          emailCount: number;
        }>,
      create: vi.fn(),
      saveCentroid: vi.fn(),
    };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      emails as never,
      embeddings as never,
      categories as never,
      silentLogger,
    );

    const out = await svc.suggest('a', 'bge-m3', 'gen');

    expect(out.newCategories.map((c) => c.label)).toEqual(['Receipts & Invoices']);
    expect(categories.create).not.toHaveBeenCalled();
    expect(categories.saveCentroid).not.toHaveBeenCalled();
  });

  /** Wires a CategoryImprovementService over stubs and exposes its chat spy for provider assertions. */
  const improveChatProbe = () => {
    const chat = vi.fn(
      async (_o: {
        provider?: string;
        model?: string;
        think?: boolean;
        messages?: Array<{ content: string }>;
      }) => JSON.stringify({ newCategories: [], merges: [] }),
    );
    const llm = { chat, embed: vi.fn(async () => [0.1, 0.2]) };
    const emails = {
      listUncategorizedSummaries: () => ['m1', 'm2'].map((id) => uncat(id, 'a@x.com')),
    };
    const embeddings = {
      listForAccount: () => [] as Array<{ messageId: string; vector: Float32Array }>,
      search: () => [] as Array<{ messageId: string; distance: number }>,
    };
    const categories = {
      countUncategorized: () => 50,
      listActive() {
        return this.listForAccount();
      },
      listForAccount: () => [] as Array<{ id: string; label: string; description: string }>,
      getCentroidEntries: () =>
        [] as Array<{
          categoryId: string;
          label: string;
          vector: Float32Array;
          emailCount: number;
        }>,
      create: vi.fn(),
      saveCentroid: vi.fn(),
    };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      emails as never,
      embeddings as never,
      categories as never,
      silentLogger,
    );
    return { svc, chat };
  };

  it('routes the suggestion call to the cloud chat provider, omitting the local-only think param', async () => {
    const { svc, chat } = improveChatProbe();
    await svc.suggest('a', 'bge-m3', 'gpt-4o-mini', 'chat');
    const arg = chat.mock.calls[0]![0];
    expect(arg.provider).toBe('chat');
    expect(arg.model).toBe('gpt-4o-mini');
    expect(arg.think).toBeUndefined();
    expect(arg.messages?.[0]?.content).not.toContain('/no_think');
  });

  it('keeps the local provider and the think param by default', async () => {
    const { svc, chat } = improveChatProbe();
    await svc.suggest('a', 'bge-m3', 'gen');
    const arg = chat.mock.calls[0]![0];
    expect(arg.provider).toBe('main');
    expect(arg.think).toBe(false);
    expect(arg.messages?.[0]?.content).toContain('/no_think');
  });

  it('suggest drops a semantic duplicate of an existing category and a thinly-covered one', async () => {
    const llm = {
      chat: vi.fn(async () =>
        JSON.stringify({
          newCategories: [
            { label: 'Developer Tools & Platforms', description: 'dev platforms' },
            { label: 'Travel & Accommodation', description: 'flights and hotels' },
          ],
          merges: [],
        }),
      ),
      embed: vi.fn(async (text: string) => (text.startsWith('Developer') ? [1, 0] : [0, 1])),
    };
    const emails = {
      listUncategorizedSummaries: () => ['m1', 'm2'].map((id) => uncat(id, 'a@x.com')),
    };
    const embeddings = {
      listForAccount: () => [] as Array<{ messageId: string; vector: Float32Array }>,
      search: () => [{ messageId: 'm1', distance: 0.3 }],
    };
    const categories = {
      countUncategorized: () => 50,
      listActive() {
        return this.listForAccount();
      },
      listForAccount: () => [{ id: 'c1', label: 'Developer Code Reviews', description: 'dev' }],
      getCentroidEntries: () => [
        {
          categoryId: 'c1',
          label: 'Developer Code Reviews',
          vector: Float32Array.from([1, 0]),
          emailCount: 10,
        },
      ],
      create: vi.fn(),
      saveCentroid: vi.fn(),
    };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      emails as never,
      embeddings as never,
      categories as never,
      silentLogger,
    );

    const out = await svc.suggest('a', 'bge-m3', 'gen');

    expect(out.newCategories).toHaveLength(0);
  });

  it('suggest returns existing-category expansions for approved uncategorized clusters', async () => {
    const llm = {
      chat: vi.fn(async () =>
        JSON.stringify({
          existingCategoryExpansions: [
            {
              category: 'Job Opportunities',
              clusterNumbers: [1],
              reason: 'Recurring job alerts belong in the existing jobs category.',
            },
          ],
          newCategories: [],
          merges: [],
        }),
      ),
      embed: vi.fn(),
    };
    const uncats = ['m1', 'm2', 'm3', 'm4'].map((id) =>
      uncat(id, 'LinkedIn Job Alerts <jobalerts-noreply@linkedin.com>'),
    );
    const vec = unit({ 0: 1 });
    const embeddings = {
      listForAccount: () => uncats.map((u) => ({ messageId: u.messageId, vector: vec })),
      search: () => [] as Array<{ messageId: string; distance: number }>,
    };
    const categories = {
      countUncategorized: () => 50,
      listActive() {
        return this.listForAccount();
      },
      listForAccount: () => [
        { id: 'job', label: 'Job Opportunities', description: 'job alerts', emailCount: 10 },
      ],
      getCentroidEntries: () => [],
    };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      { listUncategorizedSummaries: () => uncats } as never,
      embeddings as never,
      categories as never,
      silentLogger,
    );

    const out = await svc.suggest('a', 'bge-m3', 'gen');

    expect(out.existingCategoryExpansions).toHaveLength(1);
    expect(out.existingCategoryExpansions[0]).toMatchObject({
      categoryId: 'job',
      categoryLabel: 'Job Opportunities',
      estimatedCount: 4,
    });
    expect(out.existingCategoryExpansions[0]!.messageIds.sort()).toEqual(['m1', 'm2', 'm3', 'm4']);
    expect(llm.chat).toHaveBeenCalledWith(expect.objectContaining({ temperature: 0 }));
  });

  it('does not report clusters as leave-uncategorized when they are already suggested for filing', async () => {
    const llm = {
      chat: vi.fn(async () =>
        JSON.stringify({
          existingCategoryExpansions: [
            {
              category: 'Job Opportunities',
              clusterNumbers: [1],
              reason: 'Recurring job alerts belong in the existing jobs category.',
            },
          ],
          leaveUncategorized: {
            clusterNumbers: [1],
            reason: 'model accidentally repeated a used cluster',
          },
        }),
      ),
      embed: vi.fn(),
    };
    const uncats = ['m1', 'm2', 'm3', 'm4'].map((id) =>
      uncat(id, 'LinkedIn Job Alerts <jobalerts-noreply@linkedin.com>'),
    );
    const vec = unit({ 0: 1 });
    const embeddings = {
      listForAccount: () => uncats.map((u) => ({ messageId: u.messageId, vector: vec })),
      search: () => [] as Array<{ messageId: string; distance: number }>,
    };
    const categories = {
      countUncategorized: () => 50,
      listActive() {
        return this.listForAccount();
      },
      listForAccount: () => [
        { id: 'job', label: 'Job Opportunities', description: 'job alerts', emailCount: 10 },
      ],
      getCentroidEntries: () => [],
    };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      { listUncategorizedSummaries: () => uncats } as never,
      embeddings as never,
      categories as never,
      silentLogger,
    );

    const out = await svc.suggest('a', 'bge-m3', 'gen');

    expect(out.existingCategoryExpansions).toHaveLength(1);
    expect(out.leaveUncategorized).toBeUndefined();
  });

  it('converts duplicate new-category suggestions into existing-category expansions', async () => {
    const llm = {
      chat: vi.fn(async () =>
        JSON.stringify({
          newCategories: [
            {
              label: 'Job Alerts',
              description: 'job alert emails',
              clusterNumbers: [1],
            },
          ],
          merges: [],
        }),
      ),
      embed: vi.fn(),
    };
    const uncats = ['m1', 'm2', 'm3', 'm4'].map((id) =>
      uncat(id, 'Glassdoor Jobs <noreply@glassdoor.com>'),
    );
    const vec = unit({ 0: 1 });
    const embeddings = {
      listForAccount: () => uncats.map((u) => ({ messageId: u.messageId, vector: vec })),
      search: () => [] as Array<{ messageId: string; distance: number }>,
    };
    const categories = {
      countUncategorized: () => 50,
      listActive() {
        return this.listForAccount();
      },
      listForAccount: () => [
        { id: 'job', label: 'Job Opportunities', description: 'job alerts', emailCount: 10 },
      ],
      getCentroidEntries: () => [],
    };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      { listUncategorizedSummaries: () => uncats } as never,
      embeddings as never,
      categories as never,
      silentLogger,
    );

    const out = await svc.suggest('a', 'bge-m3', 'gen');

    expect(out.newCategories).toHaveLength(0);
    expect(out.existingCategoryExpansions).toHaveLength(1);
    expect(out.existingCategoryExpansions[0]!.categoryId).toBe('job');
  });

  it('suggest returns backlog diagnostics when no new categories but emails fit existing ones', async () => {
    const llm = {
      chat: vi.fn(async () => JSON.stringify({ newCategories: [], merges: [] })),
      embed: vi.fn(),
    };
    const uncats = ['m1', 'm2'].map((id) => uncat(id, 'a@x.com'));
    const vec = Float32Array.from([1, 0]);
    const embeddings = {
      listForAccount: () => uncats.map((u) => ({ messageId: u.messageId, vector: vec })),
      search: () => [] as Array<{ messageId: string; distance: number }>,
    };
    const categories = {
      countUncategorized: () => 50,
      listActive() {
        return this.listForAccount();
      },
      listForAccount: () => [{ id: 'c1', label: 'Banking', description: 'd' }],
      getCentroidEntries: () => [
        { categoryId: 'c1', label: 'Banking', vector: vec, emailCount: 10 },
      ],
    };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      { listUncategorizedSummaries: () => uncats } as never,
      embeddings as never,
      categories as never,
      silentLogger,
    );

    const out = await svc.suggest('a', 'bge-m3', 'gen');

    expect(out.newCategories).toHaveLength(0);
    expect(out.diagnostics?.existingCategoriesLikelyCoverBacklog).toBe(true);
  });

  it('suggest returns nothing when the uncategorized backlog is small', async () => {
    const llm = { chat: vi.fn(), embed: vi.fn() };
    const categories = { countUncategorized: () => 5, listForAccount: () => [] };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      {} as never,
      {} as never,
      categories as never,
      silentLogger,
    );

    const out = await svc.suggest('a', 'bge-m3', 'gen');

    expect(out.newCategories).toHaveLength(0);
    expect(llm.chat).not.toHaveBeenCalled();
  });

  it('apply creates approved categories and merges, skipping vague labels', async () => {
    const llm = { embed: vi.fn(async () => [0.1, 0.2]) };
    const embeddings = {
      listForAccount: () => [] as Array<{ messageId: string; vector: Float32Array }>,
      search: () => [] as Array<{ messageId: string; distance: number }>,
    };
    const categories = {
      listForAccount: () => [] as Array<{ label: string }>,
      create: vi.fn((i: { label: string }) => ({ id: `new-${i.label}`, accountId: 'a' })),
      saveCentroid: vi.fn(),
      findById: (id: string) => ({ id, accountId: 'a', status: 'active' }),
      mergeInto: vi.fn(() => ({ reassigned: 1 })),
    };
    const svc = new CategoryImprovementService(
      dbMock as never,
      llm as never,
      {} as never,
      embeddings as never,
      categories as never,
      silentLogger,
    );

    const out = await svc.apply('a', 'bge-m3', {
      newCategories: [
        { label: 'Receipts & Invoices', description: 'd' },
        { label: 'Notifications', description: 'vague, must be skipped' },
      ],
      merges: [{ sourceId: 'c2', targetId: 'c1' }],
    });

    expect(out.created).toBe(1);
    expect(out.merged).toBe(1);
    expect(categories.create).toHaveBeenCalledTimes(1);
    expect(categories.saveCentroid).toHaveBeenCalledTimes(1);
    expect(categories.mergeInto).toHaveBeenCalledWith('c2', 'c1');
  });
});

describe('salvageSuggestions (tolerant Improve parsing)', () => {
  it('parses existing-category expansions with cluster references', () => {
    const raw = JSON.stringify({
      existingCategoryExpansions: [
        { category: 'Job Opportunities', clusterNumbers: [1, '2'], reason: 'fits jobs' },
      ],
    });
    const out = salvageSuggestions(raw);
    expect(out?.existingCategoryExpansions).toEqual([
      { category: 'Job Opportunities', clusterNumbers: [1, 2], reason: 'fits jobs' },
    ]);
  });

  it('trims an over-long description instead of discarding the whole set', () => {
    const raw = JSON.stringify({
      newCategories: [{ label: 'Travel', description: 'x'.repeat(400) }],
      merges: [],
    });
    const out = salvageSuggestions(raw);
    expect(out?.newCategories).toHaveLength(1);
    expect(out?.newCategories[0]!.description.length).toBe(300);
  });

  it('drops only the unusable item, keeping valid suggestions', () => {
    const raw = JSON.stringify({
      newCategories: [{ label: 'Banking', description: 'd' }, { description: 'no label' }],
    });
    expect(salvageSuggestions(raw)?.newCategories.map((c) => c.label)).toEqual(['Banking']);
  });

  it('returns null only when the reply is not parseable JSON, never on a valid empty set', () => {
    expect(salvageSuggestions('sorry, I could not help')).toBeNull();
    expect(salvageSuggestions('<think></think>{"newCategories":[]}')).toEqual({
      existingCategoryExpansions: [],
      newCategories: [],
      merges: [],
    });
  });

  it('reads suggestions returned as a bare top-level array (unwrapped)', () => {
    const raw = JSON.stringify([{ label: 'Invoices', description: 'bills' }]);
    expect(salvageSuggestions(raw)?.newCategories.map((c) => c.label)).toEqual(['Invoices']);
  });

  it('keeps an empty description empty instead of echoing the label', () => {
    const raw = JSON.stringify({ newCategories: [{ label: 'Banking', description: '   ' }] });
    expect(salvageSuggestions(raw)?.newCategories[0]).toEqual({
      label: 'Banking',
      description: '',
      clusterNumbers: [],
    });
  });
});

describe('rateLimitDelayMs', () => {
  it('honors a seconds delay from the 429 body, padded', () => {
    expect(rateLimitDelayMs('Rate limit reached. Please try again in 1.095s. Visit ...', 0)).toBe(
      1345,
    );
  });
  it('floors a tiny delay to 500ms', () => {
    expect(rateLimitDelayMs('Please try again in 98ms.', 0)).toBe(500);
  });
  it('falls back to clamped exponential backoff when the body has no hint', () => {
    expect(rateLimitDelayMs('rate limited', 0)).toBe(1000);
    expect(rateLimitDelayMs('rate limited', 3)).toBe(8000);
    expect(rateLimitDelayMs('rate limited', 20)).toBe(30_000);
  });
});

describe('LlmClient 429 backoff', () => {
  it('retries a rate-limited request and then succeeds', async () => {
    let calls = 0;
    const fetchMock = vi.fn(async () => {
      calls += 1;
      if (calls === 1) {
        return new Response(
          '{"error":{"message":"Rate limit reached. Please try again in 10ms.","code":"rate_limit_exceeded"}}',
          { status: 429 },
        );
      }
      return new Response(
        JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' } }] }),
        { status: 200, headers: { 'content-type': 'application/json' } },
      );
    });
    vi.stubGlobal('fetch', fetchMock);
    try {
      const client = createLlmClient(
        () =>
          ({
            baseUrl: 'http://local/v1',
            embeddingModel: 'bge-m3',
            generationModel: 'g',
            embeddingDimensions: 1024,
            chatRerank: true,
            categorizeUseChatProvider: false,
          }) as never,
      );
      const answer = await client.chat({ messages: [{ role: 'user', content: 'hi' }] });
      expect(answer).toBe('ok');
      expect(calls).toBe(2);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe('config PATCH llm merge', () => {
  it('a partial llm patch preserves keys the caller did not send', () => {
    const stored = LlmConfigSchema.parse({
      baseUrl: 'http://192.168.1.50:1234/v1',
      embeddingDimensions: 768,
      chatRerank: false,
    });
    const rawLlm: Record<string, unknown> = {
      generationModel: 'qwen3:8b',
      categorizeUseChatProvider: true,
      priorityUseChatProvider: true,
    };
    const parsed = LlmConfigSchema.partial().parse(rawLlm) as Record<string, unknown>;
    const llmPatch: Record<string, unknown> = {};
    for (const key of Object.keys(rawLlm)) {
      if (key in parsed) llmPatch[key] = parsed[key];
    }
    Object.assign(stored, llmPatch);

    expect(stored.baseUrl).toBe('http://192.168.1.50:1234/v1');
    expect(stored.embeddingDimensions).toBe(768);
    expect(stored.chatRerank).toBe(false);
    expect(stored.generationModel).toBe('qwen3:8b');
    expect(stored.categorizeUseChatProvider).toBe(true);
    expect(stored.priorityUseChatProvider).toBe(true);
  });
});

describe('features config flag (Phase 4)', () => {
  it('defaults features.multiPrototypeCategories to false', () => {
    const config = AppConfigSchema.parse({});
    expect(config.features.multiPrototypeCategories).toBe(false);
  });

  it('parses a legacy config that has no features block, defaulting the flag false', () => {
    const legacy = {
      version: 1,
      locale: 'en',
      autoIndex: true,
      indexedFolders: [],
      llm: {
        baseUrl: 'http://localhost:11434/v1',
        embeddingModel: 'bge-m3',
        generationModel: 'qwen3:8b',
      },
      authToken: 'tok',
    };
    const config = AppConfigSchema.parse(legacy);
    expect(config.features.multiPrototypeCategories).toBe(false);
    // The rest of the legacy config still parses unchanged.
    expect(config.autoIndex).toBe(true);
    expect(config.llm.embeddingModel).toBe('bge-m3');
  });

  it('respects an explicit features.multiPrototypeCategories=true and keeps the flag out of llm', () => {
    const config = AppConfigSchema.parse({ features: { multiPrototypeCategories: true } });
    expect(config.features.multiPrototypeCategories).toBe(true);
    expect('multiPrototypeCategories' in config.llm).toBe(false);
  });
});

describe('redactConfig (M8)', () => {
  it('strips authToken, llm.apiKey, and imap while keeping public fields', () => {
    const config = AppConfigSchema.parse({
      authToken: 'super-secret-token',
      llm: {
        baseUrl: 'http://localhost:11434/v1',
        apiKey: 'PROVIDER_KEY',
        embeddingModel: 'bge-m3',
        generationModel: 'qwen3:8b',
        embeddingDimensions: 1024,
        chatBaseUrl: 'https://api.openai.com/v1',
        chatApiKey: 'CLOUD_CHAT_KEY',
        priorityUseChatProvider: true,
      },
      imap: { host: 'mail', user: 'u', password: 'p', port: 993, tls: true },
    });

    const safe = redactConfig(config) as Record<string, unknown>;
    const json = JSON.stringify(safe);
    const safeLlm = safe.llm as Record<string, unknown>;

    expect(json).not.toContain('super-secret-token');
    expect(json).not.toContain('PROVIDER_KEY');
    expect(json).not.toContain('CLOUD_CHAT_KEY');
    expect(json).not.toContain('password');
    expect(safe.authToken).toBeUndefined();
    expect(safe.imap).toBeUndefined();
    expect(safeLlm.apiKey).toBeUndefined();
    expect(safeLlm.chatApiKey).toBeUndefined();
    expect(safeLlm.embeddingModel).toBe('bge-m3');
    expect(safeLlm.chatBaseUrl).toBe('https://api.openai.com/v1');
    expect(safeLlm.priorityUseChatProvider).toBe(true);
    expect(safeLlm.chatApiKeySet).toBe(true);
    expect(safe.locale).toBe('en');
  });

  it('enforces chat tuning bounds and accepts null to clear back to default', () => {
    const base = {
      baseUrl: 'http://localhost:11434/v1',
      embeddingModel: 'bge-m3',
      generationModel: 'qwen3:8b',
      embeddingDimensions: 1024,
    };
    expect(
      AppConfigSchema.parse({ llm: { ...base, chatTopK: null, chatSnippetChars: null } }).llm
        .chatTopK,
    ).toBeNull();
    expect(
      AppConfigSchema.parse({ llm: { ...base, chatTopK: 12, chatSnippetChars: 1400 } }).llm
        .chatSnippetChars,
    ).toBe(1400);
    expect(() => AppConfigSchema.parse({ llm: { ...base, chatSnippetChars: 50 } })).toThrow();
    expect(() => AppConfigSchema.parse({ llm: { ...base, chatTopK: 99 } })).toThrow();
  });
});

describe('CategorizationService.categorizeBatch scored count (M3)', () => {
  it('reports every scored email and only matched ones in the map', () => {
    const v = (a: number, b: number): Float32Array => {
      const arr = new Float32Array(EMBEDDING_DIM);
      arr[0] = a;
      arr[1] = b;
      return arr;
    };
    const entries = [{ categoryId: 'c1', label: 'L', vector: v(1, 0), emailCount: 1 }];
    const fakeCategories = {
      getCentroidEntries: () => entries,
      getEffectivePrototypeEntries: () => entries.map((e) => ({ ...e, prototypeIndex: 0 })),
    };
    const fakeEmbeddings = {
      listForAccount: () => [
        { messageId: 'm1', vector: v(1, 0) },
        { messageId: 'm2', vector: v(0, 1) },
        { messageId: 'm3', vector: v(1, 0.1) },
      ],
    };
    const svc = new CategorizationService(fakeCategories as never, fakeEmbeddings as never);

    const { matches, scored } = svc.categorizeBatch('acct', 'bge-m3');
    expect(scored).toBe(3);
    expect(matches.size).toBe(2);
    expect(matches.has('m1')).toBe(true);
    expect(matches.has('m3')).toBe(true);
    expect(matches.has('m2')).toBe(false);
  });
});

describe('CategorizationService nearest-prototype matching (Phase 4)', () => {
  const pvec = (a: number, b: number): Float32Array => {
    const x = new Float32Array(EMBEDDING_DIM);
    x[0] = a;
    x[1] = b;
    return x;
  };
  // Category A: aggregate near axis 0, plus two sub-prototypes (one near axis 0, one near axis 1).
  const aggregate = [
    { categoryId: 'A', label: 'A', vector: pvec(1, 0), emailCount: 8, prototypeIndex: 0 },
  ];
  const subs = [
    { categoryId: 'A', label: 'A', vector: pvec(1, 0), emailCount: 5, prototypeIndex: 1 },
    { categoryId: 'A', label: 'A', vector: pvec(0, 1), emailCount: 3, prototypeIndex: 2 },
  ];
  const makeCategories = () => ({
    getCentroidEntries: () => aggregate.map(({ prototypeIndex: _p, ...e }) => e),
    getEffectivePrototypeEntries: (_a: string, _m: string, multi: boolean) =>
      multi ? subs : aggregate,
  });

  it('flag on rescues an email near a sub-prototype that the aggregate alone would miss', () => {
    const cats = makeCategories();
    const emb = { getEmbedding: () => pvec(0, 1) }; // near sub-prototype 2, far from the aggregate

    // Flag off: only the aggregate (axis 0) is compared; axis-1 email is beyond the hard threshold.
    const off = new CategorizationService(cats as never, emb as never, () => false);
    expect(off.categorize('m', 'acct', 'bge-m3')).toHaveLength(0);

    // Flag on: the axis-1 sub-prototype matches; A is assigned once with the winning prototype cosine.
    const on = new CategorizationService(cats as never, emb as never, () => true);
    const matches = on.categorize('m', 'acct', 'bge-m3');
    expect(matches).toHaveLength(1);
    expect(matches[0]!.categoryId).toBe('A');
    expect(matches[0]!.confidence).toBeGreaterThan(0.9);
  });

  it('flag on: a category with several matching prototypes appears once, ranked by its nearest', () => {
    const cats = {
      getEffectivePrototypeEntries: () => [
        { categoryId: 'A', label: 'A', vector: pvec(1, 0), emailCount: 5, prototypeIndex: 1 },
        { categoryId: 'A', label: 'A', vector: pvec(0.9, 0.1), emailCount: 3, prototypeIndex: 2 },
        { categoryId: 'B', label: 'B', vector: pvec(0, 1), emailCount: 4, prototypeIndex: 1 },
      ],
    };
    const emb = { getEmbedding: () => pvec(1, 0) };
    const svc = new CategorizationService(cats as never, emb as never, () => true);
    const matches = svc.categorize('m', 'acct', 'bge-m3', { maxLabels: 5, relativeMargin: 2 });
    // A appears exactly once (deduped to its nearest prototype), and it is the top match.
    expect(matches.filter((x) => x.categoryId === 'A')).toHaveLength(1);
    expect(matches[0]!.categoryId).toBe('A');
  });
});

describe('TriageOrchestrator priority enrichment', () => {
  it('does NOT start or clear triage data when a run is already in progress', async () => {
    let releaseClassify: () => void = () => {};
    const classifyGate = new Promise<void>((resolve) => {
      releaseClassify = resolve;
    });

    const service = {
      classify: vi.fn(async () => {
        await classifyGate;
        return { bucket: 'urgent' as const, reasoning: 'r' };
      }),
    };
    let servedBatch = false;
    const repo = {
      countPendingTriage: vi.fn(() => (servedBatch ? 0 : 1)),
      findPendingTriageEmails: vi.fn(() => {
        if (servedBatch) return [];
        servedBatch = true;
        return [
          { messageId: 'm1', subject: 's', fromAddr: 'f', date: 1, body: 'b', bodyFormat: 'text' },
        ];
      }),
      upsert: vi.fn(),
      clearForAccount: vi.fn(),
    };
    const failures = {
      countPermanentlyFailed: vi.fn(() => 0),
      permanentlyFailedIds: vi.fn(() => []),
      clearForAccount: vi.fn(),
      clearFailure: vi.fn(),
      recordFailure: vi.fn(() => 1),
    };

    const orch = new TriageOrchestrator(
      service as never,
      repo as never,
      failures as never,
      silentLogger,
    );

    const first = orch.start('acct', 'model');
    expect(first.started).toBe(true);
    await Promise.resolve();

    const second = orch.start('acct', 'model', { force: true });
    expect(second.started).toBe(false);
    expect(repo.clearForAccount).not.toHaveBeenCalled();

    releaseClassify();
    await vi.waitFor(() => expect(orch.getProgress().status).toBe('completed'));
  });

  it('force reclassifies without deleting triage rows', () => {
    const service = { classify: vi.fn() };
    const repo = {
      countPendingTriage: vi.fn(() => 0),
      findPendingTriageEmails: vi.fn(() => []),
      upsert: vi.fn(),
      clearForAccount: vi.fn(),
    };
    const failures = {
      countPermanentlyFailed: vi.fn(() => 0),
      permanentlyFailedIds: vi.fn(() => []),
      clearForAccount: vi.fn(),
      clearFailure: vi.fn(),
      recordFailure: vi.fn(() => 1),
    };
    const orch = new TriageOrchestrator(
      service as never,
      repo as never,
      failures as never,
      silentLogger,
    );
    orch.start('acct', 'model', { force: true });
    expect(repo.clearForAccount).not.toHaveBeenCalled();
    expect(failures.clearForAccount).toHaveBeenCalledTimes(1);
    expect(repo.countPendingTriage).toHaveBeenCalledWith('acct', true, expect.any(Set));
  });

  it('uses batched classification for priority throughput', async () => {
    const result = {
      bucket: 'summarize' as const,
      reasoning: 'r',
      metadata: {
        actionRequired: false,
        needsReply: false,
        deadlineAt: null,
        importanceScore: 35,
        suggestedAction: null,
        shortSummary: null,
        confidence: null,
      },
    };
    const service = {
      classifyBatch: vi.fn(
        async (emails: Array<{ messageId: string }>) =>
          new Map(emails.map((e) => [e.messageId, result])),
      ),
      classify: vi.fn(),
    };
    let servedBatch = false;
    const repo = {
      countPendingTriage: vi.fn(() => (servedBatch ? 0 : 2)),
      findPendingTriageEmails: vi.fn(() => {
        if (servedBatch) return [];
        servedBatch = true;
        return [
          { messageId: 'm1', subject: 's1', fromAddr: 'f', date: 1, body: 'b', bodyFormat: 'text' },
          { messageId: 'm2', subject: 's2', fromAddr: 'f', date: 2, body: 'b', bodyFormat: 'text' },
        ];
      }),
      upsert: vi.fn(),
      clearForAccount: vi.fn(),
    };
    const failures = {
      countPermanentlyFailed: vi.fn(() => 0),
      permanentlyFailedIds: vi.fn(() => []),
      clearForAccount: vi.fn(),
      clearFailure: vi.fn(),
      recordFailure: vi.fn(() => 1),
    };

    const orch = new TriageOrchestrator(
      service as never,
      repo as never,
      failures as never,
      silentLogger,
    );
    expect(orch.start('acct', 'model').started).toBe(true);
    await vi.waitFor(() => expect(orch.getProgress().status).toBe('completed'));

    expect(service.classifyBatch).toHaveBeenCalledTimes(1);
    expect(service.classify).not.toHaveBeenCalled();
    expect(repo.upsert).toHaveBeenCalledTimes(2);
  });
});

describe('LlmCategorizeOrchestrator.start (retryUncategorized guard + messageIds scoping)', () => {
  /** Builds the repository and service stubs an orchestrator run depends on, seeded with embeddings. */
  const buildMocks = (embedded: Array<{ messageId: string; vector: Float32Array }>) => ({
    categorizer: { decideBatch: vi.fn() },
    emails: {
      listSenders: vi.fn(() => []),
      findById: vi.fn((id: string) => ({
        messageId: id,
        subject: 's',
        fromAddr: 'f',
        body: 'b',
        bodyFormat: 'text',
      })),
    },
    embeddings: {
      countForModel: vi.fn(() => embedded.length),
      listForAccount: vi.fn(() => embedded),
    },
    categories: {
      listActive: vi.fn(() => [{ id: 'c1', label: 'A', description: null }]),
      listForAccount: vi.fn(() => [{ id: 'c1', label: 'A', description: null }]),
      getLlmProtectedMessageIds: vi.fn(() => new Set<string>()),
      getNoneDecisionIds: vi.fn(() => new Set<string>()),
      getUserAssignedMessageIds: vi.fn(() => new Set<string>()),
      clearNoneDecisions: vi.fn(),
      getCentroidEntries: vi.fn(() => []),
      getEffectivePrototypeEntries: vi.fn(() => []),
      getUserCorrectionExamples: vi.fn(() => []),
      bulkReplaceForCluster: vi.fn(),
      recordNoneDecisions: vi.fn(),
      clearDecisions: vi.fn(),
    },
  });
  /** Builds a jobs-repository stub with no prior persisted run. */
  const jobsMock = () => ({
    markRunningInterrupted: vi.fn(() => 0),
    getMostRecent: vi.fn(() => null),
    save: vi.fn(),
    get: vi.fn(() => null),
  });
  /** Constructs an orchestrator wired to the given mocks plus a fresh jobs stub. */
  const orchestrator = (m: ReturnType<typeof buildMocks>) =>
    new LlmCategorizeOrchestrator(
      m.categorizer as never,
      m.emails as never,
      m.embeddings as never,
      m.categories as never,
      jobsMock() as never,
      silentLogger,
    );

  it('clears none-decisions on retryUncategorized when idle, but NEVER when already running', () => {
    const m = buildMocks([{ messageId: 'm1', vector: new Float32Array([1, 0]) }]);
    m.categorizer.decideBatch.mockReturnValue(new Promise(() => {}));
    const orch = orchestrator(m);

    expect(orch.start('a', 'gen', 'embed', { retryUncategorized: true }).started).toBe(true);
    expect(m.categories.clearNoneDecisions).toHaveBeenCalledTimes(1);

    expect(orch.start('a', 'gen', 'embed', { retryUncategorized: true }).started).toBe(false);
    expect(m.categories.clearNoneDecisions).toHaveBeenCalledTimes(1);
  });

  it('returns immediately in the preparing phase, before the deferred clustering runs', () => {
    const m = buildMocks([{ messageId: 'm1', vector: new Float32Array([1, 0]) }]);
    m.categorizer.decideBatch.mockResolvedValue([['c1']]);
    const orch = orchestrator(m);

    const res = orch.start('a', 'gen', 'embed');

    expect(res.started).toBe(true);
    expect(orch.getProgress().phase).toBe('preparing');
  });

  it('honors a stop requested during preparing, skipping clustering and any model call', async () => {
    const m = buildMocks([{ messageId: 'm1', vector: new Float32Array([1, 0]) }]);
    const orch = orchestrator(m);

    orch.start('a', 'gen', 'embed');
    expect(orch.stop()).toBe(true);

    await vi.waitFor(() => expect(orch.getProgress().status).toBe('stopped'));
    expect(m.categorizer.decideBatch).not.toHaveBeenCalled();
  });

  it('messageIds scopes the run to only those emails, not the whole account', async () => {
    const m = buildMocks([
      { messageId: 'm1', vector: new Float32Array([1, 0]) },
      { messageId: 'm2', vector: new Float32Array([0, 1]) },
    ]);
    m.categorizer.decideBatch.mockResolvedValue([['c1']]);
    const orch = orchestrator(m);

    orch.start('a', 'gen', 'embed', { messageIds: ['m1'] });
    await vi.waitFor(() => expect(m.categorizer.decideBatch).toHaveBeenCalled());

    const entries = m.categorizer.decideBatch.mock.calls[0]![0] as Array<unknown>;
    expect(entries).toHaveLength(1);
    expect(m.emails.findById).toHaveBeenCalledWith('m1', 'a');
    expect(m.emails.findById).not.toHaveBeenCalledWith('m2', 'a');
  });

  it('getProgress(accountId) returns that account own persisted job, not the global last run', () => {
    const m = buildMocks([]);
    const persisted: LlmCategorizeProgress = {
      status: 'interrupted',
      accountId: 'b',
      modelId: 'q',
      total: 10,
      processed: 4,
      assigned: 4,
      uncategorized: 0,
      failed: 0,
      clusters: 5,
      clustersProcessed: 2,
      gatedClusters: 0,
      llmCalls: 0,
    };
    const jobs = {
      markRunningInterrupted: vi.fn(() => 0),
      getMostRecent: vi.fn(() => null),
      save: vi.fn(),
      get: vi.fn((id: string) => (id === 'b' ? persisted : null)),
    };
    const orch = new LlmCategorizeOrchestrator(
      m.categorizer as never,
      m.emails as never,
      m.embeddings as never,
      m.categories as never,
      jobs as never,
      silentLogger,
    );
    expect(orch.getProgress('b').status).toBe('interrupted');
    expect(orch.getProgress('b').processed).toBe(4);
    expect(jobs.get).toHaveBeenCalledWith('b');
  });

  it('routes decisions to the cloud chat provider when the run opts in, else local', async () => {
    const cloud = buildMocks([{ messageId: 'm1', vector: new Float32Array([1, 0]) }]);
    cloud.categorizer.decideBatch.mockResolvedValue([['c1']]);
    orchestrator(cloud).start('a', 'gpt-4o-mini', 'embed', { provider: 'chat' });
    await vi.waitFor(() => expect(cloud.categorizer.decideBatch).toHaveBeenCalled());
    const cloudArgs = cloud.categorizer.decideBatch.mock.calls[0]!;
    expect(cloudArgs[2]).toBe('gpt-4o-mini');
    expect(cloudArgs[3]).toBe('chat');

    const local = buildMocks([{ messageId: 'm1', vector: new Float32Array([1, 0]) }]);
    local.categorizer.decideBatch.mockResolvedValue([['c1']]);
    orchestrator(local).start('a', 'qwen3:8b', 'embed');
    await vi.waitFor(() => expect(local.categorizer.decideBatch).toHaveBeenCalled());
    expect(local.categorizer.decideBatch.mock.calls[0]![3]).toBe('main');
  });

  it('counts one llm call per batch, not one per cluster', async () => {
    const m = buildMocks([
      { messageId: 'm1', vector: new Float32Array([1, 0]) },
      { messageId: 'm2', vector: new Float32Array([0, 1]) },
    ]);
    m.categorizer.decideBatch.mockImplementation(async (entries: unknown[]) =>
      entries.map(() => ['c1']),
    );
    const orch = orchestrator(m);

    orch.start('a', 'gen', 'embed');
    await vi.waitFor(() => expect(orch.isRunning()).toBe(false));

    const p = orch.getProgress();
    expect(p.clusters).toBeGreaterThanOrEqual(1);
    expect(p.llmCalls).toBe(1);
    expect(p.gatedClusters).toBe(0);
    expect(p.clustersProcessed).toBe(p.clusters);
  });

  it('aborts the whole run with an actionable error when the generation model is missing', async () => {
    const m = buildMocks([
      { messageId: 'm1', vector: new Float32Array([1, 0]) },
      { messageId: 'm2', vector: new Float32Array([0, 1]) },
    ]);
    m.categorizer.decideBatch.mockRejectedValue(
      new LlmApiError(404, '/chat/completions', 'model "qwen3:4b" does not exist'),
    );
    const orch = orchestrator(m);

    orch.start('a', 'qwen3:4b', 'embed');
    await vi.waitFor(() => expect(orch.isRunning()).toBe(false));

    const p = orch.getProgress();
    expect(p.status).toBe('error');
    expect(p.error).toContain('qwen3:4b');
    expect(p.error).toMatch(/ollama pull/i);
    expect(p.failed).toBe(0);
  });

  it('reports an error, not a soft retry, when every LLM batch fails transiently', async () => {
    const m = buildMocks([{ messageId: 'm1', vector: new Float32Array([1, 0]) }]);
    m.categorizer.decideBatch.mockRejectedValue(new Error('LLM API: timed out after 180000ms'));
    const orch = orchestrator(m);

    orch.start('a', 'qwen3:8b', 'embed');
    await vi.waitFor(() => expect(orch.isRunning()).toBe(false));

    const p = orch.getProgress();
    expect(p.status).toBe('error');
    expect(p.error).toMatch(/local LLM server failed/i);
  });
});
