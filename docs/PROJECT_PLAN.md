# AI MailPilot: Implementation Status and Roadmap

**Project:** CPS2 master's personal project
**Repository:** <https://github.com/muhammadnoorulainroy/ai-mailpilot>

This document records the current engineering state of AI MailPilot. It replaces the original
schedule-based project plan, which described several ideas that were later rejected or deferred.
Claims below are based on the current TypeScript and Python source, database migrations, extension
manifest, and automated tests.

## 1. Product Goal

AI MailPilot addresses a practical problem: a large mailbox contains useful information, deadlines,
and actions, but the user must spend time searching and maintaining folders before that information
is usable.

The product adds four forms of assistance to Thunderbird:

1. Personalized semantic categories that can be corrected and maintained without silently
   replacing the user's organization.
2. A date-scoped priority view that separates actionable mail from important updates and routine
   summaries.
3. Grounded question answering across email, extracted attachment text, and captured calendar
   events.
4. Per-message summaries and editable reply drafts.

The governing product rule is conservative automation: leaving an uncertain message uncategorized
is preferable to giving it a confident but incorrect label. Taxonomy changes require review when
they may alter the visible organization.

## 2. Runtime Architecture

AI MailPilot is an npm-workspaces monorepo with three shipped packages:

| Package | Responsibility |
| --- | --- |
| `@ai-mailpilot/extension` | Thunderbird mailbox access, synchronization, settings, dashboard, message assistant, tag application, and folder moves |
| `@ai-mailpilot/core` | Authenticated Fastify API, AI orchestration, retrieval, categorization, triage, persistence, and background jobs |
| `@ai-mailpilot/shared` | TypeScript API contracts, configuration types, model presets, and shared constants |

The extension reads mail through Thunderbird WebExtension APIs and pushes selected content to Core
over authenticated HTTP on `localhost:3420`. Core has no active IMAP client and stores no mailbox
password. This keeps Thunderbird as the source of truth for message state.

Core uses one SQLCipher-encrypted SQLite database. Normal tables, FTS5 indexes, and `sqlite-vec`
vector tables share the same transaction boundary and backup unit. The application does not require
a separate vector database.

### 2.1 Main component flow

```text
Thunderbird WebExtension
  -> authenticated localhost API
  -> Fastify routes and validation
  -> application services
  -> account-scoped repositories
  -> encrypted SQLite, FTS5, and sqlite-vec

Application services
  -> local OpenAI-compatible endpoint for embeddings and default generation
  -> optional cloud OpenAI-compatible endpoint for explicitly enabled generation tasks
```

### 2.2 Local and cloud model roles

The model client exposes two logical providers:

- `main` is the configured local endpoint and is always used for embeddings.
- `chat` uses the optional chat endpoint when configured and otherwise resolves to the local
  generation endpoint.

Cloud configuration is not one global permission. Chat uses the selected chat provider;
opened-message summary and draft endpoints additionally request confirmation; Refine and Priority
have separate toggles; and cloud discovery requires a global option plus per-account consent.

## 3. Implemented Features

### 3.1 Account selection and synchronization

**Status: implemented**

- Discovers supported Thunderbird mail accounts and applies per-account indexing controls.
- Work and institutional accounts are indexable by default. Personal accounts require explicit
  enablement for indexing and have an additional discovery-consent control.
- Synchronizes configured folders in bounded batches and avoids fetching unchanged message bodies.
- Auto-indexing can process newly arrived mail; optional auto-triage can update the priority view in
  the same workflow.
- Force sync repairs incomplete or stale body capture.
- Core receives message data from the extension; it never logs in to the mail server independently.

### 3.2 Local embeddings and searchable storage

**Status: implemented**

- `bge-m3` is the standard embedding model and must return 1024 values.
- Embeddings are generated locally, canonicalized by model identifier, and updated only when message
  content changes.
- FTS5 indexes sender, subject, and body text. `sqlite-vec` stores email, category, and attachment
  vectors.
- PDF, DOCX, and text-like attachments are extracted, chunked, indexed lexically, and embedded.
- Extraction is bounded by page, character, and chunk limits. Unsupported or encrypted formats are
  recorded without blocking the rest of synchronization.

### 3.3 Two-pass categorization

**Status: implemented**

The categorizer separates cheap, high-precision decisions from expensive, ambiguous ones.

1. **Organize (fast)** ranks active category prototypes by cosine similarity. It assigns the top
   category only when the top score is at least `0.78` and leads the runner-up by at least `0.10`.
2. **Refine (accurate AI)** clusters unresolved mail for efficient processing, shortlists plausible
   existing categories, and uses a strong deterministic gate for unambiguous cases. Remaining cases
   are proposed by the selected generation model.
3. A deterministic adjudicator verifies that each model result belongs to the email's shortlist and
   is supported by embedding rank, margin, and independent text evidence. It may accept, override,
   or abstain.
4. Automatic Refine results are capped at two labels. User corrections can express a wider personal
   organization and are never overwritten by automatic reruns.

The language model cannot create a category during Refine. It chooses among existing candidates;
new taxonomy elements use the discovery workflow described below.

### 3.4 Category discovery and maintenance

**Status: implemented, with experimental options**

There are two related discovery paths:

- **Initial Re-discover** selects a bounded, deterministic view of the inbox and asks the model to
  name useful purpose-based topics. Cluster-representative sampling is available as an experimental
  alternative to sender-oriented sampling.
- **Residual discovery** selects uncategorized and low-confidence automatic assignments, excludes
  every user-confirmed assignment, and groups the residual with bounded deterministic leader
  clustering. A local model names candidate clusters; deterministic cohesion, separation,
  duplicate, and naming checks decide whether a proposal is safe to show.

Accepted candidates are stored as inactive suggestions. The review queue supports:

- adding a new category;
- merging overlapping automatic categories;
- splitting a loose automatic category into validated children; and
- retiring an empty automatic category.

Applying a proposal is transactional. Category identifiers and canonical keys remain stable;
aliases preserve prior names; user provenance survives approved moves; and no structural proposal
changes the taxonomy before approval.

### 3.5 Learning from corrections

**Status: implemented**

- A user correction atomically replaces the selected message's category assignments with
  `assigned_by = user` provenance.
- Automatic category jobs skip user-confirmed messages.
- A correction immediately nudges the relevant aggregate prototype. When at least three trusted
  examples exist, a full rebuild derives the prototype from user-confirmed members.
- If experimental multi-prototype categories are enabled, a broad category can retain its aggregate
  prototype and several sub-prototypes. Matching uses the nearest prototype for that category.
- Automatic evidence may bootstrap a category without trusted examples, but it does not displace a
  user-grounded prototype.

### 3.6 Existing Thunderbird organization

**Status: partially implemented**

- Synchronization records non-MailPilot Thunderbird tags and meaningful folder names separately
  from AI-owned categories.
- The dashboard summarizes existing organization, and discovery can use aggregated labels and
  representative subjects as weak hints.
- AI-created Thunderbird tags use a `mailpilot_` namespace and collision-safe canonical keys.
- Applying AI tags does not remove the user's own tags.
- Folder organization is previewed before messages move and records a ledger that supports rollback.

The API can propose existing labels that may seed categories, but a complete select-and-import UI is
not implemented.

### 3.7 Priority triage and calendar capture

**Status: implemented**

- The triage model returns a bucket plus structured action, reply, deadline, importance, suggested
  action, and short-summary fields.
- Deterministic normalization prevents routine code-platform notifications from becoming urgent
  solely because they contain urgent-sounding words.
- Today's Focus supports Today, Last 7 days, and All ranges; Needs Action, Important Updates,
  Summaries, and Low Priority sections; and a 14-day carryover for unresolved actionable mail.
- Done, Snooze, Dismiss, and Reset are persisted and survive reclassification.
- Event extraction rides on the triage call. Captured meetings, appointments, and dated deadlines
  appear in priority results, are retrievable by chat, and can be exported as `.ics` files.

Calendar capture does not write directly to Google Calendar, Outlook Calendar, or a CalDAV server.

### 3.8 Chat and retrieval-augmented generation

**Status: implemented**

- Retrieves lexical and semantic candidates from emails and attachments, fuses rankings, and can
  rerank the result before generation.
- Understands explicit dates and relative time windows, sender and topic references, aggregate
  questions, named documents, follow-up questions, and English/French term expansion.
- Deduplicates versions of similar documents and surfaces conflicting or rescheduled information
  rather than silently choosing one version.
- Includes captured calendar events in relevant date and schedule questions.
- Stores conversations and a bounded summary of older turns for context management.
- Streams grounded answers with source metadata that the extension can open in Thunderbird.

Experimental options provide a local cross-encoder reranker and a model-based query-understanding
step. Both are disabled by default.

### 3.9 Opened-message assistant

**Status: implemented**

- Generates and caches a structured summary for the opened message, including key points, action
  required, reply required, deadline, suggested action, and attachment status.
- Invalid or truncated model JSON degrades to safe defaults instead of crashing the panel.
- Generates a professional, editable reply draft with an optional instruction for tone, language,
  or content.
- Opens the Thunderbird reply composer with the draft. It never sends mail automatically.
- Summary cache keys include message content, attachment state, model, and provider, so stale results
  are regenerated after relevant input changes.

## 4. Data Model

The schema currently contains 26 forward migrations. Major storage groups are:

| Area | Stored data |
| --- | --- |
| Accounts and email | Account metadata, headers, bodies, sync state, and user label snapshots |
| Search | Email and attachment FTS5 indexes; email, attachment, and category vectors |
| Categorization | Categories, aliases, assignments, prototype indexes, model decisions, jobs, and corrections |
| Discovery | Suggestions, structural proposal children, evidence, status, and audit records |
| Priority | Triage verdicts, structured metadata, resolution state, and captured calendar events |
| Assistance | Conversations, conversation messages and summaries, and cached email summaries |
| Reliability | Model-scoped failure records and job progress |

All account-owned repository operations are scoped by `account_id`. Destructive taxonomy changes
run inside SQLite transactions.

## 5. Security Boundaries

### 5.1 Controls implemented

- Core binds to `localhost` and protects every route except health and pairing with a 256-bit bearer
  token.
- Pairing codes are six digits, single-use, valid for 15 minutes, and limited to ten attempts.
- API input is validated with Zod or explicit route validation.
- API responses expose a safe configuration shape and never return authentication or model API keys.
- The database is encrypted with a random 256-bit key. Existing plaintext databases are migrated
  through a recoverable backup-and-rekey procedure.
- Database, key, configuration, and log paths use platform-specific application-data directories;
  creation requests owner-only permissions.
- Discovery has an audit trail and separate account eligibility checks.
- Retrieved email and attachment text is marked as untrusted reference data in model prompts.
- Markdown rendering escapes untrusted HTML before displaying model output.

### 5.2 Residual risks

- The database key and encrypted database live under the same operating-system account. Full account
  compromise can expose both. This is encryption at rest, not hardware-backed secret isolation.
- CORS currently accepts all origins. Bearer authentication still protects the API, but a stricter
  extension-origin policy would reduce exposure if the token is compromised.
- Some privacy guarantees rely on callers selecting the correct logical model provider. Additional
  type-level routing constraints would reduce the chance of a future call-site error.
- Cloud providers receive the request-specific context needed for enabled cloud tasks. Their own
  retention and processing terms remain outside AI MailPilot's control.
- Logs avoid routine body content, but operational errors and model diagnostics still require a
  deliberate redaction review before broad distribution.

## 6. Reliability and Performance

- Incremental synchronization, embedding caching, batched work, bounded clustering coresets, and
  per-account progress reduce repeat work on large mailboxes.
- Processing failures are model-scoped and retried with bounded behavior. Malformed structured model
  output is salvaged or rejected conservatively.
- Long categorization jobs can be stopped and retain completed progress. Interrupted jobs are
  detected after Core restarts.
- Proposal application uses transactions so category, assignment, alias, prototype, and proposal
  state commit or roll back together.
- Local model timeouts remain possible. Batch triage falls back to smaller work units, but repeated
  local inference timeouts still require a faster model, more context capacity, or a cloud opt-in.

## 7. Verification

The repository uses Vitest for Core and extension behavior, strict TypeScript, ESLint, Prettier, and
build checks. Tests cover repositories and migrations, encryption and pairing, categorization gates
and adversarial label cases, corrections, category proposals and rollback, attachment extraction,
retrieval and conversation memory, priority state, event capture, message-assistant behavior, tag
collisions, folder matching, and UI formatting helpers.

The CI matrix runs on Node.js 20 and 22. The local release gate is:

```bash
npm run typecheck
npm run lint
npm run format:check
npm test
npm run build
```

The test suite establishes deterministic behavior and guards regressions. It does not establish
real-world classification accuracy. A labelled multilingual evaluation corpus and measured
precision, recall, calibration, and latency remain necessary for a quantitative claim.

## 8. Deliberately Deferred or Experimental Work

| Item | Current position |
| --- | --- |
| Multi-prototype category matching | Implemented behind an experimental flag; off by default |
| Cluster-representative initial discovery | Implemented behind an experimental flag; off by default |
| Local cross-encoder reranking | Implemented behind an experimental flag; requires Python dependencies |
| Model-based query understanding | Implemented behind an experimental flag; off by default |
| Python UMAP/HDBSCAN clustering | Standalone evaluation sidecar, not wired into the production Core path |
| Existing-label category import | Suggestion API exists; apply UI is incomplete |
| Direct calendar-provider write | Not implemented; `.ics` export is implemented |
| Parent/child category hierarchy | Not implemented |
| Taxonomy version-diff UI | Not implemented |
| Federated learning | Not implemented; research/future work only |
| MCP server | Not implemented |
| Institutional control plane | Not implemented |

## 9. Next Engineering Priorities

1. Build a privacy-safe labelled evaluation set and publish categorization, retrieval, triage, and
   latency measurements by language and model.
2. Complete the user-label-to-category import flow with preview, explicit selection, and rollback.
3. Harden localhost transport with a restricted CORS policy, broader log redaction, and clearer key
   backup/recovery guidance.
4. Add browser-level end-to-end tests for onboarding, synchronization, proposal review, Priority,
   chat citations, message assistance, tag collisions, and folder rollback.
5. Validate experimental multi-prototype and cross-encoder modes before considering either as a
   default.

These priorities improve evidence, safety, and usability without expanding the product into
unimplemented federation or external-agent integrations.
