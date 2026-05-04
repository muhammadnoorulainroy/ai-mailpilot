# AI MailPilot - Project Plan

**Date**: April 2026 - revised
**Author**: Muhammad Noor Ul Ain
**Program**: CPS2 Master's - University Jean Monnet & Ecole des Mines de Saint-Etienne
**Repository**: https://github.com/muhammadnoorulainroy/ai-mailpilot

---

## Vision

AI MailPilot is an open-source, privacy-first Thunderbird extension that uses local AI to intelligently organize your inbox. It automatically classifies incoming emails into actionable categories (urgent, summarize, spam, personal), discovers semantic clusters of related emails, lets users chat with their mailbox using RAG, and can federate spam and category knowledge across an organization without sharing email content.

**Target**: Thunderbird Add-on Store (ATN), millions of potential users in academic and professional organizations.

---

## Architecture Overview

AI MailPilot is a two-piece system: a Thunderbird extension that owns the user interface and email access, and a Core Server that owns the AI intelligence. The two communicate via a local REST API. An optional MCP wrapper exposes the same intelligence to external AI clients (Claude Desktop, VS Code Copilot) for power users, but this is configuration-only and not part of the primary product.

```
+--------------------------------------------------------+
|                                                        |
|           THUNDERBIRD EXTENSION (Primary UI)           |
|                                                        |
|  - Dashboard (urgent, summaries, mailing lists)        |
|  - Categories sidebar (multi-label tags)               |
|  - Chat panel (Ask your mailbox)                       |
|  - Context menus (categorize, find similar)            |
|  - Settings (LLM URL, models, federation toggle)       |
|  - Onboarding wizard                                   |
|                                                        |
|  Reads emails directly from Thunderbird APIs           |
|  (no separate IMAP path)                               |
|                                                        |
+--------------+-----------------------------------------+
               |
               | REST (HTTP/JSON)
               | - push email text + metadata
               | - request embeddings, classifications, RAG answers
               v
+--------------+-----------------------------------------+
|                                                        |
|        AI MAILPILOT CORE SERVER (Local Process)        |
|                                                        |
|  Interfaces:                                           |
|  - REST API (Fastify)                                  |
|  - MCP wrapper (optional, low priority)                |
|                                                        |
|  Engines:                                              |
|  - Embedding engine (calls LLM runtime)                |
|  - Triage classifier (urgent/summarize/spam/personal)  |
|  - Personal email detector (dual-signal)               |
|  - LLM topic discovery + centroid similarity           |
|    (multi-label, NOT K-means or HDBSCAN)               |
|  - Category manager (centroids + learning loop)        |
|  - RAG engine (retrieval + generation)                 |
|  - Draft reply generator (for urgent emails)           |
|  - Awaiting Response tracker                           |
|  - Mailing list detector                               |
|                                                        |
|  Storage:                                              |
|  - SQLite with sqlite-vec extension (one DB file)      |
|    - account_id-scoped tables                          |
|    - vectors + metadata in single transaction model    |
|                                                        |
+--------------+-----------------------------------------+
               |
               | OpenAI-compatible API
               v
+--------------+-----------------------------------------+
|                                                        |
|              LLM RUNTIME (Configurable)                |
|                                                        |
|  Mode 1: Local Ollama (default, fully private)         |
|  Mode 2: Institutional server (VPN/local network)      |
|  Mode 3: LM Studio, vLLM, or any OpenAI-compatible     |
|                                                        |
|  Same API contract for all modes                       |
|                                                        |
+--------------------------------------------------------+

OPTIONAL: MCP wrapper on Core Server lets external AI
clients (Claude Desktop, VS Code Copilot) call the same
intelligence. Wired up via configuration docs in README.
```

### Key Architectural Decisions (Meeting 2)

1. **No separate IMAP connector in Core**. Thunderbird already has IMAP. The extension reads emails via Thunderbird APIs and pushes content to Core. This avoids double-fetching and keeps Thunderbird as source of truth for email state.

2. **OpenAI-compatible LLM API**. Core treats the LLM runtime as a black box with a standard API. Works with local Ollama, institutional servers, LM Studio, vLLM. User configures URL, port, optional credentials.

3. **sqlite-vec for vectors, not a separate vector DB**. SQLite with the sqlite-vec extension handles both regular persistence and vector search in a single file. Production-grade (used by Spotify and others), simpler stack, easier backup/restore. Avoids running a separate Qdrant/LanceDB process.

4. **LLM topic discovery, not K-means or HDBSCAN**. K-means is single-label and requires K upfront. HDBSCAN has no production JS library. Instead: sample emails, ask LLM to identify topics with descriptions, compute centroids from matched emails, classify new emails by cosine similarity to all centroids (multi-label via threshold). Naturally multi-label, human-meaningful from day one.

5. **External MCP clients are configuration-only**. Drop the planned Local MCP Client and CLI. The MCP wrapper on Core is low priority. README documents how to connect Claude Desktop or Copilot for power users.

6. **Two deployment modes supported, MVP defaults to local**. Core Server is deployable both locally (per-user, default) and institutionally (org-hosted, future). Same code, transport-agnostic. MVP ships local mode; institutional mode is a configuration swap, not a rewrite.

7. **account_id baked into every data model from day one**. Multi-account support (Gmail + work institutional + personal) is cheap to add now, expensive to retrofit. All records carry account scope.

8. **Bootstrap from existing folder structure on first run**. The professor already has years of folder curation. We treat each existing folder as an implicit category, compute its centroid from current contents, and use it as training data. No clean slate. His investment becomes training.

---

## Functional Requirements

### Always-On Email Triage (NEW from Meeting 2)

Inspired by the n8n workflow the professor referenced, every incoming email is classified into one of four buckets:

| Bucket | Action | What user sees |
|--------|--------|----------------|
| Urgent | Summarize + draft reply | Top of dashboard, ready-to-edit response |
| Summarize | Brief summary | Mailing list digest section of dashboard |
| Spam | Filter out | Hidden by default, optional review |
| Personal | Excluded from work workflows | Separate filter, never federated |

This classification is in addition to the multi-label semantic categories that the user discovers through clustering.

### Dashboard View (NEW)

First thing the user sees when opening the extension. Shows:
- Urgent items requiring response (with auto-drafts)
- Mailing list summaries (digests of recent activity)
- Recent activity overview
- Spam count (collapsed)
- Personal emails (separate view)

### Multi-Label Semantic Categories

Beyond the four triage buckets, emails get semantic tags discovered through LLM topic discovery and centroid similarity:
- "Student Applications", "Research - AAAI", "Department Meetings", etc.
- An email can have multiple tags (multi-label via similarity threshold)
- Category names support `/` as hierarchy separator (e.g., `CPS2/Applications/2026`). UI renders as tree.
- User can rename, merge, split, delete, create tags
- System learns from confirmations and corrections (centroid running mean)

### Existing Folder Structure Import (NEW)

On first run, AI MailPilot scans the user's existing folder tree in Thunderbird and offers to use it as a starting point:
- Each existing folder becomes an implicit category
- Centroid computed from emails currently in that folder
- New emails get suggested for these existing folders, not just newly discovered categories
- User can keep, rename, merge, or ignore each imported category
- Respects existing organization rather than replacing it

### Bulk Operations (NEW)

Single-email categorization is too slow for inboxes with thousands of emails. The dashboard supports:
- Select multiple emails -> apply tag or move to folder in one action
- "Auto-categorize this folder right now" (background batch with progress)
- Per-rule actions: "All emails from this mailing list always go to Folder X"

### Mailing List Detection and Per-List Rules (NEW)

Mailing lists deserve different treatment than one-off emails:
- Auto-detect mailing lists via `List-Unsubscribe` header and `From` patterns
- Group emails by list
- Per-list digest in dashboard ("This week on the AAAI list: 12 emails, key topics...")
- Per-list rules: "Always summarize, never urgent" or "Always move to Newsletters folder"

### Awaiting Response Tracking (NEW)

The professor explicitly wants to ensure he replies to everyone who needs an answer:
- Emails classified as Urgent are tracked
- System detects when a reply has been sent (Sent folder check)
- "Awaiting Response" view shows urgent emails that have not received a reply
- Removes false anxiety: he can see what genuinely needs his attention

### Chat with Mailbox (Inside Thunderbird)

- Sidebar tab in Thunderbird extension
- Natural language queries
- RAG: retrieves relevant emails, feeds to LLM, returns answer with sources
- Conversation history for follow-ups
- Sources are clickable to jump to source emails

### Federated Learning (Tier 3)

Two primary use cases:
- **Spam improvement**: collective spam detection. Each user sees different spam, federation helps catch new patterns faster.
- **Cross-team awareness**: knowing if a colleague has already acted on a shared institutional email.

Privacy guarantees:
- Personal emails never federated
- Only centroids and aggregated signals shared
- Email content stays local

---

## Scope Tiers

### Tier 1 - Core (MUST)
The Thunderbird-only product. Ships as a complete app.

- Thunderbird extension scaffold (sidebar, context menus, settings, dashboard)
- Onboarding wizard (detect LLM runtime, model presets, choose folders, import existing folder structure)
- Email push from extension to Core (text + metadata via Thunderbird APIs)
- Multi-account support (account_id on every record)
- Embedding computation via OpenAI-compatible API (local Ollama or institutional)
- sqlite-vec for vector storage (no separate vector DB process)
- Triage classification (urgent / summarize / spam / personal)
- Personal email detector (conservative defaults, strictly excluded from federation)
- LLM topic discovery + centroid similarity for multi-label categorization
- Category names support `/` hierarchy separator, UI renders as tree
- Category manager (rename, merge, split, delete, create)
- Email categorization with confidence scores and explanations
- Multi-label tag and folder assignment
- Bulk operations (select N emails -> apply tag or move)
- Existing folder structure bootstrap (use current folders as initial categories)
- Mailing list detection and per-list rules
- Awaiting Response tracking (urgent items without a reply yet)
- Learning loop (centroid updates, classification feedback)
- Dashboard view (urgent + drafts, mailing list digests, awaiting response, recent)
- Auto-draft reply generation for urgent emails (always user-reviewed, never auto-sent)
- Auto-classify on new email arrival (opt-in)
- DB migration framework from day 1 (schema versioning in Core)
- Structured logging (pino) to local file, exportable diagnostics
- Category export/import (JSON, GDPR compliance and backup safety)
- i18n (English + French)
- Progress indicators for all long-running tasks
- Graceful degradation (LLM offline, model missing, Core not running)

### Tier 2 - RAG Chat (SHOULD)
Chat panel inside Thunderbird extension.

- Chat UI tab in Thunderbird sidebar
- Query embedding -> sqlite-vec similarity search -> LLM answer
- Source attribution with clickable links
- Conversation history (multi-turn)
- Search filters (date range, folder, sender)

### Tier 3 - Shared Vocabulary + Spam Federation (COULD)
Org-wide intelligence sharing, scoped conservatively. Not full federated learning (that has Byzantine robustness, differential privacy, and other research-grade requirements out of scope here). Instead, two concrete features that deliver most of the value safely.

- Lightweight central server (Node.js)
- Organization authentication (OIDC or org SSO, deferred until institutional mode)
- **Shared category vocabulary**: organization-curated category labels (e.g., "Student Applications", "AAAI Submissions") that all users can adopt. Centroids are local per user, but vocabulary is shared. Avoids the privacy/robustness issues of model federation while still solving the "everyone reinvents the same categories" problem.
- **Shared spam signals**: aggregated spam fingerprints (sender domains, message hashes, not content). Each user's spam corrections improve everyone's spam detection.
- **Cross-team awareness**: when multiple users receive the same email (CC chains, mailing lists), surface who has already acted on it. No email content shared, just message-id + action-taken signals.
- Personal emails strictly excluded from federation
- Extension works standalone OR with org server (configurable)

### Optional - MCP Wrapper (LOW PRIORITY)
- Thin MCP server interface on Core
- Exposes existing REST endpoints as MCP tools
- Documented in README for Claude Desktop / Copilot users
- No custom MCP client built by us

### Future Work (NOT in this iteration)
- Knowledge graph / Graphify (alternative to vector retrieval)
- Federated knowledge graph
- Architecture supports plugging these in later

---

## Model and Runtime Choices

### LLM Runtime Modes

The user picks one of three modes during onboarding:

| Mode | Description | Privacy |
|------|-------------|---------|
| Local Ollama (default) | Runs models on user's machine | Fully private |
| Institutional Server | Org-hosted LLM accessible via VPN | Stays within org |
| Custom OpenAI-compatible | User provides URL + credentials | Depends on provider |

All modes use the same OpenAI-compatible API. Configuration includes URL, port, optional auth.

### Model Presets (when using local Ollama)

| Preset | Embedding Model | Generation Model | Total Size | Min RAM |
|--------|----------------|-----------------|-----------|---------|
| Lightweight | nomic-embed-text (274 MB) | phi3.5:3.8b (2.2 GB) | ~2.5 GB | 4 GB |
| Recommended | bge-m3 (1.2 GB) | mistral:7b (4.1 GB) | ~5.3 GB | 8 GB |
| Maximum Quality | bge-m3 (1.2 GB) | mistral-nemo:12b (7.1 GB) | ~8.3 GB | 16 GB |
| Custom | User's choice | User's choice | Varies | Varies |

### Why These Models

- **bge-m3** (embedding): Best multilingual quality. Handles French + English natively. 1024 dimensions.
- **nomic-embed-text** (embedding fallback): Smaller, faster. 768 dimensions.
- **mistral:7b** (generation): French company (Mistral AI). Excellent French support. Fits 8B hardware limit.
- **phi3.5:3.8b** (generation fallback): Half the size, runs on weaker hardware.
- **mistral-nemo:12b** (generation premium): Best quality, needs 16GB+ RAM.

---

## Storage Architecture

**Single database**: SQLite via better-sqlite3 with the **sqlite-vec** extension for vector search. One file, one transaction model, easy backup, no separate process.

### Schema (high level)

- `accounts` - account_id, address, type (personal/work/institutional)
- `categories` - id, account_id, label (supports `/` hierarchy), description, source (auto/user/imported), created_at, updated_at
- `emails` - message_id, account_id, folder, subject, from, date, has_attachments
- `embeddings` (sqlite-vec virtual table) - email_id, model_id, vector
- `centroids` (sqlite-vec virtual table) - category_id, model_id, vector
- `email_categories` - email_id, category_id, confidence, assigned_by (user/auto)
- `triage` - email_id, bucket (urgent/summarize/spam/personal), reasoning, classified_at
- `awaiting_response` - email_id, marked_urgent_at, replied_at (null if no reply yet)
- `drafts` - email_id, draft_body, model_used, approved (bool)
- `mailing_lists` - list_id, account_id, name, rules (json)
- `conversations` - id, account_id, history (json), updated_at
- `migrations` - version, applied_at

All vectors keyed by `(email_id, model_id)` or `(category_id, model_id)` so model changes invalidate cleanly without data loss.

---

## Timeline - 12 Weeks

### Phase 0: Foundation (Weeks 1-2)
**Goal**: Working Core Server + extension skeleton + LLM runtime integration

| Week | Deliverables |
|------|-------------|
| W1 | DONE: Repository setup, monorepo, build pipeline, CI, ESLint, Prettier, Vitest, Vite (extension), tsup (core), Fastify, i18n (EN+FR). Health endpoint working. |
| W2 | OpenAI-compatible client in Core (configurable URL, optional auth header). sqlite-vec integration with schema migration framework. pino logging to local file. account_id baked into schema. Onboarding wizard UI in extension (detect runtime, pick mode, model preset, choose folders, scan existing folder structure). Settings page (LLM URL, models, mode toggle). REST API contracts finalized. Email-push endpoint in Core. Thunderbird API integration in extension (read folders, read messages, fetch full body). |

**Milestone 1**: Core Server runs locally with persistent DB. Extension installs in Thunderbird, talks to Core, user can configure runtime mode, select folders, and existing folder structure is imported as categories.

---

### Phase 1: Embedding Pipeline (Weeks 3-4)
**Goal**: Emails get embedded and cached efficiently

| Week | Deliverables |
|------|-------------|
| W3 | Extension fetches email content via Thunderbird APIs and pushes to Core. Text preprocessing in Core (strip HTML, normalize, truncate to model token limit). Embedding service via OpenAI-compatible API. sqlite-vec persistence (store/retrieve by message_id + model_id). |
| W4 | Batch embedding orchestrator (paginated processing, progress tracking, pause/resume). Auto-index listener for new email events (opt-in). Cache invalidation on model change (re-index workflow with user confirmation). Progress UI in extension sidebar with cancel/pause controls. |

**Milestone 2**: User indexes a folder, sees real-time progress, embeddings cached in sqlite-vec. Auto-index works for new emails. Switching models triggers controlled re-index.

---

### Phase 2: Triage + Multi-Label Categorization + Dashboard (Weeks 5-7)
**Goal**: Triage buckets + LLM topic discovery + dashboard with actionable views

| Week | Deliverables |
|------|-------------|
| W5 | Triage classifier in Core (LLM with structured JSON output: urgent / summarize / spam / personal + reasoning). Personal email detector (conservative defaults, dual-signal: heuristic + LLM). Spam detection. Mailing list detection (List-Unsubscribe header, sender patterns). Auto-draft reply generation for urgent emails. REST endpoints for triage and drafts. Awaiting Response tracking (Sent folder watcher). |
| W6 | LLM topic discovery (sample N emails -> identify topics -> compute centroids from matched emails). Centroid storage in sqlite-vec. Threshold-based multi-label assignment. Category UI in extension (list with hierarchy support via `/` separator). Manual category editing (rename, merge, split, delete, create). Existing folder structure import wizard. |
| W7 | Email categorization flow (right-click -> ranked categories with confidence + explanations). Multi-label tag and folder suggestions. Learning loop (centroid running mean on confirmations, correction handling). Dashboard view (urgent + drafts, mailing list digests, awaiting response, recent activity, spam count, personal section). Bulk operations UI (select N emails -> apply tag or move). Per-mailing-list rules. |

**Milestone 3**: Full Tier 1 loop works end-to-end. Professor opens Thunderbird, dashboard shows urgent items with drafts + mailing list digests + awaiting response. Existing folders imported as categories. Bulk operations functional.

---

### Phase 3: Polish & Ship Core (Weeks 8-9)
**Goal**: Production-quality release

| Week | Deliverables |
|------|-------------|
| W8 | Thunderbird tag and folder sync (changes from extension reflect on IMAP server). Edge cases (empty folders, single-email categories, LLM unavailable, Core not running, DB locked, disk full, network drop mid-batch). Error handling and structured logging (pino). Performance optimization on 5000+ email mailbox. Category export/import (JSON). Diagnostics export button in settings. |
| W9 | Full i18n pass (EN + FR). Accessibility (keyboard nav, screen reader labels, focus management). Help panel inside extension. Onboarding polish. ATN packaging and manifest compliance. Core Server installer/launcher per platform (Windows/Mac/Linux). First ATN submission. |

**Milestone 4**: Tier 1 submitted to Thunderbird Add-on Store. Production-quality on 5000+ email mailbox.

---

### Phase 4: RAG Chat (Weeks 10-11)
**Goal**: Ask your mailbox inside Thunderbird

| Week | Deliverables |
|------|-------------|
| W10 | Chat UI tab in Thunderbird sidebar. Query embedding -> sqlite-vec similarity search -> top-k retrieval -> context assembly -> LLM generation. REST endpoints for chat. Conversation persistence in DB. |
| W11 | Source attribution with clickable references to source emails. Multi-turn conversation history. Search filters (date range, folder, sender, category). Personal emails excluded from chat results by default (toggle to include). Performance optimization for large result sets. Optional: thin MCP wrapper on Core + README docs for Claude Desktop / Copilot integration (configuration-only, no custom client). |

**Milestone 5**: Professor asks natural language questions inside Thunderbird and gets sourced answers. Personal emails respected.

---

### Phase 5: Shared Vocabulary + Spam Federation + Final Polish (Week 12)
**Goal**: Org-wide intelligence (conservative scope) + release

| Week | Deliverables |
|------|-------------|
| W12 | If time allows: lightweight central server (Node.js + Fastify) for shared vocabulary and spam fingerprint aggregation. Strict personal email exclusion enforced client-side before any upload. Cross-team awareness signals (message_id-based, no content). Final docs (README, CONTRIBUTING, user guide, privacy policy). Demo video. ATN listing update with screenshots. |

**Milestone 6**: Final product delivered. Org-wide federation prototype runs at Ecole des Mines if time permits, else clearly documented for future work.

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | TypeScript (strict mode) | Type safety, maintainability |
| Core Server runtime | Node.js 20+ | Ecosystem, libraries, OpenAI-compatible clients |
| Core Server framework | Fastify | Lightweight, fast, schema-based |
| Extension API | Thunderbird MailExtension API (WebExtension) | Only supported API for TB 115+ |
| Build tool | Vite (extension) + tsup (core) | Fast TS compilation |
| Monorepo | npm workspaces | Shared types between core and extension |
| Linting | ESLint + Prettier | Code consistency |
| Testing | Vitest | Fast, TS-native |
| LLM runtime | OpenAI-compatible API (Ollama default, institutional / LM Studio / vLLM / custom) | Same contract for all modes, no vendor lock-in |
| Embedding model (default) | bge-m3 | Best multilingual (FR+EN), 1024 dim |
| Generation model (default) | mistral:7b | French company, fits 8B limit |
| Storage (everything) | SQLite via better-sqlite3 + sqlite-vec extension | Single file, vector + SQL in one DB, no separate process, easy backup |
| Migrations | Lightweight versioned migrations in Core | Required from day 1 |
| Logging | pino to local file | Structured, exportable diagnostics |
| Clustering / categorization | LLM topic discovery + cosine similarity to centroids | Multi-label, no preset K, no missing JS library |
| Email source | Thunderbird APIs (NOT separate IMAP) | Single source of truth, no double-fetching |
| IPC (Extension <-> Core) | HTTP REST localhost:3420 with bearer token (MVP) | Simple. Native Messaging upgrade path for production. |
| Deployment modes | Local Core (MVP default) OR institutional Core (config swap) | Both supported, MVP ships local |
| Config storage (Extension) | browser.storage.local | Native to WebExtension |
| Heavy computation | Worker threads (Core) | Non-blocking embeddings, similarity batches |
| i18n | Thunderbird i18n API (browser.i18n) | Native, follows ATN conventions |
| CI/CD | GitHub Actions | Automated build, lint, test |
| Distribution | ATN (extension) + npm/binary (core) | Official channels |

---

## Repository Structure

```
ai-mailpilot/
├── .github/
│   └── workflows/
│       └── ci.yml
├── packages/
│   ├── shared/        # Shared types and constants
│   ├── core/          # Core server (Fastify + AI engines + sqlite-vec)
│   └── extension/     # Thunderbird extension (UI + Thunderbird API integration)
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── COMPETITIVE_ANALYSIS.md
│   └── AI_MailPilot_Overview.docx
├── package.json
├── LICENSE
├── README.md
└── CONTRIBUTING.md
```

The previously-planned `cli` package has been removed - it was redundant given Claude Desktop and VS Code Copilot already exist as MCP clients, and per the professor's guidance the project should focus on Thunderbird.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Thunderbird WebExtension API does not expose required email data (full body, threading, etc.) | Medium | High | Prototype in W2 before committing. If gaps exist, fall back to a read-only IMAP adapter only for missing fields (extension still owns state). |
| LLM topic discovery quality on small mailboxes | Medium | Medium | Allow manual category creation as primary escape hatch. Use existing folder import to bootstrap. Re-run topic discovery as more emails are categorized. |
| Triage classifier accuracy | High | Medium | Use few-shot prompts with structured JSON output. User corrections feed back into per-user thresholds. Track accuracy metrics. |
| Personal email detection accuracy | High | High | Dual-signal (heuristic + LLM both must agree to mark "work"). Conservative default to personal when uncertain. Audit log of federated items. |
| LLM runtime latency for bulk triage on 5000+ emails | High | High | Run as background batch with progress UI. Allow overnight indexing. Document expected time. Plan v2 with classifier head for speed. |
| sqlite-vec performance at 100k+ vectors | Low | Medium | Production-tested at much larger scale (Spotify). Benchmark in W2 with synthetic data. Fallback: switch to Qdrant only if hit ceiling. |
| Auto-draft reply quality or inappropriate content | Medium | Medium | Always require user review. Show "AI Draft" banner. Never auto-send. Track accept/reject rates. |
| Embedding model change invalidates all data | Medium | High | Lock model after first index. Re-index requires explicit user confirmation. Document trade-off. |
| Category labels lost on uninstall or migration | Low | High | Export/import as JSON from settings. Document backup workflow. |
| Cross-team awareness leaks personal info | Medium | High | Strict opt-in. Personal emails strictly excluded. Audit log of what got shared. |
| Core Server lifecycle UX on user laptops | High | High | MVP uses HTTP REST on localhost (simple). Document startup. Plan Native Messaging Host upgrade for production (auto-spawned by Thunderbird). |
| Localhost REST API accessible to other apps | Medium | High | Bearer token auth (token exchanged via browser.storage.local). DNS rebinding protection. Upgrade to Native Messaging removes this concern entirely. |
| Multi-account schema retrofitting | Low | High | Already mitigated: account_id baked in from day 1. |
| Scope creep into Tier 3 at expense of Tier 1 quality | High | High | Strict milestone gates. Tier 1 must ship to ATN before Tier 3 starts. |
| MCP wrapper drifts from REST API | Low | Low | Keep MCP wrapper as thin adapter generated from REST schema. |

---

## Success Criteria

| Criterion | Metric |
|-----------|--------|
| Technical depth | Two-piece architecture with Thunderbird Extension + Core. sqlite-vec for vector storage. LLM topic discovery for multi-label categorization. OpenAI-compatible runtime abstraction. Learning loop. RAG. Multi-account. Migration framework. Optional MCP wrapper. |
| Usability | Zero-to-organized in under 5 minutes. Dashboard shows urgent items + drafts + mailing list digests + awaiting response on first launch after indexing. Existing folder structure imported as starter categories. |
| Problem-solving | Professor sees urgent emails with auto-drafts on dashboard. Categories are meaningful and respect his existing folder organization. Personal emails correctly filtered. Bulk operations let him categorize hundreds of emails in seconds. Mailing lists get their own digests. |
| Impact | Published on ATN. Handles 5000+ email mailbox. Works on 8B model hardware. Can be deployed locally (default) OR institutionally (config swap). |
| Code quality | TypeScript strict mode. > 80% test coverage on Core services. CI passes on all PRs. Migrations versioned. Logs structured. |
| Privacy | Email content never leaves the local machine in default mode. Personal emails strictly excluded from any external traffic. Configurable runtime modes. Bearer token auth on localhost. Category export for GDPR compliance. |
