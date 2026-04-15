# AI MailPilot - Project Plan

**Date**: April 2026
**Author**: Muhammad Noor Ul Ain
**Program**: CPS2 Master's - University Jean Monnet & Ecole des Mines de Saint-Etienne
**Repository**: https://github.com/muhammadnoorulainroy/ai-mailpilot

---

## Vision

AI MailPilot is an open-source, privacy-first email intelligence platform. It consists of a local core server (with MCP + REST interfaces) and a Thunderbird extension UI. It computes semantic embeddings, discovers categories through clustering, suggests labels/folders with confidence scores, learns from user feedback, and lets users chat with their mailbox via RAG - all while keeping email data fully private on the local machine.

**Target**: Thunderbird Add-on Store (ATN) + MCP ecosystem, millions of potential users.

---

## Architecture Overview

AI MailPilot is a hybrid system with three pieces:

```
+-----------------------------------------------------+
|                MCP CLIENTS (optional)                |
|  +--------------+  +--------------+                  |
|  |Claude Desktop|  | Local Client |                  |
|  |  (cloud LLM) |  | (Ollama LLM) |                  |
|  +------+-------+  +------+-------+                  |
|         +--------+--------+                          |
|                  | MCP Protocol                       |
+------------------+----------------------------------+
                   |
+------------------+----------------------------------+
|                  v                                   |
|  +---------------------------------+                 |
|  |    AI MailPilot Core Server     |                 |
|  |    (Node.js / TypeScript)       |                 |
|  |                                 |                 |
|  |  Interfaces:                    |                 |
|  |  - MCP server (tools)           |                 |
|  |  - REST API (for extension)     |                 |
|  |                                 |                 |
|  |  Engines:                       |                 |
|  |  - IMAP connector               |                 |
|  |  - Embedding engine (Ollama)    |                 |
|  |  - Clustering engine (K-means)  |                 |
|  |  - Category manager             |                 |
|  |  - RAG engine                   |                 |
|  |  - Persistence (SQLite)         |                 |
|  +---------------------------------+                 |
|         ^              |           |                 |
|         |              |           |                 |
|  +------+-------+  +---v----+ +---v-----+           |
|  |  Thunderbird |  | Ollama | |  IMAP   |           |
|  |  Extension   |  |(local) | | Server  |           |
|  |  (thin UI)   |  +--------+ +---------+           |
|  +--------------+                                    |
|                                                      |
|          ALL OF THIS RUNS ON LOCAL MACHINE            |
+------------------------------------------------------+
```

**Piece 1 - AI MailPilot Core Server**: The brain. A local Node.js/TypeScript process that handles all AI logic, IMAP access, persistence, and exposes both a REST API and an MCP interface. This is where most of the code lives.

**Piece 2 - Thunderbird Extension**: A thin UI client that talks to the Core Server via REST. Handles sidebar, context menus, settings page, onboarding wizard.

**Piece 3 - MCP Interface**: A thin protocol layer on the Core Server that exposes email intelligence tools to any MCP-compatible client (Claude Desktop, VS Code, etc.). Optional - the system works fully without it.

### Privacy Modes

| Mode | How It Works | Privacy Level |
|------|-------------|---------------|
| Thunderbird + Ollama (default) | Extension -> Core Server -> Ollama. All local. | Full privacy |
| MCP + Local Client | Local MCP client (Ollama-powered) -> Core Server -> Ollama | Full privacy |
| MCP + Cloud Client | Claude Desktop -> Core Server -> Ollama. Answers may reach cloud. | Partial (user's choice) |

### MCP Tools Exposed

| Tool | Description | Returns |
|------|------------|---------|
| `index_folder` | Embed all emails in a folder | Progress status |
| `list_categories` | Show discovered categories | Category names, descriptions, counts |
| `categorize_email` | Classify an email into categories | Ranked categories with confidence |
| `search_emails` | Semantic search across mailbox | Matching emails (metadata + snippet) |
| `ask_mailbox` | RAG - answer a question using emails | Answer + source references |
| `get_category_emails` | List emails in a category | Email list |
| `update_category` | Rename, merge, delete a category | Updated category |
| `get_status` | Check system health | Ollama status, index progress |

---

## Scope Tiers

### Tier 1 - Core: Semantic Email Organization (MUST)
The foundation. A complete, polished product on its own.

- AI MailPilot Core Server (Node.js/TypeScript, local process)
- REST API for Thunderbird extension
- Thunderbird extension scaffolding (sidebar, context menus, settings page)
- Onboarding wizard (detect Ollama, model presets, folder picker)
- IMAP connector (read emails directly)
- Email text extraction + preprocessing pipeline
- Local embedding computation via Ollama (incremental, cached)
- Auto-index on new email arrival (opt-in)
- K-means clustering with auto-K (silhouette score / elbow method)
- LLM-generated labels and descriptions for clusters
- Manual category management (rename, merge, split, delete)
- Email categorization with confidence scores and explanations
- Tag and folder assignment (both supported)
- Learning loop (user confirms/edits -> centroid updates)
- IMAP label/folder sync
- i18n support from day 1 (English + French)
- Progress indicators for all long-running tasks
- Graceful degradation (Ollama offline, model missing, etc.)

### Tier 2 - RAG Chat: Ask Your Mailbox (SHOULD)
Natural extension since embeddings already exist.

- MCP interface on Core Server (expose tools via MCP protocol)
- Chat UI inside Thunderbird extension (sidebar tab)
- `ask_mailbox` tool: query embedding -> retrieve closest emails -> LLM answer
- `search_emails` tool: semantic search with filters
- Source attribution (clickable references to source emails)
- Conversation history (multi-turn context)
- Search filters (date range, folder, sender)
- Works from Thunderbird UI AND any MCP client (Claude Desktop, etc.)

### Tier 3 - Federated: Organization-Wide Categories (COULD)
Stretch goal. Only if Tier 1 and 2 are solid.

- Lightweight central server (Node.js API)
- Organization authentication
- Shared category centroids (not email content - privacy preserved)
- Federated centroid aggregation (averaging across users)
- Extension works standalone OR with org server (configurable)

### Low Priority - Future Work
- Knowledge graph / Graphify (alternative retrieval via graph traversal instead of vector similarity)
- Federated knowledge graph
- Architecture supports plugging these in later - retrieval layer is abstracted

---

## Model Recommendations

### Presets (offered during onboarding)

| Preset | Embedding Model | Generation Model | Total Size | Min RAM |
|--------|----------------|-----------------|-----------|---------|
| Lightweight | nomic-embed-text (274 MB) | phi3.5:3.8b (2.2 GB) | ~2.5 GB | 4 GB |
| Recommended | bge-m3 (1.2 GB) | mistral:7b (4.1 GB) | ~5.3 GB | 8 GB |
| Maximum Quality | bge-m3 (1.2 GB) | mistral-nemo:12b (7.1 GB) | ~8.3 GB | 16 GB |
| Custom | User's choice | User's choice | Varies | Varies |

### Why These Models

- **bge-m3** (embedding): Best multilingual quality. Handles French + English natively. 1024 dimensions.
- **nomic-embed-text** (embedding fallback): Smaller, faster. 768 dimensions. Good for low-resource machines.
- **mistral:7b** (generation): French company (Mistral AI). Excellent French support. Ecole des Mines has partnership. Fits 8B hardware limit.
- **phi3.5:3.8b** (generation fallback): Half the size, runs on weaker hardware.
- **mistral-nemo:12b** (generation premium): Best quality, needs 16GB+ RAM.

---

## Timeline - 12 Weeks

### Phase 0: Foundation (Weeks 1-2)
**Goal**: Working Core Server + extension skeleton + Ollama integration

| Week | Deliverables |
|------|-------------|
| W1 | Repository setup (TypeScript, monorepo with npm workspaces, build pipeline, linting, CI). AI MailPilot Core Server scaffold (Node.js, Fastify). Thunderbird extension scaffold (manifest, background script, empty sidebar). Dev workflow: build -> install -> reload. Study Thunderbird MailExtension APIs + Ollama API. |
| W2 | Ollama client service in Core (health check, model list, pull model, embeddings API, generate API). IMAP connector in Core (connect, list folders, fetch email headers + bodies). Onboarding wizard UI in extension (detect Core Server, detect Ollama, model preset selection, folder picker). Settings page (Ollama URL, model config, auto-index toggle). REST API: Core <-> Extension communication. |

**Milestone 1 checkpoint**: Core Server runs locally, connects to Ollama and IMAP. Extension installs in Thunderbird, communicates with Core, user can configure models and select folders.

---

### Phase 1: Embedding Pipeline (Weeks 3-4)
**Goal**: Emails get embedded and cached efficiently

| Week | Deliverables |
|------|-------------|
| W3 | Email text extraction via IMAP in Core (subject + body). Text preprocessing (strip HTML, normalize whitespace, truncate to model's token limit). Embedding Service in Core (call Ollama, return Float32Array). SQLite persistence layer in Core (store/retrieve/invalidate embeddings by messageId + modelId). |
| W4 | Batch embedding orchestrator (paginated folder processing, progress tracking, pause/resume). Auto-index listener (new email event -> embed if opt-in). Embedding cache invalidation (model change -> mark stale). REST endpoints for progress + status. Progress UI in extension sidebar. |

**Milestone 2 checkpoint**: User can index a folder, see progress, embeddings are cached in Core's SQLite. Auto-index works for new emails.

---

### Phase 2: Clustering & Categorization (Weeks 5-7)
**Goal**: Automatic category discovery + manual management + email categorization

| Week | Deliverables |
|------|-------------|
| W5 | K-means implementation (in Core, can offload to worker thread). Auto-K selection (silhouette score). Cluster result data structure (centroid, member messageIds, metadata). REST endpoints for clustering operations. |
| W6 | Generation Service in Core (LLM calls for cluster labels and descriptions). Category UI in extension (list view with labels, descriptions, email counts, representative emails). Manual category editing (rename, merge, delete, create). Persistent centroid storage in SQLite. |
| W7 | Email categorization flow (right-click -> extension calls Core API -> ranked categories with scores). Explanation generation (why this category?). Tag and folder suggestion UI (user confirms or edits). Learning loop (confirmation -> centroid update via running mean). Re-clustering trigger. |

**Milestone 3 checkpoint**: Full core loop works - index -> cluster -> label -> categorize -> learn. This is a shippable product.

---

### Phase 3: Polish & Ship Core (Weeks 8-9)
**Goal**: Production-quality core ready for release

| Week | Deliverables |
|------|-------------|
| W8 | IMAP sync (tags and folder moves reflect on mail server). Edge cases (empty folders, single-email clusters, model unavailable, Ollama disconnected, Core Server not running). Error handling and graceful degradation throughout. Performance optimization (large mailbox testing with 5000+ emails). |
| W9 | i18n (English + French at minimum). Accessibility pass (keyboard navigation, screen reader labels). User-facing documentation (help panel inside extension). ATN packaging and manifest compliance. Core Server installer/launcher. First ATN submission. |

**Milestone 4 checkpoint**: Core extension + server submitted to Thunderbird Add-on Store. Works end-to-end on real mailbox.

---

### Phase 4: RAG Chat + MCP (Weeks 10-11)
**Goal**: "Ask your mailbox" via Thunderbird UI AND MCP clients

| Week | Deliverables |
|------|-------------|
| W10 | MCP server interface on Core (expose tools via MCP protocol). `search_emails` tool (query embedding + cosine similarity over cached embeddings). `ask_mailbox` tool (retrieve -> assemble context -> LLM generate). Chat UI in Thunderbird extension (sidebar tab). REST endpoints for chat. |
| W11 | Source attribution (clickable references to source emails in responses). Conversation history (multi-turn context). Search filters (date range, folder, sender). Test with Claude Desktop as MCP client. Performance optimization for large result sets. |

**Milestone 5 checkpoint**: Users can ask natural language questions from Thunderbird UI or any MCP client and get answers sourced from their emails.

---

### Phase 5: Federated + Final Polish (Week 12)
**Goal**: Stretch features + documentation + release

| Week | Deliverables |
|------|-------------|
| W12 | If time allows: federated server prototype (shared centroids API, org auth). Final documentation (README, CONTRIBUTING, user guide). Performance benchmarks. Demo video/screenshots for ATN listing. Updated ATN submission with RAG + MCP features. |

**Milestone 6 checkpoint**: Final product delivered.

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | TypeScript (strict mode) | Type safety, maintainability, contributor-friendliness |
| Core Server runtime | Node.js | Native TypeScript support, excellent IMAP libraries, MCP SDK available |
| Core Server framework | Fastify | Lightweight, fast, schema-based REST API |
| MCP SDK | @modelcontextprotocol/sdk | Official MCP TypeScript SDK for building MCP servers |
| Extension API | Thunderbird MailExtension API (WebExtension) | Only supported API for TB 115+ |
| Build tool | Vite (extension) + tsup (core server) | Fast TS compilation, extension-compatible bundling |
| Monorepo | npm workspaces | Shared types between core and extension, zero extra tooling |
| Linting | ESLint + Prettier | Code consistency |
| Testing | Vitest | Fast, TS-native |
| Local AI runtime | Ollama | OpenAI-compatible API, easy model management, cross-platform |
| Embedding model (default) | bge-m3 via Ollama | Best multilingual (French+English), 1024 dim |
| Generation model (default) | mistral:7b via Ollama | French company, org partnership, fits 8B limit |
| Persistence (Core) | SQLite (via better-sqlite3) | Lightweight, no external DB server, handles vectors well |
| Config storage (Extension) | browser.storage.local | Thunderbird's native extension storage |
| IMAP client | imapflow | Modern, Promise-based IMAP library for Node.js |
| Heavy computation | Worker threads (Core) | Non-blocking K-means, similarity search |
| i18n | Thunderbird i18n API (browser.i18n) | Native, follows ATN conventions |
| CI/CD | GitHub Actions | Automated build, lint, test on every PR |
| Package/Distribution | ATN (extension) + npm/binary (core server) | Official channels |

---

## Repository Structure

```
ai-mailpilot/
├── .github/
│   └── workflows/
│       └── ci.yml
├── packages/
│   ├── shared/
│   │   ├── src/
│   │   │   ├── index.ts
│   │   │   ├── constants.ts
│   │   │   └── types/
│   │   │       ├── email.ts
│   │   │       ├── embedding.ts
│   │   │       ├── category.ts
│   │   │       ├── api.ts
│   │   │       └── config.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   ├── core/
│   │   ├── src/
│   │   │   └── server.ts
│   │   ├── tsup.config.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── extension/
│       ├── src/
│       │   ├── background/
│       │   │   └── background.ts
│       │   ├── api-client/
│       │   │   └── core-client.ts
│       │   ├── types/
│       │   │   └── thunderbird.d.ts
│       │   └── ui/
│       │       └── sidebar/
│       │           ├── sidebar.html
│       │           ├── sidebar.css
│       │           └── sidebar.ts
│       ├── _locales/
│       │   ├── en/messages.json
│       │   └── fr/messages.json
│       ├── manifest.json
│       ├── vite.config.ts
│       ├── package.json
│       └── tsconfig.json
├── docs/
│   ├── PROJECT_PLAN.md
│   └── COMPETITIVE_ANALYSIS.md
├── .editorconfig
├── .gitattributes
├── .gitignore
├── .nvmrc
├── .prettierrc
├── eslint.config.js
├── tsconfig.base.json
├── package.json
├── LICENSE
├── README.md
└── CONTRIBUTING.md
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Thunderbird WebExtension API limitations | High | Medium | Extension is thin UI client. Heavy logic lives in Core Server, not extension. API limitations affect only UI, not intelligence. |
| Ollama embedding speed too slow for 50k emails | Medium | High | Batch processing with progress UI. Pause/resume. Allow overnight indexing. Optimize text preprocessing. |
| IMAP complexity (auth methods, server quirks) | Medium | High | Use battle-tested imapflow library. Test against multiple IMAP servers early (Gmail, Outlook, Renater). |
| Core Server <-> Extension communication reliability | Medium | Medium | Well-defined REST API with shared types. Health check endpoint. Extension detects if Core is down and shows friendly message. |
| SQLite storage limits for large vector sets | Low | Medium | 50k emails x 1024-dim float32 = ~195MB - well within SQLite limits. Add cleanup for old model embeddings. |
| K-means quality poor on email embeddings | Medium | Medium | Test with real email data early. Implement fallback algorithms. Auto-K selection helps. |
| MCP protocol changes (still evolving) | Low | Low | Use official SDK. MCP interface is a thin layer - easy to update. |
| Scope creep into Tier 3 at expense of core quality | Medium | High | Strict milestone gates. Tier 2 only starts when Tier 1 passes manual QA. |

---

## Success Criteria

| Criterion | Metric |
|-----------|--------|
| Technical depth | Hybrid architecture (Core Server + Extension + MCP). Custom K-means with auto-K. Model-agnostic abstraction. Learning loop. IMAP integration. |
| Usability | Zero-to-organized in < 5 minutes. Non-technical user completes onboarding without external docs. |
| Problem-solving | Professor can categorize 1000+ emails with < 20 manual corrections. Categories are meaningful and improve over time. |
| Impact | Published on ATN. Works with any MCP client. Handles 5000+ email mailbox. Performance acceptable on 8B model hardware. |
| Code quality | TypeScript strict mode. >80% test coverage on services layer. CI passes on all PRs. Clean monorepo structure. |
| Privacy | Zero email content leaves machine in default mode. Clear privacy documentation. Configurable privacy levels. |
