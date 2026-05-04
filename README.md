# AI MailPilot

**Intelligent email organization for Thunderbird, powered by local AI. Triage, categorize, and chat with your inbox privately.**

AI MailPilot is an open-source Thunderbird extension that uses local AI to intelligently organize your inbox. It automatically classifies emails into actionable buckets (urgent, summarize, spam, personal), discovers semantic categories, drafts replies for urgent items, and lets you chat with your mailbox using RAG. Email content never leaves your machine.

## Features

- **Triage dashboard** - urgent items with auto-drafted replies, mailing list summaries, and spam filtering on first launch
- **Multi-label semantic categories** - emails can belong to multiple categories, discovered automatically
- **Personal email filter** - work and personal emails are kept separate, never federated
- **Chat with your emails** - ask natural language questions, get answers sourced from your mailbox (RAG)
- **Learning loop** - the system improves as you confirm or correct suggestions
- **Privacy-first** - all processing happens locally via Ollama (or institutional LLM). Email content never leaves your machine.
- **Configurable LLM runtime** - works with local Ollama, institutional servers, LM Studio, vLLM, or any OpenAI-compatible endpoint
- **Optional federated learning** - share spam signals and category centroids across an organization without sharing email content

## Architecture

```
+----------------------------------+
|   Thunderbird Extension          |
|   - Dashboard (urgent + drafts)  |
|   - Categories sidebar           |
|   - Chat panel                   |
|   - Settings + onboarding        |
+----------------+-----------------+
                 | REST (HTTP/JSON)
                 v
+----------------+-----------------+
|   AI MailPilot Core Server       |
|   - Triage classifier            |
|   - Multi-label clustering       |
|   - RAG engine                   |
|   - SQLite + Vector DB           |
+----------------+-----------------+
                 | OpenAI-compatible API
                 v
+----------------------------------+
|   LLM Runtime                    |
|   (Ollama / Institutional /      |
|    LM Studio / Custom)           |
+----------------------------------+
```

The Thunderbird extension owns the user interface and reads email content via Thunderbird's native APIs. The Core Server handles all AI logic and storage. The LLM runtime is configurable and treated as a black box behind an OpenAI-compatible API.

For power users: the Core Server can optionally expose its tools via MCP, so Claude Desktop, VS Code Copilot, or any MCP-compatible client can interact with your email intelligence. See [docs/MCP_INTEGRATION.md](docs/MCP_INTEGRATION.md) (coming soon).

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) >= 20
- An OpenAI-compatible LLM runtime ([Ollama](https://ollama.ai), LM Studio, or institutional server)
- [Thunderbird](https://www.thunderbird.net/) >= 115

### Setup

```bash
# Clone the repo
git clone https://github.com/muhammadnoorulainroy/ai-mailpilot.git
cd ai-mailpilot

# Install dependencies
npm install

# If using local Ollama, pull recommended models
ollama pull bge-m3
ollama pull mistral:7b

# Start the core server
npm run dev:core

# Build the extension (in another terminal)
npm run dev:extension
```

### Install the Extension in Thunderbird

1. Open Thunderbird
2. Go to **Add-ons Manager** (Menu > Add-ons and Themes)
3. Click the gear icon > **Install Add-on From File...**
4. Select `packages/extension/dist/ai-mailpilot.xpi`

The onboarding wizard will guide you through:
- Detecting your LLM runtime
- Choosing a model preset
- Selecting which folders to index

## Project Structure

```
ai-mailpilot/
├── packages/
│   ├── shared/          # Shared types and constants
│   ├── core/            # Core server (Fastify + AI engines + storage)
│   └── extension/       # Thunderbird extension (UI)
├── docs/                # Architecture documentation
└── .github/workflows/   # CI/CD
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | TypeScript (strict mode) |
| Core Server | Node.js + Fastify |
| Extension | Thunderbird WebExtension API |
| LLM Runtime | OpenAI-compatible (Ollama / institutional / custom) |
| Embedding model | bge-m3 (multilingual, 1024 dim) |
| Generation model | mistral:7b (8B, French + English) |
| SQL Storage | SQLite (better-sqlite3) |
| Vector Storage | Qdrant or LanceDB |
| Clustering | Multi-label (HDBSCAN or threshold-based) |
| Email source | Thunderbird APIs (no separate IMAP path) |
| Build | tsup (core), Vite (extension) |
| Testing | Vitest |
| CI/CD | GitHub Actions |

## Model Presets (for local Ollama)

| Preset | Embedding | Generation | RAM Required |
|--------|-----------|-----------|-------------|
| Lightweight | nomic-embed-text | phi3.5:3.8b | 4 GB |
| Recommended | bge-m3 | mistral:7b | 8 GB |
| Maximum | bge-m3 | mistral-nemo:12b | 16 GB |

## Roadmap

- [x] Project scaffolding and architecture
- [ ] Core server with OpenAI-compatible LLM client
- [ ] Vector database integration
- [ ] Embedding pipeline (incremental, cached)
- [ ] Triage classifier (urgent / summarize / spam / personal)
- [ ] Auto-draft reply generation
- [ ] Multi-label clustering (HDBSCAN or threshold-based)
- [ ] Category management and learning loop
- [ ] Thunderbird extension UI (dashboard, sidebar, context menus)
- [ ] Tag and folder sync
- [ ] RAG chat ("Ask your mailbox")
- [ ] Federated learning (spam + category sharing)
- [ ] MCP wrapper (optional, for external AI clients)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
