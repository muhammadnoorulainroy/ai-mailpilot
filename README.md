# AI MailPilot

**Intelligent email organization powered by local AI. Categorize, search, and chat with your inbox privately.**

AI MailPilot is an open-source email intelligence platform that uses local AI models (via [Ollama](https://ollama.ai)) to semantically organize your emails. It consists of a local core server, a Thunderbird extension, and an MCP interface for integration with any AI assistant.

## Features

- **Semantic email embeddings** - understand what your emails are about, not just keyword matching
- **Auto-discovery of categories** - K-means clustering with automatic cluster count selection
- **AI-generated labels** - local LLM names and describes each category
- **Smart categorization** - right-click any email to see which categories it matches with confidence scores
- **Learning loop** - the system improves as you confirm or correct suggestions
- **Chat with your emails** - ask natural language questions, get answers sourced from your mailbox (RAG)
- **MCP interface** - use from Claude Desktop, VS Code Copilot, or any MCP-compatible client
- **Privacy-first** - all processing happens locally via Ollama. No email content ever leaves your machine.
- **Works with any IMAP provider** - Gmail, Outlook, institutional email, self-hosted

## Architecture

```
+------------------+     +------------------+     +--------------+
|   Thunderbird    |     |   MCP Clients    |     |  CLI (soon)  |
|   Extension      |     |  (Claude, etc.)  |     |              |
+--------+---------+     +--------+---------+     +------+-------+
         | REST                   | MCP                   | REST
         +------------------------+-------------------+---+
                                  |
                        +---------v----------+
                        |  AI MailPilot Core  |
                        |  (local server)     |
                        +--+----------+-------+
                           |          |
                      +----v---+ +----v---+
                      | Ollama | |  IMAP  |
                      |(local) | | Server |
                      +--------+ +--------+
```

## Quick Start

### Prerequisites

- [Node.js](https://nodejs.org/) >= 20
- [Ollama](https://ollama.ai) installed and running
- [Thunderbird](https://www.thunderbird.net/) >= 115

### Setup

```bash
# Clone the repo
git clone https://github.com/muhammadnoorulainroy/ai-mailpilot.git
cd ai-mailpilot

# Install dependencies
npm install

# Pull recommended models
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

## Project Structure

```
ai-mailpilot/
├── packages/
│   ├── shared/          # Shared types and constants
│   ├── core/            # Core server (REST API + MCP + AI engine)
│   ├── extension/       # Thunderbird extension (UI)
│   └── cli/             # CLI tool (coming soon)
├── docs/                # Architecture documentation
└── .github/workflows/   # CI/CD
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | TypeScript (strict mode) |
| Core Server | Node.js + Fastify |
| Extension | Thunderbird WebExtension API |
| Local AI | Ollama (bge-m3 for embeddings, mistral:7b for generation) |
| Storage | SQLite (embeddings, categories) |
| IMAP | imapflow |
| MCP | @modelcontextprotocol/sdk |
| Build | tsup (core), Vite (extension) |
| Testing | Vitest |
| CI/CD | GitHub Actions |

## Model Presets

| Preset | Embedding | Generation | RAM Required |
|--------|-----------|-----------|-------------|
| Lightweight | nomic-embed-text | phi3.5:3.8b | 4 GB |
| Recommended | bge-m3 | mistral:7b | 8 GB |
| Maximum | bge-m3 | mistral-nemo:12b | 16 GB |

## Roadmap

- [x] Project scaffolding and architecture
- [ ] Core server with Ollama integration
- [ ] Embedding pipeline (incremental, cached)
- [ ] K-means clustering with auto-K
- [ ] Category management and learning loop
- [ ] Thunderbird extension UI (sidebar, context menus)
- [ ] IMAP sync (tags and folders)
- [ ] MCP server interface
- [ ] RAG chat ("Ask your mailbox")
- [ ] CLI tool
- [ ] Federated learning (org-wide categories)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
