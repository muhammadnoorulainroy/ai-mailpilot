# Contributing to AI MailPilot

AI MailPilot is a privacy-sensitive Thunderbird extension and local service. Contributions should
preserve conservative automation, account isolation, and explicit user control over mailbox and
taxonomy changes.

## Development Setup

### Requirements

- Node.js 20 or 22
- npm
- Thunderbird 115 or later for extension testing
- Ollama or another local OpenAI-compatible endpoint
- `bge-m3`, which must return 1024-dimensional embeddings
- A local generation model such as `llama3.1` or one of the presets exposed in Settings

Install the repository and default local models:

```bash
npm install
ollama pull bge-m3
ollama pull llama3.1
```

Start Core in watch mode:

```bash
npm run dev:core
```

Build the extension once or watch for changes:

```bash
npm run build -w @ai-mailpilot/extension
# or
npm run dev:extension
```

Load `packages/extension/dist/manifest.json` as a temporary Thunderbird add-on. Enter the six-digit
pairing code printed by Core in the extension Settings page.

Python dependencies are optional and are needed only for the experimental tools documented in
[`ml/README.md`](ml/README.md).

## Repository Layout

```text
packages/shared/     Shared API types, configuration types, and constants
packages/core/       Fastify API, services, repositories, migrations, and tests
packages/extension/  Thunderbird WebExtension, dashboard, settings, and extension tests
ml/                  Optional clustering and cross-encoder experiments
docs/                Current implementation status and competitive analysis
```

There is no CLI package, Core IMAP connector, MCP server, or federated-learning service in the
current product.

## Working Agreements

- Follow the existing TypeScript strict-mode, repository, service, and route boundaries.
- Validate external input before it reaches services or repositories.
- Scope every account-owned read and write by `account_id`.
- Preserve `assigned_by = user` provenance and never let an automatic job overwrite a user-confirmed
  category assignment.
- Keep structural taxonomy writes transactional and require review for generated add, merge, split,
  or retire operations.
- Keep embeddings local. New cloud-capable generation calls need an explicit policy decision and a
  visible user opt-in.
- Do not log email bodies, attachment text, API keys, database keys, or bearer tokens.
- Keep edits focused. Do not combine behavior changes with unrelated refactors.
- Add comments only for non-obvious invariants or control flow.

## Tests

Scale tests to the risk of the change:

- Pure algorithms: unit tests for normal, boundary, malformed, multilingual, and deterministic
  behavior.
- Repositories and migrations: tests against a real temporary SQLite database, including rollback
  and account-isolation cases.
- Taxonomy mutations: success, stale proposal, user-provenance, duplicate, and rollback tests.
- Extension integration: tests for Thunderbird API failures, tag collisions, user-tag preservation,
  and visible formatting helpers.
- Privacy changes: prove both the allowed path and the denied/default path.

Run the complete local gate before submitting a pull request:

```bash
npm run typecheck
npm run lint
npm run format:check
npm test
npm run build
```

For optional Python tooling:

```bash
python -m pytest ml/tests -q
```

## Pull Requests

1. Create a focused branch from `main`.
2. Describe the user-visible behavior and the invariant being protected.
3. Include tests that would fail without the change.
4. Call out schema migrations, privacy effects, model-provider changes, and destructive mailbox
   operations explicitly.
5. Update `README.md`, `docs/`, and the relevant local knowledge-vault note when behavior or
   architecture changes.
6. Report the verification commands that were actually run. Do not claim live Thunderbird or
   real-mailbox testing unless it was performed.

## Database Changes

- Add a forward migration; do not rewrite a migration that may already exist in a user database.
- Preserve existing rows and support upgrades from the immediately preceding schema.
- Keep changes idempotent where a previous development migration may have partially landed.
- Test both a fresh database and the upgrade path.
- Never add real mailbox databases, keys, logs, decrypted exports, or copied user data to Git.

## Documentation

The [README](README.md) is the public entry point. [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
records shipped, partial, experimental, and deferred behavior. Historical design notes may explain
why a decision changed, but they must be labelled as historical and must not be presented as the
current runtime.

Use relative links for repository files and verify that every local link resolves.

## Reporting Issues

Open a GitHub issue with:

- operating system and version;
- Thunderbird version;
- Node.js and Ollama versions;
- configured embedding and generation model names;
- whether the failing feature used local or cloud generation;
- steps to reproduce and the expected behavior;
- sanitized Core logs that contain no message content or secrets; and
- whether the failure is repeatable or intermittent.

For a suspected security or privacy issue, avoid attaching real emails, database files, API keys,
pairing codes, or authentication tokens to a public issue.
