# Contributing to AI MailPilot

Thank you for your interest in contributing to AI MailPilot!

## Development Setup

1. Fork and clone the repository
2. Install dependencies: `npm install`
3. Install Ollama and pull models: `ollama pull bge-m3 && ollama pull mistral:7b`
4. Start the core server: `npm run dev:core`
5. Build the extension: `npm run dev:extension`

## Project Structure

This is a monorepo using npm workspaces:

- `packages/shared` - Shared TypeScript types and constants
- `packages/core` - Core server (REST API, AI engine, IMAP, MCP)
- `packages/extension` - Thunderbird extension (UI)
- `packages/cli` - CLI tool

## Code Style

- TypeScript strict mode
- ESLint + Prettier (run `npm run lint` and `npm run format`)
- No `any` types without justification
- Minimal comments, only where logic is non-obvious

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Ensure `npm run lint` and `npm run test` pass
4. Submit a PR with a clear description

## Reporting Issues

Use GitHub Issues. Include:
- Thunderbird version
- OS and version
- Ollama version and models
- Steps to reproduce
