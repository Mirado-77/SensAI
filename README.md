# SENSAI

Command-line AI assistant built on a local model served by Ollama.
Every call to the model goes through the Ollama HTTP API, with no LLM framework.

> Bootstrap in progress: the CLI chat (B1) is not available yet.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com), installed locally or started with `make ollama` (requires Docker)

## Installation

```bash
make install
```

This creates `.venv/` and installs the project with its development dependencies.

## Development commands

| Command | Effect |
|---|---|
| `make test` | Unit and contract tests (none of them needs Ollama or the network) |
| `make test-integration` | Tests marked `integration` (Ollama or Docker required) |
| `make lint` | `ruff check` and `ruff format --check` |
| `make format` | Fix style with ruff |
| `make ollama` | Start an Ollama server in Docker on port 11434 |

## Repository layout

- `src/sensai/core/`: types, events, contracts, runtime. Never imports `cli` or `api`.
- `src/sensai/{model,tools,safety,memory,observability,db,cli}/`: one package per responsibility.
- `configs/`: TOML configuration (`default.toml`, profiles, personas, policies).
- `docker/ollama/`: Dockerfile and launch script for the Ollama server.
- `tests/{unit,contract,integration,fixtures}/`: tests.
- `docs/`: user stories and demo documents.

## Features

| ID | Feature | Status |
|---|---|---|
| B1 | Streaming CLI chat | to do |
