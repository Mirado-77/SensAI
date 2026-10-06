# SENSAI

Command-line AI assistant built on a local model served by Ollama.
Every call to the model goes through the Ollama HTTP API, with no LLM framework.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com), installed locally or started with `make ollama` (requires Docker)
- At least one model pulled in Ollama, for example `ollama pull gemma4:e2b`

## Installation

```bash
make install
```

This creates `.venv/` and installs the project with its development dependencies.

## Usage

```bash
make chat ARGS="--model gemma4:e2b"
```

Or, with the virtual environment active (`source .venv/bin/activate`): `sensai chat`.

```text
$ sensai chat --model gemma4:e2b
SENSAI | model: gemma4:e2b | /exit or Ctrl-D to quit
you> My name is Phil. Reply with just: OK
sensai> OK
128 tokens (prompt 25, completion 103)
you> What is my name? Reply with just the name.
sensai> Phil
49 tokens (prompt 47, completion 2)
```

| Input | Effect |
|---|---|
| a line of text | one turn: the answer streams as the model produces it |
| an empty line | ignored, the model is not called |
| `Ctrl-C` during an answer | stops the generation and returns to the prompt; the text received so far is kept in the history |
| `/exit`, `/quit`, `Ctrl-D` or `Ctrl-C` at the prompt | quit |

Options of `sensai chat`:

| Option | Effect |
|---|---|
| `--model NAME` | Ollama model to use, overriding the configuration |
| `--config FILE` | TOML file that overrides `configs/default.toml`; it only lists the keys to change |

Errors are shown without a stack trace and the chat stays open: Ollama not running, model not
installed, a limit reached (steps, tool calls, tokens, time).

## Configuration

`configs/default.toml` holds every default. A file passed with `--config` only needs the keys
to change, for example:

```toml
[model]
name = "llama3.2:3b"
```

| Section | Keys | Meaning |
|---|---|---|
| `[model]` | `name`, `host`, `timeout_s` | Ollama model, server address, maximum time without a response |
| `[runtime]` | `max_steps`, `max_tool_calls`, `timeout_s`, `max_tokens` | Limits of one turn |
| `[storage]` | `db_path` | sqlite database holding sessions, messages and the event journal |
| `[policy]` | `read`, `write`, `exec` | Before a tool runs: `allow`, `ask` (user approval) or `deny` |

## How it works

```text
CLI -> AgentRuntime -> OllamaGateway -> Ollama
            |-> ToolRegistry, approval policy, UserChannel
            |-> SessionStore (sqlite3: messages stored as a tree)
            `-> EventBus -> console renderer, event journal (sqlite3)
```

- `AgentRuntime` runs one turn as a state machine: model stream, optional tool calls
  (policy, approval, execution), then persistence. By default a turn makes a single model call.
- Four contracts decouple the parts: `ModelGateway`, `Tool`, `EventSink` / `UserChannel` and
  `SessionStore` (`src/sensai/core/contracts.py`). Each has a fake in `core/fakes.py` and a
  contract test in `tests/contract/`, which also runs against the real implementations.
- Everything that happens is an event (`run.started`, `model.chunk`, `tool.finished`, ...). The
  console, the sqlite journal and the tests consume the same stream. Payloads are documented at
  the top of `src/sensai/core/runtime.py`.
- Content coming from outside (a file, an MCP tool) is marked as untrusted data before it
  reaches the model.

## Development

| Command | Effect |
|---|---|
| `make test` | Unit and contract tests (none of them needs Ollama or the network) |
| `make test-integration` | Tests marked `integration`, against a real Ollama server (set `SENSAI_TEST_MODEL` to pick the model) |
| `make lint` | `ruff check` and `ruff format --check` |
| `make format` | Fix style with ruff |
| `make ollama` | Start an Ollama server in Docker on port 11434 |
| `make chat ARGS="..."` | Start the chat |

Database changes go in a new `src/sensai/db/migrations/NNN_<feature-id>_<description>.sql`
file, one migration per pull request. Migrations are applied when the database is opened.

## Repository layout

- `src/sensai/core/`: types, events, contracts, config, event bus, runtime, fakes. Never imports `cli` or `api`.
- `src/sensai/model/`: `OllamaGateway`.
- `src/sensai/tools/`: tool registry.
- `src/sensai/safety/`: approval policy, untrusted content marking.
- `src/sensai/memory/`: sqlite `SessionStore`.
- `src/sensai/observability/`: sqlite event journal.
- `src/sensai/db/`: connection and migrations.
- `src/sensai/cli/`: chat loop, console rendering, approval prompt.
- `configs/`: TOML configuration (`default.toml`, profiles, personas, policies).
- `docker/ollama/`: Dockerfile and launch script for the Ollama server.
- `tests/{unit,contract,integration,fixtures}/`: tests.
- `docs/user-stories/`: user stories, one file per feature, plus `_template.md`.

## Features

| ID | Feature | Status |
|---|---|---|
| B1 | Streaming CLI chat with in-session history, clean error handling, selectable model | done ([user stories](docs/user-stories/B1.md)) |

Foundations already in place for the next features: the tool-call path of the agent loop
(registry, approval by risk level, untrusted content marking), the event journal, and the
message tree of the session store. No tool is registered yet.
