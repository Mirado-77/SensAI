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
| `/tools` | list the tools the model can call, with their risk level |
| `/profile` | show your profile; `/profile instructions <text>`, `/profile set <key> <value>`, `/profile unset <key>`, `/profile clear` edit it |
| `/exit`, `/quit`, `Ctrl-D` or `Ctrl-C` at the prompt | quit |

The profile (custom instructions and preferences, e.g. `/profile set language Python`) is saved in
the database and sent to the model as a system message at every turn, in every session.

Every chat is saved. `sensai sessions` lists them (newest first), and `sensai chat --resume`
continues the most recent one; `--resume <id>` continues another, the first characters of the
id are enough.

Options of `sensai chat` (also accepted by `sensai ask`):

| Option | Effect |
|---|---|
| `--model NAME` | Ollama model to use, overriding the configuration |
| `--config FILE` | TOML file that overrides `configs/default.toml`; it only lists the keys to change |
| `--resume [ID]` | `chat` only: continue a saved session, the most recent one without `ID` |

### Structured answers: `sensai ask`

`sensai ask --schema FILE QUESTION` asks one question and prints a JSON answer that conforms to
the JSON Schema in `FILE`, on stdout only, so it can feed another tool. Pass `-` as the question
to read it from stdin. Example (the exact answer depends on the model):

```text
$ sensai ask --schema configs/schemas/bug_report.json "calc.py sums range(1, n): why is total(3) 3?"
{
  "file": "calc.py",
  "cause": "range(1, n) stops before n, so the last value is never added",
  "fix": "use range(1, n + 1)",
  "severity": "medium"
}
```

The schema is sent to Ollama (`format`) and the answer is validated. A non-conforming answer is
sent back to the model with the validation errors for one repair attempt (`[structured]
max_repairs`). Exit codes: 0 success, 1 model failure or answer still invalid, 2 unusable schema
or configuration.

The same validation protects tools: arguments that do not match a tool's JSON Schema are never
executed nor submitted for approval; the model gets the errors back and can correct its call.

### System prompt versions

The system prompt sent before the profile is a versioned file of `configs/prompts/system/`
(`v1.txt`, `v2.txt`, ...). A new version is a new file; `[prompts] system` picks the active one.

```bash
sensai prompts                  # list the versions, * marks the active one
sensai prompts diff v1 v2       # what changed between two versions
sensai prompts stats            # runs, failures, tokens, duration and tool errors per version
sensai chat --prompt v1         # roll back for this chat
```

In the chat, `/prompt` shows the active version and `/prompt v1` switches to it for the next
turns, without losing the history. Each run records `<version>@<sha>`, so an edited file shows
up as a new line in `stats`.

### Tools from MCP servers

The model can call tools served over MCP (stdio). Declare servers in a `--config` file:

```toml
[[mcp.servers]]
name = "code"                     # tools are named code__<tool>
command = "sensai-mcp"            # our own server, installed with SENSAI
args = ["--root", "demo-repo"]
risk = "read"

[[mcp.servers]]
name = "fs"                       # an existing server: the official filesystem server
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem@2026.8.31", "demo-repo"]
risk = "auto"                     # readOnlyHint decides between read and exec
```

`sensai-mcp --root DIR` serves two read-only tools on the Python files under `DIR`:
`python_symbols` (classes, functions and methods of a file, with line numbers) and
`find_references` (lines that mention an identifier). Paths that leave `DIR` are refused.

MCP tools go through the same path as native tools: argument validation, approval policy by risk
level, event journal. Their output is marked as untrusted data. A server that fails to start is
reported and skipped; a server that crashes or stops answering (`[mcp] timeout_s`) turns into a
tool error the model sees.

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
| `[structured]` | `max_repairs` | Repair attempts when a structured answer does not match its schema (0 to 3) |
| `[prompts]` | `system` | Active system prompt version, a file of `configs/prompts/system/` without `.txt` (`""` for none); also `sensai chat --prompt v1` |
| `[questions]` | `enabled`, `max_per_turn` | Structured questions: the model may call `ask_user` to ask one short question with options (1 to 5 per turn) |
| `[react]` | `enabled`, `max_plan_steps` | Bounded ReAct loop: a short structured plan before acting (1 to 10 steps); `enabled = false` turns it off |
| `[mcp]` | `timeout_s`, `[[mcp.servers]]` | Maximum time of one MCP request; MCP servers: `name`, `command`, `args`, `env`, `cwd`, `enabled`, `risk` (`read`, `write`, `exec` or `auto`) |
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
  (policy, approval, execution), then persistence. With `[react]` enabled (A1), a structured
  planning call comes first, and each action and observation of the tool loop is an event
  (`react.plan`, `react.action`, `react.observation`), bounded by `[runtime] max_steps`.
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
- `src/sensai/model/`: `OllamaGateway`, structured output (`structured.py`).
- `src/sensai/tools/`: tool registry, MCP client and adapter (`mcp/`).
- `src/sensai/mcp_server/`: `sensai-mcp`, our own MCP server.
- `src/sensai/safety/`: approval policy, untrusted content marking.
- `src/sensai/memory/`: sqlite `SessionStore`.
- `src/sensai/observability/`: sqlite event journal.
- `src/sensai/db/`: connection and migrations.
- `src/sensai/cli/`: chat loop, console rendering, approval prompt.
- `configs/`: TOML configuration (`default.toml`, profiles, personas, policies) and example JSON Schemas (`schemas/`).
- `docker/ollama/`: Dockerfile and launch script for the Ollama server.
- `tests/{unit,contract,integration,fixtures}/`: tests.
- `docs/user-stories/`: user stories, one file per feature, plus `_template.md`.

## Features

| ID | Feature | Status |
|---|---|---|
| B1 | Streaming CLI chat with in-session history, clean error handling, selectable model | done ([user stories](docs/user-stories/B1.md)) |
| T1 | MCP: hand-written stdio client, our own server `sensai-mcp`, existing servers by configuration, same permissions as native tools | done ([user stories](docs/user-stories/T1.md)) |
| M1 | Session persistence and user profile: `sensai sessions`, `sensai chat --resume`, `/profile` sent at every turn | done ([user stories](docs/user-stories/M1.md)) |
| A4 | Versioned system prompts: one file per version, active version in config or `/prompt` in the chat, version recorded on every run, `sensai prompts list`, `diff` and `stats` to compare and roll back | done ([user stories](docs/user-stories/A4.md)) |
| X4 | Structured questions: when the request is ambiguous the model asks one short question with numbered options, the answer goes back to it and is saved | done ([user stories](docs/user-stories/X4.md)) |
| A1 | Bounded ReAct loop: short structured plan shown before acting, one event per action and observation, bounded by `max_steps` | done ([user stories](docs/user-stories/A1.md)) |
| T5 | Structured output: JSON validated against a schema with one repair attempt (`sensai ask --schema`), tool arguments validated before any execution | done ([user stories](docs/user-stories/T5.md)) |

Foundations already in place for the next features: the tool-call path of the agent loop
(registry, approval by risk level, untrusted content marking), the event journal, and the
message tree of the session store.
