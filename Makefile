VENV := .venv
BIN := $(VENV)/bin

DEMO := $(BIN)/sensai chat --config configs/demo.toml
# sensai-mcp (MCP server of the demo) must be on the PATH even without an activated venv.
DEMO_ENV := PATH="$(abspath $(BIN)):$$PATH"

.PHONY: install lint format test test-integration ollama chat eval eval-quality eval-compare bench \
	demo demo-reset demo-rehearse web web-dev web-test

install:
	python3 -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:
	$(BIN)/pytest

test-integration:
	$(BIN)/pytest -m integration

ollama:
	./docker/ollama/run-ollama.sh

searxng:
	docker compose -f docker/searxng/docker-compose.yml up -d

chat:
	$(BIN)/sensai chat $(ARGS)

web:
	cd web && npm ci && npm run build

web-dev:
	cd web && npm run dev

web-test:
	cd web && npm test && npm run typecheck && npm run lint

sandbox-image:
	docker build -t sensai-sandbox:latest sandbox/

eval:
	$(BIN)/python -m sensai.evals $(ARGS)

eval-quality:
	$(BIN)/python -m sensai.evals.ev1 $(ARGS)

eval-compare:
	$(BIN)/python -m sensai.evals.compare $(ARGS)

bench:
	$(BIN)/python -m sensai.benchmarks $(ARGS)

# Keynote demo (docs/demo/SCENARIO.md): reset before every run, then chat on the demo copy.
demo-reset:
	./docs/demo/reset.sh
	$(DEMO_ENV) $(BIN)/sensai index --config configs/demo.toml

demo:
	$(DEMO_ENV) $(DEMO) $(ARGS)

# Plays the scenario unattended and records it for `scriptreplay` (backup recording).
demo-rehearse: demo-reset
	$(DEMO_ENV) $(BIN)/python docs/demo/rehearse.py $(ARGS)
