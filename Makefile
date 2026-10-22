VENV := .venv
BIN := $(VENV)/bin

.PHONY: install lint format test test-integration ollama chat eval

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

chat:
	$(BIN)/sensai chat $(ARGS)

sandbox-image:
	docker build -t sensai-sandbox:latest sandbox/

eval:
	$(BIN)/python -m sensai.evals $(ARGS)
