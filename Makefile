# Works on Windows (cmd.exe or sh) and Linux/macOS. All tools run from .venv.
.PHONY: setup test lint format up down scan bench schema

ifeq ($(OS),Windows_NT)
BOOTSTRAP_PY ?= py -3.11
PY := .venv/Scripts/python.exe
else
BOOTSTRAP_PY ?= python3.11
PY := .venv/bin/python
endif

PIPELINE ?= fixtures/dvp

setup:
	$(BOOTSTRAP_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy src

format:
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

up:
	$(PY) -c "import os, shutil; os.path.exists('.env') or shutil.copy('.env.example', '.env')"
	docker compose up -d --wait

down:
	docker compose down

scan:
	$(PY) -m faultline.cli scan $(PIPELINE)

bench:
	$(PY) -m faultline.bench

schema:
	$(PY) -m faultline.schema.export
