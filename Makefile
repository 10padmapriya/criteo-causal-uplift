# Usage: make setup && make java && make data && make all
PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: help setup java data lint format test all

help:
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

setup: ## create .venv and install pinned requirements
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

java: ## check that Java 17 is the active JDK (needed by PySpark)
	@java -version 2>&1 | head -1
	@java -version 2>&1 | grep -q '"17\.' || (echo "Java 17 required. Mac: brew install openjdk@17, then export JAVA_HOME=$$(/usr/libexec/java_home -v 17)" && exit 1)

data: ## download Criteo v2.1, write full and 10% Parquet, check published rates
	$(PY) -m src.data

lint: ## ruff check and format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## auto fix lint and format
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

test: ## run pytest
	$(PY) -m pytest

# Grows as phases land: experiment, observational, uplift, policy, report.
all: lint test data ## full pipeline from a fresh clone
