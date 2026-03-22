SHELL := /bin/bash

.PHONY: help setup env models backend frontend run dev lint format typecheck test check clean

help:
	@printf "Available targets:\n"
	@printf "  make setup      - Sync deps and create .env if missing\n"
	@printf "  make models     - Download all models\n"
	@printf "  make backend    - Run FastAPI backend\n"
	@printf "  make frontend   - Run Streamlit frontend\n"
	@printf "  make run        - Run backend + frontend in one command\n"
	@printf "  make lint       - Run ruff check\n"
	@printf "  make format     - Run ruff format\n"
	@printf "  make typecheck  - Run mypy\n"
	@printf "  make test       - Run pytest\n"
	@printf "  make check      - Run lint + typecheck + test\n"

setup: env
	uv sync

env:
	@if [ ! -f .env ]; then cp .env.example .env; fi

models:
	uv run python scripts/download_models.py --all

backend:
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	uv run streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0

run:
	@set -euo pipefail; \
	trap 'kill 0' INT TERM EXIT; \
	uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000 & \
	BACKEND_PID=$$!; \
	sleep 2; \
	uv run streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0; \
	kill $$BACKEND_PID

dev: run

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

test:
	uv run pytest -q

check: lint typecheck test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
