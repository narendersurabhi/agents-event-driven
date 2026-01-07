# Default recipe: run full check
default: check

set shell := ["bash", "-c"]

format:
    python -m black .

lint:
    python -m ruff check .

typecheck:
    python -m mypy agents core api

test:
    python -m pytest

check:
    just format
    just lint
    just typecheck
    just test

complexity:
    python -m scripts.complexity_report

run-api:
    uvicorn api.app:app --reload
