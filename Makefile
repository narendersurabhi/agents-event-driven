PYTHON ?= python

.PHONY: format lint typecheck test check run-api
.PHONY: complexity

format:
	$(PYTHON) -m black .

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy agents core api

test:
	$(PYTHON) -m pytest

complexity:
	$(PYTHON) -m scripts.complexity_report

check: format lint typecheck test

run-api:
	uvicorn api.app:app --reload
