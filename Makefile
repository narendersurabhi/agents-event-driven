PYTHON ?= python
PYTHONPATH ?= src

.PHONY: format lint typecheck test check demo-happy demo-blocked demo-rollback
.PHONY: compose-up compose-down install
.PHONY: complexity

format:
	$(PYTHON) -m ruff format .

lint:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m ruff check .

typecheck:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m mypy

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest

complexity:
	$(PYTHON) -m scripts.complexity_report

check: format lint typecheck test

install:
	$(PYTHON) -m pip install -r requirements.txt

demo-happy:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m govguard.demo.cli happy

demo-blocked:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m govguard.demo.cli blocked

demo-rollback:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m govguard.demo.cli rollback

compose-up:
	docker compose up -d

compose-down:
	docker compose down
