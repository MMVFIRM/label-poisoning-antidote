.PHONY: test lint audit fed-audit build

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m mypy

audit:
	python -m lpa audit

fed-audit:
	python -m lpa federated-audit

build:
	python -m build
