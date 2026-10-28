.PHONY: check test build

check:
	git diff --check
	uv run --no-sync ruff check src tests scripts
	uv run --no-sync ruff format --check src tests scripts

test:
	uv run --no-sync pytest -q

build:
	uv build
