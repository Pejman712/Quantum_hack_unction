.PHONY: setup test lint lint-imports format run

# One-command bootstrap for a fresh clone: install deps + enable the pre-push hook.
setup:
	uv sync --dev
	git config core.hooksPath .githooks
	@echo "Setup complete. Tests will run automatically before every 'git push'."

test:
	uv run pytest

lint: lint-imports
	uv run ruff check .

# Dedicated import-sorting check (ruff's isort "I" rules). Run with --fix to apply.
lint-imports:
	uv run ruff check --select I .

format:
	uv run ruff format .

# Render diagrams for a circuit, e.g.: make run QASM=qasm_data/easy/challenge-16_12.qasm
run:
	uv run quantum-hack $(QASM)
