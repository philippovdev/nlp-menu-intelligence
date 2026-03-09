.PHONY: frontend-install frontend-check backend-install backend-check ci-local

frontend-install:
	cd frontend && npm install

frontend-check:
	cd frontend && npm run typecheck
	cd frontend && npm run build

backend-install:
	cd backend && python3 -m venv .venv
	cd backend && . .venv/bin/activate && pip install -e ".[dev]"

backend-check:
	cd backend && . .venv/bin/activate && ruff check .
	cd backend && . .venv/bin/activate && pytest

ci-local: frontend-check backend-check

