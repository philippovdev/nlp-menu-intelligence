# NLP Menu Intelligence

Monorepo for the NLP course project: a web app and inference API that turns raw menu text or menu documents into a structured menu.

## Stack

- Frontend: Vue 3 + Vite + TypeScript
- Backend: FastAPI
- Current category model runtime: scikit-learn sparse text pipeline
- OCR/document extraction: PDF text extraction + Tesseract/RapidOCR image OCR

## Repository Layout

```text
frontend/   Vue application
backend/    FastAPI application
.github/    CI workflows
docs/       report and research notes
data/       annotated subsets and evaluation fixtures
```

## Quick Start

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## Local Checks

```bash
make ci-local
```

This runs the frontend tests, typecheck, and production build, plus backend
lint, backend tests, and backend runtime verification.

## Current API

- `GET /api/health`
- `GET /api/v1/health`
- `GET /api/version`
- `GET /api/v1/version`
- `GET /api/status`
- `POST /api/v1/menu/parse`
- `POST /api/v1/menu/parse-file`

## Research Materials

- [Report source](report/main.tex)
- [Problem statement](docs/course/problem-statement.md)
- [Annotation guide](docs/course/annotation-guide.md)
- [Dataset](docs/course/dataset.md)
- [Related work](docs/course/related-work.md)
- [Experiments](docs/course/experiments.md)
- [Results log](docs/course/results-log.md)
- [Live smoke log](docs/course/live-smoke-log.md)

## Deployment

- CI is defined in `.github/workflows/ci.yml`.
- VPS sync deployment is defined in `.github/workflows/deploy.yml`.
- Remote sync logic lives in `ops/remote_sync.sh`.
- Remote app deploy logic lives in `ops/remote_deploy.sh`.
