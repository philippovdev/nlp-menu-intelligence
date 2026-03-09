# NLP Menu Intelligence

Monorepo for the NLP course project: a web app and inference API that turns raw menu text or menu documents into a structured menu.

## Stack

- Frontend: Vue 3 + Vite + TypeScript
- Backend: FastAPI
- Inference target: ONNX Runtime

## Repository Layout

```text
frontend/   Vue application
backend/    FastAPI application
.github/    CI workflows
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

## Initial API

- `GET /api/health`
- `GET /api/version`
- `GET /api/status`

## Deployment

- CI is defined in `.github/workflows/ci.yml`.
- VPS sync deployment is defined in `.github/workflows/deploy.yml`.
- Remote sync logic lives in `ops/remote_sync.sh`.
- Remote app deploy logic lives in `ops/remote_deploy.sh`.
