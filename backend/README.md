# Backend

FastAPI service for the Slice 1 menu parsing flow.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## Endpoints

- `GET /api/health`
- `GET /api/v1/health`
- `GET /api/version`
- `GET /api/v1/version`
- `GET /api/status`
- `POST /api/v1/menu/parse`
- `POST /api/v1/menu/parse-file`

## Runtime Notes

- `POST /api/v1/menu/parse-file` requires Tesseract OCR for image uploads.
- The Docker image installs `tesseract-ocr` with English and Russian language packs.
