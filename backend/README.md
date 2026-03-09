# Backend

FastAPI service for health checks, versioning, and the future OCR / inference pipeline.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

