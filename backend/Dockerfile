FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-rus && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip && \
    pip install . && \
    python - <<'PY'
from rapidocr import LangRec, OCRVersion, RapidOCR

RapidOCR(
    params={
        "Global.log_level": "error",
        "Rec.lang_type": LangRec.CYRILLIC,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
    }
)
PY

EXPOSE 8010

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]
