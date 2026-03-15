from __future__ import annotations

import unicodedata
from functools import lru_cache
from io import BytesIO
from typing import Any

import pytesseract
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

from app.schemas import ApiError, Issue

try:
    from rapidocr import LangRec, OCRVersion, RapidOCR
except ImportError:
    LangRec = None
    OCRVersion = None
    RapidOCR = None

PRIMARY_OCR_ENGINE_ID = "rapidocr-ppocrv5-cyrillic@3.7.0"
FALLBACK_OCR_ENGINE_ID = "tesseract-rus+eng"
TESSERACT_LANGUAGES = "rus+eng"
UPSCALE_LONGEST_EDGE = 1800
DOWNSCALE_LONGEST_EDGE = 2600


class RapidOcrUnavailableError(RuntimeError):
    pass


class RapidOcrExecutionError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_rapidocr_engine() -> Any:
    if RapidOCR is None or LangRec is None or OCRVersion is None:
        raise RapidOcrUnavailableError("RapidOCR is not installed.")

    try:
        return RapidOCR(
            params={
                "Global.log_level": "error",
                "Rec.lang_type": LangRec.CYRILLIC,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
        )
    except Exception as exc:  # pragma: no cover - guarded by tests via boundary
        raise RapidOcrUnavailableError("RapidOCR could not be initialized.") from exc


def extract_image_text(file_bytes: bytes) -> tuple[str, list[Issue]]:
    try:
        image = preprocess_image_for_ocr(load_image(file_bytes))
    except (UnidentifiedImageError, OSError) as exc:
        raise ApiError(
            status_code=422,
            code="TEXT_EXTRACTION_FAILED",
            message="Unable to open the uploaded image for OCR.",
            details={"source_type": "image"},
        ) from exc

    try:
        return extract_text_with_rapidocr(image), []
    except RapidOcrUnavailableError:
        fallback_reason = "primary_unavailable"
    except RapidOcrExecutionError:
        fallback_reason = "primary_failed"

    issues = [
        Issue(
            code="OCR_FALLBACK_TESSERACT",
            level="info",
            message="RapidOCR was not used; Tesseract fallback is active.",
            path="/document",
            details={
                "primary_engine": PRIMARY_OCR_ENGINE_ID,
                "fallback_engine": FALLBACK_OCR_ENGINE_ID,
                "reason": fallback_reason,
            },
        )
    ]

    try:
        extracted_text = extract_text_with_tesseract(image)
    except TesseractNotFoundError as exc:
        raise ApiError(
            status_code=503,
            code="OCR_UNAVAILABLE",
            message="No OCR engine is available on the server.",
            details={
                "source_type": "image",
                "primary_engine": PRIMARY_OCR_ENGINE_ID,
                "fallback_engine": FALLBACK_OCR_ENGINE_ID,
            },
        ) from exc
    except TesseractError as exc:
        raise ApiError(
            status_code=422,
            code="TEXT_EXTRACTION_FAILED",
            message="OCR failed for the uploaded image.",
            details={
                "source_type": "image",
                "primary_engine": PRIMARY_OCR_ENGINE_ID,
                "fallback_engine": FALLBACK_OCR_ENGINE_ID,
            },
        ) from exc

    if not extracted_text:
        raise ApiError(
            status_code=422,
            code="TEXT_EXTRACTION_FAILED",
            message="OCR did not produce usable text.",
            details={
                "source_type": "image",
                "primary_engine": PRIMARY_OCR_ENGINE_ID,
                "fallback_engine": FALLBACK_OCR_ENGINE_ID,
            },
        )

    return extracted_text, issues


def extract_text_with_rapidocr(image: Image.Image) -> str:
    try:
        result = get_rapidocr_engine()(image)
    except RapidOcrUnavailableError:
        raise
    except Exception as exc:
        raise RapidOcrExecutionError("RapidOCR inference failed.") from exc

    extracted_text = normalize_rapidocr_output(result)
    if not extracted_text:
        raise RapidOcrExecutionError("RapidOCR did not produce usable text.")
    return extracted_text


def extract_text_with_tesseract(image: Image.Image) -> str:
    extracted_text = pytesseract.image_to_string(image, lang=TESSERACT_LANGUAGES)
    return normalize_extracted_text(extracted_text)


def load_image(file_bytes: bytes) -> Image.Image:
    with Image.open(BytesIO(file_bytes)) as image:
        return image.copy()


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    prepared = ImageOps.exif_transpose(image)
    if prepared.mode != "RGB":
        prepared = prepared.convert("RGB")

    prepared = resize_image(prepared)
    prepared = ImageOps.grayscale(prepared)
    prepared = ImageOps.autocontrast(prepared, cutoff=1)
    return ImageEnhance.Contrast(prepared).enhance(1.2)


def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge == 0:
        return image

    scale = 1.0
    if longest_edge < UPSCALE_LONGEST_EDGE:
        scale = UPSCALE_LONGEST_EDGE / longest_edge
    elif longest_edge > DOWNSCALE_LONGEST_EDGE:
        scale = DOWNSCALE_LONGEST_EDGE / longest_edge

    if scale == 1.0:
        return image

    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)


def normalize_rapidocr_output(result: Any) -> str:
    texts = getattr(result, "txts", None)
    if not texts:
        return ""

    boxes = getattr(result, "boxes", None)
    if boxes is not None and hasattr(result, "to_markdown"):
        return normalize_extracted_text(result.to_markdown())

    return normalize_extracted_text("\n".join(texts))


def normalize_extracted_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    return "\n".join(lines).strip()


def get_ocr_runtime_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "primary_engine": PRIMARY_OCR_ENGINE_ID,
        "primary_ready": False,
        "primary_error": None,
        "fallback_engine": FALLBACK_OCR_ENGINE_ID,
        "fallback_ready": False,
        "fallback_error": None,
    }

    try:
        get_rapidocr_engine()
        status["primary_ready"] = True
    except RapidOcrUnavailableError as exc:
        status["primary_error"] = str(exc)

    try:
        pytesseract.get_tesseract_version()
        status["fallback_ready"] = True
    except TesseractNotFoundError as exc:
        status["fallback_error"] = str(exc)

    return status
