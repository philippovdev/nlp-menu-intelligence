from __future__ import annotations

import unicodedata
from io import BytesIO

import pytesseract
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pytesseract.pytesseract import TesseractError, TesseractNotFoundError

from app.menu_parser import parse_menu_text
from app.schemas import (
    ApiError,
    Issue,
    MenuParseRequest,
    MenuParseResponse,
    ParsedDocument,
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
OCR_LANGUAGES = "rus+eng"
SUPPORTED_MEDIA_TYPES = {
    "application/pdf": "pdf",
    "image/jpeg": "image",
    "image/png": "image",
    "image/webp": "image",
}


async def parse_menu_file(
    file: UploadFile,
    *,
    schema_version: str = "v1",
    lang: str = "ru",
    currency_hint: str = "RUB",
    category_labels: list[str] | None = None,
) -> MenuParseResponse:
    media_type = normalize_media_type(file.content_type)
    source_type = validate_media_type(media_type)
    file_bytes = await file.read()
    validate_file_bytes(file_bytes)

    if source_type == "pdf":
        extracted_text, extraction_issues = extract_pdf_text(file_bytes)
        ocr_used = False
    else:
        extracted_text = extract_image_text(file_bytes)
        extraction_issues = []
        ocr_used = True

    response = parse_menu_text(
        MenuParseRequest(
            schema_version=schema_version,
            text=extracted_text,
            lang=lang,
            currency_hint=currency_hint,
            category_labels=category_labels,
        )
    )
    response.document = ParsedDocument(
        source_type=source_type,
        filename=file.filename or None,
        media_type=media_type,
        ocr_used=ocr_used,
        extracted_text=extracted_text,
    )
    response.issues.extend(extraction_issues)
    return response


def validate_media_type(media_type: str | None) -> str:
    if media_type not in SUPPORTED_MEDIA_TYPES:
        raise ApiError(
            status_code=422,
            code="FILE_UNSUPPORTED",
            message="Only PDF, JPEG, PNG, and WebP files are supported.",
            details={"media_type": media_type},
        )
    return SUPPORTED_MEDIA_TYPES[media_type]


def validate_file_bytes(file_bytes: bytes) -> None:
    if not file_bytes:
        raise ApiError(
            status_code=422,
            code="FILE_EMPTY",
            message="Uploaded file is empty.",
        )
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ApiError(
            status_code=422,
            code="FILE_TOO_LARGE",
            message="Uploaded file exceeds the size limit.",
            details={"max_bytes": MAX_FILE_SIZE_BYTES},
        )


def extract_pdf_text(file_bytes: bytes) -> tuple[str, list[Issue]]:
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except (PdfReadError, ValueError) as exc:
        raise ApiError(
            status_code=422,
            code="TEXT_EXTRACTION_FAILED",
            message="Unable to read text from the uploaded PDF.",
            details={"source_type": "pdf"},
        ) from exc

    extracted_pages: list[str] = []
    empty_pages: list[int] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = normalize_extracted_text(page.extract_text() or "")
        if page_text:
            extracted_pages.append(page_text)
        else:
            empty_pages.append(page_number)

    extracted_text = "\n".join(extracted_pages).strip()
    if not extracted_text:
        raise ApiError(
            status_code=422,
            code="TEXT_EXTRACTION_FAILED",
            message="PDF does not contain usable embedded text.",
            details={"source_type": "pdf"},
        )

    issues: list[Issue] = []
    if empty_pages and len(empty_pages) != len(reader.pages):
        issues.append(
            Issue(
                code="TEXT_EXTRACTION_PARTIAL",
                level="info",
                message="Some PDF pages did not yield embedded text.",
                path="/document",
                details={"empty_pages": empty_pages},
            )
        )

    return extracted_text, issues


def extract_image_text(file_bytes: bytes) -> str:
    try:
        with Image.open(BytesIO(file_bytes)) as image:
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            extracted_text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
    except (UnidentifiedImageError, OSError) as exc:
        raise ApiError(
            status_code=422,
            code="TEXT_EXTRACTION_FAILED",
            message="Unable to open the uploaded image for OCR.",
            details={"source_type": "image"},
        ) from exc
    except TesseractNotFoundError as exc:
        raise ApiError(
            status_code=503,
            code="OCR_UNAVAILABLE",
            message="Tesseract OCR is not available on the server.",
            details={"source_type": "image"},
        ) from exc
    except TesseractError as exc:
        raise ApiError(
            status_code=422,
            code="TEXT_EXTRACTION_FAILED",
            message="OCR failed for the uploaded image.",
            details={"source_type": "image"},
        ) from exc

    normalized_text = normalize_extracted_text(extracted_text)
    if not normalized_text:
        raise ApiError(
            status_code=422,
            code="TEXT_EXTRACTION_FAILED",
            message="OCR did not produce usable text.",
            details={"source_type": "image"},
        )

    return normalized_text


def normalize_media_type(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def normalize_extracted_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    return "\n".join(lines).strip()
