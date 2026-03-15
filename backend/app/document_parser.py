from __future__ import annotations

import mimetypes
from io import BytesIO

from fastapi import UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.category_classifier import CategoryClassifier
from app.image_ocr import extract_image_text, normalize_extracted_text
from app.menu_parser import parse_menu_text
from app.schemas import (
    ApiError,
    Issue,
    MenuParseRequest,
    MenuParseResponse,
    ParsedDocument,
)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
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
    category_classifier: CategoryClassifier | None = None,
) -> MenuParseResponse:
    file_bytes = await file.read()
    validate_file_bytes(file_bytes)
    media_type = resolve_media_type(
        content_type=file.content_type,
        filename=file.filename,
        file_bytes=file_bytes,
    )
    source_type = validate_media_type(media_type)

    if source_type == "pdf":
        extracted_text, extraction_issues = extract_pdf_text(file_bytes)
        ocr_used = False
    else:
        extracted_text, extraction_issues = extract_image_text(file_bytes)
        ocr_used = True

    response = parse_menu_text(
        MenuParseRequest(
            schema_version=schema_version,
            text=extracted_text,
            lang=lang,
            currency_hint=currency_hint,
            category_labels=category_labels,
        ),
        category_classifier=category_classifier,
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


def resolve_media_type(
    *,
    content_type: str | None,
    filename: str | None,
    file_bytes: bytes,
) -> str | None:
    media_type = normalize_media_type(content_type)
    if media_type in SUPPORTED_MEDIA_TYPES:
        return media_type

    filename_media_type, _ = mimetypes.guess_type(filename or "")
    normalized_filename_media_type = normalize_media_type(filename_media_type)
    if normalized_filename_media_type in SUPPORTED_MEDIA_TYPES:
        return normalized_filename_media_type

    sniffed_media_type = sniff_media_type(file_bytes)
    if sniffed_media_type is not None:
        return sniffed_media_type

    return media_type


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
            details={"source_type": "pdf", "ocr_fallback_available": False},
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
def normalize_media_type(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def sniff_media_type(file_bytes: bytes) -> str | None:
    if file_bytes.startswith(b"%PDF-"):
        return "application/pdf"
    if file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if file_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if file_bytes.startswith(b"RIFF") and file_bytes[8:12] == b"WEBP":
        return "image/webp"
    return None
