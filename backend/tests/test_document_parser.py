from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
from pytesseract.pytesseract import TesseractNotFoundError

from app.category_classifier import CONFIGURED_CATEGORY_MODEL_ID
from app.image_ocr import RapidOcrUnavailableError


def build_pdf_bytes(lines: list[str]) -> bytes:
    content_lines = ["BT", "/F1 12 Tf", "72 720 Td"]

    for index, line in enumerate(lines):
        escaped = escape_pdf_text(line)
        if index:
            content_lines.append("0 -18 Td")
        content_lines.append(f"({escaped}) Tj")

    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_number} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("ascii")
    )

    return bytes(pdf)


def escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_oriented_jpeg_bytes() -> bytes:
    image = Image.new("RGB", (16, 32), "white")
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_parse_menu_file_rejects_unsupported_media_type(client: TestClient) -> None:
    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "schema_version": "v1",
        "error": {
            "code": "FILE_UNSUPPORTED",
            "message": "Only PDF, JPEG, PNG, and WebP files are supported.",
            "details": {"media_type": "text/plain"},
        },
    }


def test_parse_menu_file_rejects_empty_file(client: TestClient) -> None:
    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "schema_version": "v1",
        "error": {
            "code": "FILE_EMPTY",
            "message": "Uploaded file is empty.",
            "details": None,
        },
    }


def test_parse_menu_file_rejects_oversized_file(
    client: TestClient,
    monkeypatch,
) -> None:
    monkeypatch.setattr("app.document_parser.MAX_FILE_SIZE_BYTES", 4)

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.pdf", b"12345", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json() == {
        "schema_version": "v1",
        "error": {
            "code": "FILE_TOO_LARGE",
            "message": "Uploaded file exceeds the size limit.",
            "details": {"max_bytes": 4},
        },
    }


def test_parse_menu_file_parses_pdf_text(client: TestClient) -> None:
    pdf_bytes = build_pdf_bytes(["SALADS", "Caesar 250 g 390 RUB"])

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["document"] == {
        "source_type": "pdf",
        "filename": "menu.pdf",
        "media_type": "application/pdf",
        "ocr_used": False,
        "extracted_text": "SALADS\nCaesar 250 g 390 RUB",
    }
    assert payload["model_version"]["category_model"] == CONFIGURED_CATEGORY_MODEL_ID
    assert payload["items"][0]["kind"] == "category_header"
    assert payload["items"][1]["kind"] == "menu_item"
    assert payload["items"][1]["fields"]["name"] == "Caesar"
    assert payload["items"][1]["fields"]["prices"] == [
        {"value": 390, "currency": "RUB", "raw": "390 RUB"}
    ]


def test_parse_menu_file_falls_back_to_filename_media_type(client: TestClient) -> None:
    pdf_bytes = build_pdf_bytes(["SALADS", "Caesar 250 g 390 RUB"])

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.pdf", pdf_bytes, "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json()["document"]["media_type"] == "application/pdf"


def test_parse_menu_file_uses_ocr_for_images(client: TestClient, monkeypatch) -> None:
    image_bytes = build_png_bytes()
    monkeypatch.setattr(
        "app.image_ocr.extract_text_with_rapidocr",
        lambda image: "SOUPS\nTom Yum 300 ml 450",
    )

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["document"] == {
        "source_type": "image",
        "filename": "menu.png",
        "media_type": "image/png",
        "ocr_used": True,
        "extracted_text": "SOUPS\nTom Yum 300 ml 450",
    }
    assert payload["model_version"]["category_model"] == CONFIGURED_CATEGORY_MODEL_ID
    assert payload["items"][1]["kind"] == "menu_item"
    assert payload["items"][1]["fields"]["name"] == "Tom Yum"
    assert payload["items"][1]["fields"]["prices"] == [
        {"value": 450, "currency": "RUB", "raw": "450"}
    ]
    assert payload["issues"] == []


def test_parse_menu_file_applies_exif_orientation_before_ocr(
    client: TestClient,
    monkeypatch,
) -> None:
    image_bytes = build_oriented_jpeg_bytes()
    seen_sizes: list[tuple[int, int]] = []

    def fake_extract_text_with_rapidocr(image: Image.Image) -> str:
        seen_sizes.append(image.size)
        return "SOUPS\nTom Yum 300 ml 450"

    monkeypatch.setattr(
        "app.image_ocr.extract_text_with_rapidocr",
        fake_extract_text_with_rapidocr,
    )

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert seen_sizes == [(1800, 900)]


def test_parse_menu_file_falls_back_to_tesseract_when_primary_ocr_is_unavailable(
    client: TestClient,
    monkeypatch,
) -> None:
    image_bytes = build_png_bytes()

    def fail_primary(image: Image.Image) -> str:
        raise RapidOcrUnavailableError("RapidOCR is not installed.")

    monkeypatch.setattr("app.image_ocr.extract_text_with_rapidocr", fail_primary)
    monkeypatch.setattr(
        "app.image_ocr.pytesseract.image_to_string",
        lambda image, lang: "BURGERS\nDouble cheeseburger 290 g 540",
    )

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["document"]["ocr_used"] is True
    assert payload["items"][1]["category"]["label"] == "burgers"
    assert payload["issues"] == [
        {
            "code": "OCR_FALLBACK_TESSERACT",
            "level": "info",
            "message": "RapidOCR was not used; Tesseract fallback is active.",
            "path": "/document",
            "details": {
                "primary_engine": "rapidocr-ppocrv5-cyrillic@3.7.0",
                "fallback_engine": "tesseract-rus+eng",
                "reason": "primary_unavailable",
            },
        }
    ]


def test_parse_menu_file_reports_ocr_unavailable_when_all_engines_are_missing(
    client: TestClient,
    monkeypatch,
) -> None:
    image_bytes = build_png_bytes()

    def fail_primary(image: Image.Image) -> str:
        raise RapidOcrUnavailableError("RapidOCR is not installed.")

    monkeypatch.setattr("app.image_ocr.extract_text_with_rapidocr", fail_primary)

    def fail_tesseract(image: Image.Image, lang: str) -> str:
        raise TesseractNotFoundError()

    monkeypatch.setattr("app.image_ocr.pytesseract.image_to_string", fail_tesseract)

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.png", image_bytes, "image/png")},
    )

    assert response.status_code == 503
    assert response.json() == {
        "schema_version": "v1",
        "error": {
            "code": "OCR_UNAVAILABLE",
            "message": "No OCR engine is available on the server.",
            "details": {
                "source_type": "image",
                "primary_engine": "rapidocr-ppocrv5-cyrillic@3.7.0",
                "fallback_engine": "tesseract-rus+eng",
            },
        },
    }
