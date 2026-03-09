from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


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


def test_parse_menu_file_rejects_unsupported_media_type() -> None:
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


def test_parse_menu_file_rejects_empty_file() -> None:
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


def test_parse_menu_file_rejects_oversized_file(monkeypatch) -> None:
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


def test_parse_menu_file_parses_pdf_text() -> None:
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
    assert payload["items"][0]["kind"] == "category_header"
    assert payload["items"][1]["kind"] == "menu_item"
    assert payload["items"][1]["fields"]["name"] == "Caesar"
    assert payload["items"][1]["fields"]["prices"] == [
        {"value": 390, "currency": "RUB", "raw": "390 RUB"}
    ]


def test_parse_menu_file_falls_back_to_filename_media_type() -> None:
    pdf_bytes = build_pdf_bytes(["SALADS", "Caesar 250 g 390 RUB"])

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.pdf", pdf_bytes, "application/octet-stream")},
    )

    assert response.status_code == 200
    assert response.json()["document"]["media_type"] == "application/pdf"


def test_parse_menu_file_uses_ocr_for_images(monkeypatch) -> None:
    image_bytes = build_png_bytes()
    monkeypatch.setattr(
        "app.document_parser.pytesseract.image_to_string",
        lambda image, lang: "SOUPS\nTom Yum 300 ml 450",
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
    assert payload["items"][1]["kind"] == "menu_item"
    assert payload["items"][1]["fields"]["name"] == "Tom Yum"
    assert payload["items"][1]["fields"]["prices"] == [
        {"value": 450, "currency": "RUB", "raw": "450"}
    ]


def test_parse_menu_file_applies_exif_orientation_before_ocr(monkeypatch) -> None:
    image_bytes = build_oriented_jpeg_bytes()
    seen_sizes: list[tuple[int, int]] = []

    def fake_image_to_string(image: Image.Image, lang: str) -> str:
        seen_sizes.append(image.size)
        return "SOUPS\nTom Yum 300 ml 450"

    monkeypatch.setattr(
        "app.document_parser.pytesseract.image_to_string",
        fake_image_to_string,
    )

    response = client.post(
        "/api/v1/menu/parse-file",
        files={"file": ("menu.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert seen_sizes == [(32, 16)]
