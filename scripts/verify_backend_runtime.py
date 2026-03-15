from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.category_classifier import DEFAULT_CLASSIFIER_PATH, DEFAULT_METADATA_PATH
from app.main import create_app
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-fallback", action="store_true")
    return parser.parse_args()


def build_pdf_bytes(lines: list[str]) -> bytes:
    content_lines = ["BT", "/F1 12 Tf", "72 720 Td"]

    for index, line in enumerate(lines):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
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


def assert_ok(response, endpoint: str) -> dict[str, object]:
    if response.status_code != 200:
        raise RuntimeError(f"{endpoint} returned {response.status_code}: {response.text}")
    return response.json()


def main() -> int:
    args = parse_args()
    assets = {
        "classifier_file": str(DEFAULT_CLASSIFIER_PATH.relative_to(REPO_ROOT)),
        "classifier_exists": DEFAULT_CLASSIFIER_PATH.is_file(),
        "metadata_file": str(DEFAULT_METADATA_PATH.relative_to(REPO_ROOT)),
        "metadata_exists": DEFAULT_METADATA_PATH.is_file(),
    }

    with TestClient(create_app()) as client:
        version_payload = assert_ok(client.get("/api/v1/version"), "/api/v1/version")
        parse_payload = assert_ok(
            client.post(
                "/api/v1/menu/parse",
                json={
                    "schema_version": "v1",
                    "text": "Americano 300 ml 180",
                    "currency_hint": "RUB",
                    "category_labels": [
                        "salads",
                        "soups",
                        "mains",
                        "desserts",
                        "drinks",
                        "other",
                    ],
                },
            ),
            "/api/v1/menu/parse",
        )
        parse_file_payload = assert_ok(
            client.post(
                "/api/v1/menu/parse-file",
                files={
                    "file": (
                        "menu.pdf",
                        build_pdf_bytes(["SALADS", "Caesar 250 g 390 RUB"]),
                        "application/pdf",
                    )
                },
            ),
            "/api/v1/menu/parse-file",
        )

    if not args.allow_fallback and not version_payload["category_model_ready"]:
        raise RuntimeError("Category classifier is not ready.")

    if (
        parse_payload["model_version"]["category_model"] != version_payload["category_model"]
        or (
            parse_file_payload["model_version"]["category_model"]
            != version_payload["category_model"]
        )
    ):
        raise RuntimeError(
            "Version endpoint and parse endpoints disagree on the active category model."
        )

    result = {
        "cwd": os.getcwd(),
        "assets": assets,
        "version": version_payload,
        "parse_category": parse_payload["items"][0]["category"],
        "parse_file_category_model": parse_file_payload["model_version"]["category_model"],
        "parse_file_document": parse_file_payload["document"],
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
