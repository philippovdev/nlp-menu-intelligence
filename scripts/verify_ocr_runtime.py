from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.image_ocr import extract_image_text, get_ocr_runtime_status
from app.schemas import ApiError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify OCR runtime readiness for Menu Intelligence."
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Optional image file to run through the OCR pipeline.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    payload: dict[str, object] = {"runtime": get_ocr_runtime_status()}

    if args.image is not None:
        file_bytes = args.image.read_bytes()
        try:
            extracted_text, issues = extract_image_text(file_bytes)
        except ApiError as exc:
            payload["sample"] = {
                "image": str(args.image),
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                },
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1

        payload["sample"] = {
            "image": str(args.image),
            "extracted_text": extracted_text,
            "issues": [issue.model_dump(mode="json") for issue in issues],
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
