from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
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
COLUMN_MIN_BOX_WIDTH = 80.0
COLUMN_GAP_THRESHOLD = 120.0
OCR_LINE_VERTICAL_TOLERANCE = 6.0
OCR_LINE_VERTICAL_RATIO = 0.35
OCR_SUBCOLUMN_GAP_THRESHOLD = 72.0
OCR_PRICE_DOMINANT_RATIO = 0.6
OCR_PRICE_ATTACH_MAX_VERTICAL_DISTANCE = 32.0
OCR_PRICE_ATTACH_AMBIGUITY_DELTA = 6.0
OCR_PRICE_ATTACH_LEFT_TOLERANCE = 24.0
OCR_DESCRIPTOR_MAX_VERTICAL_GAP = 22.0
OCR_DESCRIPTOR_INDENT_TOLERANCE = 28.0
OCR_MASTHEAD_TOP_RATIO = 0.16
OCR_NUMBER_RE = r"(?:\d{1,3}(?:[ \u00A0]\d{3})+|\d+)(?:[.,]\d+)?"
OCR_PRICE_GROUP_RE = rf"{OCR_NUMBER_RE}(?:\s*/\s*{OCR_NUMBER_RE})*"
OCR_NUMERIC_FRAGMENT_PATTERN = re.compile(
    rf"^\s*(?:[$€£₽]\s*)?{OCR_PRICE_GROUP_RE}\s*(?:₽|руб\.?|р\.?|rub|[PР])?\s*$",
    re.IGNORECASE,
)
OCR_RUBLE_SUFFIX_PATTERN = re.compile(
    rf"(?<!\w)(?P<value>{OCR_PRICE_GROUP_RE})\s*(?P<currency>[PР])(?!\w)",
    re.IGNORECASE,
)
OCR_NON_LETTER_PATTERN = re.compile(r"[^A-Za-zА-Яа-я]+")
OCR_DESCRIPTOR_PATTERN = re.compile(
    (
        r"\b(?:"
        r"белое|красное|розовое|сухое|полусухое|полусладкое|сладкое|брют|"
        r"white|red|rose|dry|semi[- ]dry|semi[- ]sweet|sweet|brut|"
        r"france|italy|spain|germany|chile|argentina|australia|"
        r"франция|италия|испания|германия|чили|аргентина|австралия|"
        r"венето|пьемонт|тоскана|бургундия|мальборо|шампань|риоха|мозель"
        r")\b"
    ),
    re.IGNORECASE,
)
OCR_PRICE_PATTERN = re.compile(
    rf"(?<!\w)(?:[$€£₽]|{OCR_PRICE_GROUP_RE}\s*(?:₽|руб\.?|р\.?|rub|[PР]))",
    re.IGNORECASE,
)
OCR_SIZE_PATTERN = re.compile(
    rf"(?P<values>{OCR_NUMBER_RE}(?:\s*/\s*{OCR_NUMBER_RE})*)\s*(?P<unit>kg|g|gr|ml|l|cl|oz|cm|pc|pcs|шт\.?|кг|гр|г|мл|л|см)\b",
    re.IGNORECASE,
)
OCR_SERVICE_SCOPE_PATTERN = re.compile(
    (
        rf"^\s*(?:по\s+(?:бокалам|бутылкам|стаканам|порциям)|"
        rf"(?:by\s+the\s+)?glass(?:es)?|bottle(?:s)?)\b.*"
        rf"{OCR_NUMBER_RE}\s*(?:ml|l|cl|мл|л)\b"
    ),
    re.IGNORECASE,
)
OCR_CONTEXT_NOTE_PATTERN = re.compile(
    r"^\s*(?:цены\s+указаны|все\s+цены|all\s+prices|serving\s+size|объем\s+подачи)\b",
    re.IGNORECASE,
)
OCR_MASTHEAD_PATTERN = re.compile(
    r"\b(?:ресторан|restaurant|кафе|cafe|бар|bistro|brasserie|кухня|kitchen|menu)\b",
    re.IGNORECASE,
)
OCR_CONTINUATION_START_PATTERN = re.compile(
    r"^\s*(?:из|с|со|под|на|в|к|от|with|served|from|aged|аромат|вкус|ноты)\b",
    re.IGNORECASE,
)


class RapidOcrUnavailableError(RuntimeError):
    pass


class RapidOcrExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrBox:
    text: str
    x1: float
    x2: float
    y1: float
    y2: float
    width: float
    height: float
    x_center: float
    y_center: float
    score: float | None = None


@dataclass(frozen=True)
class OcrLine:
    text: str
    bbox: tuple[float, float, float, float]
    column_id: int
    source_boxes: tuple[OcrBox, ...]
    score: float | None = None

    @property
    def x1(self) -> float:
        return self.bbox[0]

    @property
    def x2(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[2]

    @property
    def y2(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def x_center(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def y_center(self) -> float:
        return (self.y1 + self.y2) / 2


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
    return normalize_image_ocr_text(extracted_text)


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
    if texts is None or len(texts) == 0:
        return ""

    ocr_boxes = build_ocr_boxes(result)
    if ocr_boxes:
        return reconstruct_ocr_text(ocr_boxes)

    return normalize_image_ocr_text("\n".join(texts))


def build_ocr_boxes(result: Any) -> list[OcrBox]:
    texts = getattr(result, "txts", None)
    boxes = getattr(result, "boxes", None)
    if texts is None or boxes is None or len(texts) == 0 or len(boxes) == 0:
        return []

    scores = extract_ocr_scores(result, expected_count=len(texts))
    ocr_boxes: list[OcrBox] = []
    for index, (raw_box, raw_text) in enumerate(zip(boxes, texts, strict=False)):
        text = str(raw_text or "").strip()
        if not text:
            continue

        points = parse_ocr_box_points(raw_box)
        if len(points) < 2:
            continue

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1 = min(xs)
        x2 = max(xs)
        y1 = min(ys)
        y2 = max(ys)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width == 0.0 or height == 0.0:
            continue

        ocr_boxes.append(
            OcrBox(
                text=text,
                x1=x1,
                x2=x2,
                y1=y1,
                y2=y2,
                width=width,
                height=height,
                x_center=(x1 + x2) / 2,
                y_center=(y1 + y2) / 2,
                score=scores[index] if scores is not None else None,
            )
        )

    return ocr_boxes


def extract_ocr_scores(result: Any, *, expected_count: int) -> list[float] | None:
    for attr_name in ("scores", "rec_scores"):
        raw_scores = getattr(result, attr_name, None)
        if raw_scores is None or isinstance(raw_scores, (str, bytes)):
            continue

        try:
            values = list(raw_scores)
        except TypeError:
            continue

        if len(values) != expected_count:
            continue

        scores: list[float] = []
        try:
            for value in values:
                scores.append(float(value))
        except (TypeError, ValueError):
            continue

        return scores

    return None


def parse_ocr_box_points(raw_box: Any) -> list[tuple[float, float]]:
    try:
        points = []
        for point in raw_box:
            x, y = point
            points.append((float(x), float(y)))
        return points
    except (TypeError, ValueError):
        return []


def reconstruct_ocr_text(boxes: list[OcrBox]) -> str:
    if not boxes:
        return ""

    columns = detect_text_columns(boxes)
    if not columns:
        columns = [sorted(boxes, key=lambda candidate: (candidate.y_center, candidate.x1))]

    page_top = min(box.y1 for box in boxes)
    page_bottom = max(box.y2 for box in boxes)

    assembled_lines: list[OcrLine] = []
    for column_id, column_boxes in enumerate(columns):
        assembled_lines.extend(
            assemble_ocr_column(
                column_boxes,
                column_id=column_id,
                page_top=page_top,
                page_bottom=page_bottom,
            )
        )

    rendered_lines = [line.text for line in assembled_lines]
    return normalize_image_ocr_text("\n".join(filter_reconstructed_ocr_lines(rendered_lines)))


def detect_text_columns(boxes: list[OcrBox]) -> list[list[OcrBox]]:
    candidate_boxes = [
        box
        for box in boxes
        if box.width > COLUMN_MIN_BOX_WIDTH and not looks_numeric_fragment(box.text)
    ]
    if len(candidate_boxes) < 2:
        return []

    clustered: list[list[OcrBox]] = []
    for box in sorted(candidate_boxes, key=lambda candidate: candidate.x1):
        if not clustered or box.x1 - clustered[-1][-1].x1 > COLUMN_GAP_THRESHOLD:
            clustered.append([box])
        else:
            clustered[-1].append(box)

    if len(clustered) < 2:
        return []

    column_centers = [
        sum(candidate.x_center for candidate in cluster) / len(cluster)
        for cluster in clustered
    ]
    assigned_columns: list[list[OcrBox]] = [[] for _ in column_centers]
    for box in boxes:
        column_index = min(
            range(len(column_centers)),
            key=lambda index: abs(box.x_center - column_centers[index]),
        )
        assigned_columns[column_index].append(box)

    return [column for column in assigned_columns if column]


def assemble_ocr_column(
    boxes: list[OcrBox],
    *,
    column_id: int,
    page_top: float,
    page_bottom: float,
) -> list[OcrLine]:
    if not boxes:
        return []

    lines = build_ocr_lines(boxes, column_id=column_id)
    lines = suppress_context_lines(lines, page_top=page_top, page_bottom=page_bottom)
    lines = merge_descriptor_lines(lines)
    lines = pair_price_subcolumns(lines)
    lines = attach_standalone_price_lines(lines)
    lines = suppress_context_lines(lines, page_top=page_top, page_bottom=page_bottom)
    return sorted(lines, key=lambda line: (line.y1, line.x1))


def build_ocr_lines(boxes: list[OcrBox], *, column_id: int) -> list[OcrLine]:
    sorted_boxes = sorted(boxes, key=lambda candidate: (candidate.y_center, candidate.x1))
    grouped_lines: list[list[OcrBox]] = []

    for box in sorted_boxes:
        if grouped_lines and should_group_box_with_line(grouped_lines[-1], box):
            grouped_lines[-1].append(box)
        else:
            grouped_lines.append([box])

    return [make_ocr_line(line_boxes, column_id=column_id) for line_boxes in grouped_lines]


def should_group_box_with_line(line_boxes: list[OcrBox], box: OcrBox) -> bool:
    line_y1 = min(candidate.y1 for candidate in line_boxes)
    line_y2 = max(candidate.y2 for candidate in line_boxes)
    overlap = vertical_overlap(line_y1, line_y2, box.y1, box.y2)
    if overlap > 0:
        min_height = min(line_y2 - line_y1, box.height)
        if overlap >= 0.2 * min_height:
            return True

    anchor_y = average_y_center(line_boxes)
    line_height = max(candidate.height for candidate in line_boxes)
    return abs(box.y_center - anchor_y) <= max(
        OCR_LINE_VERTICAL_TOLERANCE,
        OCR_LINE_VERTICAL_RATIO * line_height,
    )


def make_ocr_line(line_boxes: list[OcrBox], *, column_id: int) -> OcrLine:
    ordered_boxes = tuple(sorted(line_boxes, key=lambda candidate: candidate.x1))
    text = " ".join(box.text.strip() for box in ordered_boxes if box.text.strip()).strip()
    scores = [box.score for box in ordered_boxes if box.score is not None]
    return OcrLine(
        text=text,
        bbox=(
            min(box.x1 for box in ordered_boxes),
            max(box.x2 for box in ordered_boxes),
            min(box.y1 for box in ordered_boxes),
            max(box.y2 for box in ordered_boxes),
        ),
        column_id=column_id,
        source_boxes=ordered_boxes,
        score=sum(scores) / len(scores) if scores else None,
    )


def suppress_context_lines(
    lines: list[OcrLine],
    *,
    page_top: float,
    page_bottom: float,
) -> list[OcrLine]:
    return [
        line
        for line in lines
        if not looks_context_line(line.text)
        and not looks_masthead_like_line(line, page_top=page_top, page_bottom=page_bottom)
    ]


def merge_descriptor_lines(lines: list[OcrLine]) -> list[OcrLine]:
    merged: list[OcrLine] = []

    for line in sorted(lines, key=lambda candidate: (candidate.y1, candidate.x1)):
        if merged and should_merge_descriptor_line(merged[-1], line):
            merged[-1] = merge_ocr_lines([merged[-1], line])
            continue
        merged.append(line)

    return merged


def should_merge_descriptor_line(previous: OcrLine, current: OcrLine) -> bool:
    if current.column_id != previous.column_id:
        return False
    if line_has_price_signal(current.text):
        return False
    if not looks_descriptor_or_continuation_text(current.text):
        return False
    if looks_header_like_text(previous.text) or not looks_item_anchor_text(previous.text):
        return False
    if abs(current.x1 - previous.x1) > OCR_DESCRIPTOR_INDENT_TOLERANCE:
        return False

    max_gap = max(
        OCR_DESCRIPTOR_MAX_VERTICAL_GAP,
        0.7 * max(previous.height, current.height),
    )
    return vertical_gap_between_lines(previous, current) <= max_gap


def pair_price_subcolumns(lines: list[OcrLine]) -> list[OcrLine]:
    if len(lines) < 2:
        return lines

    ordered_lines = sorted(lines, key=lambda candidate: (candidate.x1, candidate.y1))
    subcolumns = cluster_line_subcolumns(ordered_lines)
    if len(subcolumns) < 2:
        return sorted(lines, key=lambda candidate: (candidate.y1, candidate.x1))

    updated_lines = ordered_lines[:]
    consumed_indices: set[int] = set()

    for subcolumn_index in range(1, len(subcolumns)):
        left_indices = subcolumns[subcolumn_index - 1]
        right_indices = subcolumns[subcolumn_index]
        if not is_text_dominant_subcolumn(updated_lines, left_indices):
            continue
        if not is_price_dominant_subcolumn(updated_lines, right_indices):
            continue

        attachments = collect_price_attachments(
            updated_lines,
            price_indices=[index for index in right_indices if index not in consumed_indices],
            candidate_indices=[index for index in left_indices if index not in consumed_indices],
        )
        if not attachments:
            continue

        updated_lines, newly_consumed = apply_price_attachments(updated_lines, attachments)
        consumed_indices.update(newly_consumed)

    return [
        line
        for index, line in enumerate(updated_lines)
        if index not in consumed_indices
    ]


def cluster_line_subcolumns(lines: list[OcrLine]) -> list[list[int]]:
    clustered: list[list[int]] = []
    for index in sorted(range(len(lines)), key=lambda line_index: lines[line_index].x1):
        if (
            not clustered
            or lines[index].x1 - lines[clustered[-1][-1]].x1 > OCR_SUBCOLUMN_GAP_THRESHOLD
        ):
            clustered.append([index])
        else:
            clustered[-1].append(index)
    return clustered


def is_price_dominant_subcolumn(lines: list[OcrLine], indices: list[int]) -> bool:
    if not indices:
        return False
    price_like_count = sum(looks_price_like_line(lines[index].text) for index in indices)
    return price_like_count / len(indices) >= OCR_PRICE_DOMINANT_RATIO


def is_text_dominant_subcolumn(lines: list[OcrLine], indices: list[int]) -> bool:
    if not indices:
        return False
    text_like_count = sum(looks_item_anchor_text(lines[index].text) for index in indices)
    return text_like_count / len(indices) >= OCR_PRICE_DOMINANT_RATIO


def collect_price_attachments(
    lines: list[OcrLine],
    *,
    price_indices: list[int],
    candidate_indices: list[int],
) -> dict[int, list[int]]:
    attachments: dict[int, list[int]] = {}

    for price_index in sorted(price_indices, key=lambda index: (lines[index].y1, lines[index].x1)):
        target_index = find_best_price_target(lines[price_index], lines, candidate_indices)
        if target_index is None:
            continue
        attachments.setdefault(target_index, []).append(price_index)

    return attachments


def apply_price_attachments(
    lines: list[OcrLine],
    attachments: dict[int, list[int]],
) -> tuple[list[OcrLine], set[int]]:
    updated_lines = lines[:]
    consumed_indices: set[int] = set()

    for target_index, price_indices in attachments.items():
        attached_lines = sorted(
            (lines[index] for index in price_indices),
            key=lambda candidate: (candidate.y1, candidate.x1),
        )
        updated_lines[target_index] = merge_ocr_lines(
            [updated_lines[target_index], *attached_lines]
        )
        consumed_indices.update(price_indices)

    return updated_lines, consumed_indices


def attach_standalone_price_lines(lines: list[OcrLine]) -> list[OcrLine]:
    if len(lines) < 2:
        return lines

    ordered_lines = sorted(lines, key=lambda candidate: (candidate.y1, candidate.x1))
    updated_lines = ordered_lines[:]
    consumed_indices: set[int] = set()

    for price_index, price_line in enumerate(updated_lines):
        if looks_price_like_line(price_line.text) is False or price_index in consumed_indices:
            continue

        candidate_indices = [
            index
            for index, candidate in enumerate(updated_lines)
            if index != price_index and index not in consumed_indices
        ]
        target_index = find_best_price_target(price_line, updated_lines, candidate_indices)
        if target_index is None:
            continue

        updated_lines[target_index] = merge_ocr_lines([updated_lines[target_index], price_line])
        consumed_indices.add(price_index)

    return [
        line
        for index, line in enumerate(updated_lines)
        if index not in consumed_indices
    ]


def find_best_price_target(
    price_line: OcrLine,
    lines: list[OcrLine],
    candidate_indices: list[int],
) -> int | None:
    scored_candidates: list[tuple[tuple[int, float, float, float], int]] = []

    for candidate_index in candidate_indices:
        candidate = lines[candidate_index]
        if candidate.column_id != price_line.column_id:
            continue
        if candidate.x1 > price_line.x1 + OCR_PRICE_ATTACH_LEFT_TOLERANCE:
            continue
        if candidate.y_center > price_line.y_center + 0.35 * max(
            candidate.height,
            price_line.height,
        ):
            continue
        if not looks_price_target_line(candidate.text):
            continue

        overlap = vertical_overlap_between_lines(candidate, price_line)
        gap = vertical_gap_between_lines(candidate, price_line)
        max_gap = max(
            OCR_PRICE_ATTACH_MAX_VERTICAL_DISTANCE,
            1.4 * max(candidate.height, price_line.height),
        )
        if overlap <= 0 and gap > max_gap:
            continue

        scored_candidates.append(
            (
                (
                    0 if overlap > 0 else 1,
                    gap,
                    abs(candidate.y_center - price_line.y_center),
                    abs(price_line.x1 - candidate.x2),
                ),
                candidate_index,
            )
        )

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda candidate: candidate[0])
    best_score, best_index = scored_candidates[0]

    if len(scored_candidates) > 1:
        second_score, _ = scored_candidates[1]
        if (
            best_score[0] == second_score[0]
            and abs(best_score[1] - second_score[1]) <= OCR_PRICE_ATTACH_AMBIGUITY_DELTA
            and abs(best_score[2] - second_score[2]) <= OCR_PRICE_ATTACH_AMBIGUITY_DELTA
        ):
            return None

    return best_index


def merge_ocr_lines(lines: list[OcrLine]) -> OcrLine:
    source_boxes = tuple(box for line in lines for box in line.source_boxes)
    scores = [box.score for box in source_boxes if box.score is not None]
    return OcrLine(
        text=" ".join(line.text.strip() for line in lines if line.text.strip()),
        bbox=(
            min(line.x1 for line in lines),
            max(line.x2 for line in lines),
            min(line.y1 for line in lines),
            max(line.y2 for line in lines),
        ),
        column_id=lines[0].column_id,
        source_boxes=source_boxes,
        score=sum(scores) / len(scores) if scores else None,
    )


def average_y_center(line_boxes: list[OcrBox]) -> float:
    return sum(box.y_center for box in line_boxes) / len(line_boxes)


def vertical_overlap(y1: float, y2: float, other_y1: float, other_y2: float) -> float:
    return max(0.0, min(y2, other_y2) - max(y1, other_y1))


def vertical_overlap_between_lines(left: OcrLine, right: OcrLine) -> float:
    return vertical_overlap(left.y1, left.y2, right.y1, right.y2)


def vertical_gap_between_lines(top: OcrLine, bottom: OcrLine) -> float:
    if vertical_overlap_between_lines(top, bottom) > 0:
        return 0.0
    return min(abs(bottom.y1 - top.y2), abs(top.y1 - bottom.y2))


def looks_numeric_fragment(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return bool(OCR_NUMERIC_FRAGMENT_PATTERN.fullmatch(normalized))


def looks_price_like_line(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized or looks_context_line(normalized):
        return False
    if looks_numeric_fragment(normalized):
        return True

    words = split_words(normalized)
    digits = sum(char.isdigit() for char in normalized)
    letters = count_letters(normalized)
    has_currency = OCR_PRICE_PATTERN.search(normalized) is not None
    return bool(words) and len(words) <= 4 and digits >= 2 and has_currency and digits > letters


def looks_header_like_text(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    words = split_words(normalized)
    if not words or len(words) > 4:
        return False
    return normalized == normalized.upper() and count_letters(normalized) >= 3


def looks_descriptor_like_text(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return normalized.count("/") >= 2 or OCR_DESCRIPTOR_PATTERN.search(normalized) is not None


def looks_descriptor_or_continuation_text(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized or line_has_price_signal(normalized):
        return False
    if looks_context_line(normalized) or looks_header_like_text(normalized):
        return False
    if looks_descriptor_like_text(normalized):
        return True

    words = split_words(normalized)
    if len(words) < 2 or count_letters(normalized) < 6:
        return False

    first_alpha = next((char for char in normalized if char.isalpha()), "")
    return bool(first_alpha and first_alpha.islower()) or bool(
        OCR_CONTINUATION_START_PATTERN.search(normalized)
    )


def looks_item_anchor_text(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized or looks_context_line(normalized):
        return False
    if looks_header_like_text(normalized) or looks_price_like_line(normalized):
        return False
    return count_letters(normalized) >= 3


def looks_price_target_line(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    if not normalized:
        return False
    if looks_context_line(normalized) or looks_header_like_text(normalized):
        return False
    if looks_price_like_line(normalized):
        return False
    return count_letters(normalized) >= 3


def looks_context_line(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return (
        looks_service_scope_like_text(normalized)
        or OCR_CONTEXT_NOTE_PATTERN.search(normalized) is not None
    )


def looks_service_scope_like_text(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return OCR_SERVICE_SCOPE_PATTERN.search(normalized) is not None


def looks_masthead_like_line(
    line: OcrLine,
    *,
    page_top: float,
    page_bottom: float,
) -> bool:
    text = unicodedata.normalize("NFKC", line.text).strip()
    if not text or line_has_price_signal(text) or line_has_size_signal(text):
        return False
    if looks_header_like_text(text) or count_letters(text) < 10:
        return False

    words = split_words(text)
    if len(words) < 2:
        return False

    page_height = max(1.0, page_bottom - page_top)
    if line.y_center > page_top + OCR_MASTHEAD_TOP_RATIO * page_height:
        return False

    return OCR_MASTHEAD_PATTERN.search(text) is not None


def line_has_price_signal(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return OCR_PRICE_PATTERN.search(normalized) is not None or looks_numeric_fragment(normalized)


def line_has_size_signal(text: str) -> bool:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return OCR_SIZE_PATTERN.search(normalized) is not None


def split_words(text: str) -> list[str]:
    return [part for part in OCR_NON_LETTER_PATTERN.split(text) if part]


def count_letters(text: str) -> int:
    return sum(char.isalpha() for char in text)


def normalize_image_ocr_text(text: str) -> str:
    return normalize_extracted_text(normalize_ocr_currency_suffixes(text))


def normalize_ocr_currency_suffixes(text: str) -> str:
    return OCR_RUBLE_SUFFIX_PATTERN.sub(r"\g<value> ₽", text)


def filter_reconstructed_ocr_lines(lines: Iterable[str]) -> list[str]:
    filtered: list[str] = []

    for raw_line in lines:
        line = unicodedata.normalize("NFKC", str(raw_line or "")).strip()
        if not line:
            continue
        if looks_numeric_fragment(line):
            continue
        if looks_context_line(line):
            continue

        has_price = line_has_price_signal(line)
        has_size = line_has_size_signal(line)
        if looks_descriptor_like_text(line) and not has_price and not has_size:
            continue

        words = split_words(line)
        if not has_price and not has_size and not looks_header_like_text(line):
            if count_letters(line) < 5:
                continue
            if len(words) <= 2 and line == line.upper():
                continue

        filtered.append(line)

    return filtered


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
