from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from app.image_ocr import normalize_extracted_text
from app.main import create_app
from dataset_common import AnnotatedPrice, AnnotatedSize
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "data/eval/realworld-manifest.v1.csv"
DEFAULT_GOLD = REPO_ROOT / "data/eval/realworld-gold.v1.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs/course/artifacts/realworld-eval-v1.json"
INPUT_TYPE_ORDER = ("text", "pdf", "image")
NormalizedFieldList = list[dict[str, object]]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalManifestRow(StrictModel):
    eval_id: str
    source_id: str
    restaurant_id: str
    source_type: str
    input_type: Literal["text", "pdf", "image"]
    fixture_path: str | None = None
    source_url: str
    subset: str
    notes: str

    @field_validator("fixture_path", mode="before")
    @classmethod
    def normalize_fixture_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class EvalGoldItem(StrictModel):
    text: str
    category: str
    slots: EvalGoldSlots


class EvalGoldSlots(StrictModel):
    name: str | None = None
    prices: list[AnnotatedPrice] = Field(default_factory=list)
    sizes: list[AnnotatedSize] = Field(default_factory=list)


class EvalGoldCase(StrictModel):
    eval_id: str
    source_id: str
    restaurant_id: str
    input_type: Literal["text", "pdf", "image"]
    fixture_path: str | None = None
    input_text: str | None = None
    gold_extracted_text: str
    gold_items: list[EvalGoldItem]
    notes: str

    @field_validator("fixture_path", "input_text", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ExactMatchMetric(StrictModel):
    correct: int
    total: int
    accuracy: float


class TextQualityMetrics(StrictModel):
    exact_match_examples: int
    example_count: int
    exact_match_rate: float
    token_precision: float
    token_recall: float
    token_f1: float


class ItemCountMetrics(StrictModel):
    exact_match_examples: int
    example_count: int
    exact_match_rate: float
    gold_item_count: int
    predicted_item_count: int


class CategoryMetrics(StrictModel):
    item_count: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]


class QualitySlice(StrictModel):
    example_count: int
    text_quality: TextQualityMetrics
    item_count: ItemCountMetrics
    category: CategoryMetrics
    field_quality: FieldQualityMetrics


class FieldQualityMetrics(StrictModel):
    price_exact_match: ExactMatchMetric
    size_exact_match: ExactMatchMetric


class ExampleSummary(StrictModel):
    eval_id: str
    input_type: Literal["text", "pdf", "image"]
    source_id: str
    request_ok: bool
    status_code: int
    error_code: str | None = None
    text_exact_match: bool
    text_token_f1: float
    gold_item_count: int
    predicted_item_count: int
    correct_category_count: int
    correct_price_count: int
    correct_size_count: int
    issue_codes: list[str]
    category_model: str | None = None
    ocr_used: bool | None = None


class ErrorEntry(StrictModel):
    eval_id: str
    input_type: Literal["text", "pdf", "image"]
    item_index: int | None = None
    gold: str | list[dict[str, object]] | None = None
    predicted: str | list[dict[str, object]] | None = None


class ErrorSummary(StrictModel):
    failed_requests: list[dict[str, object]]
    text_mismatch_eval_ids: list[str]
    item_count_mismatch_eval_ids: list[str]
    category_mismatches: list[ErrorEntry]
    price_mismatch_examples: list[ErrorEntry]
    size_mismatch_examples: list[ErrorEntry]


class RealworldEvalArtifact(StrictModel):
    run_id: str
    run_date: str
    commit_sha: str | None = None
    eval_manifest_file: str
    eval_gold_file: str
    example_count: int
    input_type_counts: dict[str, int]
    runtime: dict[str, object]
    metrics: dict[str, QualitySlice]
    error_summary: ErrorSummary
    examples: list[ExampleSummary]
    notes: str


class CaseRunResult(StrictModel):
    case: EvalGoldCase
    request_ok: bool
    status_code: int
    error_code: str | None = None
    actual_extracted_text: str = ""
    predicted_categories: list[str | None] = Field(default_factory=list)
    predicted_prices: list[list[dict[str, object]]] = Field(default_factory=list)
    predicted_sizes: list[list[dict[str, object]]] = Field(default_factory=list)
    predicted_item_count: int = 0
    issue_codes: list[str] = Field(default_factory=list)
    category_model: str | None = None
    ocr_used: bool | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="realworld-eval-v1-001")
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def detect_commit_sha() -> str | None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        return None
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def load_eval_manifest(path: Path) -> list[EvalManifestRow]:
    with resolve_repo_path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [EvalManifestRow.model_validate(row) for row in reader]


def load_eval_gold(path: Path) -> list[EvalGoldCase]:
    lines = resolve_repo_path(path).read_text(encoding="utf-8").splitlines()
    return [EvalGoldCase.model_validate_json(line) for line in lines if line.strip()]


def validate_eval_inputs(
    manifest_rows: list[EvalManifestRow],
    gold_cases: list[EvalGoldCase],
) -> None:
    manifest_by_id = {row.eval_id: row for row in manifest_rows}
    gold_by_id = {case.eval_id: case for case in gold_cases}

    if manifest_by_id.keys() != gold_by_id.keys():
        raise ValueError("Manifest and gold eval IDs do not match.")

    for eval_id, case in gold_by_id.items():
        manifest_row = manifest_by_id[eval_id]
        if case.source_id != manifest_row.source_id:
            raise ValueError(f"Source mismatch for {eval_id}.")
        if case.restaurant_id != manifest_row.restaurant_id:
            raise ValueError(f"Restaurant mismatch for {eval_id}.")
        if case.input_type != manifest_row.input_type:
            raise ValueError(f"Input type mismatch for {eval_id}.")
        if case.fixture_path != manifest_row.fixture_path:
            raise ValueError(f"Fixture path mismatch for {eval_id}.")
        if case.input_type == "text" and not case.input_text:
            raise ValueError(f"Text example {eval_id} is missing input_text.")
        if case.input_type != "text" and not case.fixture_path:
            raise ValueError(f"File example {eval_id} is missing fixture_path.")
        if not case.gold_items:
            raise ValueError(f"Eval case {eval_id} has no gold items.")
        if case.fixture_path is not None and not resolve_repo_path(
            Path(case.fixture_path)
        ).is_file():
            raise ValueError(f"Fixture file does not exist for {eval_id}.")


def detect_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    raise ValueError(f"Unsupported fixture extension: {path.suffix}")


def normalize_number(value: int | float) -> int | float:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 4)


def normalize_prices(
    prices: Sequence[AnnotatedPrice] | Sequence[dict[str, object]],
) -> NormalizedFieldList:
    normalized: NormalizedFieldList = []
    for price in prices:
        value = price.value if isinstance(price, AnnotatedPrice) else price["value"]
        currency = price.currency if isinstance(price, AnnotatedPrice) else price["currency"]
        normalized.append({"value": normalize_number(value), "currency": str(currency)})
    return normalized


def normalize_sizes(
    sizes: Sequence[AnnotatedSize] | Sequence[dict[str, object]],
) -> NormalizedFieldList:
    normalized: NormalizedFieldList = []
    for size in sizes:
        value = size.value if isinstance(size, AnnotatedSize) else size["value"]
        unit = size.unit if isinstance(size, AnnotatedSize) else size["unit"]
        normalized.append({"value": normalize_number(value), "unit": str(unit)})
    return normalized


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def compute_text_scores(gold: str, predicted: str) -> tuple[bool, int, int, int]:
    gold_tokens = Counter(tokenize(gold))
    predicted_tokens = Counter(tokenize(predicted))
    true_positive = sum((gold_tokens & predicted_tokens).values())
    predicted_total = sum(predicted_tokens.values())
    gold_total = sum(gold_tokens.values())
    return gold == predicted, true_positive, predicted_total, gold_total


def compute_accuracy(gold: Sequence[str], predicted: Sequence[str | None]) -> float:
    if not gold:
        return 0.0
    correct = sum(expected == observed for expected, observed in zip(gold, predicted))
    return round(correct / len(gold), 4)


def compute_macro_f1(
    gold: Sequence[str],
    predicted: Sequence[str | None],
) -> tuple[float, dict[str, float]]:
    labels = sorted({*gold, *(label for label in predicted if label is not None)})
    if not labels:
        return 0.0, {}

    per_class: dict[str, float] = {}
    for label in labels:
        true_positive = sum(
            expected == label and observed == label
            for expected, observed in zip(gold, predicted)
        )
        false_positive = sum(
            expected != label and observed == label
            for expected, observed in zip(gold, predicted)
        )
        false_negative = sum(
            expected == label and observed != label
            for expected, observed in zip(gold, predicted)
        )

        if true_positive + false_positive:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0.0
        if true_positive + false_negative:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = round(f1, 4)

    macro_f1 = round(sum(per_class.values()) / len(per_class), 4)
    return macro_f1, per_class


def build_exact_match_metric(
    gold: Sequence[list[dict[str, object]]],
    predicted: Sequence[list[dict[str, object]]],
) -> ExactMatchMetric:
    correct = sum(expected == observed for expected, observed in zip(gold, predicted))
    total = len(gold)
    accuracy = round(correct / total, 4) if total else 0.0
    return ExactMatchMetric(correct=correct, total=total, accuracy=accuracy)


def build_item_count_metric(
    gold_counts: Sequence[int],
    predicted_counts: Sequence[int],
) -> ItemCountMetrics:
    exact_match_examples = sum(
        expected == observed for expected, observed in zip(gold_counts, predicted_counts)
    )
    example_count = len(gold_counts)
    exact_match_rate = round(exact_match_examples / example_count, 4) if example_count else 0.0
    return ItemCountMetrics(
        exact_match_examples=exact_match_examples,
        example_count=example_count,
        exact_match_rate=exact_match_rate,
        gold_item_count=sum(gold_counts),
        predicted_item_count=sum(predicted_counts),
    )


def run_text_case(client: TestClient, case: EvalGoldCase) -> CaseRunResult:
    response = client.post(
        "/api/v1/menu/parse",
        json={
            "schema_version": "v1",
            "text": case.input_text,
            "lang": "en",
            "currency_hint": "RUB",
        },
    )
    return build_case_result(case, response.status_code, response.json())


def run_file_case(client: TestClient, case: EvalGoldCase) -> CaseRunResult:
    fixture_path = resolve_repo_path(Path(case.fixture_path or ""))
    response = client.post(
        "/api/v1/menu/parse-file",
        data={"schema_version": "v1", "lang": "en", "currency_hint": "RUB"},
        files={
            "file": (
                fixture_path.name,
                fixture_path.read_bytes(),
                detect_media_type(fixture_path),
            )
        },
    )
    return build_case_result(case, response.status_code, response.json())


def build_case_result(
    case: EvalGoldCase,
    status_code: int,
    payload: dict[str, object],
) -> CaseRunResult:
    if status_code != 200:
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        return CaseRunResult(
            case=case,
            request_ok=False,
            status_code=status_code,
            error_code=error.get("code") if isinstance(error, dict) else None,
        )

    items = payload["items"]
    menu_items = [item for item in items if item["kind"] == "menu_item"]
    predicted_categories = [item["category"]["label"] for item in menu_items]
    predicted_prices = [normalize_prices(item["fields"]["prices"]) for item in menu_items]
    predicted_sizes = [normalize_sizes(item["fields"]["sizes"]) for item in menu_items]
    issue_codes = [issue["code"] for issue in payload.get("issues", [])]
    document = payload.get("document")

    if isinstance(document, dict):
        extracted_text = normalize_extracted_text(document.get("extracted_text", ""))
        ocr_used = document.get("ocr_used")
    else:
        extracted_text = normalize_extracted_text(case.input_text or "")
        ocr_used = None

    return CaseRunResult(
        case=case,
        request_ok=True,
        status_code=status_code,
        actual_extracted_text=extracted_text,
        predicted_categories=predicted_categories,
        predicted_prices=predicted_prices,
        predicted_sizes=predicted_sizes,
        predicted_item_count=len(menu_items),
        issue_codes=issue_codes,
        category_model=payload["model_version"]["category_model"],
        ocr_used=ocr_used,
    )


def align_predictions(
    values: Sequence[object],
    expected_length: int,
    *,
    fill_value: object,
) -> list[object]:
    aligned = list(values[:expected_length])
    if len(aligned) < expected_length:
        aligned.extend([fill_value] * (expected_length - len(aligned)))
    return aligned


def build_quality_slice(results: Sequence[CaseRunResult]) -> QualitySlice:
    text_exact_match_examples = 0
    token_true_positive = 0
    token_predicted_total = 0
    token_gold_total = 0
    gold_counts: list[int] = []
    predicted_counts: list[int] = []
    gold_categories: list[str] = []
    predicted_categories: list[str | None] = []
    gold_prices: list[list[dict[str, object]]] = []
    predicted_prices: list[list[dict[str, object]]] = []
    gold_sizes: list[list[dict[str, object]]] = []
    predicted_sizes: list[list[dict[str, object]]] = []

    for result in results:
        exact_match, true_positive, predicted_total, gold_total = compute_text_scores(
            result.case.gold_extracted_text,
            result.actual_extracted_text,
        )
        if exact_match:
            text_exact_match_examples += 1
        token_true_positive += true_positive
        token_predicted_total += predicted_total
        token_gold_total += gold_total

        gold_counts.append(len(result.case.gold_items))
        predicted_counts.append(result.predicted_item_count)

        gold_case_categories = [item.category for item in result.case.gold_items]
        predicted_case_categories = align_predictions(
            result.predicted_categories,
            len(gold_case_categories),
            fill_value=None,
        )
        gold_categories.extend(gold_case_categories)
        predicted_categories.extend(predicted_case_categories)

        gold_case_prices = [normalize_prices(item.slots.prices) for item in result.case.gold_items]
        gold_case_sizes = [normalize_sizes(item.slots.sizes) for item in result.case.gold_items]
        predicted_case_prices = align_predictions(
            result.predicted_prices,
            len(gold_case_prices),
            fill_value=[],
        )
        predicted_case_sizes = align_predictions(
            result.predicted_sizes,
            len(gold_case_sizes),
            fill_value=[],
        )

        gold_prices.extend(gold_case_prices)
        predicted_prices.extend(predicted_case_prices)
        gold_sizes.extend(gold_case_sizes)
        predicted_sizes.extend(predicted_case_sizes)

    example_count = len(results)
    token_precision = (
        round(token_true_positive / token_predicted_total, 4)
        if token_predicted_total
        else 0.0
    )
    token_recall = round(token_true_positive / token_gold_total, 4) if token_gold_total else 0.0
    if token_precision + token_recall == 0:
        token_f1 = 0.0
    else:
        token_f1 = round(2 * token_precision * token_recall / (token_precision + token_recall), 4)

    macro_f1, per_class_f1 = compute_macro_f1(gold_categories, predicted_categories)

    return QualitySlice(
        example_count=example_count,
        text_quality=TextQualityMetrics(
            exact_match_examples=text_exact_match_examples,
            example_count=example_count,
            exact_match_rate=(
                round(text_exact_match_examples / example_count, 4)
                if example_count
                else 0.0
            ),
            token_precision=token_precision,
            token_recall=token_recall,
            token_f1=token_f1,
        ),
        item_count=build_item_count_metric(gold_counts, predicted_counts),
        category=CategoryMetrics(
            item_count=len(gold_categories),
            accuracy=compute_accuracy(gold_categories, predicted_categories),
            macro_f1=macro_f1,
            per_class_f1=per_class_f1,
        ),
        field_quality=FieldQualityMetrics(
            price_exact_match=build_exact_match_metric(gold_prices, predicted_prices),
            size_exact_match=build_exact_match_metric(gold_sizes, predicted_sizes),
        ),
    )


def build_error_summary(results: Sequence[CaseRunResult]) -> ErrorSummary:
    failed_requests: list[dict[str, object]] = []
    text_mismatch_eval_ids: list[str] = []
    item_count_mismatch_eval_ids: list[str] = []
    category_mismatches: list[ErrorEntry] = []
    price_mismatch_examples: list[ErrorEntry] = []
    size_mismatch_examples: list[ErrorEntry] = []

    for result in results:
        case = result.case
        if not result.request_ok:
            failed_requests.append(
                {
                    "eval_id": case.eval_id,
                    "input_type": case.input_type,
                    "status_code": result.status_code,
                    "error_code": result.error_code,
                }
            )
            text_mismatch_eval_ids.append(case.eval_id)
            item_count_mismatch_eval_ids.append(case.eval_id)
            continue

        if normalize_extracted_text(case.gold_extracted_text) != result.actual_extracted_text:
            text_mismatch_eval_ids.append(case.eval_id)

        if len(case.gold_items) != result.predicted_item_count:
            item_count_mismatch_eval_ids.append(case.eval_id)

        aligned_categories = align_predictions(
            result.predicted_categories,
            len(case.gold_items),
            fill_value=None,
        )
        aligned_prices = align_predictions(
            result.predicted_prices,
            len(case.gold_items),
            fill_value=[],
        )
        aligned_sizes = align_predictions(
            result.predicted_sizes,
            len(case.gold_items),
            fill_value=[],
        )

        for index, gold_item in enumerate(case.gold_items):
            predicted_category = aligned_categories[index]
            if gold_item.category != predicted_category:
                category_mismatches.append(
                    ErrorEntry(
                        eval_id=case.eval_id,
                        input_type=case.input_type,
                        item_index=index,
                        gold=gold_item.category,
                        predicted=predicted_category,
                    )
                )

            expected_prices = normalize_prices(gold_item.slots.prices)
            expected_sizes = normalize_sizes(gold_item.slots.sizes)
            predicted_price_list = aligned_prices[index]
            predicted_size_list = aligned_sizes[index]

            if expected_prices != predicted_price_list:
                price_mismatch_examples.append(
                    ErrorEntry(
                        eval_id=case.eval_id,
                        input_type=case.input_type,
                        item_index=index,
                        gold=expected_prices,
                        predicted=predicted_price_list,
                    )
                )
            if expected_sizes != predicted_size_list:
                size_mismatch_examples.append(
                    ErrorEntry(
                        eval_id=case.eval_id,
                        input_type=case.input_type,
                        item_index=index,
                        gold=expected_sizes,
                        predicted=predicted_size_list,
                    )
                )

    return ErrorSummary(
        failed_requests=failed_requests,
        text_mismatch_eval_ids=text_mismatch_eval_ids,
        item_count_mismatch_eval_ids=item_count_mismatch_eval_ids,
        category_mismatches=category_mismatches,
        price_mismatch_examples=price_mismatch_examples,
        size_mismatch_examples=size_mismatch_examples,
    )


def build_example_summaries(results: Sequence[CaseRunResult]) -> list[ExampleSummary]:
    summaries: list[ExampleSummary] = []
    for result in results:
        gold_items = result.case.gold_items
        predicted_categories = align_predictions(
            result.predicted_categories,
            len(gold_items),
            fill_value=None,
        )
        predicted_prices = align_predictions(
            result.predicted_prices,
            len(gold_items),
            fill_value=[],
        )
        predicted_sizes = align_predictions(
            result.predicted_sizes,
            len(gold_items),
            fill_value=[],
        )

        correct_category_count = sum(
            gold_item.category == predicted_categories[index]
            for index, gold_item in enumerate(gold_items)
        )
        correct_price_count = sum(
            normalize_prices(gold_item.slots.prices) == predicted_prices[index]
            for index, gold_item in enumerate(gold_items)
        )
        correct_size_count = sum(
            normalize_sizes(gold_item.slots.sizes) == predicted_sizes[index]
            for index, gold_item in enumerate(gold_items)
        )

        exact_text_match, true_positive, predicted_total, gold_total = compute_text_scores(
            result.case.gold_extracted_text,
            result.actual_extracted_text,
        )
        if predicted_total == 0 or gold_total == 0 or true_positive == 0:
            token_f1 = 0.0 if gold_total or predicted_total else 1.0
        else:
            precision = true_positive / predicted_total
            recall = true_positive / gold_total
            token_f1 = round(2 * precision * recall / (precision + recall), 4)

        summaries.append(
            ExampleSummary(
                eval_id=result.case.eval_id,
                input_type=result.case.input_type,
                source_id=result.case.source_id,
                request_ok=result.request_ok,
                status_code=result.status_code,
                error_code=result.error_code,
                text_exact_match=exact_text_match,
                text_token_f1=token_f1,
                gold_item_count=len(gold_items),
                predicted_item_count=result.predicted_item_count,
                correct_category_count=correct_category_count,
                correct_price_count=correct_price_count,
                correct_size_count=correct_size_count,
                issue_codes=result.issue_codes,
                category_model=result.category_model,
                ocr_used=result.ocr_used,
            )
        )
    return summaries


def build_artifact(
    *,
    manifest_path: Path,
    gold_path: Path,
    run_id: str,
    run_date: str | None,
    commit_sha: str | None,
    runtime: dict[str, object],
    results: list[CaseRunResult],
) -> RealworldEvalArtifact:
    results_by_type = {
        input_type: [result for result in results if result.case.input_type == input_type]
        for input_type in INPUT_TYPE_ORDER
    }

    resolved_manifest_path = resolve_repo_path(manifest_path)
    resolved_gold_path = resolve_repo_path(gold_path)

    return RealworldEvalArtifact(
        run_id=run_id,
        run_date=run_date or datetime.now(UTC).date().isoformat(),
        commit_sha=commit_sha or detect_commit_sha(),
        eval_manifest_file=str(resolved_manifest_path.relative_to(REPO_ROOT)),
        eval_gold_file=str(resolved_gold_path.relative_to(REPO_ROOT)),
        example_count=len(results),
        input_type_counts={
            input_type: len(results_by_type[input_type]) for input_type in INPUT_TYPE_ORDER
        },
        runtime=runtime,
        metrics={
            "overall": build_quality_slice(results),
            **{
                input_type: build_quality_slice(type_results)
                for input_type, type_results in results_by_type.items()
            },
        },
        error_summary=build_error_summary(results),
        examples=build_example_summaries(results),
        notes=(
            "Small source-grounded real-world evaluation slice. Text cases are pasted snippets; "
            "PDF and image cases use compact rendered fixtures anchored to the same public-source "
            "menu lines, so OCR quality here is informative but easier than arbitrary phone "
            "photos. "
            "Category, price, and size comparisons are aligned by predicted menu-item order."
        ),
    )


def save_artifact(path: Path, artifact: RealworldEvalArtifact) -> None:
    resolved_path = resolve_repo_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def format_output_path(path: Path) -> str:
    try:
        return str(resolve_repo_path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def print_summary(artifact: RealworldEvalArtifact, output_path: Path) -> None:
    overall = artifact.metrics["overall"]
    print(f"Run ID: {artifact.run_id}")
    print(f"Examples: {artifact.example_count}")
    print(
        "Input types: "
        + ", ".join(f"{key}={value}" for key, value in artifact.input_type_counts.items())
    )
    print(
        "Runtime: "
        f"category_model={artifact.runtime['category_model']}, "
        f"ready={artifact.runtime['category_model_ready']}"
    )
    print(
        "Overall: "
        f"text exact={overall.text_quality.exact_match_rate:.4f}, "
        f"text token F1={overall.text_quality.token_f1:.4f}, "
        f"category acc={overall.category.accuracy:.4f}, "
        f"category macro-F1={overall.category.macro_f1:.4f}, "
        f"price exact={overall.field_quality.price_exact_match.accuracy:.4f}, "
        f"size exact={overall.field_quality.size_exact_match.accuracy:.4f}"
    )
    for input_type in INPUT_TYPE_ORDER:
        slice_metrics = artifact.metrics[input_type]
        print(
            f"{input_type.title()}: "
            f"text token F1={slice_metrics.text_quality.token_f1:.4f}, "
            f"category acc={slice_metrics.category.accuracy:.4f}, "
            f"price exact={slice_metrics.field_quality.price_exact_match.accuracy:.4f}, "
            f"size exact={slice_metrics.field_quality.size_exact_match.accuracy:.4f}"
        )
    print(f"Artifact: {format_output_path(output_path)}")


def main() -> None:
    args = parse_args()
    manifest_rows = load_eval_manifest(args.manifest)
    gold_cases = load_eval_gold(args.gold)
    validate_eval_inputs(manifest_rows, gold_cases)

    with TestClient(create_app()) as client:
        runtime = client.get("/api/v1/version").json()
        results = [
            run_text_case(client, case)
            if case.input_type == "text"
            else run_file_case(client, case)
            for case in gold_cases
        ]

    artifact = build_artifact(
        manifest_path=args.manifest,
        gold_path=args.gold,
        run_id=args.run_id,
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        runtime=runtime,
        results=results,
    )
    save_artifact(args.output, artifact)
    print_summary(artifact, args.output)


if __name__ == "__main__":
    main()
