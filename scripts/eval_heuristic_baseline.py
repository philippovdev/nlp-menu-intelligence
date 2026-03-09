from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from app.menu_parser import parse_menu_text
from app.schemas import FULL_CATEGORY_LABELS, MenuParseRequest
from dataset_common import (
    AnnotatedItem,
    AnnotatedPrice,
    AnnotatedSize,
    load_annotated_items,
)
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.sample.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs/course/artifacts/baseline-heuristic-results.json"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExactMatchMetric(StrictModel):
    correct: int
    total: int
    accuracy: float


class BaselineMetrics(StrictModel):
    item_count: int
    category_accuracy: float
    category_macro_f1: float
    category_per_class_f1: dict[str, float]
    price_exact_match: ExactMatchMetric
    size_exact_match: ExactMatchMetric


class BaselineArtifact(StrictModel):
    run_id: str
    run_date: str
    commit_sha: str | None = None
    dataset_file: str
    method: str
    metrics: BaselineMetrics
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="baseline-heuristic-001")
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    return parser.parse_args()


def load_items(path: Path) -> list[AnnotatedItem]:
    return load_annotated_items(path)


def infer_currency_hint(item: AnnotatedItem) -> str:
    currencies = {price.currency for price in item.slots.prices if price.currency}
    if len(currencies) == 1:
        return next(iter(currencies))
    return "RUB"


def predict_item(
    item: AnnotatedItem,
) -> tuple[str | None, list[dict[str, object]], list[dict[str, object]]]:
    response = parse_menu_text(
        MenuParseRequest(
            schema_version="v1",
            text=item.text,
            lang=item.language,
            currency_hint=infer_currency_hint(item),
            category_labels=list(FULL_CATEGORY_LABELS),
        )
    )

    predicted_item = next((entry for entry in response.items if entry.kind == "menu_item"), None)
    if predicted_item is None:
        return None, [], []

    return (
        predicted_item.category.label,
        normalize_prices(predicted_item.fields.prices),
        normalize_sizes(predicted_item.fields.sizes),
    )


def normalize_prices(prices: Sequence[AnnotatedPrice]) -> list[dict[str, object]]:
    return [
        {"value": normalize_number(price.value), "currency": price.currency}
        for price in prices
    ]


def normalize_sizes(sizes: Sequence[AnnotatedSize]) -> list[dict[str, object]]:
    return [
        {"value": normalize_number(size.value), "unit": size.unit}
        for size in sizes
    ]


def normalize_number(value: int | float) -> int | float:
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return round(numeric, 4)


def compute_accuracy(gold: Sequence[str], predicted: Sequence[str | None]) -> float:
    correct = sum(expected == observed for expected, observed in zip(gold, predicted))
    return round(correct / len(gold), 4) if gold else 0.0


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
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
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


def detect_commit_sha() -> str | None:
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


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def evaluate(items: list[AnnotatedItem]) -> BaselineMetrics:
    gold_categories = [item.category for item in items]
    predicted_categories: list[str | None] = []
    gold_prices = [normalize_prices(item.slots.prices) for item in items]
    predicted_prices: list[list[dict[str, object]]] = []
    gold_sizes = [normalize_sizes(item.slots.sizes) for item in items]
    predicted_sizes: list[list[dict[str, object]]] = []

    for item in items:
        category, prices, sizes = predict_item(item)
        predicted_categories.append(category)
        predicted_prices.append(prices)
        predicted_sizes.append(sizes)

    category_macro_f1, per_class_f1 = compute_macro_f1(gold_categories, predicted_categories)

    return BaselineMetrics(
        item_count=len(items),
        category_accuracy=compute_accuracy(gold_categories, predicted_categories),
        category_macro_f1=category_macro_f1,
        category_per_class_f1=per_class_f1,
        price_exact_match=build_exact_match_metric(gold_prices, predicted_prices),
        size_exact_match=build_exact_match_metric(gold_sizes, predicted_sizes),
    )


def build_artifact(
    *,
    dataset_path: Path,
    run_id: str,
    run_date: str | None,
    commit_sha: str | None,
    metrics: BaselineMetrics,
) -> BaselineArtifact:
    resolved_dataset_path = resolve_repo_path(dataset_path)
    return BaselineArtifact(
        run_id=run_id,
        run_date=run_date or datetime.now(UTC).date().isoformat(),
        commit_sha=commit_sha or detect_commit_sha(),
        dataset_file=str(resolved_dataset_path.relative_to(REPO_ROOT)),
        method="heuristic_pipeline",
        metrics=metrics,
        notes=(
            "Item-level heuristic baseline using the current deterministic parser "
            "and keyword categorizer. Macro-F1 is computed over labels observed "
            "in this evaluation file. Price and size use exact-match on normalized lists."
        ),
    )


def save_artifact(path: Path, artifact: BaselineArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def format_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def print_summary(artifact: BaselineArtifact, output_path: Path) -> None:
    metrics = artifact.metrics
    print(f"Run ID: {artifact.run_id}")
    print(f"Dataset: {artifact.dataset_file}")
    print(f"Items: {metrics.item_count}")
    print(f"Category accuracy: {metrics.category_accuracy:.4f}")
    print(f"Category macro-F1: {metrics.category_macro_f1:.4f}")
    print(
        "Price exact match: "
        f"{metrics.price_exact_match.correct}/{metrics.price_exact_match.total} "
        f"({metrics.price_exact_match.accuracy:.4f})"
    )
    print(
        "Size exact match: "
        f"{metrics.size_exact_match.correct}/{metrics.size_exact_match.total} "
        f"({metrics.size_exact_match.accuracy:.4f})"
    )
    print(f"Artifact: {format_output_path(output_path)}")


def main() -> None:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(args.output)
    items = load_items(dataset_path)
    metrics = evaluate(items)
    artifact = build_artifact(
        dataset_path=dataset_path,
        run_id=args.run_id,
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        metrics=metrics,
    )
    save_artifact(output_path, artifact)
    print_summary(artifact, output_path)


if __name__ == "__main__":
    main()
