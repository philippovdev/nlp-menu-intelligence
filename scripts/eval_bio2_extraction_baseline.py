from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.bio2_extraction import (
    ACTIVE_ENTITY_LABELS,
    build_gold_bio2_tags,
    build_predicted_bio2_tags,
    compute_entity_scores,
    compute_token_scores,
)
from app.menu_parser import parse_menu_text
from app.schemas import FULL_CATEGORY_LABELS, MenuParseRequest
from classification_baseline_common import REPO_ROOT, format_output_path, resolve_repo_path, split_items
from dataset_common import AnnotatedItem, load_annotated_items
from pydantic import BaseModel, ConfigDict

DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v2.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs/course/artifacts/bio2-extraction-baseline-items-v2.json"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractionSplitMetrics(StrictModel):
    item_count: int
    token_count: int
    entity_count: int
    token_micro_precision: float
    token_micro_recall: float
    token_micro_f1: float
    token_macro_f1: float
    token_per_label_f1: dict[str, float]
    entity_micro_precision: float
    entity_micro_recall: float
    entity_micro_f1: float
    entity_macro_f1: float
    entity_per_label_f1: dict[str, float]


class ExtractionBaselineArtifact(StrictModel):
    run_id: str
    run_date: str
    commit_sha: str | None = None
    dataset_file: str
    method: str
    entity_labels: list[str]
    valid_metrics: ExtractionSplitMetrics
    test_metrics: ExtractionSplitMetrics
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="bio2-extraction-baseline-v2-001")
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    return parser.parse_args()


def infer_currency_hint(item: AnnotatedItem) -> str:
    currencies = {price.currency for price in item.slots.prices if price.currency}
    if len(currencies) == 1:
        return next(iter(currencies))
    return "RUB"


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


def evaluate_split(items: list[AnnotatedItem]) -> ExtractionSplitMetrics:
    gold_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []

    for item in items:
        gold_tags, gold_tokens = build_gold_bio2_tags(
            item.text,
            name=item.slots.name,
            prices=item.slots.prices,
            sizes=item.slots.sizes,
            default_currency=infer_currency_hint(item),
        )

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
            predicted_tags = ["O"] * len(gold_tags)
        else:
            predicted_tags, predicted_tokens = build_predicted_bio2_tags(
                item.text,
                name=predicted_item.fields.name,
                prices=predicted_item.fields.prices,
                sizes=predicted_item.fields.sizes,
                default_currency=infer_currency_hint(item),
            )
            if [token.text for token in predicted_tokens] != [token.text for token in gold_tokens]:
                raise RuntimeError("BIO2 tokenization mismatch between gold and predicted sequences.")

        gold_sequences.append(gold_tags)
        predicted_sequences.append(predicted_tags)

    token_metrics = compute_token_scores(gold_sequences, predicted_sequences)
    entity_metrics = compute_entity_scores(gold_sequences, predicted_sequences)

    return ExtractionSplitMetrics(
        item_count=len(items),
        token_count=int(token_metrics["token_count"]),
        entity_count=int(entity_metrics["entity_count"]),
        token_micro_precision=float(token_metrics["micro_precision"]),
        token_micro_recall=float(token_metrics["micro_recall"]),
        token_micro_f1=float(token_metrics["micro_f1"]),
        token_macro_f1=float(token_metrics["macro_f1"]),
        token_per_label_f1=dict(token_metrics["per_label_f1"]),
        entity_micro_precision=float(entity_metrics["micro_precision"]),
        entity_micro_recall=float(entity_metrics["micro_recall"]),
        entity_micro_f1=float(entity_metrics["micro_f1"]),
        entity_macro_f1=float(entity_metrics["macro_f1"]),
        entity_per_label_f1=dict(entity_metrics["per_label_f1"]),
    )


def build_artifact(
    *,
    dataset_path: Path,
    run_id: str,
    run_date: str | None,
    commit_sha: str | None,
    valid_metrics: ExtractionSplitMetrics,
    test_metrics: ExtractionSplitMetrics,
) -> ExtractionBaselineArtifact:
    resolved_dataset_path = resolve_repo_path(dataset_path)
    return ExtractionBaselineArtifact(
        run_id=run_id,
        run_date=run_date or datetime.now(UTC).date().isoformat(),
        commit_sha=commit_sha or detect_commit_sha(),
        dataset_file=str(resolved_dataset_path.relative_to(REPO_ROOT)),
        method="deterministic_bio2_extraction_baseline",
        entity_labels=list(ACTIVE_ENTITY_LABELS),
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        notes=(
            "Deterministic extraction baseline converted into BIO2 tags on the fixed dataset "
            "splits. Gold tags are derived from structured slot annotations with substring "
            "alignment for name and surface-fragment alignment for price and size spans."
        ),
    )


def print_summary(artifact: ExtractionBaselineArtifact, output_path: Path) -> None:
    print(f"Run ID: {artifact.run_id}")
    print(f"Dataset: {artifact.dataset_file}")
    print(
        "Valid: "
        f"token micro-F1={artifact.valid_metrics.token_micro_f1:.4f}, "
        f"entity micro-F1={artifact.valid_metrics.entity_micro_f1:.4f}"
    )
    print(
        "Test: "
        f"token micro-F1={artifact.test_metrics.token_micro_f1:.4f}, "
        f"entity micro-F1={artifact.test_metrics.entity_micro_f1:.4f}"
    )
    print(f"Artifact: {format_output_path(output_path)}")


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(args.output)
    items = load_annotated_items(dataset_path)
    items_by_split = split_items(items)

    artifact = build_artifact(
        dataset_path=dataset_path,
        run_id=args.run_id,
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        valid_metrics=evaluate_split(items_by_split["valid"]),
        test_metrics=evaluate_split(items_by_split["test"]),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    print_summary(artifact, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
