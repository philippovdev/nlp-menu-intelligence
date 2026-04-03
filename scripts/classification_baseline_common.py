from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from app.category_model_features import (
    RECORD_INPUT_FORMAT,
    TEXT_INPUT_FORMAT,
    build_category_model_record,
)
from dataset_common import AnnotatedItem
from pydantic import BaseModel, ConfigDict
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_ORDER = ("train", "valid", "test")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitMetrics(StrictModel):
    item_count: int
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]


class BaselineArtifact(StrictModel):
    run_id: str
    run_date: str
    commit_sha: str | None = None
    scikit_learn_version: str
    dataset_file: str
    method: str
    label_order: list[str]
    train_item_count: int
    parameters: dict[str, object]
    valid_metrics: SplitMetrics
    test_metrics: SplitMetrics
    notes: str


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def dataset_version_tag(dataset_path: Path) -> str:
    stem = dataset_path.stem
    if "." in stem:
        return stem.split(".")[-1]
    return stem


def default_output_path(*, prefix: str, dataset_path: Path) -> Path:
    dataset_tag = dataset_version_tag(dataset_path)
    return REPO_ROOT / f"docs/course/artifacts/{prefix}-items-{dataset_tag}.json"


def default_run_id(*, prefix: str, dataset_path: Path) -> str:
    dataset_tag = dataset_version_tag(dataset_path)
    return f"{prefix}-{dataset_tag}-001"


def default_notes(*, dataset_path: Path, classifier_label: str) -> str:
    return (
        "Text-only category classification baseline trained on the fixed train split of "
        f"{dataset_path.name} and evaluated on the fixed valid and test splits using "
        f"{classifier_label}."
    )


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


def split_items(items: list[AnnotatedItem]) -> dict[str, list[AnnotatedItem]]:
    grouped = {split: [item for item in items if item.split == split] for split in SPLIT_ORDER}
    missing = [split for split, split_items_ in grouped.items() if not split_items_]
    if missing:
        missing_splits = ", ".join(missing)
        raise ValueError(f"Dataset is missing required splits: {missing_splits}")
    return grouped


def build_label_order(items: list[AnnotatedItem]) -> list[str]:
    return sorted({item.category for item in items})


def build_model_inputs(
    items: list[AnnotatedItem],
    *,
    input_format: str = TEXT_INPUT_FORMAT,
) -> list[object]:
    if input_format == TEXT_INPUT_FORMAT:
        return [item.text for item in items]
    if input_format == RECORD_INPUT_FORMAT:
        return [
            build_category_model_record(
                text=item.text,
                name=item.slots.name,
                prices=item.slots.prices,
                sizes=item.slots.sizes,
            )
            for item in items
        ]
    raise ValueError(f"Unsupported category model input format: {input_format}")


def evaluate_split(
    *,
    pipeline: Pipeline,
    items: list[AnnotatedItem],
    label_order: list[str],
    input_format: str = TEXT_INPUT_FORMAT,
) -> SplitMetrics:
    texts = build_model_inputs(items, input_format=input_format)
    gold = [item.category for item in items]
    predicted = pipeline.predict(texts).tolist()
    return build_split_metrics(gold=gold, predicted=predicted, label_order=label_order)


def build_split_metrics(
    *,
    gold: list[str],
    predicted: list[str],
    label_order: list[str],
) -> SplitMetrics:
    report = classification_report(
        gold,
        predicted,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )

    return SplitMetrics(
        item_count=len(gold),
        accuracy=round(accuracy_score(gold, predicted), 4),
        macro_f1=round(
            f1_score(gold, predicted, labels=label_order, average="macro", zero_division=0),
            4,
        ),
        per_class_f1={label: round(report[label]["f1-score"], 4) for label in label_order},
        confusion_matrix=confusion_matrix(gold, predicted, labels=label_order).tolist(),
    )


def build_artifact(
    *,
    dataset_path: Path,
    run_id: str,
    run_date: str | None,
    commit_sha: str | None,
    method: str,
    label_order: list[str],
    train_item_count: int,
    parameters: dict[str, object],
    valid_metrics: SplitMetrics,
    test_metrics: SplitMetrics,
    notes: str,
) -> BaselineArtifact:
    resolved_dataset_path = resolve_repo_path(dataset_path)
    return BaselineArtifact(
        run_id=run_id,
        run_date=run_date or datetime.now(UTC).date().isoformat(),
        commit_sha=commit_sha or detect_commit_sha(),
        scikit_learn_version=sklearn_version,
        dataset_file=str(resolved_dataset_path.relative_to(REPO_ROOT)),
        method=method,
        label_order=label_order,
        train_item_count=train_item_count,
        parameters=parameters,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        notes=notes,
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
    print(f"Run ID: {artifact.run_id}")
    print(f"Dataset: {artifact.dataset_file}")
    print(f"Train items: {artifact.train_item_count}")
    print(
        "Valid: "
        f"accuracy={artifact.valid_metrics.accuracy:.4f}, "
        f"macro-F1={artifact.valid_metrics.macro_f1:.4f}"
    )
    print(
        "Test: "
        f"accuracy={artifact.test_metrics.accuracy:.4f}, "
        f"macro-F1={artifact.test_metrics.macro_f1:.4f}"
    )
    print(f"Artifact: {format_output_path(output_path)}")
