from __future__ import annotations

import argparse
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dataset_common import AnnotatedItem, load_annotated_items
from pydantic import BaseModel, ConfigDict
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v1.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs/course/artifacts/tfidf-logreg-items-v1.json"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="tfidf-logreg-v1-001")
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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


def split_items(items: list[AnnotatedItem]) -> dict[str, list[AnnotatedItem]]:
    grouped = {split: [item for item in items if item.split == split] for split in SPLIT_ORDER}
    missing = [split for split, split_items_ in grouped.items() if not split_items_]
    if missing:
        missing_splits = ", ".join(missing)
        raise ValueError(f"Dataset is missing required splits: {missing_splits}")
    return grouped


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_label_order(items: list[AnnotatedItem]) -> list[str]:
    return sorted({item.category for item in items})


def evaluate_split(
    *,
    pipeline: Pipeline,
    items: list[AnnotatedItem],
    label_order: list[str],
) -> SplitMetrics:
    texts = [item.text for item in items]
    gold = [item.category for item in items]
    predicted = pipeline.predict(texts).tolist()
    report = classification_report(
        gold,
        predicted,
        labels=label_order,
        output_dict=True,
        zero_division=0,
    )

    return SplitMetrics(
        item_count=len(items),
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
    label_order: list[str],
    train_item_count: int,
    valid_metrics: SplitMetrics,
    test_metrics: SplitMetrics,
) -> BaselineArtifact:
    resolved_dataset_path = resolve_repo_path(dataset_path)
    return BaselineArtifact(
        run_id=run_id,
        run_date=run_date or datetime.now(UTC).date().isoformat(),
        commit_sha=commit_sha or detect_commit_sha(),
        scikit_learn_version=sklearn_version,
        dataset_file=str(resolved_dataset_path.relative_to(REPO_ROOT)),
        method="tfidf_logistic_regression",
        label_order=label_order,
        train_item_count=train_item_count,
        parameters={
            "vectorizer": {"type": "tfidf", "ngram_range": [1, 2]},
            "classifier": {
                "type": "logistic_regression",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        notes=(
            "Text-only category classification baseline trained on the fixed train split of "
            "items.v1.jsonl and evaluated on the fixed valid and test splits."
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


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(args.output)
    items = load_annotated_items(dataset_path)
    items_by_split = split_items(items)
    label_order = build_label_order(items)

    pipeline = build_pipeline()
    train_items = items_by_split["train"]
    pipeline.fit(
        [item.text for item in train_items],
        [item.category for item in train_items],
    )

    artifact = build_artifact(
        dataset_path=dataset_path,
        run_id=args.run_id,
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        label_order=label_order,
        train_item_count=len(train_items),
        valid_metrics=evaluate_split(
            pipeline=pipeline,
            items=items_by_split["valid"],
            label_order=label_order,
        ),
        test_metrics=evaluate_split(
            pipeline=pipeline,
            items=items_by_split["test"],
            label_order=label_order,
        ),
    )

    save_artifact(output_path, artifact)
    print_summary(artifact, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
