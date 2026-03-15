from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from classification_baseline_common import (
    REPO_ROOT,
    build_artifact,
    build_label_order,
    default_notes,
    evaluate_split,
    format_output_path,
    resolve_repo_path,
    split_items,
)
from dataset_common import load_annotated_items
from run_tfidf_logreg_baseline import build_pipeline

DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v2.jsonl"
DEFAULT_MODEL_OUTPUT = REPO_ROOT / "backend/app/model_assets/category_classifier.pkl"
DEFAULT_METADATA_OUTPUT = REPO_ROOT / "backend/app/model_assets/category_classifier.json"
DEFAULT_MODEL_ID = "tfidf-logreg-items-v2@1.0.0"
DEFAULT_MINIMUM_CONFIDENCE = 0.35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--minimum-confidence", type=float, default=DEFAULT_MINIMUM_CONFIDENCE)
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    return parser.parse_args()


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    model_output = resolve_repo_path(args.model_output)
    metadata_output = resolve_repo_path(args.metadata_output)
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
        run_id=f"export-category-classifier-{dataset_path.stem.split('.')[-1]}-001",
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        method="tfidf_logistic_regression",
        label_order=label_order,
        train_item_count=len(train_items),
        parameters={
            "vectorizer": {"type": "tfidf", "ngram_range": [1, 2]},
            "classifier": {
                "type": "logistic_regression",
                "max_iter": 1000,
                "random_state": 42,
            },
        },
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
        notes=default_notes(dataset_path=dataset_path, classifier_label="LogisticRegression"),
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as file_pointer:
        pickle.dump(pipeline, file_pointer, protocol=5)

    metadata = artifact.model_dump(mode="json")
    metadata.update(
        {
            "model_id": args.model_id,
            "minimum_confidence": args.minimum_confidence,
            "model_file": relative_to_repo(model_output),
        }
    )
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Dataset: {artifact.dataset_file}")
    print(f"Model ID: {args.model_id}")
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
    print(f"Model file: {format_output_path(model_output)}")
    print(f"Metadata file: {format_output_path(metadata_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
