from __future__ import annotations

import argparse
from pathlib import Path

from classification_baseline_common import (
    REPO_ROOT,
    build_artifact,
    build_label_order,
    evaluate_split,
    print_summary,
    resolve_repo_path,
    save_artifact,
    split_items,
)
from dataset_common import load_annotated_items
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v1.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "docs/course/artifacts/tfidf-linear-svm-items-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="tfidf-linear-svm-v1-001")
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    return parser.parse_args()


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            (
                "classifier",
                LinearSVC(
                    max_iter=10000,
                    random_state=42,
                ),
            ),
        ]
    )


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
        method="tfidf_linear_svm",
        label_order=label_order,
        train_item_count=len(train_items),
        parameters={
            "vectorizer": {"type": "tfidf", "ngram_range": [1, 2]},
            "classifier": {
                "type": "linear_svc",
                "max_iter": 10000,
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
        notes=(
            "Text-only category classification baseline trained on the fixed train split of "
            "items.v1.jsonl and evaluated on the fixed valid and test splits using LinearSVC."
        ),
    )

    save_artifact(output_path, artifact)
    print_summary(artifact, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
