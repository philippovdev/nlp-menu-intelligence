from __future__ import annotations

import argparse
from pathlib import Path

from category_model_builders import (
    build_tfidf_calibrated_linear_svm_parameters,
    build_tfidf_calibrated_linear_svm_pipeline,
    resolve_calibration_cv,
    sparse_compute_backend_description,
)
from classification_baseline_common import (
    REPO_ROOT,
    build_artifact,
    build_label_order,
    default_output_path,
    default_run_id,
    evaluate_split,
    print_summary,
    resolve_repo_path,
    save_artifact,
    split_items,
)
from dataset_common import load_annotated_items
from sklearn.pipeline import Pipeline

DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v2.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--run-date")
    parser.add_argument("--commit-sha")
    return parser.parse_args()


def build_pipeline() -> Pipeline:
    return build_tfidf_calibrated_linear_svm_pipeline()


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(
        args.output
        or default_output_path(prefix="tfidf-calibrated-linear-svm", dataset_path=dataset_path)
    )
    items = load_annotated_items(dataset_path)
    items_by_split = split_items(items)
    label_order = build_label_order(items)

    train_items = items_by_split["train"]
    calibration_cv = resolve_calibration_cv([item.category for item in train_items])
    pipeline = build_tfidf_calibrated_linear_svm_pipeline(calibration_cv=calibration_cv)
    pipeline.fit(
        [item.text for item in train_items],
        [item.category for item in train_items],
    )

    artifact = build_artifact(
        dataset_path=dataset_path,
        run_id=args.run_id
        or default_run_id(prefix="tfidf-calibrated-linear-svm", dataset_path=dataset_path),
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        method="tfidf_calibrated_linear_svm",
        label_order=label_order,
        train_item_count=len(train_items),
        parameters=build_tfidf_calibrated_linear_svm_parameters(calibration_cv=calibration_cv),
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
            f"{dataset_path.name} and evaluated on the fixed valid and test splits using "
            "a sigmoid-calibrated LinearSVC."
        ),
    )

    save_artifact(output_path, artifact)
    print_summary(artifact, output_path)
    print(f"Compute backend: {sparse_compute_backend_description()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
