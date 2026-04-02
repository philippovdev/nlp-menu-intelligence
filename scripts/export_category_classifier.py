from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from category_model_builders import MODEL_FAMILIES
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
    evaluate_split,
    format_output_path,
    resolve_repo_path,
    split_items,
)
from dataset_common import load_annotated_items

DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v2.jsonl"
DEFAULT_MODEL_OUTPUT = REPO_ROOT / "backend/app/model_assets/category_classifier.pkl"
DEFAULT_METADATA_OUTPUT = REPO_ROOT / "backend/app/model_assets/category_classifier.json"
DEFAULT_MINIMUM_CONFIDENCE = 0.35
DEFAULT_MODEL_FAMILY = "tfidf_union_logreg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--model-family", choices=sorted(MODEL_FAMILIES), default=DEFAULT_MODEL_FAMILY)
    parser.add_argument("--model-id")
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
    model_family = MODEL_FAMILIES[args.model_family]
    model_id = args.model_id or model_family.default_model_id
    items = load_annotated_items(dataset_path)
    items_by_split = split_items(items)
    label_order = build_label_order(items)

    train_items = items_by_split["train"]
    if args.model_family == "tfidf_calibrated_linear_svm":
        calibration_cv = resolve_calibration_cv([item.category for item in train_items])
        pipeline = build_tfidf_calibrated_linear_svm_pipeline(calibration_cv=calibration_cv)
        parameters = build_tfidf_calibrated_linear_svm_parameters(calibration_cv=calibration_cv)
    else:
        pipeline = model_family.build_pipeline()
        parameters = model_family.build_parameters()
    pipeline.fit(
        [item.text for item in train_items],
        [item.category for item in train_items],
    )

    artifact = build_artifact(
        dataset_path=dataset_path,
        run_id=f"export-category-classifier-{dataset_path.stem.split('.')[-1]}-001",
        run_date=args.run_date,
        commit_sha=args.commit_sha,
        method=model_family.method,
        label_order=label_order,
        train_item_count=len(train_items),
        parameters=parameters,
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
            f"{model_family.classifier_label}."
        ),
    )

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with model_output.open("wb") as file_pointer:
        pickle.dump(pipeline, file_pointer, protocol=5)

    metadata = artifact.model_dump(mode="json")
    metadata.update(
        {
            "model_id": model_id,
            "minimum_confidence": args.minimum_confidence,
            "model_file": relative_to_repo(model_output),
        }
    )
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Dataset: {artifact.dataset_file}")
    print(f"Model ID: {model_id}")
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
    print(f"Compute backend: {sparse_compute_backend_description()}")
    print(f"Model file: {format_output_path(model_output)}")
    print(f"Metadata file: {format_output_path(metadata_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
