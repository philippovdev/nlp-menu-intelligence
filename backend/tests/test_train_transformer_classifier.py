import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"

DATASET_V2_PATH = REPO_ROOT / "data/annotated/items.v2.jsonl"


def load_script_modules():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    baseline_common = importlib.import_module("classification_baseline_common")
    dataset_common = importlib.import_module("dataset_common")
    transformer_classifier = importlib.import_module("train_transformer_classifier")
    return baseline_common, dataset_common, transformer_classifier


def test_build_label_to_id_is_sorted_and_stable() -> None:
    _, dataset_common, transformer_classifier = load_script_modules()
    items = dataset_common.load_annotated_items(DATASET_V2_PATH)
    label_to_id = transformer_classifier.build_label_to_id(items)

    assert list(label_to_id) == sorted(label_to_id)
    assert label_to_id["breakfast"] == 0
    assert label_to_id["soups"] == len(label_to_id) - 1


def test_build_transformer_artifact_contains_expected_fields(tmp_path: Path) -> None:
    baseline_common, dataset_common, transformer_classifier = load_script_modules()
    items = dataset_common.load_annotated_items(DATASET_V2_PATH)
    items_by_split = baseline_common.split_items(items)
    label_to_id = transformer_classifier.build_label_to_id(items)
    label_order = sorted(label_to_id, key=label_to_id.get)
    metrics = baseline_common.SplitMetrics(
        item_count=72,
        accuracy=0.75,
        macro_f1=0.74,
        per_class_f1={label: 0.7 for label in label_order},
        confusion_matrix=[[1 if row == column else 0 for column in range(12)] for row in range(12)],
    )

    artifact = transformer_classifier.build_artifact(
        dataset_path=DATASET_V2_PATH,
        run_id="transformer-classifier-v2-001",
        run_date="2026-03-15",
        commit_sha="test-sha",
        model_name="distilbert-base-uncased",
        label_order=label_order,
        label_to_id=label_to_id,
        device="cpu",
        parameters={"seed": 42},
        model_dir=tmp_path / "best-model",
        best_checkpoint="checkpoint-1",
        valid_metrics=metrics,
        test_metrics=metrics,
        items_by_split=items_by_split,
    )

    assert artifact.dataset_file == "data/annotated/items.v2.jsonl"
    assert artifact.method == "transformer_sequence_classification"
    assert artifact.model_name == "distilbert-base-uncased"
    assert artifact.train_item_count == 288
    assert artifact.valid_item_count == 72
    assert artifact.test_item_count == 72
    assert artifact.label_to_id["breakfast"] == 0
    assert artifact.best_checkpoint == "checkpoint-1"
