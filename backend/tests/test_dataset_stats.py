import json
import os
import subprocess
import sys
from collections import defaultdict
from csv import DictReader
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_V1_PATH = REPO_ROOT / "data/annotated/items.v1.jsonl"
MANIFEST_V1_PATH = REPO_ROOT / "data/raw/dataset-manifest.v1.csv"
DATASET_V2_PATH = REPO_ROOT / "data/annotated/items.v2.jsonl"
MANIFEST_V2_PATH = REPO_ROOT / "data/raw/dataset-manifest.v2.csv"
STATS_SCRIPT_PATH = REPO_ROOT / "scripts/generate_dataset_stats.py"
EVAL_SCRIPT_PATH = REPO_ROOT / "scripts/eval_heuristic_baseline.py"


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    return list(DictReader(path.read_text(encoding="utf-8").splitlines()))


def test_items_v1_matches_documented_schema() -> None:
    lines = DATASET_V1_PATH.read_text(encoding="utf-8").splitlines()
    manifest_rows = load_manifest_rows(MANIFEST_V1_PATH)
    manifest_source_ids = {row["source_id"] for row in manifest_rows}
    splits_by_source_id: dict[str, set[str]] = defaultdict(set)
    splits_by_restaurant_id: dict[str, set[str]] = defaultdict(set)
    categories_by_split: dict[str, set[str]] = defaultdict(set)

    assert len(lines) == 72
    for line in lines:
        item = json.loads(line)
        assert set(item) == {
            "id",
            "source_id",
            "restaurant_id",
            "split",
            "language",
            "text",
            "category",
            "slots",
        }
        assert item["split"] in {"train", "valid", "test"}
        assert set(item["slots"]) == {"name", "description", "prices", "sizes"}
        assert item["source_id"] in manifest_source_ids
        splits_by_source_id[item["source_id"]].add(item["split"])
        splits_by_restaurant_id[item["restaurant_id"]].add(item["split"])
        categories_by_split[item["split"]].add(item["category"])

    assert all(len(splits) == 1 for splits in splits_by_source_id.values())
    assert all(len(splits) == 1 for splits in splits_by_restaurant_id.values())
    assert {split: len(categories) for split, categories in categories_by_split.items()} == {
        "train": 12,
        "valid": 12,
        "test": 12,
    }

    for row in manifest_rows:
        assert row["source_url"]
        assert row["collected_at"]
        assert row["license_note"]
        assert row["notes"]


def test_items_v2_meets_training_ready_constraints() -> None:
    lines = DATASET_V2_PATH.read_text(encoding="utf-8").splitlines()
    manifest_rows = load_manifest_rows(MANIFEST_V2_PATH)
    manifest_source_ids = {row["source_id"] for row in manifest_rows}
    splits_by_source_id: dict[str, set[str]] = defaultdict(set)
    splits_by_restaurant_id: dict[str, set[str]] = defaultdict(set)
    categories_by_split: dict[str, set[str]] = defaultdict(set)

    assert len(lines) == 432
    for line in lines:
        item = json.loads(line)
        assert set(item) == {
            "id",
            "source_id",
            "restaurant_id",
            "split",
            "language",
            "text",
            "category",
            "slots",
        }
        assert item["split"] in {"train", "valid", "test"}
        assert set(item["slots"]) == {"name", "description", "prices", "sizes"}
        assert item["source_id"] in manifest_source_ids
        splits_by_source_id[item["source_id"]].add(item["split"])
        splits_by_restaurant_id[item["restaurant_id"]].add(item["split"])
        categories_by_split[item["split"]].add(item["category"])

    assert all(len(splits) == 1 for splits in splits_by_source_id.values())
    assert all(len(splits) == 1 for splits in splits_by_restaurant_id.values())
    assert {split: len(categories) for split, categories in categories_by_split.items()} == {
        "train": 12,
        "valid": 12,
        "test": 12,
    }

    for row in manifest_rows:
        assert row["source_url"]
        assert row["collected_at"]
        assert row["license_note"]
        assert row["notes"]


def test_generate_dataset_stats_script_runs_and_writes_expected_artifact(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "dataset-stats.v1.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(STATS_SCRIPT_PATH),
            "--dataset",
            str(DATASET_V1_PATH),
            "--manifest",
            str(MANIFEST_V1_PATH),
            "--output",
            str(output_path),
            "--generated-at",
            "2026-03-09",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Items: 72" in completed.stdout
    assert "Split counts: {'train': 36, 'valid': 18, 'test': 18}" in completed.stdout

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact == {
        "dataset_version": "v1",
        "generated_at": "2026-03-09",
        "dataset_file": "data/annotated/items.v1.jsonl",
        "manifest_file": "data/raw/dataset-manifest.v1.csv",
        "item_count": 72,
        "restaurant_count": 12,
        "document_count": 12,
        "split_counts": {"train": 36, "valid": 18, "test": 18},
        "source_counts_by_split": {"train": 6, "valid": 3, "test": 3},
        "restaurant_counts_by_split": {"train": 6, "valid": 3, "test": 3},
        "category_counts": {
            "breakfast": 6,
            "burgers": 6,
            "desserts": 6,
            "drinks_cold": 6,
            "drinks_hot": 6,
            "mains": 6,
            "other": 6,
            "pasta": 6,
            "pizza": 6,
            "salads": 6,
            "sides": 6,
            "soups": 6,
        },
        "category_coverage_by_split": {"train": 12, "valid": 12, "test": 12},
        "category_counts_by_split": {
            "train": {
                "breakfast": 3,
                "burgers": 3,
                "desserts": 3,
                "drinks_cold": 3,
                "drinks_hot": 3,
                "mains": 3,
                "other": 3,
                "pasta": 3,
                "pizza": 3,
                "salads": 4,
                "sides": 2,
                "soups": 3,
            },
            "valid": {
                "breakfast": 2,
                "burgers": 2,
                "desserts": 1,
                "drinks_cold": 1,
                "drinks_hot": 2,
                "mains": 1,
                "other": 1,
                "pasta": 2,
                "pizza": 1,
                "salads": 1,
                "sides": 2,
                "soups": 2,
            },
            "test": {
                "breakfast": 1,
                "burgers": 1,
                "desserts": 2,
                "drinks_cold": 2,
                "drinks_hot": 1,
                "mains": 2,
                "other": 2,
                "pasta": 1,
                "pizza": 2,
                "salads": 1,
                "sides": 2,
                "soups": 1,
            },
        },
        "source_type_counts": {"html": 6, "image": 2, "pdf": 4},
        "average_tokens_per_item": 7.58,
        "average_tokens_by_split": {"train": 7.5, "valid": 7.61, "test": 7.72},
        "class_balance_summary": {
            "min_count": 6,
            "max_count": 6,
            "mean_count": 6.0,
            "max_to_min_ratio": 1.0,
        },
        "notes": (
            "Dataset v1 is the first in-repo item-level subset with source-level "
            "split assignment. Counts are derived from annotated JSONL plus the "
            "paired source manifest."
        ),
    }


def test_generate_dataset_stats_script_writes_expected_v2_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset-stats.v2.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(STATS_SCRIPT_PATH),
            "--dataset",
            str(DATASET_V2_PATH),
            "--manifest",
            str(MANIFEST_V2_PATH),
            "--output",
            str(output_path),
            "--generated-at",
            "2026-03-09",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Items: 432" in completed.stdout
    assert "Split counts: {'train': 288, 'valid': 72, 'test': 72}" in completed.stdout

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact == {
        "dataset_version": "v2",
        "generated_at": "2026-03-09",
        "dataset_file": "data/annotated/items.v2.jsonl",
        "manifest_file": "data/raw/dataset-manifest.v2.csv",
        "item_count": 432,
        "restaurant_count": 12,
        "document_count": 12,
        "split_counts": {"train": 288, "valid": 72, "test": 72},
        "source_counts_by_split": {"train": 8, "valid": 2, "test": 2},
        "restaurant_counts_by_split": {"train": 8, "valid": 2, "test": 2},
        "category_counts": {
            "breakfast": 36,
            "burgers": 36,
            "desserts": 36,
            "drinks_cold": 36,
            "drinks_hot": 36,
            "mains": 36,
            "other": 36,
            "pasta": 36,
            "pizza": 36,
            "salads": 36,
            "sides": 36,
            "soups": 36,
        },
        "category_coverage_by_split": {"train": 12, "valid": 12, "test": 12},
        "category_counts_by_split": {
            "train": {
                "breakfast": 24,
                "burgers": 24,
                "desserts": 24,
                "drinks_cold": 24,
                "drinks_hot": 24,
                "mains": 24,
                "other": 24,
                "pasta": 24,
                "pizza": 24,
                "salads": 24,
                "sides": 24,
                "soups": 24,
            },
            "valid": {
                "breakfast": 6,
                "burgers": 6,
                "desserts": 6,
                "drinks_cold": 6,
                "drinks_hot": 6,
                "mains": 6,
                "other": 6,
                "pasta": 6,
                "pizza": 6,
                "salads": 6,
                "sides": 6,
                "soups": 6,
            },
            "test": {
                "breakfast": 6,
                "burgers": 6,
                "desserts": 6,
                "drinks_cold": 6,
                "drinks_hot": 6,
                "mains": 6,
                "other": 6,
                "pasta": 6,
                "pizza": 6,
                "salads": 6,
                "sides": 6,
                "soups": 6,
            },
        },
        "source_type_counts": {"html": 6, "pdf": 6},
        "average_tokens_per_item": 8.7,
        "average_tokens_by_split": {"train": 8.7, "valid": 8.72, "test": 8.68},
        "class_balance_summary": {
            "min_count": 36,
            "max_count": 36,
            "mean_count": 36.0,
            "max_to_min_ratio": 1.0,
        },
        "notes": (
            "Dataset v2 is the training-ready source-grounded synthetic expansion "
            "with source-level split assignment, normalized English text, "
            "normalized RUB prices, manifest-backed provenance, and full 12-label "
            "coverage in train, valid, and test."
        ),
    }


def test_eval_heuristic_baseline_accepts_items_v1(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline-v1.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(EVAL_SCRIPT_PATH),
            "--dataset",
            str(DATASET_V1_PATH),
            "--output",
            str(output_path),
            "--run-date",
            "2026-03-09",
            "--commit-sha",
            "test-sha",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "backend")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["dataset_file"] == "data/annotated/items.v1.jsonl"
    assert artifact["run_date"] == "2026-03-09"
    assert artifact["commit_sha"] == "test-sha"
    assert artifact["metrics"]["item_count"] == 72
