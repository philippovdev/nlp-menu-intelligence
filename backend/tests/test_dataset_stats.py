import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_V1_PATH = REPO_ROOT / "data/annotated/items.v1.jsonl"
MANIFEST_V1_PATH = REPO_ROOT / "data/raw/dataset-manifest.v1.csv"
STATS_SCRIPT_PATH = REPO_ROOT / "scripts/generate_dataset_stats.py"
EVAL_SCRIPT_PATH = REPO_ROOT / "scripts/eval_heuristic_baseline.py"


def test_items_v1_matches_documented_schema() -> None:
    lines = DATASET_V1_PATH.read_text(encoding="utf-8").splitlines()
    manifest_rows = MANIFEST_V1_PATH.read_text(encoding="utf-8").splitlines()
    manifest_source_ids = {row.split(",", 1)[0] for row in manifest_rows[1:]}
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

    for row in manifest_rows[1:]:
        columns = row.split(",")
        assert columns[4]


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
        "source_type_counts": {"html": 6, "image": 2, "pdf": 4},
        "average_tokens_per_item": 7.58,
        "average_tokens_by_split": {"train": 7.5, "valid": 7.61, "test": 7.72},
        "notes": (
            "Dataset v1 is the first in-repo item-level subset with source-level "
            "split assignment. Counts are derived from annotated JSONL plus the "
            "paired source manifest."
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
