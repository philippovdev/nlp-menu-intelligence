import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data/annotated/items.sample.jsonl"
SCRIPT_PATH = REPO_ROOT / "scripts/eval_heuristic_baseline.py"


def test_sample_annotation_file_matches_documented_schema() -> None:
    lines = DATASET_PATH.read_text(encoding="utf-8").splitlines()

    assert lines
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
        assert set(item["slots"]) == {"name", "description", "prices", "sizes"}


def test_eval_heuristic_baseline_script_runs_and_writes_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dataset",
            str(DATASET_PATH),
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
    assert "Run ID: baseline-heuristic-001" in completed.stdout
    assert "Category accuracy: 0.9000" in completed.stdout
    assert "Category macro-F1: 0.8182" in completed.stdout

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["dataset_file"] == "data/annotated/items.sample.jsonl"
    assert artifact["run_date"] == "2026-03-09"
    assert artifact["commit_sha"] == "test-sha"
    assert artifact["method"] == "heuristic_pipeline"
    assert artifact["metrics"] == {
        "item_count": 10,
        "category_accuracy": 0.9,
        "category_macro_f1": 0.8182,
        "category_per_class_f1": {
            "breakfast": 0.0,
            "desserts": 1.0,
            "drinks_cold": 1.0,
            "drinks_hot": 1.0,
            "mains": 1.0,
            "other": 0.0,
            "pasta": 1.0,
            "pizza": 1.0,
            "salads": 1.0,
            "sides": 1.0,
            "soups": 1.0,
        },
        "price_exact_match": {"correct": 10, "total": 10, "accuracy": 1.0},
        "size_exact_match": {"correct": 10, "total": 10, "accuracy": 1.0},
    }
