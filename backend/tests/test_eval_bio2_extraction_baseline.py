import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data/annotated/items.v1.jsonl"
SCRIPT_PATH = REPO_ROOT / "scripts/eval_bio2_extraction_baseline.py"
PYTHONPATH = os.pathsep.join([str(REPO_ROOT / "backend"), str(REPO_ROOT / "scripts")])


def test_eval_bio2_extraction_baseline_script_writes_expected_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "bio2-extraction-baseline.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dataset",
            str(DATASET_PATH),
            "--output",
            str(output_path),
            "--run-date",
            "2026-04-03",
            "--commit-sha",
            "test-sha",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": PYTHONPATH},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run ID: bio2-extraction-baseline-v2-001" in completed.stdout
    assert "Valid: token micro-F1=1.0000, entity micro-F1=1.0000" in completed.stdout
    assert "Test: token micro-F1=1.0000, entity micro-F1=1.0000" in completed.stdout

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["dataset_file"] == "data/annotated/items.v1.jsonl"
    assert artifact["run_date"] == "2026-04-03"
    assert artifact["commit_sha"] == "test-sha"
    assert artifact["method"] == "deterministic_bio2_extraction_baseline"
    assert artifact["entity_labels"] == ["NAME", "PRICE", "SIZE"]
    assert artifact["valid_metrics"]["item_count"] == 18
    assert artifact["valid_metrics"]["token_micro_f1"] == 1.0
    assert artifact["valid_metrics"]["entity_micro_f1"] == 1.0
    assert artifact["test_metrics"]["item_count"] == 18
    assert artifact["test_metrics"]["token_micro_f1"] == 1.0
    assert artifact["test_metrics"]["entity_micro_f1"] == 1.0
