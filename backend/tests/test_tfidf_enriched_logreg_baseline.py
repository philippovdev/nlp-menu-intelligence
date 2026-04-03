import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data/annotated/items.v1.jsonl"
SCRIPT_PATH = REPO_ROOT / "scripts/run_tfidf_enriched_logreg_baseline.py"


def test_tfidf_enriched_logreg_baseline_script_writes_expected_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "tfidf-enriched-logreg.json"
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
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Run ID: tfidf-enriched-logreg-v1-001" in completed.stdout
    assert "Valid: accuracy=" in completed.stdout
    assert "Test: accuracy=" in completed.stdout

    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["dataset_file"] == "data/annotated/items.v1.jsonl"
    assert artifact["run_date"] == "2026-04-03"
    assert artifact["commit_sha"] == "test-sha"
    assert artifact["method"] == "tfidf_enriched_logistic_regression"
    assert artifact["train_item_count"] == 36
    assert artifact["valid_metrics"]["item_count"] == 18
    assert artifact["valid_metrics"]["accuracy"] == 0.5
    assert artifact["valid_metrics"]["macro_f1"] == 0.4794
    assert artifact["test_metrics"]["item_count"] == 18
    assert artifact["test_metrics"]["accuracy"] == 0.6667
    assert artifact["test_metrics"]["macro_f1"] == 0.6667
    assert artifact["parameters"]["input_format"] == "record"
    assert artifact["parameters"]["features"]["structured"]["type"] == "dict_vectorizer"
