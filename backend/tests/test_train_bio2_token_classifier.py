import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET_PATH = REPO_ROOT / "data/annotated/items.v1.jsonl"
EXPORT_SCRIPT_PATH = REPO_ROOT / "scripts/export_bio2_dataset.py"
TRAIN_SCRIPT_PATH = REPO_ROOT / "scripts/train_bio2_token_classifier.py"
PYTHONPATH = os.pathsep.join([str(REPO_ROOT / "backend"), str(REPO_ROOT / "scripts")])


def test_train_bio2_token_classifier_script_writes_artifact_and_model(tmp_path: Path) -> None:
    bio2_dataset_path = tmp_path / "items.v1.bio2.jsonl"
    artifact_path = tmp_path / "bio2-token-logreg.json"
    model_path = tmp_path / "bio2-token-logreg.pkl"

    export_completed = subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT_PATH),
            "--dataset",
            str(SOURCE_DATASET_PATH),
            "--output",
            str(bio2_dataset_path),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": PYTHONPATH},
        capture_output=True,
        text=True,
        check=False,
    )
    assert export_completed.returncode == 0, export_completed.stderr

    train_completed = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT_PATH),
            "--dataset",
            str(bio2_dataset_path),
            "--artifact",
            str(artifact_path),
            "--model-output",
            str(model_path),
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

    assert train_completed.returncode == 0, train_completed.stderr
    assert "Run ID: bio2-token-logreg-v2-001" in train_completed.stdout
    assert "Artifact:" in train_completed.stdout
    assert "Model:" in train_completed.stdout
    assert model_path.exists()

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["dataset_file"].endswith("items.v1.bio2.jsonl")
    assert artifact["run_date"] == "2026-04-03"
    assert artifact["commit_sha"] == "test-sha"
    assert artifact["method"] == "contextual_logistic_regression_token_classifier"
    assert artifact["train_item_count"] == 36
    assert artifact["train_token_count"] > 0
    assert artifact["valid_metrics"]["item_count"] == 18
    assert artifact["test_metrics"]["item_count"] == 18
    assert artifact["valid_metrics"]["token_micro_f1"] >= 0.8
    assert artifact["test_metrics"]["token_micro_f1"] >= 0.8
