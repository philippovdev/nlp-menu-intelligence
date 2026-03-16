import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "data/eval/realworld-manifest.v1.csv"
GOLD_PATH = REPO_ROOT / "data/eval/realworld-gold.v1.jsonl"
SCRIPT_PATH = REPO_ROOT / "scripts/eval_realworld_pipeline.py"


def load_eval_module():
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("eval_realworld_pipeline", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_realworld_eval_inputs_are_valid_and_balanced() -> None:
    eval_module = load_eval_module()
    manifest_rows = eval_module.load_eval_manifest(MANIFEST_PATH)
    gold_cases = eval_module.load_eval_gold(GOLD_PATH)

    eval_module.validate_eval_inputs(manifest_rows, gold_cases)

    assert len(manifest_rows) == 12
    assert len(gold_cases) == 12
    assert Counter(row.input_type for row in manifest_rows) == {
        "text": 4,
        "pdf": 4,
        "image": 4,
    }
    assert all(len(case.gold_items) == 2 for case in gold_cases)


def test_realworld_eval_script_runs_and_writes_artifact(tmp_path: Path) -> None:
    output_path = tmp_path / "realworld-eval.json"
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    pythonpath_parts = [str(REPO_ROOT / "backend"), str(REPO_ROOT / "scripts")]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    completed = subprocess.run(
        [
            str(REPO_ROOT / "backend/.venv/bin/python"),
            str(SCRIPT_PATH),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "realworld-eval-v1-001"
    assert payload["example_count"] == 12
    assert payload["metrics"]["overall"]["category"]["accuracy"] == 1.0
