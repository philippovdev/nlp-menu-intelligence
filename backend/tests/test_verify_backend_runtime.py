from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from app.category_classifier import CONFIGURED_CATEGORY_MODEL_ID

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/verify_backend_runtime.py"


def test_verify_backend_runtime_script_runs() -> None:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    backend_path = str(REPO_ROOT / "backend")
    environment["PYTHONPATH"] = (
        backend_path
        if not existing_pythonpath
        else f"{backend_path}{os.pathsep}{existing_pythonpath}"
    )

    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["assets"]["classifier_exists"] is True
    assert payload["assets"]["metadata_exists"] is True
    assert payload["version"]["category_model_ready"] is True
    assert payload["version"]["category_model"] == CONFIGURED_CATEGORY_MODEL_ID
    assert payload["parse_file_category_model"] == payload["version"]["category_model"]
