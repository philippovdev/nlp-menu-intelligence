import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "data/annotated/items.sample.jsonl"
SCRIPT_PATH = REPO_ROOT / "scripts/export_bio2_dataset.py"
PYTHONPATH = os.pathsep.join([str(REPO_ROOT / "backend"), str(REPO_ROOT / "scripts")])


def test_export_bio2_dataset_script_writes_expected_rows(tmp_path: Path) -> None:
    output_path = tmp_path / "items.sample.bio2.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--dataset",
            str(DATASET_PATH),
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": PYTHONPATH},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Records: 10" in completed.stdout
    assert "Entity labels: NAME, PRICE, SIZE" in completed.stdout

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 10

    first = rows[0]
    assert first["id"] == "item_000001"
    assert first["tokens"] == ["Caesar", "with", "chicken", "250", "g", "-", "390", "RUB"]
    assert first["tags"] == [
        "B-NAME",
        "I-NAME",
        "I-NAME",
        "B-SIZE",
        "I-SIZE",
        "O",
        "B-PRICE",
        "I-PRICE",
    ]

    multi_value = rows[3]
    assert multi_value["id"] == "item_000004"
    assert multi_value["tokens"] == ["Pepperoni", "Pizza", "30/40", "cm", "690", "/", "890"]
    assert multi_value["tags"] == [
        "B-NAME",
        "I-NAME",
        "B-SIZE",
        "I-SIZE",
        "B-PRICE",
        "I-PRICE",
        "I-PRICE",
    ]
