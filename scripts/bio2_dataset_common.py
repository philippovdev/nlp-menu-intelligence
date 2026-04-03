from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Bio2Record(StrictModel):
    id: str
    source_id: str
    restaurant_id: str
    split: str
    language: str
    category: str
    text: str
    tokens: list[str]
    tags: list[str]

    @model_validator(mode="after")
    def validate_lengths(self) -> "Bio2Record":
        if len(self.tokens) != len(self.tags):
            raise ValueError("BIO2 record must contain the same number of tokens and tags.")
        return self


def load_bio2_records(path: Path) -> list[Bio2Record]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Bio2Record.model_validate_json(line) for line in lines if line.strip()]


def split_bio2_records(records: list[Bio2Record]) -> dict[str, list[Bio2Record]]:
    grouped: dict[str, list[Bio2Record]] = {"train": [], "valid": [], "test": []}
    for record in records:
        grouped.setdefault(record.split, []).append(record)
    return grouped


def write_bio2_records(path: Path, records: list[Bio2Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record.model_dump(), ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
