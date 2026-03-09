from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnnotatedPrice(StrictModel):
    value: int | float
    currency: str
    raw: str | None = None


class AnnotatedSize(StrictModel):
    value: int | float
    unit: str
    raw: str | None = None


class AnnotatedSlots(StrictModel):
    name: str | None = None
    description: str | None = None
    prices: list[AnnotatedPrice] = Field(default_factory=list)
    sizes: list[AnnotatedSize] = Field(default_factory=list)


class AnnotatedItem(StrictModel):
    id: str
    source_id: str
    restaurant_id: str
    split: str
    language: str
    text: str
    category: str
    slots: AnnotatedSlots


def load_annotated_items(path: Path) -> list[AnnotatedItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AnnotatedItem.model_validate_json(line) for line in lines if line.strip()]


def count_tokens(text: str) -> int:
    return len(text.split())
