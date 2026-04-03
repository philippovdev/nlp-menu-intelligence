from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

TEXT_INPUT_FORMAT = "text"
RECORD_INPUT_FORMAT = "record"

_SIDES_LEXICON = (
    "bread",
    "basket",
    "spinach",
    "fries",
    "rice",
    "potato",
    "broccoli",
    "asparagus",
    "cob",
    "mushroom",
    "edamame",
    "focaccia",
    "naan",
    "pita",
    "slaw",
)
_SALAD_LEXICON = (
    "salad",
    "greens",
    "arugula",
    "lettuce",
    "vinaigrette",
    "caesar",
)
_PASTA_LEXICON = (
    "pasta",
    "spaghetti",
    "linguine",
    "ziti",
    "rigatoni",
    "gnocchi",
    "orzo",
    "ravioli",
    "cacio",
)
_BREAKFAST_LEXICON = (
    "granola",
    "waffle",
    "omelet",
    "breakfast",
    "toast",
    "quesadilla",
)
_HOT_DRINK_TERMS = ("latte", "tea", "espresso", "americano", "cappuccino", "mocha")
_COLD_DRINK_TERMS = ("kombucha", "soda", "spritz", "lemonade", "juice")


def build_category_model_record(
    *,
    text: str,
    name: str | None = None,
    prices: Sequence[object] | None = None,
    sizes: Sequence[object] | None = None,
) -> dict[str, Any]:
    return {
        "text": text,
        "name": name or text,
        "prices": [normalize_slot_entry(entry) for entry in prices or ()],
        "sizes": [normalize_slot_entry(entry) for entry in sizes or ()],
    }


def select_record_text(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(record.get("text", "")) for record in records]


def select_record_name(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(record.get("name", "") or record.get("text", "")) for record in records]


def build_structured_slot_features(records: Iterable[Mapping[str, Any]]) -> list[dict[str, int]]:
    features: list[dict[str, int]] = []

    for record in records:
        text = str(record.get("text", "")).lower()
        name = str(record.get("name", "") or record.get("text", "")).lower()
        prices = _normalize_slot_entries(record.get("prices"))
        sizes = _normalize_slot_entries(record.get("sizes"))
        first_size = sizes[0] if sizes else None
        unit = str(first_size.get("unit", "none")) if first_size else "none"
        value = _coerce_float(first_size.get("value")) if first_size else None

        features.append(
            {
                f"size_unit={unit}": 1,
                f"size_bucket={bucket_size(value=value, unit=unit)}": 1,
                "has_size": int(bool(sizes)),
                "has_price": int(bool(prices)),
                f"token_bucket={bucket_token_count(name)}": 1,
                f"char_bucket={bucket_char_count(name)}": 1,
                "contains_iced": int("iced" in text),
                "contains_hot_drink_term": int(contains_any_fragment(text, _HOT_DRINK_TERMS)),
                "contains_cold_drink_term": int(contains_any_fragment(text, _COLD_DRINK_TERMS)),
                "sides_lexicon": int(contains_any_fragment(name, _SIDES_LEXICON)),
                "salad_lexicon": int(contains_any_fragment(name, _SALAD_LEXICON)),
                "pasta_lexicon": int(contains_any_fragment(name, _PASTA_LEXICON)),
                "breakfast_lexicon": int(contains_any_fragment(name, _BREAKFAST_LEXICON)),
            }
        )

    return features


def bucket_size(*, value: float | None, unit: str) -> str:
    if value is None:
        return "none"
    if unit in {"ml", "g"}:
        if value < 200:
            return "lt200"
        if value <= 400:
            return "200_400"
        return "gt400"
    if unit == "cm":
        if value < 30:
            return "lt30"
        if value <= 32:
            return "30_32"
        return "gt32"
    return "present"


def bucket_token_count(text: str) -> str:
    token_count = len(text.split())
    if token_count <= 2:
        return "short"
    if token_count <= 4:
        return "medium"
    return "long"


def bucket_char_count(text: str) -> str:
    char_count = len(text)
    if char_count <= 15:
        return "short"
    if char_count <= 30:
        return "medium"
    return "long"


def contains_any_fragment(text: str, fragments: Sequence[str]) -> bool:
    return any(fragment in text for fragment in fragments)


def normalize_slot_entry(entry: object) -> dict[str, Any]:
    if isinstance(entry, Mapping):
        return dict(entry)
    if hasattr(entry, "model_dump"):
        dumped = entry.model_dump()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    if hasattr(entry, "__dict__"):
        payload = vars(entry)
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError(f"Unsupported slot entry type: {type(entry)!r}")


def _normalize_slot_entries(entries: object) -> list[dict[str, Any]]:
    if entries is None:
        return []
    if not isinstance(entries, Sequence):
        return []
    return [normalize_slot_entry(entry) for entry in entries]


def _coerce_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None
