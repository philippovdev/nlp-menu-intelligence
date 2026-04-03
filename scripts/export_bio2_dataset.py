from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.bio2_extraction import ACTIVE_ENTITY_LABELS, build_gold_bio2_tags
from classification_baseline_common import REPO_ROOT, format_output_path, resolve_repo_path
from dataset_common import AnnotatedItem, load_annotated_items

DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v2.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/interim/items.v2.bio2.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def infer_currency_hint(item: AnnotatedItem) -> str:
    currencies = {price.currency for price in item.slots.prices if price.currency}
    if len(currencies) == 1:
        return next(iter(currencies))
    return "RUB"


def export_record(item: AnnotatedItem) -> dict[str, object]:
    tags, tokens = build_gold_bio2_tags(
        item.text,
        name=item.slots.name,
        prices=item.slots.prices,
        sizes=item.slots.sizes,
        default_currency=infer_currency_hint(item),
    )
    return {
        "id": item.id,
        "source_id": item.source_id,
        "restaurant_id": item.restaurant_id,
        "split": item.split,
        "language": item.language,
        "category": item.category,
        "text": item.text,
        "tokens": [token.text for token in tokens],
        "tags": tags,
    }


def main() -> int:
    args = parse_args()
    dataset_path = resolve_repo_path(args.dataset)
    output_path = resolve_repo_path(args.output)
    items = load_annotated_items(dataset_path)
    records = [export_record(item) for item in items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    split_counts: dict[str, int] = {}
    for record in records:
        split = str(record["split"])
        split_counts[split] = split_counts.get(split, 0) + 1

    print(f"Dataset: {dataset_path.relative_to(REPO_ROOT)}")
    print(f"Records: {len(records)}")
    print(f"Entity labels: {', '.join(ACTIVE_ENTITY_LABELS)}")
    print(
        "Splits: "
        + ", ".join(f"{split}={count}" for split, count in sorted(split_counts.items()))
    )
    print(f"Output: {format_output_path(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
