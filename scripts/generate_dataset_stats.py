from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from dataset_common import count_tokens, load_annotated_items
from pydantic import BaseModel, ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "data/annotated/items.v1.jsonl"
DEFAULT_MANIFEST = REPO_ROOT / "data/raw/dataset-manifest.v1.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data/interim/dataset-stats.v1.json"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetStatsArtifact(StrictModel):
    dataset_version: str
    generated_at: str
    dataset_file: str
    manifest_file: str
    item_count: int
    restaurant_count: int
    document_count: int
    split_counts: dict[str, int]
    category_counts: dict[str, int]
    category_coverage_by_split: dict[str, int]
    source_type_counts: dict[str, int]
    average_tokens_per_item: float
    average_tokens_by_split: dict[str, float]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at")
    return parser.parse_args()


def load_manifest_source_types(path: Path) -> dict[str, str]:
    rows = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
    return {
        row["source_id"]: row["source_type"]
        for row in rows
        if row.get("source_id") and row.get("source_type")
    }


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def build_artifact(
    *,
    dataset_path: Path,
    manifest_path: Path,
    generated_at: str | None,
) -> DatasetStatsArtifact:
    resolved_dataset_path = resolve_repo_path(dataset_path)
    resolved_manifest_path = resolve_repo_path(manifest_path)
    items = load_annotated_items(resolved_dataset_path)
    source_type_by_source = load_manifest_source_types(resolved_manifest_path)

    split_order = ("train", "valid", "test")
    split_counts_raw = Counter(item.split for item in items)
    category_counts = Counter(item.category for item in items)
    token_counts = [count_tokens(item.text) for item in items]
    tokens_by_split: dict[str, list[int]] = defaultdict(list)
    categories_by_split: dict[str, set[str]] = defaultdict(set)
    source_type_counts_raw = Counter(source_type_by_source.values())

    for item, token_count in zip(items, token_counts):
        tokens_by_split[item.split].append(token_count)
        categories_by_split[item.split].add(item.category)

    average_tokens_by_split: dict[str, float] = {}
    for split in split_order:
        if split not in tokens_by_split:
            continue
        values = tokens_by_split[split]
        average_tokens_by_split[split] = round(sum(values) / len(values), 2)
    split_counts = {
        split: split_counts_raw[split]
        for split in split_order
        if split in split_counts_raw
    }
    category_coverage_by_split = {
        split: len(categories_by_split[split])
        for split in split_order
        if split in categories_by_split
    }
    source_type_counts = dict(sorted(source_type_counts_raw.items()))

    return DatasetStatsArtifact(
        dataset_version="v1",
        generated_at=generated_at or datetime.now(UTC).date().isoformat(),
        dataset_file=str(resolved_dataset_path.relative_to(REPO_ROOT)),
        manifest_file=str(resolved_manifest_path.relative_to(REPO_ROOT)),
        item_count=len(items),
        restaurant_count=len({item.restaurant_id for item in items}),
        document_count=len({item.source_id for item in items}),
        split_counts=split_counts,
        category_counts=dict(sorted(category_counts.items())),
        category_coverage_by_split=category_coverage_by_split,
        source_type_counts=source_type_counts,
        average_tokens_per_item=round(sum(token_counts) / len(token_counts), 2),
        average_tokens_by_split=average_tokens_by_split,
        notes=(
            "Dataset v1 is the first in-repo item-level subset with source-level "
            "split assignment. Counts are derived from annotated JSONL plus the "
            "paired source manifest."
        ),
    )


def save_artifact(path: Path, artifact: DatasetStatsArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


def print_summary(artifact: DatasetStatsArtifact) -> None:
    print(f"Dataset version: {artifact.dataset_version}")
    print(f"Dataset file: {artifact.dataset_file}")
    print(f"Manifest file: {artifact.manifest_file}")
    print(f"Items: {artifact.item_count}")
    print(f"Restaurants: {artifact.restaurant_count}")
    print(f"Documents: {artifact.document_count}")
    print(f"Average tokens per item: {artifact.average_tokens_per_item:.2f}")
    print(f"Split counts: {artifact.split_counts}")
    print(f"Source types: {artifact.source_type_counts}")


def main() -> None:
    args = parse_args()
    output_path = resolve_repo_path(args.output)
    artifact = build_artifact(
        dataset_path=args.dataset,
        manifest_path=args.manifest,
        generated_at=args.generated_at,
    )
    save_artifact(output_path, artifact)
    print_summary(artifact)


if __name__ == "__main__":
    main()
