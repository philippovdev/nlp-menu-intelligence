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
SPLIT_ORDER = ("train", "valid", "test")
REQUIRED_MANIFEST_FIELDS = ("source_url", "license_note", "notes", "collected_at")


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
    source_counts_by_split: dict[str, int]
    restaurant_counts_by_split: dict[str, int]
    category_counts: dict[str, int]
    category_coverage_by_split: dict[str, int]
    category_counts_by_split: dict[str, dict[str, int]]
    source_type_counts: dict[str, int]
    average_tokens_per_item: float
    average_tokens_by_split: dict[str, float]
    class_balance_summary: dict[str, int | float]
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at")
    return parser.parse_args()


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))


def build_manifest_index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["source_id"]: row for row in rows if row.get("source_id")}


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def infer_dataset_version(path: Path) -> str:
    stem = path.stem
    if "." in stem:
        return stem.split(".")[-1]
    return stem


def build_notes(dataset_version: str) -> str:
    if dataset_version == "v1":
        return (
            "Dataset v1 is the first in-repo item-level subset with source-level "
            "split assignment. Counts are derived from annotated JSONL plus the "
            "paired source manifest."
        )
    if dataset_version == "v2":
        return (
            "Dataset v2 is the training-ready source-grounded synthetic expansion "
            "with source-level split assignment, normalized English text, "
            "normalized RUB prices, manifest-backed provenance, and full 12-label "
            "coverage in train, valid, and test."
        )
    return (
        f"Dataset {dataset_version} statistics are derived from annotated JSONL and "
        "the paired source manifest."
    )


def validate_inputs(
    *,
    items,
    manifest_rows: list[dict[str, str]],
) -> None:
    manifest_index = build_manifest_index(manifest_rows)
    manifest_source_ids = set(manifest_index)
    dataset_source_ids = {item.source_id for item in items}
    missing_source_ids = sorted(dataset_source_ids - manifest_source_ids)
    if missing_source_ids:
        missing = ", ".join(missing_source_ids)
        raise ValueError(f"Dataset references source IDs missing from manifest: {missing}")

    for source_id in sorted(dataset_source_ids):
        row = manifest_index[source_id]
        missing_fields = [field for field in REQUIRED_MANIFEST_FIELDS if not row.get(field)]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Manifest row for {source_id} is missing required fields: {missing}")

    source_splits: dict[str, set[str]] = defaultdict(set)
    restaurant_splits: dict[str, set[str]] = defaultdict(set)
    categories_by_split: dict[str, set[str]] = defaultdict(set)
    all_categories = {item.category for item in items}

    for item in items:
        source_splits[item.source_id].add(item.split)
        restaurant_splits[item.restaurant_id].add(item.split)
        categories_by_split[item.split].add(item.category)

    leaking_sources = sorted(
        source_id for source_id, splits in source_splits.items() if len(splits) > 1
    )
    if leaking_sources:
        leaking = ", ".join(leaking_sources)
        raise ValueError(f"Source split leakage detected for source IDs: {leaking}")

    leaking_restaurants = sorted(
        restaurant_id
        for restaurant_id, splits in restaurant_splits.items()
        if len(splits) > 1
    )
    if leaking_restaurants:
        leaking = ", ".join(leaking_restaurants)
        raise ValueError(f"Restaurant split leakage detected for restaurant IDs: {leaking}")

    missing_splits = [split for split in SPLIT_ORDER if split not in categories_by_split]
    if missing_splits:
        missing = ", ".join(missing_splits)
        raise ValueError(f"Dataset is missing required splits: {missing}")

    for split in SPLIT_ORDER:
        missing_categories = sorted(all_categories - categories_by_split[split])
        if missing_categories:
            missing = ", ".join(missing_categories)
            raise ValueError(f"Split {split} is missing categories: {missing}")


def build_artifact(
    *,
    dataset_path: Path,
    manifest_path: Path,
    generated_at: str | None,
) -> DatasetStatsArtifact:
    resolved_dataset_path = resolve_repo_path(dataset_path)
    resolved_manifest_path = resolve_repo_path(manifest_path)
    items = load_annotated_items(resolved_dataset_path)
    manifest_rows = load_manifest_rows(resolved_manifest_path)
    validate_inputs(items=items, manifest_rows=manifest_rows)
    manifest_index = build_manifest_index(manifest_rows)
    dataset_version = infer_dataset_version(resolved_dataset_path)

    split_counts_raw = Counter(item.split for item in items)
    category_counts = Counter(item.category for item in items)
    token_counts = [count_tokens(item.text) for item in items]
    tokens_by_split: dict[str, list[int]] = defaultdict(list)
    categories_by_split: dict[str, set[str]] = defaultdict(set)
    category_counts_by_split_raw: dict[str, Counter[str]] = defaultdict(Counter)
    source_ids_by_split: dict[str, set[str]] = defaultdict(set)
    restaurant_ids_by_split: dict[str, set[str]] = defaultdict(set)
    source_type_counts_raw = Counter(
        manifest_index[source_id]["source_type"] for source_id in {item.source_id for item in items}
    )

    for item, token_count in zip(items, token_counts):
        tokens_by_split[item.split].append(token_count)
        categories_by_split[item.split].add(item.category)
        category_counts_by_split_raw[item.split][item.category] += 1
        source_ids_by_split[item.split].add(item.source_id)
        restaurant_ids_by_split[item.split].add(item.restaurant_id)

    average_tokens_by_split: dict[str, float] = {}
    for split in SPLIT_ORDER:
        if split not in tokens_by_split:
            continue
        values = tokens_by_split[split]
        average_tokens_by_split[split] = round(sum(values) / len(values), 2)
    split_counts = {
        split: split_counts_raw[split]
        for split in SPLIT_ORDER
        if split in split_counts_raw
    }
    source_counts_by_split = {
        split: len(source_ids_by_split[split])
        for split in SPLIT_ORDER
        if split in source_ids_by_split
    }
    restaurant_counts_by_split = {
        split: len(restaurant_ids_by_split[split])
        for split in SPLIT_ORDER
        if split in restaurant_ids_by_split
    }
    category_coverage_by_split = {
        split: len(categories_by_split[split])
        for split in SPLIT_ORDER
        if split in categories_by_split
    }
    category_counts_by_split = {
        split: dict(sorted(category_counts_by_split_raw[split].items()))
        for split in SPLIT_ORDER
        if split in category_counts_by_split_raw
    }
    source_type_counts = dict(sorted(source_type_counts_raw.items()))
    class_balance_summary = {
        "min_count": min(category_counts.values()),
        "max_count": max(category_counts.values()),
        "mean_count": round(sum(category_counts.values()) / len(category_counts), 2),
        "max_to_min_ratio": round(max(category_counts.values()) / min(category_counts.values()), 2),
    }

    return DatasetStatsArtifact(
        dataset_version=dataset_version,
        generated_at=generated_at or datetime.now(UTC).date().isoformat(),
        dataset_file=str(resolved_dataset_path.relative_to(REPO_ROOT)),
        manifest_file=str(resolved_manifest_path.relative_to(REPO_ROOT)),
        item_count=len(items),
        restaurant_count=len({item.restaurant_id for item in items}),
        document_count=len({item.source_id for item in items}),
        split_counts=split_counts,
        source_counts_by_split=source_counts_by_split,
        restaurant_counts_by_split=restaurant_counts_by_split,
        category_counts=dict(sorted(category_counts.items())),
        category_coverage_by_split=category_coverage_by_split,
        category_counts_by_split=category_counts_by_split,
        source_type_counts=source_type_counts,
        average_tokens_per_item=round(sum(token_counts) / len(token_counts), 2),
        average_tokens_by_split=average_tokens_by_split,
        class_balance_summary=class_balance_summary,
        notes=build_notes(dataset_version),
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
