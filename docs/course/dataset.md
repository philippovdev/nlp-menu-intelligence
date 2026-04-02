# Dataset

## Overview

The project uses two related data layers:

- a document-level evaluation slice for end-to-end parsing checks on text, PDF,
  and image inputs
- an item-level dataset for category classification and slot extraction

The item-level corpus is the main research dataset.

## Sources And Policy

Sources are public restaurant menu pages and public PDF menus. The repository
stores manifests, annotations, small fixtures, and derived statistics. Raw
captures stay outside git.

Tracked source metadata includes:

- source URL
- collection date
- source type: `html`, `pdf`, or `image`
- restaurant identifier
- language

The manifest format starts from
[dataset-manifest-template.csv](../../data/raw/dataset-manifest-template.csv).

## Annotation Format

The source of truth is JSONL with one item per line.

```json
{
  "id": "item_000123",
  "source_id": "menu_000017",
  "restaurant_id": "rest_000042",
  "split": "train",
  "language": "ru",
  "text": "Caesar with chicken 250 g - 390 RUB",
  "category": "salads",
  "slots": {
    "name": "Caesar with chicken",
    "description": null,
    "prices": [
      {
        "value": 390,
        "currency": "RUB",
        "raw": "390 RUB"
      }
    ],
    "sizes": [
      {
        "value": 250,
        "unit": "g",
        "raw": "250 g"
      }
    ]
  }
}
```

Derived exports are used for classification tables and optional BIO2-style
extraction experiments. Annotation rules are documented in
[annotation-guide.md](annotation-guide.md).

## Label Set

The active category set contains 12 labels:

- `salads`
- `soups`
- `mains`
- `pizza`
- `pasta`
- `burgers`
- `sides`
- `desserts`
- `breakfast`
- `drinks_hot`
- `drinks_cold`
- `other`

## Split Policy

Splits are assigned by source and restaurant, not by random row. This reduces
leakage between near-duplicate menu items and produces a more credible held-out
evaluation.

## Releases

### Sample

Small starter files remain in the repository for tests and schema examples:

- [dataset-manifest.sample.csv](../../data/raw/dataset-manifest.sample.csv)
- [items.sample.jsonl](../../data/annotated/items.sample.jsonl)

### v1

The first in-repo experimental subset is:

- [items.v1.jsonl](../../data/annotated/items.v1.jsonl)
- [dataset-manifest.v1.csv](../../data/raw/dataset-manifest.v1.csv)
- [dataset-stats.v1.json](../../data/interim/dataset-stats.v1.json)

Summary:

- 72 annotated menu items
- 12 source documents from 12 restaurants
- split sizes: `train=36`, `valid=18`, `test=18`
- full 12-label coverage in every split
- source type mix: `html=6`, `pdf=4`, `image=2`
- average item length: `7.58` whitespace tokens

### v2

The main experimental release is:

- [items.v2.jsonl](../../data/annotated/items.v2.jsonl)
- [dataset-manifest.v2.csv](../../data/raw/dataset-manifest.v2.csv)
- [dataset-stats.v2.json](../../data/interim/dataset-stats.v2.json)

Summary:

- 432 annotated menu items
- 12 source documents from 12 restaurants
- split sizes: `train=288`, `valid=72`, `test=72`
- full 12-label coverage in every split
- no source leakage and no restaurant leakage across splits
- source type mix: `html=6`, `pdf=6`
- average item length: `8.70` whitespace tokens
- balanced class counts at `36` items per label overall

`v2` is a derived item-level release built from public menu sources. The
released texts are normalized into consistent English forms with normalized
`RUB` prices. That makes the repository lightweight and reproducible, but it
also means the public dataset is cleaner than raw menu captures.
