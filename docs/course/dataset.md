# Dataset Plan

## Dataset Layers

The project should maintain two related datasets:

### 1. Document-Level Set

Used for end-to-end checks and product validation.

- input: image or PDF or pasted menu text
- output: extracted text and structured menu JSON

### 2. Item-Level Set

Used for NLP experiments.

- input: single menu line or item text
- output:
  - category label
  - structured slots
  - optional BIO2 sequence export

The item-level set is the main dataset for the course experiments.

## Data Sources

Preferred sources:

- public restaurant menu pages
- public restaurant PDF menus
- public menu screenshots or images when reuse is allowed for research

For each source, keep:

- source URL
- collection date
- source type: `html`, `pdf`, `image`
- restaurant identifier
- language

Use [dataset-manifest-template.csv](/Users/philippovdev/WebstormProjects/nlp/data/raw/dataset-manifest-template.csv)
as the starting manifest format.

## Legal and Repository Policy

- Only use data that is publicly accessible for research use.
- Do not commit large raw files directly into the repository.
- Keep raw data outside git if needed and commit only:
  - source manifests
  - annotation schema
  - small example fixtures
  - scripts and instructions needed to reproduce the dataset

## Working Data Layout

The intended structure under `data/`:

```text
data/
  raw/         source manifests or small public fixtures
  interim/     normalized intermediate exports
  annotated/   gold JSONL item annotations
  external/    optional large data kept outside git
```

## Annotation Format

Primary source of truth: JSONL.

One line per item:

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

Derived exports:

- classification TSV or CSV
- BIO2 CoNLL-style export for extraction

See [annotation-guide.md](/Users/philippovdev/WebstormProjects/nlp/docs/course/annotation-guide.md)
for labeling rules.

## Split Policy

Preferred split: by source or restaurant, not by random line.

Reason:

- avoids leakage between very similar menu items
- better reflects domain shift between establishments
- makes the evaluation more credible in the report

Initial target:

- `train`: 70%
- `valid`: 15%
- `test`: 15%

Adjust later if the number of distinct restaurants is still small.

## Statistics to Report

Prepare the following table for the final report.

| Statistic | Train | Valid | Test |
| --- | --- | --- | --- |
| Restaurants | TBD | TBD | TBD |
| Documents | TBD | TBD | TBD |
| Items | TBD | TBD | TBD |
| Avg. tokens per item | TBD | TBD | TBD |
| Categories | TBD | TBD | TBD |

Additional figures to prepare:

- class distribution bar chart
- line length histogram
- source type distribution (`text`, `pdf`, `image`)

## Annotation Checklist

Each annotated item should answer:

- Is this a real menu item, a header, or noise?
- Which category label is correct?
- What is the canonical item name?
- Is there a price? If yes, what is the normalized numeric value?
- Is there a size? If yes, what is the normalized numeric value and unit?

## Immediate Dataset Work

1. Freeze the first annotation guideline.
2. Collect 30 to 50 documents for the first gold evaluation set.
3. Convert those documents into item-level examples.
4. Annotate the first 300 to 500 items.
5. Export the first classification dataset.

## Current Starter Files

The repository now includes a minimal starter set for the first measured
baseline:

- [dataset-manifest.sample.csv](/Users/philippovdev/WebstormProjects/nlp/data/raw/dataset-manifest.sample.csv)
- [items.sample.jsonl](/Users/philippovdev/WebstormProjects/nlp/data/annotated/items.sample.jsonl)

The current sample file contains 10 realistic item-level annotations and is
used only as a reproducible starter baseline, not as the final course dataset.

## Dataset v1

The repository now also includes the first real in-repo item-level subset for
course experiments:

- [items.v1.jsonl](/Users/philippovdev/WebstormProjects/nlp/data/annotated/items.v1.jsonl)
- [dataset-manifest.v1.csv](/Users/philippovdev/WebstormProjects/nlp/data/raw/dataset-manifest.v1.csv)
- [dataset-stats.v1.json](/Users/philippovdev/WebstormProjects/nlp/data/interim/dataset-stats.v1.json)

Current factual status of `v1`:

- 72 annotated menu items
- 12 source documents from 12 restaurants
- split by source/restaurant, not by random row
- split sizes: `train=36`, `valid=18`, `test=18`
- all 12 category labels are present in every split
- source type mix: `html=6`, `pdf=4`, `image=2`
- average item length: `7.58` whitespace tokens

This is still a first curated subset, not the final course dataset, but it now
supports honest first heuristic and classical baseline runs because the
validation and test splits both cover the full 12-label taxonomy.

The public repository keeps source metadata and split structure, but does not
re-publish raw crawls or direct source captures. For `v1`, the manifest keeps
source identifiers, restaurant identifiers, source types, collection dates, and
public source URLs, while raw captures remain outside git.
