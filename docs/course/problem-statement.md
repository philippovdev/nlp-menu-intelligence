# Problem Statement

## Product Goal

The service receives raw menu content and returns a structured menu with
categories and menu items. The input may be pasted text, a PDF, or an image.
The output is a normalized JSON structure that can be reviewed and corrected in
the web interface.

## NLP Goal

The system is evaluated as a pipeline of measurable NLP tasks, not only as a
web application.

### Task A. Menu Item Categorization

Given a menu item text `x`, predict a category label `y` from a fixed set `C`.

Example:

- input: `Caesar with chicken 250 g - 390 RUB`
- output: `salads`

### Task B. Menu Item Information Extraction

Given a tokenized menu item text `x = (t1, ..., tn)`, predict BIO2 tags
`z = (z1, ..., zn)` so that the model can recover structured attributes such as
name, price, size, and optional description.

### Task C. Menu Structuring

Given a raw menu text or OCR output `M`, split it into candidate lines, run
classification and extraction for each line, and aggregate the items into a
structured menu JSON response.

## Category Label Set

The working label set for item-level classification:

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

This set is stable enough for the first dataset version and broad enough to
group common restaurant menus.

## Extraction Label Schema

### BIO2 Entity Set

- `B-NAME`, `I-NAME`
- `B-DESC`, `I-DESC`
- `B-PRICE`, `I-PRICE`
- `B-SIZE`, `I-SIZE`
- `O`

This is the default extraction schema for the first dataset version. It is
coarse enough to annotate quickly and still supports a useful structured output.

## Source-of-Truth JSON Item

The item-level annotation format should stay JSON-first.

```json
{
  "id": "item_000123",
  "source_id": "menu_000017",
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

## Current Service Contract

The current service already exposes:

- `POST /api/v1/menu/parse`
- `POST /api/v1/menu/parse-file`

Both routes should end in the same `items[]` response shape so that dataset
annotation, manual review, and later model inference all stay aligned.

## Evaluation Scope

The current system is measured as a pipeline:

1. document ingestion and text extraction
2. line normalization
3. category prediction
4. structured field extraction
5. JSON aggregation and human review

The main quantitative results should focus on tasks A and B. OCR quality may be
reported separately as an upstream ablation or preprocessing study.
