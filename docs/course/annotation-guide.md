# Annotation Guide

This guide is the working annotation policy for the first item-level dataset
version.

## Unit of Annotation

Annotate one row per semantic menu line.

Examples of valid annotation units:

- `Caesar with chicken 250 g - 390 RUB`
- `Tom Yum 450 RUB`
- `Americano 250 ml - 180 RUB`

Examples that are usually not annotated as menu items:

- `SALADS`
- `Prices may vary`
- `Ask the waiter about allergens`

Headers and noise may still be kept in a document-level dataset, but the
item-level classification dataset should focus on real menu items.

## Item Types

When preparing the full service JSON, each line may be labeled as:

- `menu_item`
- `category_header`
- `noise`

For the item-level NLP dataset, prefer keeping only `menu_item` rows unless a
specific experiment needs headers or noise.

## Category Labels

Use exactly one category label per item:

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

### Category Rules

- Pick the most specific available label from the list above.
- Use `other` only if no label fits reasonably.
- Do not invent new labels during annotation.
- If the item is ambiguous, keep the best label and note the ambiguity in a
  separate review note if needed.

## Slot Schema

Each annotated item should contain:

- `name`
- `description`
- `prices`
- `sizes`

### Name

`name` is the canonical dish or drink title without trailing price or size
fragments.

Examples:

- raw: `Caesar with chicken 250 g - 390 RUB`
- name: `Caesar with chicken`

- raw: `Americano 250 ml 180 RUB`
- name: `Americano`

### Description

`description` is optional. Use it only when the line clearly contains a
secondary descriptive fragment that is not part of the item name.

If the line has no reliable description, set it to `null`.

### Prices

Use a list, even when there is only one price.

Normalize each price into:

```json
{
  "value": 390,
  "currency": "RUB",
  "raw": "390 RUB"
}
```

Rules:

- keep numeric value as a number
- normalize currency to ISO-like code where possible
- preserve the original snippet in `raw`

### Sizes

Use a list, even when there is only one size.

Normalize each size into:

```json
{
  "value": 250,
  "unit": "g",
  "raw": "250 g"
}
```

Canonical units:

- `g`
- `kg`
- `ml`
- `l`
- `pcs`

## BIO2 Export Rules

When exporting to BIO2:

- `NAME` covers the normalized dish or drink name span
- `DESC` covers optional description text
- `PRICE` covers numeric price plus currency if adjacent
- `SIZE` covers numeric amount plus unit if adjacent
- everything else is `O`

## Tricky Cases

### Multiple prices

If a line contains multiple prices:

- keep all prices in `prices`
- preserve order of appearance
- note that this row may later require a more specific modeling decision

### Multiple sizes

If a line contains multiple sizes:

- keep all sizes in `sizes`
- preserve order of appearance

### Combo or set items

If the item is clearly a single menu position, annotate it as one item even if
it mentions multiple subcomponents.

### Missing price

If a real item has no visible price:

- keep the row
- set `prices` to an empty list

### Section headers mixed with items

If a line is only a section title such as `SALADS`, do not include it in the
item-level training dataset unless a specific experiment needs header detection.

## Quality Checklist

Before accepting an annotated item, verify:

- the category label exists in the fixed label set
- the name does not contain leftover price or size noise
- the price value is numeric and normalized
- the size unit is canonical
- the raw text can still be reconstructed from the stored fields plus `text`

## Initial Annotation Target

For the first dataset version:

- annotate 300 to 500 items
- keep at least 20 to 30 items per major category where possible
- keep a small difficult subset with OCR noise for later evaluation
