# Related Work Plan

## Why This Section Matters

The course grading explicitly rewards a strong `Related Work` section and a
comparison against other approaches. This file is the working place for that
material before it is moved into the final report.

## Literature Buckets

### 1. Short Text Classification

Use this bucket for methods close to `menu item -> category`.

Look for:

- product categorization
- catalog title classification
- short sentence classification

### 2. Food and Recipe Classification

Use this bucket for food-domain methods, even if they are not menu-specific.

Look for:

- recipe type classification
- cuisine or dish type classification
- food menu understanding

### 3. Information Extraction

Use this bucket for extracting prices, sizes, attributes, or structured fields
from short retail-like texts.

Look for:

- price list parsing
- receipt or invoice extraction
- named entity recognition in short noisy text

## Working Matrix

Fill this table as sources are collected.

| Reference key | Problem | Dataset | Method | Metric | Reported result | Relevance |
| --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## What to Capture for Each Source

- bibliographic entry for `report/lit.bib`
- exact task definition
- dataset name and size
- strongest baseline
- strongest reported model
- metric used
- limitation relative to this project

## Comparison Strategy

Not every paper will match the project exactly. The report should separate:

- direct comparators on the same or very similar task
- conceptual comparators that justify model and baseline choice

If no paper matches the exact menu dataset, be explicit about it and compare on:

- task similarity
- text length similarity
- domain similarity

## Minimum Target Before Writing the Final Report

- 7 to 12 sources total
- at least 3 sources for classification
- at least 2 sources for extraction
- at least 1 source for document or OCR pipeline motivation
- at least 1 source for a transformer-based baseline
