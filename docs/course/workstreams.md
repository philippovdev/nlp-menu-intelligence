# Workstreams

This file keeps the next course-oriented work in a concrete order.

## 1. Dataset Tasks

### Goal

Produce the first item-level gold dataset and a small document-level evaluation
set.

### Tasks

1. Freeze the category label set.
2. Freeze the JSONL annotation schema.
3. Collect 30 to 50 public menu documents or pages.
4. Convert those sources into item-level candidate rows.
5. Annotate the first 300 to 500 items.
6. Split by restaurant or document source.
7. Export a classification dataset.
8. Export a BIO2 dataset if extraction training is pursued.

### Exit Criteria

- at least one annotated JSONL file exists
- train, valid, and test splits are defined
- dataset statistics can be computed

## 2. Baseline Experiment Tasks

### Goal

Build the first honest results table for the report.

### Tasks

1. Measure the current heuristic pipeline.
2. Add TF-IDF + Logistic Regression.
3. Add TF-IDF + Linear SVM.
4. Optionally add fastText.
5. Save metrics, sample outputs, and confusion matrix artifacts.
6. Record each run with commit SHA and dataset version.

### Exit Criteria

- at least three category baselines measured
- one comparison table exists
- one error analysis note exists

## 3. Report Evidence Tasks

### Goal

Collect everything that will later be moved into the final PDF.

### Tasks

1. Fill the related work matrix.
2. Save screenshots of the product flow.
3. Save 5 to 10 representative input/output examples.
4. Generate the first dataset statistics table.
5. Generate the first results table.
6. Draft one pipeline figure.

### Exit Criteria

- `report/main.tex` can be filled without searching for missing artifacts
- all major tables already exist in draft form

## Recommended Order From Here

1. Dataset first
2. Heuristic baseline measurement
3. Classical ML baselines
4. OCR quality improvements
5. Transformer models
6. Final report filling
