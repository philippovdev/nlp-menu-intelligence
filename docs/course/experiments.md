# Experiments Plan

## Core Research Questions

1. How strong is the current heuristic pipeline as a baseline?
2. How much do simple text classifiers improve category prediction?
3. How much does a transformer improve category prediction over classical
   baselines?
4. How well can structured fields be recovered from item text?
5. How much does OCR quality affect downstream parsing and classification?

## Evaluation Layers

### Item-Level NLP Evaluation

Main layer for the course report.

- classification input: item text
- classification output: category label
- extraction input: item text
- extraction output: slots or BIO2 spans

### End-to-End Document Evaluation

Secondary layer for the product pipeline.

- document input: PDF or image
- output: extracted text plus structured menu JSON

This should be reported, but not replace the item-level NLP evaluation.

## Metrics

### Classification

- Accuracy
- Macro-F1
- Per-class F1

Macro-F1 should be the main model selection metric because the label set is not
expected to be balanced.

### Extraction

- entity-level Precision
- entity-level Recall
- entity-level F1

If the extraction remains rule-based for the first phase, still evaluate the
slot output on a labeled subset.

### End-to-End

- exact or partial match on price and size fields
- category accuracy after OCR
- document-to-item success rate

## Baselines

These should appear in the final results table.

### Classification Baselines

- heuristic keyword classifier
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- fastText

### Extraction Baselines

- regex-only extraction
- heuristic normalization rules

### Main Model Candidates

- transformer encoder fine-tuning for classification
- transformer token classifier for extraction, if the dataset is large enough

## Ablations

The first ablations to prepare:

- pasted text vs PDF embedded text vs image OCR
- heuristic categorization vs trained classifier
- regex-only extraction vs learned extraction

## Run Tracking Template

Keep one row per experiment run.

| Run ID | Date | Commit | Dataset version | Task | Method | Main metric | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Main Results Table Template

| Method | Category Accuracy | Category Macro-F1 | Extraction F1 | Notes |
| --- | --- | --- | --- | --- |
| Heuristic baseline | 0.9000 | 0.8182 | N/A | v0 sample set; price exact = 1.0000, size exact = 1.0000 |
| TF-IDF + Logistic Regression | 0.3333 | 0.2790 | N/A | items.v1 test split; valid acc = 0.2778, valid macro-F1 = 0.2472 |
| TF-IDF + Linear SVM | TBD | TBD | N/A or TBD | |
| fastText | TBD | TBD | N/A or TBD | |
| Transformer | TBD | TBD | TBD | |

## Error Analysis Checklist

For every serious run, save:

- 5 to 10 high-confidence mistakes
- 5 to 10 OCR-induced mistakes
- top confused category pairs
- examples where extraction breaks the category decision

## Immediate Experiment Work

1. Freeze the first gold evaluation subset.
2. Measure the current heuristic pipeline.
3. Add the first classical baseline for category classification.
4. Save the results in a machine-readable table.
5. Keep sample outputs for the report.

The first measured run is saved in
[baseline-heuristic-results.json](/Users/philippovdev/WebstormProjects/nlp/docs/course/artifacts/baseline-heuristic-results.json).

The next baseline runs should use
[items.v1.jsonl](/Users/philippovdev/WebstormProjects/nlp/data/annotated/items.v1.jsonl)
with the paired stats artifact
[dataset-stats.v1.json](/Users/philippovdev/WebstormProjects/nlp/data/interim/dataset-stats.v1.json).
This subset now has full 12-label coverage in `train`, `valid`, and `test`,
so Macro-F1 on the held-out splits is meaningful for the first classical
baselines.

The first classical baseline is now saved in
[tfidf-logreg-items-v1.json](/Users/philippovdev/WebstormProjects/nlp/docs/course/artifacts/tfidf-logreg-items-v1.json).
It uses word-level TF-IDF features with Logistic Regression, fits on the fixed
`train` split of `items.v1.jsonl`, and reports `0.2778` accuracy / `0.2472`
Macro-F1 on `valid` plus `0.3333` accuracy / `0.2790` Macro-F1 on `test`.
