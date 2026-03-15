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
| Heuristic baseline | 0.6181 | 0.6466 | N/A | items.v2 full set; price exact = 1.0000, size exact = 1.0000 |
| TF-IDF + Logistic Regression | 0.7222 | 0.7009 | N/A | items.v2 test split; valid acc = 0.7361, valid macro-F1 = 0.7302 |
| TF-IDF + Linear SVM | 0.7083 | 0.6930 | N/A | items.v2 test split; valid acc = 0.7500, valid macro-F1 = 0.7502 |
| fastText | TBD | TBD | N/A or TBD | |
| DistilBERT sequence classifier | 0.5556 | 0.5620 | N/A | items.v2 test split; valid acc = 0.6389, valid macro-F1 = 0.6248 |

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

The current training-ready classification dataset is now
[items.v2.jsonl](/Users/philippovdev/WebstormProjects/nlp/data/annotated/items.v2.jsonl)
with the paired stats artifact
[dataset-stats.v2.json](/Users/philippovdev/WebstormProjects/nlp/data/interim/dataset-stats.v2.json).
This expanded subset contains `432` annotated items with source-level split
assignment (`288 train / 72 valid / 72 test`), keeps the same 12-label schema
family as `v1`, and preserves full label coverage in all three splits. The
expanded release is a source-grounded synthetic expansion rather than a
source-faithful raw crawl, with normalized English text and normalized `RUB`
prices.

The first direct classical comparison set on `items.v2` is now saved in:

- [baseline-heuristic-items-v2.json](/Users/philippovdev/WebstormProjects/nlp/docs/course/artifacts/baseline-heuristic-items-v2.json)
- [tfidf-logreg-items-v2.json](/Users/philippovdev/WebstormProjects/nlp/docs/course/artifacts/tfidf-logreg-items-v2.json)
- [tfidf-linear-svm-items-v2.json](/Users/philippovdev/WebstormProjects/nlp/docs/course/artifacts/tfidf-linear-svm-items-v2.json)

On this fixed split, TF-IDF + Logistic Regression currently gives the strongest
held-out test result with `0.7222` accuracy and `0.7009` Macro-F1. Linear SVM
is slightly better on the validation split (`0.7500` accuracy / `0.7502`
Macro-F1) but slightly worse on the held-out test split (`0.7083` / `0.6930`).

The first transformer result is now saved in
[transformer-classifier-items-v2.json](/Users/philippovdev/WebstormProjects/nlp/docs/course/artifacts/transformer-classifier-items-v2.json).
It fine-tunes `distilbert-base-uncased` on the fixed `train` split of
`items.v2.jsonl`, uses early stopping on validation Macro-F1, and reaches
`0.6389` accuracy / `0.6248` Macro-F1 on `valid` plus `0.5556` accuracy /
`0.5620` Macro-F1 on `test`. On this dataset version, the first transformer
underperforms the two TF-IDF baselines, so the current evidence does not
support replacing them yet. The repository keeps the JSON metrics artifact and
training script, while local checkpoints under `models/` remain unversioned.
