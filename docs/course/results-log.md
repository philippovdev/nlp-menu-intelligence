# Results Log

Use this file as the running experiment ledger before final results are moved
into the report.

## Run Table

| Run ID | Date | Commit | Dataset Version | Task | Method | Split | Main Metric | Secondary Metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-heuristic-001 | 2026-03-09 | b5c0369 | v0-sample | classification + extraction | heuristic pipeline | test | Macro-F1 = 0.8182 | Acc = 0.9000; price exact = 1.0000; size exact = 1.0000 | artifact: `docs/course/artifacts/baseline-heuristic-results.json`; missed `breakfast` sample |
| tfidf-logreg-v1-001 | 2026-03-09 | ed9500c | v1 | classification | TF-IDF + Logistic Regression | valid | Macro-F1 = 0.2472 | Acc = 0.2778 | artifact: `docs/course/artifacts/tfidf-logreg-items-v1.json`; train = 36, labels = 12 |
| tfidf-logreg-v1-001 | 2026-03-09 | ed9500c | v1 | classification | TF-IDF + Logistic Regression | test | Macro-F1 = 0.2790 | Acc = 0.3333 | artifact: `docs/course/artifacts/tfidf-logreg-items-v1.json`; held-out split |
| tfidf-linear-svm-v1-001 | 2026-03-09 | 2f3f051 | v1 | classification | TF-IDF + Linear SVM | valid | Macro-F1 = 0.2556 | Acc = 0.2778 | artifact: `docs/course/artifacts/tfidf-linear-svm-items-v1.json`; train = 36, labels = 12 |
| tfidf-linear-svm-v1-001 | 2026-03-09 | 2f3f051 | v1 | classification | TF-IDF + Linear SVM | test | Macro-F1 = 0.3433 | Acc = 0.3889 | artifact: `docs/course/artifacts/tfidf-linear-svm-items-v1.json`; better than TF-IDF + Logistic Regression on held-out test |

## What To Save For Every Run

- commit SHA
- dataset version or manifest version
- exact method name
- metric values
- a short note on major failures
- path to saved artifacts if they exist

## Recommended Artifact Set

For classification runs:

- confusion matrix
- per-class F1 table
- 5 to 10 representative errors

For extraction runs:

- entity-level Precision, Recall, F1
- slot-level examples
- failure cases caused by OCR vs caused by parsing

## Priority Runs

1. current heuristic pipeline
2. TF-IDF + Logistic Regression
3. TF-IDF + Linear SVM
4. fastText
5. first transformer baseline

## Dataset Artifacts

- `dataset-v1-stats-001`: `data/interim/dataset-stats.v1.json`
  - 72 items
  - 12 restaurants / 12 documents
  - split = `36 train / 18 valid / 18 test`
  - full label coverage in `train`, `valid`, and `test`
  - average tokens per item = `7.58`
