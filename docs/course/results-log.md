# Results Log

Use this file as the running experiment ledger before final results are moved
into the report.

## Run Table

| Run ID | Date | Commit | Dataset Version | Task | Method | Split | Main Metric | Secondary Metrics | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline-heuristic-001 | TBD | TBD | v0 | classification + extraction | heuristic pipeline | TBD | TBD | TBD | current service baseline |

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
