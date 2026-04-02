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
| baseline-heuristic-v2-001 | 2026-03-15 | 2cd70b0 | v2 | classification + extraction | heuristic pipeline | full | Macro-F1 = 0.6466 | Acc = 0.6181; price exact = 1.0000; size exact = 1.0000 | artifact: `docs/course/artifacts/baseline-heuristic-items-v2.json`; full `items.v2` run, so not split-matched to train/valid/test model runs |
| tfidf-logreg-v2-001 | 2026-03-15 | 2cd70b0 | v2 | classification | TF-IDF + Logistic Regression | valid | Macro-F1 = 0.7302 | Acc = 0.7361 | artifact: `docs/course/artifacts/tfidf-logreg-items-v2.json`; train = 288, labels = 12 |
| tfidf-logreg-v2-001 | 2026-03-15 | 2cd70b0 | v2 | classification | TF-IDF + Logistic Regression | test | Macro-F1 = 0.7009 | Acc = 0.7222 | artifact: `docs/course/artifacts/tfidf-logreg-items-v2.json`; strongest held-out test result so far |
| tfidf-linear-svm-v2-001 | 2026-03-15 | 2cd70b0 | v2 | classification | TF-IDF + Linear SVM | valid | Macro-F1 = 0.7502 | Acc = 0.7500 | artifact: `docs/course/artifacts/tfidf-linear-svm-items-v2.json`; strongest validation result so far |
| tfidf-linear-svm-v2-001 | 2026-03-15 | 2cd70b0 | v2 | classification | TF-IDF + Linear SVM | test | Macro-F1 = 0.6930 | Acc = 0.7083 | artifact: `docs/course/artifacts/tfidf-linear-svm-items-v2.json`; slightly behind Logistic Regression on held-out test |
| tfidf-union-logreg-v2-001 | 2026-03-27 | working-tree | v2 | classification | TF-IDF word/char union + Logistic Regression | valid | Macro-F1 = 0.7749 | Acc = 0.7778 | artifact: `docs/course/artifacts/tfidf-union-logreg-items-v2.json`; strongest validation result so far |
| tfidf-union-logreg-v2-001 | 2026-03-27 | working-tree | v2 | classification | TF-IDF word/char union + Logistic Regression | test | Macro-F1 = 0.7254 | Acc = 0.7361 | artifact: `docs/course/artifacts/tfidf-union-logreg-items-v2.json`; strongest held-out test result so far and now the shipped backend classifier |
| tfidf-calibrated-linear-svm-v2-001 | 2026-03-27 | working-tree | v2 | classification | TF-IDF + sigmoid-calibrated Linear SVM | valid | Macro-F1 = 0.7431 | Acc = 0.7500 | artifact: `docs/course/artifacts/tfidf-calibrated-linear-svm-items-v2.json`; better-calibrated probability path for SVM comparison |
| tfidf-calibrated-linear-svm-v2-001 | 2026-03-27 | working-tree | v2 | classification | TF-IDF + sigmoid-calibrated Linear SVM | test | Macro-F1 = 0.6900 | Acc = 0.7083 | artifact: `docs/course/artifacts/tfidf-calibrated-linear-svm-items-v2.json`; did not beat the TF-IDF union + Logistic Regression run |
| transformer-classifier-v2-001 | 2026-03-15 | working-tree | v2 | classification | DistilBERT sequence classifier | valid | Macro-F1 = 0.6248 | Acc = 0.6389 | artifact: `docs/course/artifacts/transformer-classifier-items-v2.json`; `distilbert-base-uncased`, early stopping on validation Macro-F1 |
| transformer-classifier-v2-001 | 2026-03-15 | working-tree | v2 | classification | DistilBERT sequence classifier | test | Macro-F1 = 0.5620 | Acc = 0.5556 | artifact: `docs/course/artifacts/transformer-classifier-items-v2.json`; underperforms both TF-IDF baselines on held-out test |
| transformer-multilingual-v2-001 | 2026-03-16 | working-tree | v2 | classification | XLM-RoBERTa sequence classifier | valid | Macro-F1 = 0.6155 | Acc = 0.6250 | artifact: `docs/course/artifacts/transformer-multilingual-items-v2.json`; `FacebookAI/xlm-roberta-base`, trained on `mps`, early stopping on validation Macro-F1 |
| transformer-multilingual-v2-001 | 2026-03-16 | working-tree | v2 | classification | XLM-RoBERTa sequence classifier | test | Macro-F1 = 0.5462 | Acc = 0.5556 | artifact: `docs/course/artifacts/transformer-multilingual-items-v2.json`; still below TF-IDF + Logistic Regression and slightly below the earlier DistilBERT run on held-out Macro-F1 |
| realworld-eval-v1-001 | 2026-03-15 | working-tree | realworld-v1 | end-to-end OCR + parsing | deployed backend pipeline | overall | Text token F1 = 1.0000 | Category acc = 1.0000; category Macro-F1 = 1.0000; price exact = 1.0000; size exact = 1.0000 | artifact: `docs/course/artifacts/realworld-eval-v1.json`; `12` examples (`4` text / `4` PDF / `4` image); source-grounded rendered fixtures, so this is a regression slice rather than a full raw-capture benchmark |

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

- `dataset-v2-stats-001`: `data/interim/dataset-stats.v2.json`
  - 432 items
  - 12 restaurants / 12 documents
  - split = `288 train / 72 valid / 72 test`
  - full label coverage in `train`, `valid`, and `test`
  - no source leakage and no restaurant leakage across splits
  - average tokens per item = `8.70`
  - source-grounded synthetic expansion with normalized English text and normalized `RUB` prices

- `realworld-eval-v1-001`: `docs/course/artifacts/realworld-eval-v1.json`
  - 12 source-grounded examples
  - 4 pasted-text cases, 4 embedded-text PDF cases, 4 rendered image cases
  - paired inputs: `data/eval/realworld-manifest.v1.csv` and `data/eval/realworld-gold.v1.jsonl`
  - perfect end-to-end recovery on the current slice
  - intended as a small regression-oriented real-world check, not a raw phone-photo benchmark
