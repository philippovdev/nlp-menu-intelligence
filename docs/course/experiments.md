# Experiments

## Evaluation Layers

### Item-Level Evaluation

This is the main research setting.

- input: one menu item line
- outputs: category label and structured slots

### End-To-End Evaluation

This is the product-oriented setting.

- input: pasted text, PDF, or image
- output: extracted text plus structured menu JSON

The repository also includes a small end-to-end slice:

- [realworld-manifest.v1.csv](../../data/eval/realworld-manifest.v1.csv)
- [realworld-gold.v1.jsonl](../../data/eval/realworld-gold.v1.jsonl)
- [realworld-eval-v1.json](artifacts/realworld-eval-v1.json)

This slice contains 12 examples: 4 pasted-text inputs, 4 embedded-text PDFs,
and 4 rendered image fixtures. It is useful for regression checks after parser
changes, but it is easier than open-ended production traffic.

## Metrics

### Classification

- accuracy
- Macro-F1
- per-class F1

Macro-F1 is the main selection metric because all 12 labels matter.

### Extraction

- entity-level Precision
- entity-level Recall
- entity-level F1

### End-To-End

- exact or partial match on price and size fields
- category accuracy after OCR
- document-to-item success rate

## Compared Methods

- heuristic keyword pipeline
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF word/character union + Logistic Regression
- TF-IDF + sigmoid-calibrated Linear SVM
- DistilBERT sequence classifier
- XLM-RoBERTa sequence classifier

## Fixed Classification Dataset

The main classification runs use
[items.v2.jsonl](../../data/annotated/items.v2.jsonl)
with source-level splits:

- `train=288`
- `valid=72`
- `test=72`

## Main Results

| Method | Category Accuracy | Category Macro-F1 | Notes |
| --- | --- | --- | --- |
| Heuristic pipeline | 0.6181 | 0.6466 | full `items.v2`; price exact = `1.0000`; size exact = `1.0000` |
| TF-IDF + Logistic Regression | 0.7222 | 0.7009 | test split; valid = `0.7361 / 0.7302` |
| TF-IDF + Linear SVM | 0.7083 | 0.6930 | test split; valid = `0.7500 / 0.7502` |
| TF-IDF word/char union + Logistic Regression | 0.7361 | 0.7254 | test split; valid = `0.7778 / 0.7749`; best held-out result |
| TF-IDF + sigmoid-calibrated Linear SVM | 0.7083 | 0.6900 | test split; valid = `0.7500 / 0.7431` |
| DistilBERT sequence classifier | 0.5556 | 0.5620 | test split; valid = `0.6389 / 0.6248` |
| XLM-RoBERTa sequence classifier | 0.5556 | 0.5462 | test split; valid = `0.6250 / 0.6155` |

On this dataset, sparse lexical models outperform the tested transformer
baselines. The strongest result is the TF-IDF word/character union with
Logistic Regression.

## Result Artifacts

Classification artifacts:

- [baseline-heuristic-items-v2.json](artifacts/baseline-heuristic-items-v2.json)
- [tfidf-logreg-items-v2.json](artifacts/tfidf-logreg-items-v2.json)
- [tfidf-linear-svm-items-v2.json](artifacts/tfidf-linear-svm-items-v2.json)
- [tfidf-union-logreg-items-v2.json](artifacts/tfidf-union-logreg-items-v2.json)
- [tfidf-calibrated-linear-svm-items-v2.json](artifacts/tfidf-calibrated-linear-svm-items-v2.json)
- [transformer-classifier-items-v2.json](artifacts/transformer-classifier-items-v2.json)
- [transformer-multilingual-items-v2.json](artifacts/transformer-multilingual-items-v2.json)

End-to-end artifact:

- [realworld-eval-v1.json](artifacts/realworld-eval-v1.json)
