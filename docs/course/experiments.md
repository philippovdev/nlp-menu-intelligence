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
- token-level Precision / Recall / F1 on BIO2 tags

### End-To-End

- exact or partial match on price and size fields
- category accuracy after OCR
- document-to-item success rate

## Compared Methods

- heuristic keyword pipeline
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF word/character union + Logistic Regression
- TF-IDF enriched sparse features + Logistic Regression
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
| TF-IDF word/char union + Logistic Regression | 0.7361 | 0.7254 | test split; valid = `0.7778 / 0.7749`; previous shipped baseline |
| TF-IDF enriched sparse features + Logistic Regression | 0.8750 | 0.8708 | test split; valid = `0.8056 / 0.8065`; raw text + cleaned name + slot features + targeted lexicon flags; backend runtime model |
| TF-IDF + sigmoid-calibrated Linear SVM | 0.7083 | 0.6900 | test split; valid = `0.7500 / 0.7431` |
| DistilBERT sequence classifier | 0.5556 | 0.5620 | test split; valid = `0.6389 / 0.6248` |
| XLM-RoBERTa sequence classifier | 0.5556 | 0.5462 | test split; valid = `0.6250 / 0.6155` |

On this dataset, sparse lexical models outperform the tested transformer
baselines. The strongest result is the enriched sparse Logistic Regression
pipeline that combines text n-grams with deterministic slot-derived features.

## Result Artifacts

Classification artifacts:

- [baseline-heuristic-items-v2.json](artifacts/baseline-heuristic-items-v2.json)
- [tfidf-logreg-items-v2.json](artifacts/tfidf-logreg-items-v2.json)
- [tfidf-linear-svm-items-v2.json](artifacts/tfidf-linear-svm-items-v2.json)
- [tfidf-union-logreg-items-v2.json](artifacts/tfidf-union-logreg-items-v2.json)
- [tfidf-enriched-logreg-items-v2.json](artifacts/tfidf-enriched-logreg-items-v2.json)
- [tfidf-calibrated-linear-svm-items-v2.json](artifacts/tfidf-calibrated-linear-svm-items-v2.json)
- [transformer-classifier-items-v2.json](artifacts/transformer-classifier-items-v2.json)
- [transformer-multilingual-items-v2.json](artifacts/transformer-multilingual-items-v2.json)

End-to-end artifact:

- [realworld-eval-v1.json](artifacts/realworld-eval-v1.json)

## BIO2 Extraction

The repository also includes a BIO2 version of the item-level dataset and
evaluates both the deterministic extractor and a small learned token
classifier in BIO2 space:

- [items.v2.bio2.jsonl](../../data/interim/items.v2.bio2.jsonl)
- [bio2-extraction-baseline-items-v2.json](artifacts/bio2-extraction-baseline-items-v2.json)
- [bio2-token-logreg-items-v2.json](artifacts/bio2-token-logreg-items-v2.json)

The BIO2 export aligns the structured slots back to the original item text
using substring matching for `NAME` and surface-fragment recovery for `PRICE`
and `SIZE`. Because the released dataset is normalized and fully populated for
price and size, both the deterministic extractor and a contextual
LogisticRegression token tagger reach perfect scores on the fixed validation
and test splits:

| Method | Evaluation set | Token Micro-F1 | Entity Micro-F1 | Notes |
| --- | --- | --- | --- | --- |
| Deterministic BIO2 extraction baseline | valid split | 1.0000 | 1.0000 | `NAME`, `PRICE`, `SIZE`; tags derived from structured slot annotations |
| Deterministic BIO2 extraction baseline | test split | 1.0000 | 1.0000 | same result as validation on the fixed BIO2 test split |
| Contextual LogisticRegression BIO2 tagger | valid split | 1.0000 | 1.0000 | trained on whitespace-tokenized BIO2 records with contextual token features |
| Contextual LogisticRegression BIO2 tagger | test split | 1.0000 | 1.0000 | learned baseline matches the deterministic extractor on the released split |

These results are useful for coverage and reproducibility, but they should not
be treated as evidence that menu field extraction is solved under arbitrary OCR
or raw production formatting. On the released dataset, Task A remains the
harder modeling problem.
