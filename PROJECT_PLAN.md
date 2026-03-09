# NLP Menu Intelligence Project Plan

## 1. Project Goal

- Build a web service that transforms raw menu text or menu documents into a structured menu.
- Input formats for MVP:
  - plain text
  - image (`jpg`, `png`, `webp`)
  - PDF
- Output format:
  - categories
  - menu items inside categories
  - extracted attributes: name, price, weight/volume, optional allergens

## 2. Formal Task Definition

### Task A. Category Classification

- Given a menu item text `x`, predict a category label `y` from a fixed set `C`.
- Example:
  - input: `Цезарь с курицей 250 г — 390 ₽`
  - output: `salads`

### Task B. Attribute Extraction

- Given a tokenized menu item text `x = (t1, ..., tn)`, predict BIO2 tags `z = (z1, ..., zn)` for structured fields.
- Minimal entity set:
  - `NAME`
  - `PRICE`
  - `SIZE`
  - `ALLERGEN`
  - `DESC`

### Task C. Menu Structuring

- Given raw menu text or OCR output `M`, split it into lines, run classification and extraction, then aggregate items into a JSON menu schema.

## 3. Recommended Product Architecture

### Final stack

- Frontend: `Vue 3 + Vite + TypeScript`
- Backend API: `FastAPI`
- Model inference: `ONNX Runtime`
- Cache: `Redis`
- Storage:
  - MVP: `SQLite` or JSONL
  - better production option: `PostgreSQL`
- Reverse proxy: `Nginx`

### Why this stack

- `Laravel + Inertia` adds delivery cost without helping the NLP part.
- `FastAPI` is enough for inference, OCR orchestration, annotation endpoints, and feedback collection.
- `Vue 3` is enough for a clean UI: upload file, preview parsed lines, edit categories, export JSON/CSV.
- ONNX inference is realistic on a CPU VPS; local LLM hosting is not.

## 4. VPS Inventory Collected On March 9, 2026

### System facts

- OS: `Ubuntu 22.04.5 LTS`
- Kernel: `Linux 5.2.0`
- Virtualization: `OpenVZ`
- CPU: `2 vCPU`, `Intel Xeon E5-2670 v2`
- RAM: `8.0 GiB`
- Swap: `0`
- Disk: `50G`, free about `39G`

### Installed runtime and services

- `Docker 28.3.2`
- `Docker Compose v2.38.2`
- `nginx 1.18.0`
- `certbot 1.21.0`
- Local Redis already running on `127.0.0.1:6379`
- Local PostgreSQL already running on `127.0.0.1:5432`
- No active Docker containers at the moment

### Current network usage

- Public HTTP: `80`
- SSH: `22`
- Existing Node apps listen on:
  - `4000`
  - `4001`
  - `4002`
  - `4173`
  - `5858`
  - `13714`

### Missing pieces for NLP deployment

- `pip` is not installed for system Python
- `onnxruntime` is not installed
- `tesseract` is not installed
- `pdftotext` is not installed
- No GPU or CUDA/TensorRT path is available

### Consequences

- The inference stack must be CPU-first.
- ONNX Runtime should use `CPUExecutionProvider`.
- A local LLM on this VPS is a bad fit; use an external API for the LLM stage if needed.
- OCR should be deterministic and lightweight.

## 5. Deployment Architecture For This VPS

### Recommended layout

- Public Nginx vhost
- Frontend static build served by Nginx
- FastAPI service bound to `127.0.0.1:8010`
- Optional OCR worker bound to `127.0.0.1:8011` only if the OCR stage needs isolation
- Redis reuse from host
- PostgreSQL reuse from host only if feedback history is needed

### Why not a local hosted LLM

- `2 vCPU / 8 GB RAM / no GPU` is enough for OCR + ONNX inference, but not for robust local LLM inference with good latency.
- The stable architecture here is:
  - OCR locally
  - rules and ONNX locally
  - LLM only as an external normalization/validation step

### Recommended deployment mode

- Backend in Docker
- Frontend as static assets
- Nginx on host

This keeps Python/OCR/ONNX dependencies isolated without forcing the whole stack into containers.

## 6. OCR -> LLM Pipeline

### Production-friendly pipeline

1. Input router
   - if PDF has embedded text, extract text first
   - if image or scanned PDF, send to OCR
2. OCR preprocessing
   - grayscale
   - resize
   - deskew
   - thresholding
3. OCR stage
   - first choice for MVP on this VPS: `Tesseract`
   - fallback option if quality is not enough: `PaddleOCR` or `RapidOCR`, but only after measuring CPU latency
4. Line normalization
   - merge wrapped lines
   - normalize spaces, currency signs, separators
   - detect candidate menu rows
5. Structured extraction
   - regex and deterministic parsers for price/size
   - BIO2 NER model for ambiguous fields
6. Category prediction
   - ONNX classifier
7. Guarded LLM stage
   - only for normalization, repair, or conflict resolution
   - must output strict JSON schema
   - should run only when confidence is low or OCR output is noisy
8. Validation layer
   - reject impossible prices
   - reject malformed units
   - map unknown categories to `other`
9. Feedback loop
   - user edits incorrect fields in UI
   - corrections are stored as new gold data

### Important design rule

- Do not let the LLM generate the final menu freely.
- Use OCR and deterministic parsers for raw extraction.
- Use the LLM only to normalize and validate under schema constraints.

## 7. Dataset Plan

### Recommended dataset strategy

- Collect public menu pages and public menu PDFs/images from restaurant websites.
- Store raw source links and timestamps.
- Convert all examples into a unified item-level dataset.

### Annotation format

- Source of truth: `JSONL`
- Derived export for NER: `CoNLL/BIO2`
- Derived export for classification: `CSV` or `TSV`

### JSON item schema

```json
{
  "id": "item_000123",
  "source_id": "menu_000017",
  "text": "Цезарь с курицей 250 г — 390 ₽",
  "category": "salads",
  "slots": {
    "name": "Цезарь с курицей",
    "description": null,
    "prices": [{"value": 390, "currency": "RUB", "raw": "390 ₽"}],
    "sizes": [{"value": 250, "unit": "g", "raw": "250 г"}],
    "allergens": []
  }
}
```

### Initial label set

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

### BIO2 schema

- `B-NAME`, `I-NAME`
- `B-DESC`, `I-DESC`
- `B-PRICE`, `I-PRICE`
- `B-SIZE`, `I-SIZE`
- `B-ALLERGEN`, `I-ALLERGEN`
- `O`

## 8. Modeling Plan

### Baselines

- Regex keyword classifier
- `TF-IDF + Logistic Regression`
- `TF-IDF + Linear SVM`
- `fastText`
- Regex parser for price/weight extraction

### Main model

- Transformer encoder fine-tuned for short text classification
- Export best classifier to ONNX
- NER model:
  - either transformer token classifier
  - or hybrid: regex first, NER only for unresolved fields

### Metrics

- Classification:
  - `Accuracy`
  - `Macro-F1`
- Extraction:
  - `entity-level Precision`
  - `entity-level Recall`
  - `entity-level F1`
- Service:
  - latency per item
  - latency per document

## 9. Web App Scope

### MVP screens

- Upload or paste menu
- Parsed line preview
- Category view with grouped items
- Inline editor for category and extracted attributes
- Export to JSON and CSV

### Useful API endpoints

- `POST /api/parse-text`
- `POST /api/parse-file`
- `POST /api/ocr`
- `POST /api/feedback`
- `GET /api/health`
- `GET /api/version`

## 10. Repository Skeleton

```text
frontend/
backend/
  app/
  services/
  schemas/
  models/
nlp/
  data/
  notebooks/
  training/
  export/
scripts/
report/
docker/
```

### Minimum repo deliverables

- `README.md`
- local run instructions
- training instructions
- inference instructions
- report source
- final PDF

## 11. Report Skeleton

### Abstract

- problem
- dataset
- method
- main results
- GitHub link

### Introduction

- why menu structuring matters
- product goal
- main contributions

### Related Work

- short text classification
- food/menu categorization
- information extraction from semi-structured text
- mandatory comparison table with prior approaches

### Model Description

- pipeline diagram
- classifier description
- extraction description
- JSON aggregation step
- confidence thresholding

### Dataset

- source collection process
- annotation rules
- licensing/data availability
- train/valid/test split
- statistics table

### Experiments

- metrics
- setup
- baselines
- hyperparameters

### Results

- main comparison table
- confusion matrix
- error analysis
- output examples

### Conclusion

- what was built
- what quality was achieved
- next steps

## 12. Timeline To Deadline

Deadline: `May 30, 2026 23:59 MSK`

### Phase 1. March 9 to March 16

- finalize task scope
- create repo skeleton
- define labels and annotation guide
- prepare raw data collection scripts

### Phase 2. March 17 to March 31

- collect first dataset version
- annotate at least `800-1500` item rows
- build regex and TF-IDF baselines
- prepare first report draft for `Dataset` and `Related Work`

### Phase 3. April 1 to April 20

- train strong classifier baseline
- add extraction pipeline
- run error analysis
- decide final label set

### Phase 4. April 21 to May 10

- export best model to ONNX
- implement FastAPI inference service
- implement Vue frontend
- add feedback loop

### Phase 5. May 11 to May 20

- deploy on VPS
- connect Nginx
- validate OCR and parsing flow
- collect demo examples and screenshots

### Phase 6. May 21 to May 30

- freeze experiments
- fill results tables
- write final report
- produce PDF and final repo cleanup

## 13. Next Practical Steps

- Create `README.md` and repo structure.
- Write annotation guide with examples and edge cases.
- Collect the first `200-300` menu rows manually to validate labels before large-scale scraping.
- Implement baselines before any transformer training.
- Decide whether allergens are in MVP or postponed to phase 2.
- Choose the public domain or subdomain for deployment.
- Prepare server packages:
  - `python3-pip`
  - `tesseract-ocr`
  - `poppler-utils`
  - Docker image for FastAPI + ONNX Runtime

## 14. Open Questions

- Which public domain or subdomain will expose the service?
- Do you want OCR only for Russian menus, or Russian + English?
- Are allergens part of the first scoring target, or only a product bonus?
- Will the LLM step use an external provider or an existing OpenAI-compatible endpoint you already control?
