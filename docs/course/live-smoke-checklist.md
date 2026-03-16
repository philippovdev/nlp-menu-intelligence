# Live Smoke Checklist

Use this checklist before final submission and after any production-affecting
backend or frontend change.

## Backend Health

- open `https://nlp.philippov.dev/api/v1/health`
- confirm HTTP `200`
- confirm response is `{"status":"ok"}` or the current expected health payload

- open `https://nlp.philippov.dev/api/v1/version`
- confirm HTTP `200`
- confirm the payload includes:
  - `version`
  - `category_model`
  - `configured_category_model`
  - `category_model_ready`

## Text Flow

- open `https://nlp.philippov.dev`
- paste a compact menu sample such as:

```text
SALADS
Caesar Salad 210 g - 340 RUB
Greek Salad 180 g - 290 RUB

SOUPS
Tom Yum 300 ml - 430 RUB

DRINKS
Americano 300 ml - 180 RUB
Lemonade 330 ml - 250 RUB
```

- click `Parse`
- confirm:
  - grouped review renders
  - category labels look sensible
  - name, price, and size fields are usable
  - confidence and issues are visible but not broken

## Text Edit And Export

- edit one category label
- edit one item name
- export JSON
- export CSV
- confirm:
  - both downloads complete
  - edited values appear in exported files
  - the page remains stable after export

## PDF Flow

- upload one embedded-text PDF menu
- click `Parse file`
- confirm:
  - parsing succeeds
  - the same review screen is used
  - file metadata is visible
  - the extracted items are usable without obvious contract breakage

## Image Flow

- upload one menu image
- click `Parse file`
- confirm:
  - parsing succeeds
  - OCR-backed output is returned
  - at least part of the extracted text and item structure is usable
  - the page remains stable

## Final Decision

Mark each flow:

- text flow usable: `yes/no`
- PDF flow usable: `yes/no`
- image flow usable: `yes/no`
- export usable: `yes/no`
- version metadata usable for debugging: `yes/no`

If any item is `no`, record:

- failing input
- observed behavior
- whether the failure is in OCR, categorization, extraction, or frontend review
