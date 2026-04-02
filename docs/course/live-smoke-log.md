# Live Smoke Log

This file records the final deployed smoke pass for the submission-ready
commit.

## Run Metadata

- Date: `2026-04-02`
- Deployed commit: `c0c9b1f33e9ecb5ba3b98befbfc51ce36a9feb80`
- Deploy workflow:
  `https://github.com/philippovdev/nlp-menu-intelligence/actions/runs/23890909212`
- Deployment result: `success`

## Health And Version Endpoints

- `GET /api/health`: `200`, payload saved in
  `docs/course/artifacts/live-smoke/api-health.json`
- `GET /api/v1/health`: `200`, payload saved in
  `docs/course/artifacts/live-smoke/api-v1-health.json`
- `GET /api/status`: `200`, payload saved in
  `docs/course/artifacts/live-smoke/api-status.json`
- `GET /api/version`: `200`, payload saved in
  `docs/course/artifacts/live-smoke/api-version.json`
- `GET /api/v1/version`: `200`, payload saved in
  `docs/course/artifacts/live-smoke/api-v1-version.json`
- Active live model after deploy:
  `tfidf-union-logreg-items-v2@1.1.0`

## Product Flows

- Text flow: `yes`
  Artifact: `docs/course/artifacts/live-smoke/text-parse-response.json`
  Result: grouped response with `8` items, union-logreg model metadata, and
  usable name/price/size extraction.
- PDF flow: `yes`
  Artifact: `docs/course/artifacts/live-smoke/pdf-parse-response.json`
  Result: embedded-text PDF parsed successfully with `ocr_used = false`,
  grouped review shape, and `Margherita pizza` / `Spaghetti carbonara`
  recovered as menu items.
- Image flow: `yes`
  Artifact: `docs/course/artifacts/live-smoke/image-parse-response.json`
  Result: image parsed successfully with `ocr_used = true`, grouped review
  shape, and `Carrot cake` / `Passion fruit soda` recovered as menu items.
- Export JSON from reviewed state: `yes`
  Browser capture confirmed the exported JSON contained the edited item name
  `Caesar Salad XL` on the live text flow.
- Export CSV from reviewed state: `yes`
  Browser capture confirmed the exported CSV contained the expected live image
  review rows, including `Carrot cake`, `Passion fruit soda`, and
  `CATEGORY_MODEL_LOW_CONFIDENCE`.

## UI Evidence

- Text review screenshot:
  `docs/course/artifacts/screenshots/live-text-review.png`
- Image review screenshot:
  `docs/course/artifacts/screenshots/live-image-review.png`

## Console Notes

- No blocking runtime errors were observed during the live UI smoke.
- Chrome DevTools reported one non-blocking accessibility issue class:
  form fields missing `id` or `name` attributes.

## Final Decision

- text flow usable: `yes`
- PDF flow usable: `yes`
- image flow usable: `yes`
- export usable: `yes`
- version metadata usable for debugging: `yes`
