# Data Notes

This repository keeps the small, reproducible data assets needed for
experiments, tests, and report figures.

Commit here:

- small public fixtures for tests or examples
- manifests with source URLs and metadata
- annotation schemas
- small gold JSONL examples

Do not commit:

- large raw crawls
- private or unclear-license menu files
- generated caches that can be reproduced

Layout:

```text
data/
  raw/
  interim/
  annotated/
  external/
```
