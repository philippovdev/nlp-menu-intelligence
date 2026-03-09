# Data Notes

This repository should keep only lightweight, reproducible data artifacts.

Commit here:

- small public fixtures for tests or examples
- manifests with source URLs and metadata
- annotation schemas
- small gold JSONL examples

Do not commit:

- large raw crawls
- private or unclear-license menu files
- generated caches that can be reproduced

Planned layout:

```text
data/
  raw/
  interim/
  annotated/
  external/
```

Templates already included:

- `data/raw/dataset-manifest-template.csv`
