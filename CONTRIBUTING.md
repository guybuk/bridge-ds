# Contributing

Thanks for your interest in `bridge-ds`! This document captures the
conventions we ask contributors to follow.

## Test conventions

The test suite is split by extras matrix:

- `tests/core/` runs against the **base install** (no optional extras).
- `tests/vision/` and `tests/multimodal/` may rely on the `vision` extra.

Tests under `tests/core/` must run in a no-extras (Python-only) venv. They
must not exercise encodings that lazy-import optional deps:

- **Allowed encodings**: `pickle`, `utf8`, `npy`
- **Forbidden encodings**: `jpeg`, `pt`, anything else that requires the
  `vision` extra (or any other optional dependency)

Local dev typically runs `uv sync --extra vision`, so a `tests/core/` test
that accidentally exercises a vision-extra import will pass locally but
fail on CI's core matrix. To catch this before pushing, run the
no-extras safety hook:

```bash
bash scripts/check_core_no_extras.sh
```

The script creates a throwaway venv with only the base project
dependencies + pytest, then runs `pytest tests/core` against it. Run it
before pushing a PR that touches `bridge/` or `tests/core/`.
