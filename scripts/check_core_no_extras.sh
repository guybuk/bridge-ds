#!/usr/bin/env bash
# Run tests/core in a fresh no-extras venv to catch cross-matrix issues.
#
# Local dev is typically `uv sync --extra vision`, so a tests/core test that
# accidentally exercises a vision-extra import will pass locally but fail on
# CI's core matrix. This script creates a throwaway venv that installs only
# the project's [project] dependencies plus pytest (no `vision` or other
# extra) and runs `pytest tests/core` against it.
#
# Usage: bash scripts/check_core_no_extras.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_VENV="$(mktemp -d)"
trap 'rm -rf "$TMP_VENV"' EXIT

uv venv "$TMP_VENV"
# Install the project (no extras) into the throwaway venv.
VIRTUAL_ENV="$TMP_VENV" uv pip install --quiet -e "$REPO_ROOT"
# Install just the test runner; do NOT pull in the dev group (which would
# bring torch/torchvision and is unnecessary for tests/core).
VIRTUAL_ENV="$TMP_VENV" uv pip install --quiet pytest pytest-mock

cd "$REPO_ROOT"
"$TMP_VENV/bin/pytest" tests/core -q "$@"
