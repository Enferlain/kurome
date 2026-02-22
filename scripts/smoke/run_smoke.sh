#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-wsl/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

PYTEST_CACHE_DIR="${PYTEST_CACHE_DIR:-/tmp/kurome-pytest-cache}"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest tests/smoke -m smoke -q -s -o "cache_dir=$PYTEST_CACHE_DIR"
