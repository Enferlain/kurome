#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-wsl/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "error: python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi
# Avoid permission issues writing bytecode/cache on shared mounts.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/kurome-pyc}"
PYTEST_CACHE_DIR="${PYTEST_CACHE_DIR:-/tmp/kurome-pytest-cache}"

VENV_BIN_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
RUFF_BIN="${RUFF_BIN:-$VENV_BIN_DIR/ruff}"
TY_BIN="${TY_BIN:-$VENV_BIN_DIR/ty}"
TYPECHECK_MODE="${TYPECHECK_MODE:-auto}" # auto|required|off

if [[ ! -x "$RUFF_BIN" ]]; then
  echo "error: ruff not found: $RUFF_BIN" >&2
  exit 1
fi

LINT_TARGETS=(
  "kurome/config"
  "kurome/data"
  "kurome/models"
  "kurome/training/state_io.py"
  "kurome/training/validation.py"
  "kurome/training/wrapper.py"
  "kurome/training/checkpoint.py"
  "kurome/training/engine.py"
  "kurome/training/loops.py"
  "kurome/training/metrics.py"
  "tests"
)

echo "[quality] lint"
"$RUFF_BIN" check "${LINT_TARGETS[@]}"

echo "[quality] root-surface"
"$PYTHON_BIN" scripts/quality/check_root_surface.py

echo "[quality] wrapper-refs"
"$PYTHON_BIN" scripts/quality/check_wrapper_references.py

echo "[quality] compile"
"$PYTHON_BIN" -m compileall -q kurome tests

if [[ "$TYPECHECK_MODE" != "off" ]]; then
  if [[ -x "$TY_BIN" ]]; then
    TY_TARGETS=(
      "kurome/config"
      "kurome/data"
      "kurome/models/factory.py"
      "kurome/models/registry.py"
      "kurome/models/heads"
      "kurome/models/tasks"
      "kurome/training/checkpoint.py"
      "kurome/training/engine.py"
      "kurome/training/loops.py"
      "kurome/training/metrics.py"
      "kurome/training/state_io.py"
      "kurome/training/validation.py"
      "kurome/training/wrapper.py"
      "kurome/cli/train_embeddings.py"
      "kurome/cli/train_features.py"
    )
    echo "[quality] type-check"
    "$TY_BIN" check "${TY_TARGETS[@]}" --output-format concise
  elif [[ "$TYPECHECK_MODE" == "required" ]]; then
    echo "error: TYPECHECK_MODE=required but ty not found: $TY_BIN" >&2
    exit 1
  else
    echo "[quality] type-check skipped (ty not installed)."
  fi
fi

echo "[quality] unit"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest tests/unit -q -s -o "cache_dir=$PYTEST_CACHE_DIR"

echo "[quality] integration"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PYTHON_BIN" -m pytest tests/integration -q -s -o "cache_dir=$PYTEST_CACHE_DIR"

echo "[quality] smoke"
scripts/smoke/run_smoke.sh

echo "[quality] all checks passed"
