# Root Surface Contract

This document defines the intended repository-root footprint after wrapper and legacy-module cleanup.

## Why this exists
- Prevent accidental growth of root-level scripts.
- Keep implementation code in package modules (`kurome/*`).
- Make any root-level exceptions explicit and reviewable.

## Root categories
1. Metadata/tooling:
   - `.gitignore`
   - `CHANGELOG.md`
   - `LICENSE`
   - `README.md`
   - `pyproject.toml`
   - `uv.lock`
2. Root launcher:
   - `launch.py` (dispatches to package CLIs under `kurome/cli/*`)

## Enforcement
Root surface is checked by:
- `scripts/quality/check_root_surface.py`
- allowlist: `scripts/quality/root_surface_allowlist.txt`

Quality gate integration:
- `scripts/quality/run_quality.sh` runs the root-surface check.

If you intentionally add/remove a tracked root file, update the allowlist in the same change.
