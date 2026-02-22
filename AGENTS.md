# AGENTS

This file is for automated agents and maintainers working in this repository.

## Purpose

Keep the root `README.md` user-facing.
Put agent workflow, policy, and maintenance conventions here.

## Primary Entry Points

Use package CLIs first:

- `python -m kurome.cli.generate_embeddings`
- `python -m kurome.cli.generate_feature_sequences`
- `python -m kurome.cli.train_embeddings`
- `python -m kurome.cli.train_features`
- `python -m kurome.cli.infer`
- `python -m kurome.cli.infer_folder`

Root launcher is available when needed:

- `python launch.py train-embeddings -- ...`
- `python launch.py train-features -- ...`
- `python launch.py build-embeddings -- ...`
- `python launch.py build-features -- ...`
- `python launch.py infer-folder -- ...`

## Quality Gate

Run before finalizing refactor changes:

```bash
PYTHON_BIN=.venv-wsl/bin/python scripts/quality/run_quality.sh
```

Notes:

- Type-check diagnostics are warnings by default in `ty` config.
- Import warnings usually indicate missing environment deps, not necessarily code regressions.

## Root Surface Policy

Root tracked files are enforced by:

- `scripts/quality/check_root_surface.py`
- `scripts/quality/root_surface_allowlist.txt`

If you intentionally add a new root file, update the allowlist in the same change.

## Docs Map

Use docs by intent:

- `docs/README.md` for docs navigation
- `docs/how-it-works-now.md` for current runtime behavior
- `docs/architecture.md` for target architecture
- `docs/migration-map.md` for old-to-new mapping
- `docs/deprecations.md` for removals and policy decisions
- `docs/wrapper-removal-checklist.md` for wrapper cleanup

## Refactor Guardrails

- Prefer direct refactors over compatibility wrappers.
- Keep command surfaces package-first (`kurome.cli.*`).
- When removing legacy scripts, update docs references in the same pass.
