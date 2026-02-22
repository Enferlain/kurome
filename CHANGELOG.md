# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 2026-02-15

### Added

- Package-level CLIs for training and inference:
  - `kurome/cli/train_embeddings.py`
  - `kurome//cli/train_features.py`
  - `kurome//cli/infer.py`
- Package-level config layer with typed schema and loader:
  - `kurome//config/schema.py`
  - `kurome//config/loader.py`
- Package-level model registry/factory:
  - `kurome//models/registry.py`
  - `kurome//models/factory.py`
- Package-level model adapter modules:
  - `kurome//models/backbones/*`
  - `kurome//models/heads/*`
  - `kurome/models/tasks/*`
- Package-level data-loading layer:
  - `kurome/data/__init__.py`
  - `kurome/data/contracts.py`
  - `kurome/data/dataloaders.py`
  - `kurome/data/embeddings.py`
  - `kurome/data/images.py`
  - `kurome/data/sequences.py`
  - `kurome/data/transforms.py`
- Package-level training-loop helpers:
  - `kurome/training/engine.py`
- Package-level training setup helpers:
  - `kurome/training/optim.py`
  - `kurome/training/checkpoint.py`
- Package-level metric logging helpers:
  - `kurome/training/metrics.py`
- Package-level training loop implementations:
  - `kurome/training/loops.py`
- Package-level inference module:
  - `kurome/inference/pipeline.py`
  - `kurome/inference/postprocess.py`
- Refactor docs:
  - `docs/refactor-plan.md`
  - `docs/architecture.md`
  - `docs/migration-map.md`
- Contributor extension and lifecycle docs:
  - `docs/how-to-add-model.md`
  - `docs/how-to-add-dataset.md`
  - `docs/how-it-works-now.md`
  - `docs/deprecations.md`
  - `docs/root-surface.md`
  - `docs/wrapper-removal-checklist.md`
- Topic-oriented docs index and workflow guides:
  - `docs/README.md`
  - `docs/topics/configs.md`
  - `docs/topics/datasets.md`
  - `docs/topics/models.md`
  - `docs/topics/training.md`
  - `docs/topics/inference.md`
  - `docs/topics/checkpoints.md`
- Smoke test harness and smoke tests:
  - `scripts/smoke/run_smoke.sh`
  - `tests/smoke/test_refactor_smoke.py`
- Focused unit tests for shared training helpers:
  - `tests/unit/test_training_engine.py`
  - `tests/unit/test_training_metrics.py`
  - `tests/unit/test_training_checkpoint.py`
- Minimal synthetic integration training check:
  - `tests/integration/test_training_one_step.py`
- Quality gate command:
  - `scripts/quality/run_quality.sh`
  - `scripts/quality/check_root_surface.py`
  - `scripts/quality/root_surface_allowlist.txt`
  - `scripts/quality/check_wrapper_references.py`
  - `scripts/quality/wrapper_reference_allowlist.txt`

### Changed

- Root entrypoint usage has fully moved to package CLIs (`python -m kurome.cli.*`) and package inference modules.
- Training CLIs now load a normalized experiment config via `load_experiment_config(...)`.
- `train_features` and `train_embeddings` now use registry/factory helpers for loss and model construction.
- Embedding model selection can now be configured with `model.model_id` (defaults to `hybrid_head_model` if unspecified).
- Training CLI dataloader setup now delegates to `kurome.data` modules.
- Training CLIs now share train-mode/scheduler/postfix loop helpers via `kurome.training.engine`.
- Training CLIs now share optimizer/scheduler construction and checkpoint load wrappers via `kurome.training.*`.
- Validation metric logging and periodic/best checkpoint save flow now routes through shared `kurome.training` helpers.
- Per-step prediction/loss shaping now routes through shared helpers in `kurome.training.engine`.
- Training loops now route batch/target preparation through shared `kurome.training.engine` helpers.
- Optimizer-step execution and loss-window bookkeeping now route through shared `kurome.training.engine` helpers.
- Training loops now route global-step/progress/wrapper updates through shared `kurome.training.engine.advance_global_step`.
- Training loops now route step-limit checks and periodic log/validation interval gating through shared `kurome.training.engine` helpers.
- Validation post-run train-mode restoration and last-eval-loss tracking now route through shared `kurome.training` helpers.
- Training CLIs now delegate `train_loop(...)` execution to `kurome.training.loops`.
- Model and loss imports now route through package adapter modules instead of root-level model/loss imports in registry/factory/engine paths.
- `kurome/models/heads/sequence_head.py`, `kurome/models/heads/hybrid_head.py`, and `kurome/models/backbones/early_extract.py` now contain native implementations instead of root-import adapters.
- Dataset implementations now live in `kurome/data/datasets/*`, and package data adapters import those package-local modules.
- Runtime helpers previously imported from root `utils.py` now live under package modules:
  - `kurome/config/embed_params.py`
  - `kurome/config/runtime_args.py`
  - `kurome/training/wrapper.py`
  - `kurome/training/state_io.py`
  - `kurome/training/validation.py`
- Package runtime modules (`kurome/*`) no longer import from root `utils.py`.
- Root-surface policy now treats root implementation modules as removed, retaining only metadata/tooling plus public demo/utility scripts.
- Inference pipeline model-head imports now route through `kurome.models.heads`.
- Inference output formatting now routes through `kurome.inference.postprocess` helpers shared by single-model, multi-model, and sequence pipelines.
- Config normalization now infers mode from raw config only (no runtime-args fallback coupling), including `model.is_end_to_end` inference for image mode.
- Config schema now exposes typed mode-specific sections (`predictor_params`, `head_params`, `e2e_params`) for orchestration paths.
- Training model/criterion setup paths now consume typed normalized config sections instead of ad-hoc `getattr` defaults for core model behavior.
- Image-mode dataloader setup now lives in `kurome.data.images` and embeddings loader no longer owns end-to-end image mode branching.
- Data layer now documents explicit batch-key contracts via `kurome.data.contracts`.
- End-to-end image processor loading now routes through `kurome.data.transforms.load_image_processor` instead of direct CLI import/use.
- Data adapters now share common train/validation DataLoader construction and summary logging helpers via `kurome.data.dataloaders`.
- README now documents a single quality-gate command and links contributor extension docs.
- Integration coverage now includes synthetic one-step checks for feature-sequence and image-mode batch paths.
- Embedding-loop batch preparation now accepts image-mode keys (`pixel_values`/`label`) in addition to embedding keys (`emb`/`val`).
- Embeddings CLI now forwards `is_end_to_end` into shared loop execution (`is_e2e=...`) instead of hardcoding false.
- Quality gate now enforces a tracked root-surface contract via `check_root_surface.py`.
- Deprecation docs now include explicit root-surface policy and wrapper-removal checklist timeline.
- Quality gate now enforces wrapper-command reference scope via `check_wrapper_references.py`.
- README training examples are now package-first (`python -m kurome.cli.*`).
- Quality gate type-check now targets the refactored core package surface explicitly (instead of full-repo strict checking).
- Quality and smoke pytest invocations now disable capture (`-s`) to avoid environment-specific tmpfile capture failures.
- README and docs now provide topic-based navigation for core repo workflows (configs, datasets, models, training, inference, checkpoints).
- Dataset/model extension docs now align with post-refactor package-only paths (no root legacy module guidance).

### Fixed

- Config normalization supports both new and legacy YAML shapes, with numeric coercion for string numeric values.
- Inference local model path resolution was updated to be repo-root aware from package location.
- Smoke checks cover refactor-critical modules and integration points to catch structural regressions early.
- Feature-sequence loop no longer performs duplicate best-checkpoint save checks in the same validation pass.
- Core typed modules now pass `ty` without diagnostics on the scoped quality target set.
- Validation helpers now use `torch.amp.autocast(...)` instead of deprecated `torch.cuda.amp.autocast(...)`.

### Removed

- Root compatibility wrapper scripts:
  - `train.py`
  - `train_features.py`
  - `inference.py`
- Legacy root model/dataset modules:
  - `head_model.py`
  - `hybrid_model.py`
  - `model_early_extract.py`
  - `dataset.py`
  - `sequence_dataset.py`
  - `image_dataset.py`
- Remaining legacy root helper modules:
  - `model.py`
  - `losses.py`
  - `utils.py`
