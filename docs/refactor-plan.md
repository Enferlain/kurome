# kurome Refactor Plan

## Purpose
This plan defines the target structure and an incremental migration path to move the repo from flat, oversized scripts to a modular package with clearer ownership, easier model support, and safer long-term maintenance.

## Refactor Principles
1. Preserve runtime behavior while restructuring.
2. Make changes in small slices with verification after each slice.
3. Use compatibility layers only as a temporary bridge, then remove them once package paths are stable.
4. New model support should require adapter modules + registration, not training loop edits.
5. Update `CHANGELOG.md` (`Unreleased`) at the end of each migration slice.

## Target Repository Shape
```text
kurome/
  pyproject.toml
  README.md
  config/
  kurome/
    __init__.py
    cli/
      train_embeddings.py
      train_features.py
      infer.py
    config/
      schema.py
      loader.py
    data/
      embeddings.py
      sequences.py
      images.py
      transforms.py
    models/
      registry.py
      factory.py
      backbones/
      heads/
      tasks/
    training/
      engine.py
      bootstrap.py
      optim.py
      checkpoint.py
      metrics.py
    inference/
      pipeline.py
      postprocess.py
    utils/
      logging.py
      seed.py
  tests/
    unit/
    integration/
```

## Phase 0 - Baseline Freeze and Architecture Contract
### Goals
- Capture current behavior so refactors can be validated.
- Define boundaries and ownership for each module area.

### Deliverables
- `docs/architecture.md`:
  - module boundaries
  - data/model/training/inference contracts
  - dependency direction rules
- `docs/migration-map.md`:
  - old file/function -> new module target mapping
- `scripts/smoke/` commands or documented smoke command list:
  - train embeddings smoke
  - train features smoke
  - inference smoke

### Done Criteria
- Baseline smoke commands are documented and repeatable.
- We can compare pre/post-refactor behavior with consistent command set.

## Phase 1 - Entrypoint Thinning (Compatibility First)
### Goals
- Move orchestration into package modules.
- Keep root scripts as wrappers.

### Deliverables
- `kurome/cli/train_embeddings.py`
- `kurome/cli/train_features.py`
- `kurome/cli/infer.py`
- Root wrappers:
  - `train.py`
  - `train_features.py`
  - `inference.py`

### Done Criteria
- Root commands still work unchanged.
- Most business logic is no longer in root scripts.

## Phase 2 - Unified Configuration Layer
### Goals
- One normalized config object across all modes.
- Fail fast on invalid/missing config.

### Deliverables
- `kurome/config/schema.py`:
  - typed config models
- `kurome/config/loader.py`:
  - YAML parse + defaults + normalization + validation
- Replace ad-hoc `getattr` defaulting in core paths.

### Done Criteria
- Training/inference consume a validated config object.
- Config errors are clear and early.

## Phase 3 - Model/Task Registry and Factory
### Goals
- Remove model/loss hardcoding from train scripts.
- Make model support extensible by registration.

### Deliverables
- `kurome/models/registry.py`
  - backbone/head/task registries
- `kurome/models/factory.py`
  - construct model + criterion + task adapter
- `kurome/models/backbones/*`
- `kurome/models/heads/*`
- `kurome/models/tasks/*`

### Done Criteria
- Existing supported setups are registry-driven.
- Adding a model family requires:
  1. adapter module
  2. registry entry

## Phase 4 - Data Pipeline Normalization
### Goals
- Standardize dataset interfaces and collate contracts.
- Isolate mode-specific preprocessing.

### Deliverables
- `kurome/data/embeddings.py`
- `kurome/data/sequences.py`
- `kurome/data/images.py`
- `kurome/data/transforms.py`
- common dataloader builder util

### Done Criteria
- Training engine consumes a common batch contract.
- Mode differences live in adapter modules, not engine internals.

## Phase 5 - Training Engine Modularization
### Goals
- Centralize training loop mechanics.
- Reuse checkpoint, optimizer, and logging logic.

### Deliverables
- `kurome/training/engine.py`
- `kurome/training/optim.py`
- `kurome/training/checkpoint.py`
- `kurome/training/metrics.py`
- callback hooks:
  - validation
  - logging
  - checkpointing

### Done Criteria
- Train loops are shared and mode-agnostic.
- Checkpoint/optimizer/scheduler behavior is centralized.

## Phase 6 - Inference Pipeline Refactor
### Goals
- Reuse training-time model/config construction.
- Isolate IO from inference core logic.

### Deliverables
- `kurome/inference/pipeline.py`
- `kurome/inference/postprocess.py`
- CLI wrapper in `kurome/cli/infer.py`

### Done Criteria
- Inference logic is reusable and testable.
- Folder/single-image flows share core inference path.

## Phase 7 - Tests and Quality Gates
### Goals
- Add confidence for continuous refactoring.

### Deliverables
- `tests/unit/`:
  - config parsing/validation
  - registry resolution
  - target/loss shaping
  - checkpoint IO
- `tests/integration/`:
  - one-step training smoke per mode
  - inference smoke
- `pyproject.toml` quality tasks:
  - lint
  - type-check
  - test

### Done Criteria
- A single quality command catches structural regressions.
- Core refactor risk areas have coverage.

## Phase 8 - Surface Cleanup and Contributor Docs
### Goals
- Reduce root clutter.
- Document extension workflows.

### Deliverables
- Lean root directory with package-first entrypoints and no runtime wrapper scripts.
- `docs/how-to-add-model.md`
- `docs/how-to-add-dataset.md`
- deprecation notes for legacy internals

### Done Criteria
- New contributors can extend models/datasets without touching core engine logic.
- Structure is discoverable and documented.

## Phase 9 - Legacy Root Module Elimination
### Goals
- Remove remaining runtime dependencies on root implementation modules.
- Make `kurome/*` the source of truth for models, losses, data loaders/datasets, and runtime helpers.

### Scope (current refactor targets)
- Models/losses:
  - `model.py`
  - `head_model.py`
  - `hybrid_model.py`
  - `model_early_extract.py`
  - `losses.py`
- Data/datasets:
  - `dataset.py`
  - `sequence_dataset.py`
  - `image_dataset.py`
- Runtime helper monolith:
  - `utils.py`

### Migration Process (must follow in order)
1. Slice A - Models and losses:
   - Move full implementations into:
     - `kurome/models/heads/*`
     - `kurome/models/backbones/*`
     - `kurome/models/tasks/losses.py`
   - Remove adapter-style root imports (for example `from model import PredictorModel`).
   - Update all package imports to target package-local implementations only.
2. Slice B - Dataset implementations:
   - Move dataset classes/collate functions into package data modules (or `kurome/data/datasets/*` if split).
   - Update `kurome/data/embeddings.py`, `kurome/data/sequences.py`, `kurome/data/images.py` to import package-local dataset code only.
3. Slice C - `utils.py` decomposition:
   - Split config parsing, checkpoint helpers, validation helpers, and wrapper glue into:
     - `kurome/config/*`
     - `kurome/training/*`
     - `kurome/inference/*`
     - `kurome/utils/*` (only true utilities)
   - Remove package runtime imports from root `utils.py`.
4. Slice D - Root cleanup:
   - Delete migrated root modules only after import graph confirms no package/runtime references.
   - Update root surface allowlist and docs.

### Done Criteria
- No runtime package imports from root implementation modules.
- Root contains only intended public scripts/assets and non-runtime project files.
- Quality gate and smoke tests pass with deleted legacy root implementations.

## Recommended Execution Order
1. Phase 0
2. Phase 1
3. Phase 3
4. Phase 2
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7
9. Phase 8
10. Phase 9

## Verification Strategy Per Slice
1. Move code with compatibility shim only when strictly needed.
2. Run syntax checks.
3. Run smoke command(s) for affected path.
4. Confirm no CLI or config regressions.
5. Update `CHANGELOG.md` with Added/Changed/Fixed notes.
6. Commit slice independently.

## Progress Snapshot (2026-02-16)
1. Phase 0: completed
2. Phase 1: completed
3. Phase 2: completed
4. Phase 3: completed
5. Phase 4: completed
6. Phase 5: completed
7. Phase 6: completed
8. Phase 7: in progress
9. Phase 8: completed
10. Phase 9: completed

## Completed Highlights
1. Package CLIs are now the canonical entrypoints, and legacy root wrappers have been removed.
2. Config schema/loader now includes stricter normalization with typed mode-specific sections used by training setup paths.
3. Model registry/factory now drives both training CLIs.
4. Model package adapters now expose explicit module paths under:
   - `kurome/models/backbones/*`
   - `kurome/models/heads/*`
   - `kurome/models/tasks/*`
5. Data loader setup is extracted to `kurome/data/*` for embeddings, sequences, and images.
6. Data batch contracts are now documented centrally in `kurome/data/contracts.py`.
7. End-to-end processor loading is now isolated in `kurome/data/transforms.py`.
8. Data adapters now share reusable train/validation dataloader build + summary helpers in `kurome/data/dataloaders.py`.
9. Shared training helpers now cover:
   - train-mode + scheduler + progress postfix
   - optimizer/scheduler setup
   - checkpoint load + periodic/best-save helpers
   - validation metric logging helpers
   - per-step prediction/loss shaping
   - batch/target preparation
   - global-step/progress/wrapper propagation
   - step-limit checks and periodic interval gating
   - validation loss-state updates and post-validation mode restoration
10. Training-loop implementations live in `kurome/training/loops.py`, and CLI training loops now delegate to package code.
11. Inference postprocessing is now isolated in `kurome/inference/postprocess.py`.
12. Added focused unit tests for `training.engine`, `training.metrics`, and `training.checkpoint` in `tests/unit/`.
13. Added minimal synthetic integration tests that execute real one-step forward/backward optimizer updates for:
    - embedding loop
    - feature-sequence loop
    - image-mode batch path through embedding loop
14. Added a single quality-gate command in `scripts/quality/run_quality.sh` (lint + compile + unit + integration + smoke).
15. Added contributor extension docs and deprecation notes:
    - `docs/how-to-add-model.md`
    - `docs/how-to-add-dataset.md`
    - `docs/deprecations.md`
16. Added root-surface contract enforcement and wrapper-removal planning docs:
    - `docs/root-surface.md`
    - `docs/wrapper-removal-checklist.md`
    - `scripts/quality/check_root_surface.py`
    - `scripts/quality/root_surface_allowlist.txt`
17. Added wrapper-command reference enforcement and allowlist:
    - `scripts/quality/check_wrapper_references.py`
    - `scripts/quality/wrapper_reference_allowlist.txt`
18. Refactor smoke suite is active and passing.
19. Phase 9 Slice A started:
    - `kurome/models/tasks/losses.py` now contains native loss implementations (no root `losses.py` adapter import)
    - `kurome/models/heads/predictor.py` now contains native PredictorModel implementation (no root `model.py` adapter import)
    - root-model/loss adapters were fully replaced with package-local imports
20. Phase 9 Slice A continued:
    - `kurome/models/heads/sequence_head.py` now contains native `HeadModel` implementation (no root `head_model.py` adapter import)
    - `kurome/models/heads/hybrid_head.py` now contains native `HybridHeadModel` implementation (no root `hybrid_model.py` adapter import)
    - `kurome/models/backbones/early_extract.py` now contains native `EarlyExtractAnatomyModel` implementation (no root `model_early_extract.py` adapter import)
21. Phase 9 Slice B started:
    - dataset implementations moved into package-native modules:
      - `kurome/data/datasets/embedding_dataset.py`
      - `kurome/data/datasets/sequence_dataset.py`
      - `kurome/data/datasets/image_dataset.py`
    - data adapters now import package-local dataset modules only:
      - `kurome/data/embeddings.py`
      - `kurome/data/sequences.py`
      - `kurome/data/images.py`
    - root legacy modules removed:
      - `head_model.py`
      - `hybrid_model.py`
      - `model_early_extract.py`
      - `dataset.py`
      - `sequence_dataset.py`
      - `image_dataset.py`
22. Phase 9 Slice C started:
    - package runtime helpers extracted from root `utils.py` into package modules:
      - `kurome/config/embed_params.py`
      - `kurome/config/runtime_args.py`
      - `kurome/training/wrapper.py`
      - `kurome/training/state_io.py`
      - `kurome/training/validation.py`
    - package runtime imports now route through package modules only (no `from utils import ...` in `kurome/*`)
    - quality/smoke checks updated to guard this import boundary
23. Phase 9 Slice C/D completed:
    - removed remaining root legacy implementation modules:
      - `utils.py`
      - `model.py`
      - `losses.py`
    - root-surface allowlist/docs updated to only include metadata/tooling and public utility/demo scripts
    - smoke tests now assert these removed legacy helper modules stay deleted
24. Phase 7 quality baseline hardened:
    - quality gate `ty` step now targets refactored core package modules explicitly
    - `ty` noisy rules are downgraded to warnings for incremental adoption
    - quality/smoke/unit/integration pytest invocations now use `-s` to avoid environment-specific capture tmpfile failures
25. Runtime documentation refreshed:
    - added `docs/how-it-works-now.md` to describe current package-first runtime flow and extension points
    - README structure/docs links updated to reflect removed root implementation modules and current package layout

## Remaining Work Queue
1. Continue Phase 7:
   - decide when to raise `ty` strictness from warning-heavy baseline to error-gated rules
   - decide when to expand `ty` gate coverage to inference/optimizer and additional test paths
