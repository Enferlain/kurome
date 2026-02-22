# kurome Architecture (Target)

## 1. Objective
Define a modular architecture that preserves current training/inference behavior while making the codebase easier to maintain and extend.

Primary outcome: adding a new model family should not require editing core training loops.

For the current concrete runtime flow, see `docs/how-it-works-now.md`.
For action-oriented topic docs, see `docs/README.md`.

## 2. Design Constraints
1. Backward-compatible CLI commands during migration.
2. Config-driven runtime behavior remains supported.
3. Existing model checkpoints and resume flow continue to work.
4. Refactors must be done incrementally with smoke verification.

## 3. High-Level Module Layout
```text
kurome/
  cli/           # user-facing entrypoints
  config/        # schema + loading + normalization
  data/          # datasets, collators, transforms, loaders
  models/        # registries + adapters + factories
  training/      # engine, checkpoints, optimizer/scheduler setup, metrics
  inference/     # model loading + preprocessing + postprocessing pipelines
  utils/         # generic utilities (logging, seed, small shared helpers)
```

## 4. Module Responsibilities

### 4.1 `kurome.cli`
- Parse args and select mode.
- Load/validate config.
- Build runtime components through factory interfaces.
- Execute train/infer flows.

Must not:
- Contain model-specific branching logic.
- Implement core training loop internals.

### 4.2 `kurome.config`
- Read YAML config files.
- Produce a normalized typed config object.
- Validate required fields and cross-field constraints.

Must provide:
- clear errors for invalid config
- mode-specific normalization (embeddings/features/images)

### 4.3 `kurome.data`
- Dataset definitions and collation.
- Dataloader construction.
- Preprocessing transforms.

Must provide:
- stable batch contracts consumed by training/inference modules

### 4.4 `kurome.models`
- Registry-based model support.
- Adapters for backbones, heads, and task behavior.
- Factory that builds model + criterion + task adapter from config.

Must provide:
- extension path: adapter file + registry entry

### 4.5 `kurome.training`
- Runtime bootstrap (device, precision, optional patches).
- Optimizer/scheduler creation.
- Checkpoint load/save and resume state.
- Training engine loop and metrics logging hooks.

Must provide:
- mode-agnostic loop mechanics
- deterministic checkpoint/resume semantics

### 4.6 `kurome.inference`
- Reusable inference pipelines.
- Model loading + input preprocessing + output postprocessing.

Must provide:
- consistent behavior for single image and folder workflows

## 5. Contracts (Core Interfaces)

## 5.1 Config Contract
`ExperimentConfig` (normalized object) should expose at minimum:
- `mode`: embeddings | features | images
- `model`: backbone/head/task selections and params
- `data`: paths, split settings, preprocessing settings
- `train`: optimizer/scheduler/loss/precision/checkpoint settings
- `logging`: wandb and local logging options

## 5.2 Data Contract
Training batches should normalize to mode-specific shapes, but present consistent keys:
- embeddings mode: `{"emb": Tensor, "val": Tensor}`
- features mode: `{"sequence": Tensor, "mask": Tensor, "label": Tensor}`
- image mode: `{"pixel_values": Tensor, "label": Tensor}`

Task adapters handle target shaping, not engine internals.

## 5.3 Model/Task Contract
Registries expose callables that return implementations:
- `BackboneAdapter`: feature extractor + processor metadata
- `HeadBuilder`: trainable head module creation
- `TaskAdapter`: loss function + target conversion + output normalization + metrics

Factory output (single object) should include:
- `model`
- `criterion`
- `task_adapter`
- optional `processor` / preprocess info

## 5.4 Checkpoint Contract
Checkpoint system should treat these artifacts as one logical checkpoint group:
- model weights (`.safetensors`)
- optimizer state (`.optim`)
- scheduler state (`.sched`)
- scaler state (`.scaler`)
- training state (`.state`: epoch/global_step)

Resume logic must load available files safely and emit clear warnings for missing optional state.

## 6. Dependency Rules
1. `cli` may depend on all modules; no module depends on `cli`.
2. `training` depends on `models`, `data`, and `config`; not vice versa.
3. `models` may use `utils`; must not depend on `training` internals.
4. `data` must not depend on `training` or `cli`.
5. `inference` may depend on `models` and `config`, but not on training engine internals.

## 7. Current State vs Target
Current repo is script-heavy with significant duplication in training and setup code.
Initial migration already started by introducing shared bootstrap helpers in:
- `kurome/training/bootstrap.py`

Remaining work is to continue reducing legacy root compatibility modules.

## 8. Migration Policy
1. Keep package CLIs (`kurome/cli/*`) as canonical entrypoints.
2. Move one concern at a time (bootstrap, config, model factory, data, engine).
3. Run smoke checks after each slice.
4. Avoid behavior changes unless explicitly planned and documented.

## 9. Verification Gates Per Slice
Minimum checks after each migration slice:
1. `python3 -m py_compile` on touched modules.
2. Run affected smoke command(s) (train or infer).
3. Validate checkpoint load/resume still works for touched flows.
4. Compare key logs/metrics fields for obvious regressions.

## 10. Extension Workflow (Target)
To add a new model family:
1. Add backbone/head/task adapter module(s) under `kurome/models/`.
2. Register adapter IDs in `kurome/models/registry.py`.
3. Add config entry using registered IDs.
4. Run unit test(s) for registry resolution and one smoke run.

No training loop edits should be required.
