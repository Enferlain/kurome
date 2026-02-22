# How It Works Now

This document describes the current runtime flow after the refactor and where to extend behavior.
For action-oriented navigation by topic, see `docs/README.md`.

## 1. Entrypoints
Main commands:

1. `python launch.py <task> -- <task-specific-args>` (root launcher for common workflows)
2. `python -m kurome.cli.train_embeddings --config <config.yaml>`
3. `python -m kurome.cli.train_features --config <config.yaml>`
4. Dataset generation CLIs:
`python -m kurome.cli.generate_embeddings ...`, `python -m kurome.cli.generate_feature_sequences ...`
5. Folder inference CLI:
`python -m kurome.cli.infer_folder ...`

Root wrappers are removed. Package CLIs are the canonical path.

## 2. Config Flow

1. CLI reads config path from `--config`.
2. `kurome.config.loader.load_experiment_config(...)` loads YAML.
3. Loader normalizes legacy and modern YAML shapes into `ExperimentConfig`.
4. CLI stores normalized runtime args and writes run config snapshots via `kurome.config.runtime_args.write_config`.

Primary config modules:

1. `kurome/config/schema.py`
2. `kurome/config/loader.py`
3. `kurome/config/runtime_args.py`
4. `kurome/config/embed_params.py`

## 3. Training Flow

### 3.1 Bootstrap

1. Runtime/device/precision setup from `kurome.training.bootstrap`.
2. Optional optimizer/scheduler registry loading.
3. Optional checkpoint restore of model, optimizer, scheduler, scaler, and step state.

### 3.2 Model + Criterion Construction

1. Registry and factory logic in:
`kurome.models.registry`, `kurome.models.factory`.
2. Heads and task losses live under:
`kurome.models.heads.*`, `kurome.models.tasks.losses`.
3. For embeddings/features mode, CLI selects and instantiates model variants via factory.

### 3.3 Data Pipeline

1. Dataloader builders:
`kurome.data.embeddings`, `kurome.data.sequences`, `kurome.data.images`.
2. Shared loader helpers:
`kurome.data.dataloaders`.
3. Dataset implementations:
`kurome.data.datasets.*`.
4. Shared batch contracts:
`kurome.data.contracts`.

### 3.4 Train Loop

1. Loop implementations:
`kurome.training.loops`.
2. Shared step helpers:
`kurome.training.engine`.
3. Validation helpers:
`kurome.training.validation`.
4. Metrics helpers:
`kurome.training.metrics`.
5. Checkpoint helpers:
`kurome.training.checkpoint`, `kurome.training.state_io`.

## 4. Inference Flow

Inference core lives in:

1. `kurome.inference.pipeline`
2. `kurome.inference.postprocess`

Training and inference share model-head implementations from `kurome.models.*`.

## 5. Extension Points

### 5.1 Add a Model

1. Implement module under `kurome/models/backbones`, `kurome/models/heads`, or `kurome/models/tasks`.
2. Register it in `kurome/models/registry.py`.
3. Reference it from config (`model.model_id` or related model params).

See `docs/how-to-add-model.md`.

### 5.2 Add a Dataset/Loader

1. Add dataset/collate under `kurome/data/datasets`.
2. Wire a loader builder in `kurome/data/*.py`.
3. Keep batch contract consistent with `kurome/data/contracts.py`.

See `docs/how-to-add-dataset.md`.

## 6. Quality and Type Check Baseline

Primary quality command:

1. `scripts/quality/run_quality.sh`

Current `ty` gate is intentionally scoped to refactored core modules used by the package CLIs.
Legacy, optional-dependency-heavy, and exploratory surfaces are not part of the required type-check baseline yet.
