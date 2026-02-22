# kurome Migration Map

This document maps current files/functions to their target module locations and defines incremental slices to complete the migration.

## 1. Current -> Target File Mapping

| Current file | Current role | Target module |
|---|---|---|
| `train.py` | embeddings/e2e training script with setup + loop | `kurome/cli/train_embeddings.py` + `kurome/training/*` + `kurome/models/factory.py` |
| `train_features.py` | feature-sequence training script with setup + loop | `kurome/cli/train_features.py` + `kurome/training/*` + `kurome/models/factory.py` |
| `inference.py` | pipelines + helper functions + model loading | `kurome/cli/infer.py` + `kurome/inference/pipeline.py` + `kurome/inference/postprocess.py` |
| `utils.py` | mixed concerns: config, validation, wrapper, state loading | split into `config/loader.py`, `training/checkpoint.py`, `training/metrics.py`, `utils/*` |
| `dataset.py` | embedding dataset + validation subset | `kurome/data/embeddings.py` |
| `sequence_dataset.py` | sequence dataset + collate + validation subset | `kurome/data/sequences.py` |
| `image_dataset.py` | image dataset + collate helpers | `kurome/data/images.py` |
| `model.py` | embedding predictor head | `kurome/models/heads/predictor.py` |
| `head_model.py` | sequence head architecture | `kurome/models/heads/sequence_head.py` |
| `hybrid_model.py` | hybrid head architecture | `kurome/models/heads/hybrid_head.py` |
| `model_early_extract.py` | end-to-end model wrapper | `kurome/models/backbones/early_extract.py` or `models/assemblies/early_extract.py` |
| `losses.py` | GHMC + Focal loss | `kurome/models/tasks/losses.py` |
| `generate_embeddings.py` | removed from root surface | `kurome/cli/generate_embeddings.py` + `kurome/inference/pipeline.py` shared preprocess/model init |
| `generate_feature_sequences.py` | removed from root surface | `kurome/cli/generate_feature_sequences.py` + shared model/proc helpers |
| `demo_folder.py` | removed from root surface | `kurome/cli/infer_folder.py` |
| `demo_class_gradio.py` | removed legacy UI script | n/a |
| `demo_score_gradio.py` | removed legacy UI script | n/a |
| `optimizer/` | custom optimizer registry implementation | keep under `optimizer/`; accessed through `training/optim.py` adapter |

## 2. Function-Level Migration Targets

## 2.1 Training bootstrap/setup
- Current:
  - `train.py`: `setup_dataloaders`, `setup_model_criterion`, `setup_optimizer_scheduler`, `load_checkpoint`, `train_loop`
  - `train_features.py`: same shape
- Target:
  - `kurome/training/bootstrap.py` (already started)
  - `kurome/training/optim.py`
  - `kurome/training/checkpoint.py`
  - `kurome/training/engine.py`

## 2.2 Validation and logging helpers
- Current:
  - `utils.py`: `run_validation_embeddings`, `run_validation_sequences`, `ModelWrapper`
- Target:
  - `kurome/training/metrics.py`
  - `kurome/training/engine.py` (or callback submodule)
  - `kurome/utils/logging.py`

## 2.3 Config handling
- Current:
  - `utils.py`: `parse_and_load_args`, `write_config`, `get_embed_params`
- Target:
  - `kurome/config/loader.py`
  - `kurome/config/schema.py`
  - `kurome/config/serialization.py` (optional)

## 2.4 Inference internals
- Current:
  - `inference.py`: preprocess helpers + `BasePipeline` + many derived pipelines
- Target:
  - `kurome/inference/pipeline.py` (core classes)
  - `kurome/inference/postprocess.py`
  - `kurome/inference/preprocess.py`

## 2.5 Model and task definitions
- Current:
  - `model.py`, `head_model.py`, `hybrid_model.py`, `model_early_extract.py`, `losses.py`
- Target:
  - `kurome/models/heads/*`
  - `kurome/models/backbones/*`
  - `kurome/models/tasks/*`
  - `kurome/models/registry.py`
  - `kurome/models/factory.py`

## 3. Ordered Migration Slices

## Slice A: Docs + baseline checks (Phase 0)
- Add architecture/migration docs and smoke command checklist.
- No behavior changes.

## Slice B: CLI package wrappers (Phase 1)
- Create package CLI modules mirroring current root script entrypoints.
- Root scripts call package CLIs.

## Slice C: Model factory skeleton (Phase 3, scaffold)
- Introduce registries and factory interfaces.
- Register current heads/loss adapters first.
- Keep old branches as fallback until parity proven.

## Slice D: Config normalization (Phase 2)
- Implement typed config schema and loader.
- Route both training scripts through normalized config.

## Slice E: Data adapter extraction (Phase 4)
- Move dataset/collate/dataloader setup from scripts into `kurome/data/`.
- Keep wrapper functions in scripts temporarily forwarding to new modules.

## Slice F: Engine extraction (Phase 5)
- Move shared loop mechanics, checkpoint IO, optimizer/scheduler setup to `kurome/training/`.
- Keep mode-specific hooks in adapters.

## Slice G: Inference extraction (Phase 6)
- Move pipeline internals out of root `inference.py`.
- Root file becomes compatibility wrapper.

## Slice H: Tests and quality gates (Phase 7)
- Add focused unit/integration tests around moved components.
- Add lint/type/test tasks in `pyproject.toml`.

## Slice I: Surface cleanup and deprecation (Phase 8)
- Reduce root script surface.
- Add contributor docs for adding models/datasets.

## 4. Suggested New Modules (Concrete Paths)
```text
kurome/
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
    dataloaders.py
  models/
    registry.py
    factory.py
    backbones/
      aimv2.py
      siglip.py
      dino.py
      early_extract.py
    heads/
      predictor.py
      sequence_head.py
      hybrid_head.py
    tasks/
      classification.py
      scoring.py
      losses.py
  training/
    bootstrap.py
    optim.py
    checkpoint.py
    engine.py
    metrics.py
  inference/
    preprocess.py
    pipeline.py
    postprocess.py
```

## 5. Risks and Mitigations
1. Behavior drift in target shaping/loss handling.
   - Mitigation: add per-task adapter tests and smoke compare logs.
2. Checkpoint resume incompatibilities.
   - Mitigation: preserve checkpoint artifact naming and load order.
3. Hidden coupling in `utils.py`.
   - Mitigation: split with temporary compatibility imports and deprecate gradually.
4. Feature/image mode divergence in loops.
   - Mitigation: common engine + explicit mode hooks.

## 6. Smoke Checklist (Current Commands)
Use these as regression checks after each slice affecting the path.

1. Embeddings training path
- `python train.py --config config/anatomy_dinov2.yaml --help` (arg parsing sanity)

2. Feature training path
- `python train_features.py --config config/anatomy_so400.yaml --help` (arg parsing sanity)

3. Inference path
- `python inference.py --help` (if CLI supports it) or minimal import/constructor smoke

4. Compile checks
- `python3 -m py_compile` on touched files

## 7. Exit Criteria for Migration Completion
Migration is complete when:
1. Root scripts are thin compatibility wrappers.
2. Config, data, model factory, training engine, and inference pipeline live under `kurome/`.
3. Registry-based model support is in use for current supported model families.
4. Smoke checks and tests pass from package CLIs.
5. Contributor docs exist for model and dataset extension.
