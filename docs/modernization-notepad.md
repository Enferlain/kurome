# Kurome Modernization Notepad

Last updated: 2026-02-22
Scope: deep-dive architecture and implementation audit (beyond surface-level refactor)

## 1. Current Snapshot

- Core package layout exists (`kurome/*`) and task entrypoints are present.
- The execution paths still carry substantial legacy behavior and implicit coupling.
- Several correctness issues are mixed with modernization opportunities.

## 2. P0 Correctness Issues (Fix First)

1. Sequence preload index corruption (high risk, silent wrong-sample reads)
- Evidence: `kurome/data/datasets/sequence_dataset.py:178` filters out `None` values from `self.preloaded_data`, but later data access assumes metadata index alignment (`kurome/data/datasets/sequence_dataset.py:243`, `kurome/data/datasets/sequence_dataset.py:373`).
- Why it does not make sense: index-based lookup must remain stable; compaction breaks mapping.
- Fix direction: keep `preloaded_data` length == `len(metadata)` always. Never compact; keep `None` holes.

2. Multi-model classifier pipeline calls non-existent attribute
- Evidence: `kurome/inference/pipeline.py:1218` uses `model_head.outputs`; head models expose `num_classes`.
- Why it does not make sense: runtime path can fail despite successful model loading.
- Fix direction: use `getattr(model_head, "num_classes")` and fail explicitly if missing.

3. End-to-end validation path is a placeholder, not a real validation
- Evidence: `kurome/training/loops.py:283`-`kurome/training/loops.py:289`.
- Why it does not make sense: best-model logic and validation metrics are not meaningful in E2E mode.
- Fix direction: add real E2E validation function (image batch path), wire it exactly like embedding/sequence validation.

4. Gradient accumulation drops tail micro-batches when not divisible
- Evidence: optimizer step only happens on `(i + 1) % gradient_accumulation_steps == 0` in `kurome/training/loops.py:525`.
- Why it does not make sense: with `train_drop_last=false`, tail gradients can be ignored.
- Fix direction: flush one final optimizer step at epoch end if accumulation remainder > 0.

5. Feature config example is internally contradictory
- Evidence: `config/anatomy_aimv2_native_features.yaml:14` sets `data.mode: embeddings` while comments and keys indicate feature-sequence flow.
- Why it does not make sense: increases user error and weakens confidence in config contracts.
- Fix direction: correct template/config and add config contract tests for all shipped configs.

## 3. Architecture Gaps (P1)

1. Dual config system (typed normalization + legacy argparse namespace)
- Evidence: `kurome/config/loader.py:155`-`kurome/config/loader.py:170` writes temp YAML to feed legacy parser.
- Evidence: legacy parser has separate validation and `SystemExit` paths in `kurome/config/runtime_args.py:99`, `kurome/config/runtime_args.py:110`, `kurome/config/runtime_args.py:238`.
- Why it does not make sense: two sources of truth create drift and ambiguous behavior.
- Fix direction: replace runtime namespace parser with one typed config model + one flattening adapter.

2. CLI modules perform heavy side effects at import time
- Evidence: `kurome/cli/train_embeddings.py:57`-`kurome/cli/train_embeddings.py:64`, `kurome/cli/train_features.py:49`-`kurome/cli/train_features.py:55`.
- Why it does not make sense: importing the module mutates runtime state and loads registries before argument parsing.
- Fix direction: move side effects into `main()` startup phase only.

3. Inference and generation paths are monolithic and duplicated
- Evidence (size): `kurome/inference/pipeline.py` is 1421 lines; `kurome/cli/generate_embeddings.py` is 761 lines.
- Evidence: duplicate/overlapping preprocessing logic across inference and generation.
- Why it does not make sense: every model/preprocess addition multiplies maintenance cost.
- Fix direction: split into reusable components:
  - `vision_backends/`
  - `preprocess_strategies/`
  - `head_loaders/`
  - `predictors/`

4. Legacy control flow style (`exit`, print-debug, broad exception) still dominates
- Evidence examples: `kurome/cli/train_embeddings.py:80`, `kurome/cli/train_features.py:66`, `kurome/training/optim.py:24`.
- Why it does not make sense: makes behavior hard to test and recover.
- Fix direction: raise typed exceptions in library code; CLI translates exceptions to exit codes.

5. Optimizer registry is eagerly imported and fragile
- Evidence: `kurome/training/bootstrap.py:109` imports `optimizer` package wholesale; `optimizer/__init__.py:5`-`optimizer/__init__.py:38` eagerly imports many modules.
- Why it does not make sense: optional dependency failure can collapse the registry, silently degrading behavior.
- Fix direction: lazy registry loading by optimizer name; isolate optional extras.

6. Loss/output contract is inconsistent between model and trainer
- Evidence: model heads can apply output activations (`kurome/models/heads/predictor.py:116`, `kurome/models/heads/sequence_head.py:213`, `kurome/models/heads/hybrid_head.py:164`).
- Evidence: train_embeddings disables strict validation checks (`kurome/cli/train_embeddings.py:160`-`kurome/cli/train_embeddings.py:163`).
- Why it does not make sense: logits-based losses can accidentally be fed post-activation outputs.
- Fix direction: standardize training heads to return logits; apply activation only in inference/postprocess.

## 4. Performance Opportunities (P2)

1. Remove per-image debug prints in hot loops
- Evidence: `kurome/cli/generate_feature_sequences.py:268`-`kurome/cli/generate_feature_sequences.py:305`.
- Effect: large throughput penalty on dataset generation.

2. Eliminate duplicate preprocessing calls
- Evidence: `kurome/cli/generate_feature_sequences.py:261` and again at `kurome/cli/generate_feature_sequences.py:265`.
- Effect: unnecessary CPU/GPU overhead.

3. Prefer structured logging levels
- Current state: extensive unconditional prints across train/data/infer paths.
- Effect: overhead + poor observability control.

4. Add deterministic/performance presets
- Current state: runtime toggles are ad-hoc.
- Suggestion: explicit profiles (`dev`, `repro`, `throughput`) controlling cudnn benchmark, AMP, workers, pin_memory, compile.

## 5. Modularity Direction (Target Design)

1. One typed config pipeline
- Source model: `ExperimentConfig` + nested dataclasses (or pydantic model).
- Single normalization/validation pass.
- Explicit adapters:
  - `to_training_plan()`
  - `to_dataset_plan()`
  - `to_inference_plan()`

2. Explicit backend/plugin boundaries
- `kurome/backends/vision/{hf,timm,aimv2,dinov3}.py`
- `kurome/backends/optimizer/{torch,custom}.py`
- `kurome/preprocess/{fit_pad,center_crop,naflex,aimv2_native,...}.py`

3. Training kernel + callbacks
- Keep core loop minimal.
- Move checkpointing, logging, validation cadence, early stop into callback hooks.
- Benefits: easier correctness testing and experimentation.

4. Unified data contract by mode
- Typed batch structures per mode with runtime validators in debug mode.
- Keep collate functions simple and deterministic.

5. Inference service layer
- `HeadLoader` (checkpoint + config + schema validation)
- `FeatureExtractor` (backbone + preprocess strategy)
- `Predictor` (single model / ensemble)
- `Formatter` (score/class/sequence output adapters)

## 6. Test Strategy Upgrades

Current tests mainly exercise loop helpers and smoke structure. Add:

1. Config contract tests
- Parse every shipped YAML and assert mode-specific required keys.
- Ensure no contradictory shipped configs.

2. Data adapter tests
- Embedding, sequence, image dataset shape/dtype/label semantics.
- Preload path correctness (including failed preload entries).

3. Inference loading tests
- Predictor/Hybrid/HeadSequence loader compatibility with representative checkpoints.
- Ensemble pipeline execution path.

4. Optimizer/scheduler instantiation tests
- Parameter coercion behavior and explicit failures for invalid args.

5. E2E minimal train+val tests per mode
- embeddings, features, images (synthetic tiny datasets).

## 7. Suggested Execution Plan

Phase A (stability, 1-2 days)
- Fix P0 bugs above.
- Add regression tests for each fix.

Phase B (contract cleanup, 2-4 days)
- Unify config path (deprecate legacy parser internals).
- Remove import-time side effects in CLIs.
- Replace `exit()` in library internals with typed exceptions.

Phase C (modularization, 4-7 days)
- Split inference monolith.
- Extract shared preprocess/backbone adapters.
- Simplify generation scripts to reuse shared adapters.

Phase D (performance and UX, ongoing)
- Logging levels, profile presets, benchmark harness.
- Improve docs around mode-specific canonical workflows.

## 8. Immediate High-ROI Changes

1. Fix sequence preload indexing bug.
2. Fix multi-model pipeline `model_head.outputs` bug.
3. Implement real E2E validation.
4. Add accumulation tail flush in feature loop.
5. Remove duplicate processor call and debug spam from feature generation.
