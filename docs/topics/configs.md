# Configs

This repo is YAML-first for training configuration.

## Entry Point

1. CLI receives `--config`.
2. `kurome.config.loader.load_experiment_config(...)` loads + normalizes raw YAML.
3. The normalized object is `ExperimentConfig` (`kurome/config/schema.py`).

## Required Shape

Top-level sections expected in practice:

1. `model`
2. `data`
3. `train`
4. Optional: `predictor_params`, `head_params`, `e2e_params`

## Mode Selection

`data.mode` controls the pipeline:

1. `embeddings`: vector-per-image training path
2. `features`: sequence-of-features training path
3. `images`: end-to-end image path
4. `tensors`: spatial forensic-tensor path

## Mode-Critical Fields

1. `data.mode=features` requires either `data.feature_dir_name`, or both `data.manifest_path` and `data.artifact_key`.
2. `data.mode=tensors` requires both `data.manifest_path` and `data.artifact_key`.
3. `data.mode=images` requires `head_params.output_mode`.
4. `data.mode=embeddings` requires `predictor_params.output_mode` or `head_params.output_mode`.

Preferred modern shape:

1. Use one sample manifest as the dataset source of truth.
2. Attach derived artifacts under `artifacts.<artifact_key>`.
3. Point `features` or `tensors` configs at the manifest plus the artifact key.

The older folder-only feature-sequence setup is still supported, but it is now redundant when your feature artifacts are already attached to the manifest.

## Runtime Commands

1. Embeddings/images CLI:
```bash
python -m kurome.cli.train_embeddings --config config/your_config.yaml
```
2. Features CLI:
```bash
python -m kurome.cli.train_features --config config/your_config.yaml
```

## Related Docs

1. `docs/topics/training.md`
2. `docs/topics/checkpoints.md`
3. `docs/how-it-works-now.md`
