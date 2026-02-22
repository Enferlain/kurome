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

## Mode-Critical Fields

1. `data.mode=features` requires `data.feature_dir_name`.
2. `data.mode=images` requires `head_params.output_mode`.
3. `data.mode=embeddings` requires `predictor_params.output_mode` or `head_params.output_mode`.

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
