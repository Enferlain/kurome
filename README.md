UNTESTED POST REFACTOR!!!!!!!!!!!!!!! pending more work and actual testing currently...

# kurome

`kurome` is a modular framework for training and deploying scorers and classifiers.

The current implementation is image-first (embeddings, feature sequences, and end-to-end image training), but the project direction is task-first: reusable scoring/classification workflows with configurable encoders and heads.

## Core Concepts

1. **Training on precomputed embeddings**
   A backbone model generates one vector per image, then a lightweight head is trained on those vectors. This is usually the fastest iteration loop.

2. **Training on precomputed feature sequences**
   Instead of one vector, the full token/patch sequence is saved and used by a richer head. This usually improves capacity at higher storage/IO cost.

3. **End-to-end image training**
   A vision encoder and task head are trained together (or partially fine-tuned) directly from images.

## Key Features

- Multiple backbone families via Hugging Face and `timm`
- Configurable head architectures for classification and scoring tasks
- YAML-based training configuration
- Package-first CLI surface under `kurome.cli.*`
- Folder and single-item inference utilities
- Quality gate scripts for lint/type/test/smoke checks

## Project Structure

```text
.
├── config/                  # YAML training configs
├── kurome/
│   ├── cli/                 # task entrypoints (train/generate/infer)
│   ├── config/              # schema + config loading
│   ├── data/                # datasets + dataloader builders
│   ├── models/              # model registry/factory + heads/tasks
│   ├── training/            # training engine/checkpoint/metrics
│   └── inference/           # inference pipeline/postprocess
├── docs/                    # architecture, migration, topic docs
├── scripts/quality/         # quality checks
├── scripts/smoke/           # smoke tests
└── launch.py                # root task launcher
```

## Setup

```bash
git clone https://github.com/Enferlain/kurome.git
cd kurome
uv sync
```

If you need explicit Torch/CUDA selection, install with the appropriate extras defined in `pyproject.toml`.

Example:

```bash
uv sync --extra torch-cu130 --extra torch-v210
```

## Usage

### 1. Generate Embeddings

```bash
python -m kurome.cli.generate_embeddings \
  --image_dir path/to/images \
  --output_dir_root data \
  --model_name google/siglip-so400m-patch14-384 \
  --preprocess_mode fit_pad
```

### 2. Generate Feature Sequences

```bash
python -m kurome.cli.generate_feature_sequences \
  --image_dir path/to/images \
  --output_dir_root data \
  --model_name apple/aimv2-large-patch14-224-way-2b \
  --save_precision fp16
```

### 3. Train

```bash
python -m kurome.cli.train_features --config config/your_config.yaml
python -m kurome.cli.train_embeddings --config config/your_config.yaml
```

### 4. Inference

Folder inference:

```bash
python -m kurome.cli.infer_folder \
  --src path/to/images \
  --dst output_folder \
  --model models/your_model.safetensors \
  --arch class \
  --target_label_name "Good Anatomy" \
  --copy_passed
```

Single-item inference is also available via `python -m kurome.cli.infer`.

### 5. Optional Root Launcher

`launch.py` provides a unified root entrypoint:

```bash
python launch.py train-embeddings -- --config config/your_config.yaml
python launch.py train-features -- --config config/your_config.yaml
python launch.py build-embeddings -- --image_dir path/to/images --output_dir_root data
python launch.py build-features -- --image_dir path/to/images --output_dir_root data
python launch.py infer-folder -- --src path/to/images --dst output_folder --model models/your_model.safetensors --arch class
```

## Quality Gate

Run the full gate (lint, policy checks, compile, type-check, unit, integration, smoke):

```bash
scripts/quality/run_quality.sh
```

If your Python interpreter is not the script default, set `PYTHON_BIN` explicitly.

## Documentation

- Docs index: `docs/README.md`
- Runtime overview: `docs/how-it-works-now.md`
- Architecture target: `docs/architecture.md`
- Migration map: `docs/migration-map.md`
- Deprecations: `docs/deprecations.md`

## Agent/Maintainer Notes

Agent-oriented workflow/policy notes live in `AGENTS.md`.

## Upstream Credit

This project was originally forked from `city96/CityClassifiers`:
<https://github.com/city96/CityClassifiers>
