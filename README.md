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

### 1. Preferred Data Workflow

The primary data workflow is now:

1. build one sample manifest
2. attach derived artifacts to the same manifest
3. train from the manifest using either image, tensor, or feature mode

Use `prepare-data` as the main entrypoint:

```bash
python launch.py prepare-data -- \
  --src /abs/path/to/dataset \
  --manifest /abs/path/to/dataset/manifest.jsonl \
  --class-names normal_png normal_jpg adv_png adv_jpg \
  --split-mode random \
  --val-ratio 0.1 \
  --forensic-preset preset1 \
  --forensic-preset preset2 \
  --feature-model facebook/dinov2-giant
```

This writes one JSONL manifest where each row is a sample and optional derived data is attached under `artifacts`.

Example shape:

```jsonl
{"sample_id":"...","source_image_path":"images/post_0001.png","label":"adv_png","split":"train","metadata":{"width":1024,"height":1024},"artifacts":{"preset1":{"path":"../data/forensic/post_0001.preset1.npz","artifact_type":"forensic_tensor_npz"},"facebook_dinov2_giant_SeqFeatures_fp16":{"path":"../data/facebook_dinov2_giant_SeqFeatures_fp16/adv_png/post_0001.npz","artifact_type":"feature_sequence_npz","npz_key":"sequence"}}}
```

What is now redundant for most users:

- running `build_image_manifest`, `build_forensic_tensors`, and `generate_feature_sequences` manually in sequence
- keeping separate bookkeeping for images, forensic tensors, and feature-sequence folders

Those low-level scripts still exist, but they are now artifact builders under the same manifest system, not separate data systems.

### 2. Low-Level Builders

Use these only when you want finer control than `prepare-data`.

#### Generate Embeddings

```bash
python -m kurome.cli.generate_embeddings \
  --image_dir path/to/images \
  --output_dir_root data \
  --model_name google/siglip-so400m-patch14-384 \
  --preprocess_mode fit_pad
```

#### Generate Feature Sequences

```bash
python -m kurome.cli.generate_feature_sequences \
  --image_dir path/to/images \
  --output_dir_root data \
  --model_name apple/aimv2-large-patch14-224-way-2b \
  --save_precision fp16
```

#### Build Image Manifest

```bash
python -m kurome.cli.build_image_manifest \
  --src /abs/path/to/dataset \
  --output /abs/path/to/manifest.jsonl \
  --class-names normal_png normal_jpg adv_png adv_jpg \
  --limit-per-class 5000 \
  --split-mode random \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

The manifest builder writes JSONL and includes rich per-image metadata such as dimensions, file size, hash, label index, split, and optional merged `metadata.jsonl` sidecar data.

Supported folder layouts:

- `class/image.ext`
- `split/class/image.ext`

If `metadata.jsonl` files exist underneath `--src`, the builder will auto-discover them and merge matching records by post id or filename stem/basename. Use `--no-auto-metadata` to disable that behavior.

### 3. Train

Training now supports a shared manifest-first workflow across modalities:

- `data.mode: images`
  Reads `source_image_path` from the manifest.
- `data.mode: tensors`
  Reads `artifacts.<artifact_key>` pointing to spatial tensor `.npz` files.
- `data.mode: features`
  Reads `artifacts.<artifact_key>` pointing to sequence feature `.npz` files.

```bash
python -m kurome.cli.train_features --config config/your_config.yaml
python -m kurome.cli.train_embeddings --config config/your_config.yaml
python launch.py train-tensors -- --config config/your_tensor_config.yaml
```

For end-to-end image training, `data_root` can now contain either numeric or named class folders. You can also point image mode at a manifest file with `data.manifest_path`.

Image folder example:

```text
dataset/
├── normal_png/
├── normal_jpg/
├── adv_png/
└── adv_jpg/
```

Image manifest example:

```jsonl
{"path":"images/post_0001.png","label":"adv_png","split":"train","width":1024,"height":1024}
{"path":"images/post_0002.jpg","label":"normal_jpg","split":"val","width":1216,"height":832}
```

Config example:

```yaml
data:
  mode: images
  data_root: /abs/path/to/dataset
  manifest_path: /abs/path/to/manifest.jsonl
  class_names: [normal_png, normal_jpg, adv_png, adv_jpg]
```

Tensor config example:

```yaml
data:
  mode: tensors
  manifest_path: /abs/path/to/manifest.jsonl
  artifact_key: preset1
  class_names: [normal_png, normal_jpg, adv_png, adv_jpg]
```

Feature manifest config example:

```yaml
data:
  mode: features
  manifest_path: /abs/path/to/manifest.jsonl
  artifact_key: facebook_dinov2_giant_SeqFeatures_fp16
  class_names: [normal_png, normal_jpg, adv_png, adv_jpg]
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
python launch.py prepare-data -- --src path/to/images --manifest manifest.jsonl
python launch.py train-embeddings -- --config config/your_config.yaml
python launch.py train-tensors -- --config config/your_tensor_config.yaml
python launch.py train-features -- --config config/your_config.yaml
python launch.py build-manifest -- --src path/to/images --output manifest.jsonl
python launch.py build-embeddings -- --image_dir path/to/images --output_dir_root data
python launch.py build-features -- --image_dir path/to/images --output_dir_root data
python launch.py build-forensic-tensors -- --manifest manifest.jsonl --preset preset1 --output-root data/forensic
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
