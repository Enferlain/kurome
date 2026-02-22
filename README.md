# kurome

A flexible and powerful framework for training and deploying high-performance image classifiers and aesthetic predictors. This project utilizes modern vision transformers and advanced training techniques to achieve ~~state-of-the-art~~ results.

#### CHECK ORIGINAL README FOR THE ORIGINAL INFO, AND THE SIGLIP AND DINO BRANCHES FOR "STABLE" CHECKPOINTS

## Core Concepts

This framework is built around a modular architecture that separates feature extraction from the final prediction task. This allows for rapid experimentation and efficient training. The primary approaches supported are:

1.  **Training on Pre-computed Embeddings:** A large, pre-trained vision model (e.g., SigLIP, DINOv2) is used to generate single-vector embeddings for an entire dataset. A small, custom "head" model is then trained on these embeddings, which is very fast and resource-efficient.
2.  **Training on Pre-computed Feature Sequences:** Instead of a single vector, the full sequence of patch embeddings is extracted from the vision model. This provides richer, spatially-aware information to a more complex head model, often leading to higher accuracy at the cost of more disk space.
3.  **End-to-End Training:** The framework also supports training directly from images, where a vision model (like Apple's AIMv2) is combined with a trainable head into a single network. This allows for fine-tuning parts of the vision model for the specific task.

## Key Features

-   **Multiple Vision Backbones:** Easily use powerful, pre-trained vision models from Hugging Face and `timm`, including **SigLIP**, **DINOv2**, and **AIMv2**.
-   **Advanced Model Heads:** A collection of highly configurable head models (`PredictorModel`, `HeadModel`, `HybridHeadModel`) featuring modern components like:
    -   Self-Attention layers
    -   Residual Blocks (ResBlocks)
    -   **RMSNorm** Layer Normalization
    -   **SwiGLU** activation functions
    -   Attention Pooling
-   **Flexible Training Modes:** Train on embeddings, feature sequences, or raw images depending on your needs.
-   **Advanced Loss Functions:** Built-in support for `FocalLoss` and `GHMC_Loss` to effectively handle class imbalance and focus on hard examples.
-   **YAML-based Configuration:** A clean and powerful configuration system using YAML files allows you to define every aspect of your training run without changing the code.
-   **Efficient Inference:** Optimized package inference pipeline (`kurome/inference/pipeline.py`) for fast predictions on single images or entire folders.
-   **Package-First CLI Surface:** Run training, dataset generation, and folder inference from `kurome/cli/*` (or `launch.py`).
-   **Custom Optimizers & Schedulers:** The framework is extensible with a variety of custom optimizers (`AdamW`, `Lion`, `Sophia`, etc.) and learning rate schedulers.
-   **Weights & Biases Integration:** Log metrics, configurations, and training progress automatically to your W&B dashboard.

## Project Structure

```
.
├── config/                         # YAML configuration files for training runs
├── kurome/
│   ├── cli/                        # Package CLIs for training and inference
│   ├── config/                     # Typed schema + config loading/normalization
│   ├── data/                       # Dataset modules + dataloader builders
│   ├── models/                     # Registry/factory + backbones/heads/tasks
│   ├── training/                   # Engine/loops/bootstrap/checkpoint/metrics
│   └── inference/                  # Shared inference pipeline/postprocessing
├── optimizer/                      # Custom optimizer and scheduler implementations
├── docs/                           # Refactor docs, extension guides, architecture
├── tests/                          # Unit/integration/smoke suites
├── scripts/quality/                # Quality gate and root-surface checks
├── scripts/smoke/                  # Smoke command entrypoint
└── launch.py                       # Root task launcher (delegates to package CLIs)
```

## Usage

### 1. Setup

First, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Enferlain/kurome.git
cd kurome
uv venv .venv --python 3.13.5
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 torch==2.10.*
uv sync --python .venv/bin/python
```

### 1.1 Quality Gate

Run the refactor quality gate (lint + compile + unit + integration + smoke):

```bash
scripts/quality/run_quality.sh
```

Optional type-check behavior:
- `TYPECHECK_MODE=auto` (default): run `ty` only if installed.
- `TYPECHECK_MODE=required`: fail if `ty` is unavailable.
- `TYPECHECK_MODE=off`: skip `ty`.

### 1.2 Docs By Topic

- Docs index: `docs/README.md`
- Configs: `docs/topics/configs.md`
- Datasets: `docs/topics/datasets.md`
- Models: `docs/topics/models.md`
- Training: `docs/topics/training.md`
- Inference: `docs/topics/inference.md`
- Checkpoints: `docs/topics/checkpoints.md`
- Implementation checklists:
  - `docs/how-to-add-model.md`
  - `docs/how-to-add-dataset.md`

### 2. Data Preparation (Optional but Recommended)

For most use cases, you'll pre-compute features from your image dataset. Your images should be organized into class-based subfolders (e.g., `data/my_dataset/0/`, `data/my_dataset/1/`).

**Option A: Generate Single-Vector Embeddings**

Use `kurome.cli.generate_embeddings` to create embeddings. This is fast and uses less disk space.

```bash
python -m kurome.cli.generate_embeddings \
  --image_dir path/to/your/images \
  --output_dir_root data \
  --model_name google/siglip-so400m-patch14-384 \
  --preprocess_mode fit_pad
```

**Option B: Generate Feature Sequences**

Use `kurome.cli.generate_feature_sequences` for richer features. This can lead to higher accuracy but requires more disk space.

```bash
python -m kurome.cli.generate_feature_sequences \
  --image_dir path/to/your/images \
  --output_dir_root data \
  --model_name apple/aimv2-large-patch14-224-way-2b \
  --save_precision fp16
```

### 3. Training

Training is controlled via YAML configuration files located in the `config/` directory.

1.  **Create a Config File:** Copy an existing config (e.g., `config/anatomy_so400nf.yaml`) and modify it for your needs. Key parameters include:
    -   `data.mode`: `embeddings`, `features`, or `images`.
    -   `data.feature_dir_name`: The name of the folder generated in step 2.
    -   `model.base_vision_model`: The vision model used for feature generation.
    -   `head_params`: The architecture of the trainable head model.
    -   `train`: Training parameters like learning rate, batch size, optimizer, and loss function.

2.  **Start Training (package-first):**
    -   For feature sequences: `python -m kurome.cli.train_features --config config/your_config.yaml`
    -   For embeddings/images: `python -m kurome.cli.train_embeddings --config config/your_config.yaml`

    Root wrapper scripts have been removed; use package CLIs only.

The script will handle setting up the dataset, model, optimizer, and training loop, logging progress to the console and Weights & Biases.

### 4. Inference

Once a model is trained, you can use it for inference.

**Batch Processing a Folder**

Use `kurome.cli.infer_folder` to classify or score all images in a directory.

```bash
python -m kurome.cli.infer_folder \
  --src path/to/your/images \
  --dst output_folder \
  --model models/your_model_name.safetensors \
  --arch class \
  --target_label_name "Good Anatomy" \
  --copy_passed
```

You can also use the root launcher:

```bash
python launch.py infer-folder -- --src path/to/your/images --dst output_folder --model models/your_model_name.safetensors --arch class --target_label_name "Good Anatomy" --copy_passed
```

## Pre-trained Models

This repository includes configurations and information for several pre-trained models.

### CityAesthetics - Anime

An aesthetic predictor optimized for scoring anime images. It is trained to filter out real-life photos, text, 3D renders, and manga panels.

-   **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/city96/CityAesthetics-demo)
-   **Model Download:** [Hugging Face Hub](https://huggingface.co/city96/CityAesthetics)
-   **Config:** `config/CityAesthetics-v1.yaml`

### Anime Classifiers

A collection of classifiers trained to detect specific artifacts or styles in anime images.

-   **Live Demos:** [Hugging Face Space](https://huggingface.co/spaces/city96/AnimeClassifiers-demo)
-   **Model Downloads:** [Hugging Face Hub](https://huggingface.co/city96/AnimeClassifiers)

#### Chromatic Aberration

Detects the presence of chromatic aberration, a common post-processing effect.

-   **Config:** `config/CCAnime-ChromaticAberration-v1.yaml`

#### Image Compression

Detects artifacts from JPEG or WebP compression.

-   **Config:** `config/CCAnime-Compression-v1.yaml`

---

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Enferlain/kurome)
