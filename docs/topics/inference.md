# Inference

Inference logic is package-based and exposed through package CLIs.

## Core Modules

1. Pipelines: `kurome/inference/pipeline.py`
2. Output formatting/postprocess: `kurome/inference/postprocess.py`
3. Folder inference CLI: `kurome/cli/infer_folder.py`

## Common Actions

1. Batch folder inference:
```bash
python -m kurome.cli.infer_folder --help
```

## Notes

1. Training and inference share model heads from `kurome.models.*`.
2. Avoid introducing inference-only model logic in root scripts; keep behavior in package modules.
