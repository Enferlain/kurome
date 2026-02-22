# How To Add a Dataset

Dataset support is adapter-driven through `kurome/data/*`.
For topic-level navigation, see `docs/topics/datasets.md`.

## 1. Choose the mode
- `embeddings` -> `kurome/data/embeddings.py`
- `features` -> `kurome/data/sequences.py`
- `images` -> `kurome/data/images.py`

If your dataset maps to an existing mode, extend that adapter. If not, create a new adapter module and wire it from the relevant CLI.

## 2. Implement dataset class
Dataset classes should live under:
- `kurome/data/datasets/*`

Do not add new root-level dataset modules.

## 3. Honor batch contracts
Use explicit key contracts from `kurome/data/contracts.py`:
- embeddings: `("emb", "val")`
- sequences: `("sequence", "mask", "label")`
- images: `("pixel_values", "label")`

Keep collate output aligned to these keys so `kurome/training/engine.py` helpers continue working.

## 4. Provide validation loader behavior
Adapters should provide validation loader behavior compatible with:
- `build_validation_dataloader(...)` in `kurome/data/dataloaders.py`

Use shared builders in `kurome/data/dataloaders.py` for consistency:
- `build_training_dataloader(...)`
- `build_validation_dataloader(...)`
- `log_train_val_loader_summary(...)`

## 5. Config wiring
Define/verify required YAML fields in `config/*.yaml`:
- `data.mode`
- `data.data_root`
- `data.feature_dir_name` (features mode)
- `data.val_split_count`

Normalization/validation lives in `kurome/config/loader.py` + `kurome/config/schema.py`.

## 6. Add tests
1. Unit tests for any new transform/collate/data-shaping behavior.
2. Smoke test assertions for adapter delegation/import paths.
3. Optional integration test with synthetic tensors when training-path behavior changes.

## 7. Keep engine clean
Mode-specific logic belongs in data adapters, not in training loops.
