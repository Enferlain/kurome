# Datasets

Datasets are mode-specific adapters under `kurome/data/*` with shared contracts.

## Batch Contracts

1. Embeddings mode: `("emb", "val")`
2. Features mode: `("sequence", "mask", "label")`
3. Images mode: `("pixel_values", "label")`

Contracts are defined in `kurome/data/contracts.py`.

## Adapter Modules

1. Embeddings: `kurome/data/embeddings.py`
2. Feature sequences: `kurome/data/sequences.py`
3. End-to-end images: `kurome/data/images.py`
4. Shared loader helpers: `kurome/data/dataloaders.py`
5. Dataset implementations: `kurome/data/datasets/*`

## Expected Behavior

1. Keep mode-specific logic in adapters, not in training loops.
2. Return loaders with keys matching the contract for that mode.
3. Expose validation loader behavior compatible with adapter hooks.

## Add/Modify Dataset Workflow

1. Add or update dataset class in `kurome/data/datasets/*`.
2. Wire it in the relevant adapter module (`embeddings.py`, `sequences.py`, or `images.py`).
3. Verify batch keys match `kurome/data/contracts.py`.
4. Add or update tests (`tests/unit` and optionally `tests/integration`).

Detailed implementation checklist: `docs/how-to-add-dataset.md`.
