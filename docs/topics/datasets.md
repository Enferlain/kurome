# Datasets

Datasets are mode-specific adapters under `kurome/data/*` with shared contracts.

## Preferred Workflow

The preferred dataset workflow is now manifest-first:

1. Build one sample manifest from the source images.
2. Attach derived artifacts to the same manifest.
3. Train from the manifest by choosing the right mode/artifact key.

Primary command:

```bash
python launch.py prepare-data -- \
  --src /abs/path/to/dataset \
  --manifest /abs/path/to/dataset/manifest.jsonl \
  --forensic-preset preset1 \
  --feature-model facebook/dinov2-giant
```

The manifest is the shared dataset contract across:

1. `images`: reads `source_image_path`
2. `tensors`: reads `artifacts.<artifact_key>` pointing to `[C,H,W]` `.npz`
3. `features`: reads `artifacts.<artifact_key>` pointing to sequence `.npz`

Low-level builder CLIs still exist, but they are now artifact producers, not separate dataset systems:

1. `kurome.cli.build_image_manifest`
2. `kurome.cli.build_forensic_tensors`
3. `kurome.cli.generate_feature_sequences`

For most day-to-day usage, manually running those three in sequence is now redundant when `prepare-data` is sufficient.

## Batch Contracts

1. Embeddings mode: `("emb", "val")`
2. Features mode: `("sequence", "mask", "label")`
3. Images mode: `("pixel_values", "label")`
4. Tensors mode: `("pixel_values", "label")`

Contracts are defined in `kurome/data/contracts.py`.

## Adapter Modules

1. Embeddings: `kurome/data/embeddings.py`
2. Feature sequences: `kurome/data/sequences.py`
3. End-to-end images: `kurome/data/images.py`
4. Forensic tensors: `kurome/data/tensors.py`
5. Shared loader helpers: `kurome/data/dataloaders.py`
6. Dataset implementations: `kurome/data/datasets/*`

## Expected Behavior

1. Keep mode-specific logic in adapters, not in training loops.
2. Return loaders with keys matching the contract for that mode.
3. Expose validation loader behavior compatible with adapter hooks.
4. Prefer attaching derived artifacts to the sample manifest instead of inventing new dataset layouts.

## Add/Modify Dataset Workflow

1. Add or update dataset class in `kurome/data/datasets/*`.
2. Wire it in the relevant adapter module (`embeddings.py`, `sequences.py`, `images.py`, or `tensors.py`).
3. Verify batch keys match `kurome/data/contracts.py`.
4. Add or update tests (`tests/unit` and optionally `tests/integration`).

Detailed implementation checklist: `docs/how-to-add-dataset.md`.
