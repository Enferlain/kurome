# Deprecation Notes

These notes track compatibility surfaces that remain for transition safety.

## Compatibility wrappers (removed)
- `train.py` (removed)
- `train_features.py` (removed)
- `inference.py` (removed)

Replacement entrypoints:
- `python -m kurome.cli.train_embeddings --config <config.yaml>`
- `python -m kurome.cli.train_features --config <config.yaml>`
- package inference path via `kurome/inference/*` and `kurome/cli/infer_folder.py`

Removal execution details are tracked in `docs/wrapper-removal-checklist.md`.

## Legacy root modules
Legacy root implementation modules (`dataset.py`, `sequence_dataset.py`, `image_dataset.py`, `model.py`,
`head_model.py`, `hybrid_model.py`, `model_early_extract.py`, `losses.py`, `utils.py`) are removed.
Root-level file policy and enforcement are tracked in `docs/root-surface.md`.
Wrapper command references are constrained by `scripts/quality/check_wrapper_references.py`.

Preferred import paths for new code:
- `kurome.config.*`
- `kurome.data.*`
- `kurome.models.*`
- `kurome.training.*`
- `kurome.inference.*`

## Legacy internals policy
New code should use package modules only. Any reintroduction of root-level implementation modules should be treated
as a regression and reviewed explicitly against the root-surface contract.
