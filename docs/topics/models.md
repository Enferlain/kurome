# Models

Model support is registry-driven and split into backbones, heads, and task losses.

## Model Surfaces

1. Backbones/wrappers: `kurome/models/backbones/*`
2. Heads: `kurome/models/heads/*`
3. Task losses/helpers: `kurome/models/tasks/*`
4. Registry: `kurome/models/registry.py`
5. Factory: `kurome/models/factory.py`

## How Models Are Selected

1. YAML sets `model.model_id`.
2. CLI calls factory helpers.
3. Registry resolves `model_id` -> class.
4. Factory filters kwargs to the target constructor.

## Add/Modify Model Workflow

1. Implement class in the right package folder.
2. Export from package `__init__.py`.
3. Register in `kurome/models/registry.py`.
4. Update config (`model.model_id` and params).
5. Add tests for registry/factory resolution.

Detailed implementation checklist: `docs/how-to-add-model.md`.
