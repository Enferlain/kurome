# How To Add a Model

This repo now supports model extension through package modules + registry entries, without editing training loop internals.
For topic-level navigation, see `docs/topics/models.md`.

## 1. Pick the target family
- `kurome/models/heads/*` for embedding/sequence heads.
- `kurome/models/backbones/*` for end-to-end backbone+head wrappers.
- `kurome/models/tasks/*` for new losses or task-specific helpers.

## 2. Implement the model class
1. Add your module under the correct package path.
2. Keep constructor arguments explicit and typed where practical.
3. Keep runtime behavior self-contained in the class (no loop-side special cases).

Example paths:
- `kurome/models/heads/my_new_head.py`
- `kurome/models/backbones/my_new_backbone.py`

## 3. Export it from package init
1. Update `kurome/models/heads/__init__.py` or `kurome/models/backbones/__init__.py`.
2. Export the class in `__all__` if used.

## 4. Register the model ID
1. Add a stable ID in `kurome/models/registry.py`.
2. Point that ID to your class.

Example:
```python
MODEL_REGISTRY["my_new_head"] = MyNewHead
```

## 5. Use it from config
Set `model.model_id` in YAML:
```yaml
model:
  model_id: my_new_head
```

`kurome/models/factory.py` already uses signature-based kwarg filtering via `build_model_with_filtered_kwargs(...)`.

## 6. If you add a new loss
1. Implement it under `kurome/models/tasks/`.
2. Export it from `kurome/models/tasks/__init__.py`.
3. Add handling in `kurome/models/factory.py::build_criterion(...)`.

## 7. Add verification
1. Unit test registry/factory resolution.
2. Smoke test that package imports and wrappers still compile.
3. If applicable, add a tiny synthetic integration step under `tests/integration/`.

## 8. Keep compatibility boundaries
- Root legacy model modules were removed.
- Prefer package paths (`kurome.models.*`) for all new logic.
