# Checkpoints

Checkpoint behavior is shared across training modes.

## Artifact Group

A single logical checkpoint consists of:

1. Model weights: `.safetensors`
2. Optimizer state: `.optim`
3. Scheduler state: `.sched`
4. AMP scaler state: `.scaler`
5. Training progress state: `.state` (epoch/global_step)

## Runtime Modules

1. High-level checkpoint hooks: `kurome/training/checkpoint.py`
2. Low-level load/save helpers: `kurome/training/state_io.py`
3. Wrapper save orchestration: `kurome/training/wrapper.py`

## Resume Flow

1. Set `resume` in config runtime args.
2. CLI load path calls shared checkpoint helper.
3. Missing optional state files warn but do not hard-fail (model load remains primary).

## Operational Guidance

1. Keep artifact suffixes stable to preserve resume compatibility.
2. When changing checkpoint behavior, add/update unit tests in `tests/unit/test_training_checkpoint.py`.
