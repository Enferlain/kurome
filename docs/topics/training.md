# Training

Training is driven by package CLIs plus shared runtime modules under `kurome/training/*`.

## Commands

1. Embeddings/images:
```bash
python -m kurome.cli.train_embeddings --config config/your_config.yaml
```
2. Feature sequences:
```bash
python -m kurome.cli.train_features --config config/your_config.yaml
```
3. Forensic tensors:
```bash
python launch.py train-tensors -- --config config/your_config.yaml
```

For data preparation, prefer:

```bash
python launch.py prepare-data -- --src /abs/path/to/dataset --manifest /abs/path/to/manifest.jsonl
```

Manually running the low-level builder CLIs before every run is now redundant for most workflows when `prepare-data` is sufficient.

## Runtime Stack

1. Bootstrap/device/precision: `kurome/training/bootstrap.py`
2. Optimizer/scheduler setup: `kurome/training/optim.py`
3. Loop orchestration: `kurome/training/loops.py`
4. Shared step helpers: `kurome/training/engine.py`
5. Validation logic: `kurome/training/validation.py`
6. Metrics/checkpoint hooks: `kurome/training/metrics.py`, `kurome/training/checkpoint.py`

## Typical Training Actions

1. Start a new run from YAML config.
2. Resume from a checkpoint (`args.resume` in config).
3. Tune optimizer/lr scheduler settings in `train` config block.
4. Switch mode with `data.mode` (`embeddings`, `features`, `images`, `tensors`).

## Verification

1. Full gate:
```bash
scripts/quality/run_quality.sh
```
2. Smoke only:
```bash
scripts/smoke/run_smoke.sh
```

## Related Docs

1. `docs/topics/configs.md`
2. `docs/topics/checkpoints.md`
3. `docs/how-it-works-now.md`
