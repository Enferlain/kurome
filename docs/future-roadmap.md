# Kurome Future Roadmap

## Purpose
This document captures project-level direction beyond immediate refactors.

`kurome` should evolve as a task-first scoring/classification platform, not only a collection of vision scripts.

## North Star
Build a coherent system where scorers/classifiers are:
- semantically well-defined,
- reliable in decision-making,
- easy to extend across backbones and modalities,
- practical to operate in iterative data/model loops.

## Strategic Principles
1. Task-first architecture
Keep stable scorer/classifier task interfaces and plug encoders behind them.

2. Score semantics contract
Define exactly what a score means, how it is calibrated, and how thresholds are selected.

3. Data engine as a core product
Treat curation, relabeling, hard-negative mining, and active learning as first-class workflows.

4. Decision-quality evaluation
Optimize for practical decision metrics, not just training/validation loss.

5. Explicit model lifecycle
Version datasets/configs/checkpoints together and document intended use + failure modes.

6. Two-track development model
Use a fast track (frozen features) for iteration and a slow track (end-to-end) for final optimization.

7. Modality-agnostic expansion
Design encoder adapters so vision is current focus, not a hard project boundary.

## Capability Roadmap

### Phase A: Semantic and Contract Foundation
- Define scorer/classifier task contracts and output schemas.
- Establish score meaning and calibration policy.
- Standardize thresholding and reporting conventions.

### Phase B: Platformization
- Move to one typed configuration path.
- Isolate backbone/preprocess adapters from task logic.
- Keep training/inference behavior consistent across modes.

### Phase C: Data-Centric Loop
- Add repeatable curation and relabel workflows.
- Introduce disagreement review and hard-negative pipelines.
- Capture dataset provenance for every model artifact.

### Phase D: Evaluation and Reliability
- Track decision-oriented metrics (calibration, precision at threshold, ranking quality).
- Add stress tests for domain shift and class-imbalance drift.
- Promote model cards and release criteria for checkpoints.

### Phase E: Scope Expansion
- Support non-vision encoders through adapter contracts.
- Enable multimodal scoring/classification without rewriting core orchestration.

## Immediate Strategic Decisions to Lock
1. Whether training outputs should be raw logits only (activation in inference/postprocess).
2. The canonical score definition(s) and calibration approach.
3. Artifact policy: dataset snapshot + config + checkpoint must travel together.
4. Minimum decision metrics required before calling a model "ready".
