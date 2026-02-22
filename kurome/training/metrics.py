"""Shared validation and metric logging helpers for training loops."""

from __future__ import annotations

import math
from typing import Any


def log_eval_loss(
    *,
    wrapper: Any,
    step: int,
    eval_loss: float,
    key: str,
    error_prefix: str = "Wandb eval log error",
) -> None:
    """Log a single validation scalar to Weights & Biases when available."""
    if wrapper.wandb_run and not math.isnan(eval_loss):
        try:
            wrapper.wandb_run.log({key: eval_loss}, step=step)
        except Exception as exc:
            print(f"{error_prefix}: {exc}")


def append_and_average_validation_loss(validation_losses: list[float], eval_loss: float) -> float:
    """Append finite eval loss and return running average (NaN if unavailable)."""
    if math.isnan(eval_loss):
        return float("nan")
    validation_losses.append(eval_loss)
    return sum(validation_losses) / len(validation_losses)


def update_last_eval_loss(last_eval_loss: float, eval_loss: float) -> float:
    """Return updated last-eval value when current eval loss is finite."""
    if math.isnan(eval_loss):
        return last_eval_loss
    return eval_loss


def log_eval_and_average(
    *,
    wrapper: Any,
    step: int,
    eval_loss: float,
    avg_eval_loss: float,
    eval_key: str = "loss/eval_run_result",
    avg_key: str = "loss/average_val_loss",
    error_prefix: str = "Wandb val log error",
) -> None:
    """Log eval and running-average validation metrics when available."""
    if wrapper.wandb_run and not math.isnan(eval_loss):
        payload = {eval_key: eval_loss}
        if not math.isnan(avg_eval_loss):
            payload[avg_key] = avg_eval_loss
        try:
            wrapper.wandb_run.log(payload, step=step)
        except Exception as exc:
            print(f"{error_prefix}: {exc}")
