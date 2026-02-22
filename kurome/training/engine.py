"""Shared training-loop helpers used by CLI entrypoints."""

from __future__ import annotations

import collections
import math
import traceback
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from kurome.models.tasks import FocalLoss, GHMC_Loss


def ensure_training_mode(model: Any, optimizer: Any) -> None:
    """Set model/optimizer to training mode when supported."""
    if hasattr(model, "train"):
        model.train()
    if hasattr(optimizer, "train") and callable(optimizer.train):
        try:
            optimizer.train()
        except Exception as exc:
            print(f"Warning: Error calling optimizer.train(): {exc}")


def restore_training_mode_if_needed(model: Any, optimizer: Any) -> None:
    """Re-enable train mode only when the model is currently in eval mode."""
    if not getattr(model, "training", True):
        ensure_training_mode(model, optimizer)


def maybe_step_scheduler(scheduler: Any, *, is_schedule_free: bool, optimizer_stepped: bool) -> None:
    """Step scheduler only when the optimizer actually stepped."""
    if scheduler is None or is_schedule_free or not optimizer_stepped:
        return
    try:
        scheduler.step()
    except Exception as exc:
        print(f"Warning: Error scheduler step: {exc}")


def update_progress_postfix(progress_bar: Any, postfix_values: Mapping[str, Any]) -> None:
    """Apply ordered postfix values to a tqdm progress bar."""
    ordered = collections.OrderedDict(postfix_values.items())
    progress_bar.set_postfix(ordered_dict=ordered, refresh=False)


def prepare_prediction_for_loss(y_pred: torch.Tensor, criterion: Any, num_classes: int) -> torch.Tensor:
    """Normalize prediction tensor shape/dtype for downstream loss computation."""
    y_pred_for_loss = y_pred
    if (
        isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss))
        and num_classes == 1
        and y_pred.ndim > 1
        and y_pred.shape[1] == 1
    ):
        y_pred_for_loss = y_pred.squeeze(-1)
    return y_pred_for_loss.to(torch.float32)


def compute_loss(
    *,
    criterion: Any,
    loss_input: torch.Tensor,
    target: torch.Tensor,
    output_mode: str = "linear",
    cast_target_by_criterion: bool = False,
) -> torch.Tensor:
    """Compute criterion loss with optional NLL and target-type handling."""
    target_for_loss = target.to(loss_input.device)

    if isinstance(criterion, nn.NLLLoss) and output_mode == "linear":
        loss_input = F.log_softmax(loss_input, dim=-1)

    if cast_target_by_criterion:
        if isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, nn.NLLLoss, GHMC_Loss)):
            target_for_loss = target_for_loss.long()
        elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)):
            target_for_loss = target_for_loss.float()

    return criterion(loss_input, target_for_loss)


def prepare_embedding_sub_batch(
    sub_batch: Mapping[str, Any],
    *,
    device: str,
    num_classes: int,
    global_step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    """Extract embedding/image tensors from a sub-batch and shape targets."""
    model_input_val = sub_batch.get("emb")
    target_val = sub_batch.get("val")

    if model_input_val is None or target_val is None:
        model_input_val = sub_batch.get("pixel_values")
        target_val = sub_batch.get("label")

    if target_val is None:
        if global_step is not None:
            print(
                "Warning: Missing target ('val' or 'label') in sub-batch "
                f"at step {global_step}. Skipping sub-batch."
            )
        return None

    if model_input_val is None:
        return None

    model_input = model_input_val.to(device)
    current_sub_batch_size = model_input_val.size(0)

    target = target_val.to(device=device, dtype=torch.float32 if num_classes == 1 else torch.long).view(-1)
    if target.shape[0] != current_sub_batch_size:
        raise ValueError("Target shape mismatch")
    return model_input, target, current_sub_batch_size


def prepare_sequence_micro_batch(
    batch_data: Mapping[str, Any],
    *,
    device: str,
    num_classes: int,
    micro_batch_index: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
    """Extract sequence-mode tensors from a micro-batch and shape targets."""
    sequence_batch = batch_data.get("sequence")
    mask_batch = batch_data.get("mask")
    label_batch = batch_data.get("label")
    if sequence_batch is None or mask_batch is None or label_batch is None:
        return None

    if not torch.isfinite(sequence_batch).all():
        num_bad_elements = (~torch.isfinite(sequence_batch)).sum().item()
        if micro_batch_index is not None:
            print(
                f"\n!!! WARNING: Non-finite values detected in input sequence_batch at micro-batch "
                f"{micro_batch_index}! Num bad: {num_bad_elements}. Skipping batch."
            )
        else:
            print(
                f"\n!!! WARNING: Non-finite values detected in input sequence_batch! "
                f"Num bad: {num_bad_elements}. Skipping batch."
            )
        return None

    sequence_batch = sequence_batch.to(device)
    mask_batch = mask_batch.to(device)
    label_batch = label_batch.to(device)
    current_micro_batch_size = sequence_batch.size(0)

    target = label_batch
    if num_classes == 1:
        target = target.to(dtype=torch.float32).view(current_micro_batch_size, -1).squeeze(-1)
    else:
        target = target.to(dtype=torch.long).view(current_micro_batch_size)

    return sequence_batch, mask_batch, target, current_micro_batch_size


def run_optimizer_step_and_zero_grad(
    *,
    scaler: Any,
    optimizer: Any,
    step_label: int | None = None,
    error_prefix: str = "Error during optimizer step",
) -> bool:
    """Run scaler/optimizer step, always clear gradients, and report success."""
    optimizer_stepped = False
    try:
        scaler.step(optimizer)
        scaler.update()
        optimizer_stepped = True
    except Exception as exc:
        if step_label is not None:
            print(f"\n{error_prefix} {step_label}: {exc}")
        else:
            print(f"\n{error_prefix}: {exc}")
        traceback.print_exc()
    optimizer.zero_grad(set_to_none=True)
    return optimizer_stepped


def accumulate_scalar_loss(loss_sum: torch.Tensor, steps: int, loss_value: float) -> int:
    """Accumulate a finite scalar loss into running sum and return updated count."""
    if not math.isnan(loss_value):
        loss_sum += loss_value
        return steps + 1
    return steps


def average_and_reset_loss_window(loss_sum: torch.Tensor, steps: int) -> tuple[float, int]:
    """Compute average over a running loss window and reset the accumulator tensor."""
    avg = (loss_sum / steps).item() if steps > 0 else float("nan")
    loss_sum.zero_()
    return avg, 0


def mean_tensor_losses(losses: list[torch.Tensor]) -> float:
    """Return mean scalar value for a list of loss tensors."""
    if not losses:
        return float("nan")
    return torch.mean(torch.stack(losses)).item()


def normalize_embedding_batch_data(
    batch_data: Any,
    *,
    global_step: int | None = None,
) -> list[Mapping[str, Any]] | None:
    """Normalize embedding loader output to a non-empty list of sub-batches."""
    if isinstance(batch_data, dict):
        batch_data_list = [batch_data]
    elif isinstance(batch_data, list):
        batch_data_list = batch_data
    else:
        if global_step is not None:
            print(f"Warning: Unexpected batch_data type {type(batch_data)} at step {global_step}. Skipping.")
        return None

    if len(batch_data_list) == 0:
        if global_step is not None:
            print(f"Warning: Empty batch_data_list at step {global_step}. Skipping.")
        return None
    return batch_data_list


def advance_global_step(
    global_step: int,
    *,
    progress_bar: Any | None = None,
    wrapper: Any | None = None,
) -> int:
    """Increment global step and propagate progress/wrapper updates."""
    new_global_step = global_step + 1
    if progress_bar is not None:
        progress_bar.update(1)
    if wrapper is not None and hasattr(wrapper, "update_step"):
        wrapper.update_step(new_global_step)
    return new_global_step


def has_reached_total_steps(global_step: int, total_steps: int) -> bool:
    """Return whether the global-step budget has been exhausted."""
    return global_step >= total_steps


def should_run_interval(global_step: int, every_n: int, *, min_step_exclusive: int = 0) -> bool:
    """Return whether a periodic action should run on this global step."""
    return every_n > 0 and global_step > min_step_exclusive and global_step % every_n == 0
