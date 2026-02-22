"""Shared checkpoint load helpers for training CLIs."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def load_checkpoint(
    *,
    args: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    target_dev: str,
    load_checkpoint_state_fn: Callable[..., tuple[int, int]],
    load_optimizer_state_fn: Callable[..., bool],
    load_scheduler_state_fn: Callable[..., bool],
    load_scaler_state_fn: Callable[..., bool],
) -> tuple[int, int]:
    """Load checkpoint state through provided compatibility hooks."""
    return load_checkpoint_state_fn(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        target_dev=target_dev,
        load_optimizer_state_fn=load_optimizer_state_fn,
        load_scheduler_state_fn=load_scheduler_state_fn,
        load_scaler_state_fn=load_scaler_state_fn,
    )


def update_best_and_maybe_save(
    *,
    wrapper: Any,
    args: Any,
    global_step: int,
    epoch: int,
    eval_loss: float,
    best_eval_loss: float,
    should_save_new_best: bool,
    new_best_message: str,
    not_improved_message: str | None = None,
    nan_message: str | None = None,
) -> float:
    """Update best validation loss and save best checkpoint when needed."""
    if should_save_new_best:
        new_best = eval_loss
        print(new_best_message.format(eval_loss=eval_loss, best_eval_loss=best_eval_loss))
        wrapper.best_val_loss = new_best
        wrapper.save_model(step=global_step, epoch=epoch, suffix="_best_val", save_aux=False, args=args)
        return new_best

    if math.isnan(eval_loss):
        if nan_message:
            print(nan_message)
        return best_eval_loss

    if not_improved_message:
        print(not_improved_message.format(eval_loss=eval_loss, best_eval_loss=best_eval_loss))
    return best_eval_loss


def maybe_save_periodic_checkpoint(
    *,
    args: Any,
    wrapper: Any,
    global_step: int,
    epoch: int,
    initial_global_step: int,
) -> None:
    """Save periodic checkpoint if nsave conditions are met."""
    nsave = getattr(args, "nsave", 0)
    if nsave > 0 and global_step > 0 and global_step % nsave == 0 and global_step > initial_global_step:
        print(f"\nSaving periodic checkpoint @ step {global_step}...")
        wrapper.save_model(step=global_step, epoch=epoch, args=args, save_aux=False)
