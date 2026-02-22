from __future__ import annotations

import math
from types import SimpleNamespace

from kurome.training.checkpoint import (
    load_checkpoint,
    maybe_save_periodic_checkpoint,
    update_best_and_maybe_save,
)


class _Wrapper:
    def __init__(self) -> None:
        self.best_val_loss = float("inf")
        self.saved_calls: list[dict] = []

    def save_model(self, **kwargs) -> None:
        self.saved_calls.append(kwargs)


def test_load_checkpoint_delegates_to_compat_hook() -> None:
    calls = {}

    def fake_load_checkpoint_state_fn(**kwargs):
        calls.update(kwargs)
        return 2, 42

    model = object()
    optimizer = object()
    scheduler = object()
    scaler = object()
    args = SimpleNamespace(resume="checkpoint.safetensors")

    start_epoch, global_step = load_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        target_dev="cpu",
        load_checkpoint_state_fn=fake_load_checkpoint_state_fn,
        load_optimizer_state_fn=lambda *_args, **_kwargs: None,
        load_scheduler_state_fn=lambda *_args, **_kwargs: None,
        load_scaler_state_fn=lambda *_args, **_kwargs: None,
    )

    assert (start_epoch, global_step) == (2, 42)
    assert calls["args"] is args
    assert calls["model"] is model
    assert calls["optimizer"] is optimizer
    assert calls["scheduler"] is scheduler
    assert calls["scaler"] is scaler
    assert calls["target_dev"] == "cpu"


def test_update_best_and_maybe_save_saves_on_improvement() -> None:
    wrapper = _Wrapper()
    args = SimpleNamespace(name="exp")

    updated_best = update_best_and_maybe_save(
        wrapper=wrapper,
        args=args,
        global_step=50,
        epoch=3,
        eval_loss=0.12,
        best_eval_loss=0.20,
        should_save_new_best=True,
        new_best_message="best {eval_loss}",
    )

    assert updated_best == 0.12
    assert wrapper.best_val_loss == 0.12
    assert len(wrapper.saved_calls) == 1
    assert wrapper.saved_calls[0]["step"] == 50
    assert wrapper.saved_calls[0]["epoch"] == 3
    assert wrapper.saved_calls[0]["suffix"] == "_best_val"


def test_update_best_and_maybe_save_handles_nan_without_saving() -> None:
    wrapper = _Wrapper()
    args = SimpleNamespace(name="exp")

    updated_best = update_best_and_maybe_save(
        wrapper=wrapper,
        args=args,
        global_step=50,
        epoch=3,
        eval_loss=float("nan"),
        best_eval_loss=0.20,
        should_save_new_best=False,
        new_best_message="unused",
        nan_message="nan",
    )

    assert updated_best == 0.20
    assert math.isinf(wrapper.best_val_loss)
    assert wrapper.saved_calls == []


def test_maybe_save_periodic_checkpoint_respects_nsave_gate() -> None:
    wrapper = _Wrapper()
    args = SimpleNamespace(nsave=10)

    maybe_save_periodic_checkpoint(
        args=args,
        wrapper=wrapper,
        global_step=20,
        epoch=1,
        initial_global_step=0,
    )
    assert len(wrapper.saved_calls) == 1
    assert wrapper.saved_calls[0]["step"] == 20
    assert wrapper.saved_calls[0]["epoch"] == 1

    maybe_save_periodic_checkpoint(
        args=args,
        wrapper=wrapper,
        global_step=20,
        epoch=1,
        initial_global_step=20,
    )
    assert len(wrapper.saved_calls) == 1
