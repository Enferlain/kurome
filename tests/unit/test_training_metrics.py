from __future__ import annotations

import pytest

from kurome.training.metrics import (
    append_and_average_validation_loss,
    log_eval_and_average,
    log_eval_loss,
    update_last_eval_loss,
)


class _FakeWandbRun:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[tuple[dict, int]] = []

    def log(self, payload: dict, step: int) -> None:
        if self.should_fail:
            raise RuntimeError("log failed")
        self.calls.append((payload, step))


class _Wrapper:
    def __init__(self, wandb_run) -> None:
        self.wandb_run = wandb_run


def test_append_and_average_validation_loss_skips_nan() -> None:
    losses: list[float] = []
    avg = append_and_average_validation_loss(losses, float("nan"))
    assert avg != avg
    assert losses == []

    avg = append_and_average_validation_loss(losses, 0.2)
    assert avg == 0.2
    avg = append_and_average_validation_loss(losses, 0.4)
    assert avg == pytest.approx(0.3)


def test_update_last_eval_loss_keeps_previous_on_nan() -> None:
    assert update_last_eval_loss(1.0, float("nan")) == 1.0
    assert update_last_eval_loss(1.0, 0.7) == 0.7


def test_log_eval_loss_logs_only_finite_values() -> None:
    wandb_run = _FakeWandbRun()
    wrapper = _Wrapper(wandb_run)

    log_eval_loss(wrapper=wrapper, step=5, eval_loss=0.25, key="loss/eval")
    log_eval_loss(wrapper=wrapper, step=6, eval_loss=float("nan"), key="loss/eval")

    assert len(wandb_run.calls) == 1
    assert wandb_run.calls[0] == ({"loss/eval": 0.25}, 5)


def test_log_eval_loss_tolerates_wandb_errors() -> None:
    wrapper = _Wrapper(_FakeWandbRun(should_fail=True))
    log_eval_loss(wrapper=wrapper, step=5, eval_loss=0.25, key="loss/eval")


def test_log_eval_and_average_logs_payload() -> None:
    wandb_run = _FakeWandbRun()
    wrapper = _Wrapper(wandb_run)

    log_eval_and_average(wrapper=wrapper, step=9, eval_loss=0.5, avg_eval_loss=0.4)
    assert len(wandb_run.calls) == 1
    payload, step = wandb_run.calls[0]
    assert step == 9
    assert payload["loss/eval_run_result"] == 0.5
    assert payload["loss/average_val_loss"] == 0.4


def test_log_eval_and_average_omits_avg_when_nan() -> None:
    wandb_run = _FakeWandbRun()
    wrapper = _Wrapper(wandb_run)

    log_eval_and_average(wrapper=wrapper, step=9, eval_loss=0.5, avg_eval_loss=float("nan"))
    payload, _step = wandb_run.calls[0]
    assert "loss/eval_run_result" in payload
    assert "loss/average_val_loss" not in payload
