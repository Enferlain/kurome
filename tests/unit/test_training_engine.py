from __future__ import annotations

import math

import pytest
import torch

from kurome.training.engine import (
    mean_tensor_losses,
    maybe_step_scheduler,
    normalize_embedding_batch_data,
    prepare_embedding_sub_batch,
    prepare_sequence_micro_batch,
    run_optimizer_step_and_zero_grad,
    update_progress_postfix,
)


class _Scheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


class _Progress:
    def __init__(self) -> None:
        self.received = None

    def set_postfix(self, ordered_dict, refresh: bool = False) -> None:
        self.received = (ordered_dict, refresh)


class _FailingScaler:
    def step(self, _optimizer) -> None:
        raise RuntimeError("step failure")

    def update(self) -> None:
        raise AssertionError("update should not run after failed step")


class _Optimizer:
    def __init__(self) -> None:
        self.zero_grad_called = False

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.zero_grad_called = set_to_none


def test_maybe_step_scheduler_gates_step() -> None:
    scheduler = _Scheduler()
    maybe_step_scheduler(scheduler, is_schedule_free=False, optimizer_stepped=True)
    assert scheduler.step_calls == 1

    maybe_step_scheduler(scheduler, is_schedule_free=True, optimizer_stepped=True)
    maybe_step_scheduler(scheduler, is_schedule_free=False, optimizer_stepped=False)
    maybe_step_scheduler(None, is_schedule_free=False, optimizer_stepped=True)
    assert scheduler.step_calls == 1


def test_update_progress_postfix_passes_ordered_values() -> None:
    progress = _Progress()
    update_progress_postfix(progress, {"loss": 0.1, "lr": 1e-4})
    assert progress.received is not None
    ordered, refresh = progress.received
    assert list(ordered.keys()) == ["loss", "lr"]
    assert refresh is False


def test_prepare_embedding_sub_batch_validates_shapes() -> None:
    prepared = prepare_embedding_sub_batch(
        {"emb": torch.randn(2, 4), "val": torch.tensor([0, 1])},
        device="cpu",
        num_classes=2,
    )
    assert prepared is not None
    model_input, target, size = prepared
    assert model_input.shape == (2, 4)
    assert target.shape == (2,)
    assert size == 2

    with pytest.raises(ValueError):
        prepare_embedding_sub_batch(
            {"emb": torch.randn(2, 4), "val": torch.tensor([1])},
            device="cpu",
            num_classes=2,
        )

    prepared_image = prepare_embedding_sub_batch(
        {"pixel_values": torch.randn(2, 3, 8, 8), "label": torch.tensor([0, 1])},
        device="cpu",
        num_classes=2,
    )
    assert prepared_image is not None
    image_input, image_target, image_size = prepared_image
    assert image_input.shape == (2, 3, 8, 8)
    assert image_target.shape == (2,)
    assert image_size == 2


def test_prepare_sequence_micro_batch_skips_non_finite_inputs() -> None:
    bad_sequence = torch.tensor([[[1.0, float("nan")]]], dtype=torch.float32)
    prepared = prepare_sequence_micro_batch(
        {"sequence": bad_sequence, "mask": torch.ones(1, 1, dtype=torch.bool), "label": torch.tensor([0])},
        device="cpu",
        num_classes=2,
    )
    assert prepared is None


def test_run_optimizer_step_and_zero_grad_handles_step_failure() -> None:
    stepped = run_optimizer_step_and_zero_grad(
        scaler=_FailingScaler(),
        optimizer=_Optimizer(),
        step_label=7,
    )
    assert stepped is False


def test_run_optimizer_step_and_zero_grad_always_zeros_grads() -> None:
    optimizer = _Optimizer()
    run_optimizer_step_and_zero_grad(
        scaler=_FailingScaler(),
        optimizer=optimizer,
    )
    assert optimizer.zero_grad_called is True


def test_mean_tensor_losses_handles_empty_and_non_empty_inputs() -> None:
    assert math.isnan(mean_tensor_losses([]))
    losses = [torch.tensor(1.0), torch.tensor(3.0)]
    assert mean_tensor_losses(losses) == pytest.approx(2.0)


def test_normalize_embedding_batch_data_handles_dict_and_list() -> None:
    normalized_from_dict = normalize_embedding_batch_data({"emb": torch.randn(1, 2), "val": torch.tensor([1])})
    assert normalized_from_dict is not None
    assert len(normalized_from_dict) == 1

    normalized_from_list = normalize_embedding_batch_data(
        [{"emb": torch.randn(1, 2), "val": torch.tensor([1])}]
    )
    assert normalized_from_list is not None
    assert len(normalized_from_list) == 1

    assert normalize_embedding_batch_data("bad-type") is None
