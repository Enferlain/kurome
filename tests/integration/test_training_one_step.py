from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kurome.training.loops import (
    run_embedding_training_loop,
    run_feature_sequence_training_loop,
)

pytestmark = pytest.mark.integration


class _Wrapper:
    def __init__(self) -> None:
        self.best_val_loss = float("inf")
        self.current_epoch = 0
        self.wandb_run = None
        self.step_updates: list[int] = []
        self.logged_steps: list[int] = []
        self.saved_calls: list[dict] = []

    def update_step(self, step: int) -> None:
        self.step_updates.append(step)

    def log_main(self, *, step: int, current_train_loss: float, current_val_loss: float) -> None:
        _ = (current_train_loss, current_val_loss)
        self.logged_steps.append(step)

    def save_model(self, **kwargs) -> None:
        self.saved_calls.append(kwargs)


def _unused_validation(*_args, **_kwargs) -> tuple[float, bool]:
    return float("nan"), False


def _unused_sequence_validation(*_args, **_kwargs) -> tuple[float, bool]:
    return float("nan"), False


@pytest.mark.filterwarnings("ignore:CUDA initialization.*:UserWarning")
def test_embedding_loop_runs_single_optimizer_step_with_synthetic_batch() -> None:
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    wrapper = _Wrapper()

    args = SimpleNamespace(
        num_train_epochs=1,
        steps_per_epoch=1,
        max_train_steps=1,
        num_classes=2,
        output_mode="linear",
        log_every_n=1,
        validate_every_n=999,
        nsave=0,
    )

    train_loader = [
        {
            "emb": torch.randn(4, 4),
            "val": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        }
    ]

    before = model[0].weight.detach().clone()
    run_embedding_training_loop(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=None,
        wrapper=wrapper,
        start_epoch=0,
        initial_global_step=0,
        enabled_amp=False,
        amp_dtype=torch.float32,
        is_schedule_free=False,
        target_dev="cpu",
        run_validation_embeddings_fn=_unused_validation,
        is_e2e=False,
    )
    after = model[0].weight.detach().clone()

    assert not torch.equal(before, after), "Expected model parameters to update after one training step."
    assert wrapper.step_updates == [1]
    assert wrapper.logged_steps == [1]
    assert wrapper.saved_calls == []


@pytest.mark.filterwarnings("ignore:CUDA initialization.*:UserWarning")
def test_feature_sequence_loop_runs_single_optimizer_step_with_synthetic_batch() -> None:
    class _TinySequenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 2)

        def forward(self, sequence_batch: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
            _ = attention_mask
            pooled = sequence_batch.mean(dim=1)
            return self.proj(pooled)

    model = _TinySequenceModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    wrapper = _Wrapper()

    args = SimpleNamespace(
        num_train_epochs=1,
        steps_per_epoch=1,
        max_train_steps=1,
        num_labels=2,
        num_classes=2,
        batch=2,
        gradient_accumulation_steps=1,
        log_every_n=1,
        validate_every_n=999,
        nsave=0,
    )

    train_loader = [
        {
            "sequence": torch.randn(2, 3, 4),
            "mask": torch.ones(2, 3, dtype=torch.bool),
            "label": torch.tensor([0, 1], dtype=torch.long),
        }
    ]

    before = model.proj.weight.detach().clone()
    run_feature_sequence_training_loop(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=None,
        wrapper=wrapper,
        start_epoch=0,
        initial_global_step=0,
        enabled_amp=False,
        amp_dtype=torch.float32,
        is_schedule_free=False,
        target_dev="cpu",
        run_validation_sequences_fn=_unused_sequence_validation,
    )
    after = model.proj.weight.detach().clone()

    assert not torch.equal(before, after), "Expected model parameters to update after one training step."
    assert wrapper.step_updates == [1]
    assert wrapper.logged_steps == [1]
    assert wrapper.saved_calls == []


@pytest.mark.filterwarnings("ignore:CUDA initialization.*:UserWarning")
def test_embedding_loop_accepts_image_mode_batch_shape_for_one_step() -> None:
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 8 * 8, 2),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    wrapper = _Wrapper()

    args = SimpleNamespace(
        num_train_epochs=1,
        steps_per_epoch=1,
        max_train_steps=1,
        num_classes=2,
        output_mode="linear",
        log_every_n=1,
        validate_every_n=999,
        nsave=0,
        is_end_to_end=True,
    )

    train_loader = [
        [
            {
                "pixel_values": torch.randn(4, 3, 8, 8),
                "label": torch.tensor([0, 1, 0, 1], dtype=torch.long),
            }
        ]
    ]

    before = model[1].weight.detach().clone()
    run_embedding_training_loop(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=None,
        wrapper=wrapper,
        start_epoch=0,
        initial_global_step=0,
        enabled_amp=False,
        amp_dtype=torch.float32,
        is_schedule_free=False,
        target_dev="cpu",
        run_validation_embeddings_fn=_unused_validation,
        is_e2e=True,
    )
    after = model[1].weight.detach().clone()

    assert not torch.equal(before, after), "Expected model parameters to update after one training step."
    assert wrapper.step_updates == [1]
