"""Model and criterion factory helpers."""

from __future__ import annotations

import inspect
from typing import Any

import torch.nn as nn

from kurome.models.tasks import FocalLoss, GHMC_Loss

from .registry import get_model_class

BINARY_STYLE_LOSSES = {"bce", "l1", "mse"}
MULTI_CLASS_LOSSES = {"crossentropy", "focal", "nll", "ghm"}


def resolve_num_classes(
    *,
    arch: str,
    loss_function: str | None,
    dataset_num_labels: int,
    default_class_loss: str = "focal",
) -> tuple[str, int, list[str]]:
    """Resolve final loss name and class count from config + dataset info."""
    warnings: list[str] = []
    loss_name = (loss_function or "").lower()
    arch_name = (arch or "class").lower()

    if arch_name == "score":
        if not loss_name:
            loss_name = "l1"
        intended_num_classes = 1
        if loss_name in MULTI_CLASS_LOSSES:
            warnings.append(
                f"arch='score' with loss='{loss_name}' implies multi-class behavior; using intended num_classes=2."
            )
            intended_num_classes = 2
    elif arch_name == "class":
        if not loss_name:
            loss_name = default_class_loss
            warnings.append(f"No loss_function specified; defaulting to '{loss_name}'.")
        if loss_name in BINARY_STYLE_LOSSES:
            intended_num_classes = 1
        elif loss_name in MULTI_CLASS_LOSSES:
            intended_num_classes = 2
        else:
            raise ValueError(f"Unknown loss function '{loss_name}' for arch='{arch_name}'.")
    else:
        raise ValueError(f"Unknown architecture '{arch_name}'.")

    if dataset_num_labels <= 0:
        raise ValueError("Dataset reported 0 labels/classes.")

    final_num_classes = intended_num_classes
    if intended_num_classes == 1:
        if dataset_num_labels == 1:
            raise ValueError("Binary-style loss selected but dataset contains only one class.")
        if dataset_num_labels > 2:
            raise ValueError(
                f"Binary-style loss selected but dataset contains {dataset_num_labels} classes (>2)."
            )
        final_num_classes = 1
    else:
        if dataset_num_labels == 1:
            raise ValueError("Multi-class loss selected but dataset contains only one class.")
        if dataset_num_labels != intended_num_classes:
            warnings.append(
                f"Dataset has {dataset_num_labels} classes; overriding intended {intended_num_classes}."
            )
            final_num_classes = dataset_num_labels

    return loss_name, final_num_classes, warnings


def build_criterion(
    *,
    loss_name: str,
    num_classes: int,
    output_mode: str | None,
    class_weights_tensor: Any,
    args: Any,
    allow_bce_multiclass: bool = False,
    strict_regression_num_classes: bool = True,
    require_linear_logits_losses: bool = True,
) -> nn.Module:
    """Build criterion with validation around loss compatibility."""
    name = (loss_name or "").lower()
    output_mode_name = (output_mode or "linear").lower()

    if name == "l1":
        if strict_regression_num_classes and num_classes != 1:
            raise ValueError(f"loss_function='l1' requires num_classes=1, got {num_classes}.")
        return nn.L1Loss(reduction="mean")

    if name == "mse":
        if strict_regression_num_classes and num_classes != 1:
            raise ValueError(f"loss_function='mse' requires num_classes=1, got {num_classes}.")
        return nn.MSELoss(reduction="mean")

    if name == "focal":
        if num_classes <= 1:
            raise ValueError(f"loss_function='focal' requires num_classes > 1, got {num_classes}.")
        if require_linear_logits_losses and output_mode_name != "linear":
            raise ValueError(f"loss_function='focal' expects output_mode='linear', got '{output_mode_name}'.")
        return FocalLoss(
            gamma=getattr(args, "focal_loss_gamma", 2.0),
            weight=class_weights_tensor,
        )

    if name == "crossentropy":
        if num_classes <= 1:
            raise ValueError(f"loss_function='crossentropy' requires num_classes > 1, got {num_classes}.")
        if require_linear_logits_losses and output_mode_name != "linear":
            raise ValueError(
                f"loss_function='crossentropy' expects output_mode='linear', got '{output_mode_name}'."
            )
        return nn.CrossEntropyLoss(weight=class_weights_tensor)

    if name == "bce":
        if not allow_bce_multiclass and num_classes != 1:
            raise ValueError(f"loss_function='bce' requires num_classes=1, got {num_classes}.")
        if require_linear_logits_losses and output_mode_name != "linear":
            raise ValueError(f"loss_function='bce' expects output_mode='linear', got '{output_mode_name}'.")
        weight = class_weights_tensor if num_classes > 1 else None
        return nn.BCEWithLogitsLoss(weight=weight)

    if name == "nll":
        if num_classes <= 1:
            raise ValueError(f"loss_function='nll' requires num_classes > 1, got {num_classes}.")
        if output_mode_name != "linear":
            raise ValueError(f"loss_function='nll' requires output_mode='linear', got '{output_mode_name}'.")
        return nn.NLLLoss(weight=class_weights_tensor)

    if name == "ghm":
        if num_classes <= 1:
            raise ValueError(f"loss_function='ghm' requires num_classes > 1, got {num_classes}.")
        if output_mode_name != "linear":
            raise ValueError(f"loss_function='ghm' requires output_mode='linear', got '{output_mode_name}'.")
        return GHMC_Loss(
            bins=getattr(args, "ghm_bins", 10),
            momentum=getattr(args, "ghm_momentum", 0.75),
        )

    raise ValueError(f"Unknown loss function '{loss_name}'.")


def build_model(model_id: str, **kwargs: Any):
    """Instantiate a model class via the registry."""
    model_cls = get_model_class(model_id)
    return model_cls(**kwargs)


def build_model_with_filtered_kwargs(model_id: str, candidate_kwargs: dict[str, Any]):
    """Build a model by filtering a candidate kwargs map by constructor signature."""
    model_cls = get_model_class(model_id)
    signature = inspect.signature(model_cls.__init__)
    valid_params = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    kwargs = {
        key: value for key, value in candidate_kwargs.items() if key in valid_params and value is not None
    }
    missing_required = [
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and param.default is inspect.Parameter.empty
        and name not in kwargs
    ]
    if missing_required:
        missing = ", ".join(missing_required)
        raise ValueError(f"Model '{model_id}' missing required constructor args: {missing}.")
    return model_cls(**kwargs)
