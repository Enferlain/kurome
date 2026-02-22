"""Shared optimizer/scheduler construction helpers."""

from __future__ import annotations

import inspect
import traceback
from typing import Any

import torch


def setup_optimizer_scheduler(args, model, *, optimizers: dict[str, Any], schedulers: dict[str, Any]):
    """Set up optimizer + scheduler from args with dynamic registry support."""
    optimizer = None
    scheduler = None
    is_schedule_free = False
    optimizer_name = getattr(args, "optimizer", "AdamW").lower()

    print(f"Attempting to setup optimizer: {optimizer_name}")

    params_to_optimize = model.parameters()
    param_list_for_check = list(params_to_optimize)
    if not param_list_for_check:
        exit("Error: No parameters selected for optimization!")
    params_to_optimize = model.parameters()

    if optimizer_name in optimizers:
        optimizer_class = optimizers[optimizer_name]
        print(f"Found optimizer class: {optimizer_class.__name__}")
        try:
            sig = inspect.signature(optimizer_class.__init__)
            available_params = sig.parameters.keys()
            potential_kwargs = {}
            args_dict = vars(args)
            for param_name in available_params:
                if param_name in ["self", "params", "model", "args", "kwargs"]:
                    continue
                if param_name in args_dict and args_dict[param_name] is not None:
                    expected_type = sig.parameters[param_name].annotation
                    default_value = sig.parameters[param_name].default
                    value = args_dict[param_name]
                    try:
                        converted_value = None
                        if expected_type == inspect.Parameter.empty:
                            try:
                                converted_value = float(value)
                            except (ValueError, TypeError):
                                try:
                                    converted_value = int(value)
                                except (ValueError, TypeError):
                                    if isinstance(value, str) and value.lower() in ["true", "false"]:
                                        converted_value = value.lower() == "true"
                                    else:
                                        converted_value = value
                        elif expected_type == bool:
                            converted_value = str(value).lower() == "true" if isinstance(value, str) else bool(value)
                        elif expected_type == int:
                            converted_value = int(value)
                        elif expected_type == float:
                            converted_value = float(value)
                        elif expected_type == str:
                            converted_value = str(value)
                        elif expected_type == tuple or expected_type == list or (
                            hasattr(expected_type, "__origin__") and expected_type.__origin__ in [tuple, list]
                        ):
                            if isinstance(value, (list, tuple)):
                                inner_type = float
                                if hasattr(expected_type, "__args__") and expected_type.__args__:
                                    inner_type = expected_type.__args__[0]
                                converted_list = [inner_type(v) for v in value]
                                converted_value = (
                                    tuple(converted_list)
                                    if expected_type == tuple or expected_type.__origin__ == tuple
                                    else converted_list
                                )
                            else:
                                print(
                                    f"Warning: Arg {param_name} expects {expected_type} but got {type(value)}. Skipping."
                                )
                        else:
                            converted_value = value

                        if converted_value is not None:
                            potential_kwargs[param_name] = converted_value

                    except (ValueError, TypeError) as e_type:
                        print(
                            f"Warning: Could not convert arg '{param_name}' (value: {value}) "
                            f"to expected type {expected_type}. Error: {e_type}. Using default or skipping."
                        )
                        if default_value != inspect.Parameter.empty:
                            potential_kwargs[param_name] = default_value

            print(f"  Instantiating {optimizer_class.__name__} with args: {potential_kwargs}")
            optimizer = optimizer_class(params_to_optimize, **potential_kwargs)
            if "schedulefree" in optimizer_name:
                is_schedule_free = True
                print("  Detected schedule-free optimizer.")

        except Exception as e:
            print(f"ERROR: Failed to instantiate optimizer '{optimizer_name}' dynamically: {e}")
            traceback.print_exc()
            print("  Falling back to AdamW.")
            optimizer_name = "adamw"
            optimizer = None
            is_schedule_free = False

    if optimizer is None:
        if optimizer_name != "adamw":
            print(f"Warning: Optimizer '{optimizer_name}' failed. Falling back to AdamW.")
        optimizer_name = "adamw"
        adamw_kwargs = {
            "lr": float(getattr(args, "lr", 1e-4)),
            "betas": tuple(getattr(args, "betas", (0.9, 0.999))),
            "weight_decay": float(getattr(args, "weight_decay", 0.0)),
            "eps": float(getattr(args, "eps", 1e-8)),
        }
        print(f"Instantiating torch.optim.AdamW with args: {adamw_kwargs}")
        optimizer = torch.optim.AdamW(params_to_optimize, **adamw_kwargs)
        is_schedule_free = False

    if not is_schedule_free:
        scheduler_name = getattr(args, "scheduler_name", "CosineAnnealingLR").lower()
        print(f"Attempting to setup scheduler: {scheduler_name}")
        scheduler_class = None
        if scheduler_name in schedulers:
            scheduler_class = schedulers[scheduler_name]
            print(f"Found custom scheduler class: {scheduler_class.__name__}")
            try:
                sig = inspect.signature(scheduler_class.__init__)
                available_params = sig.parameters.keys()
                scheduler_kwargs = {}
                args_dict = vars(args)
                for param_name in available_params:
                    if param_name in ["self", "optimizer", "last_epoch", "args", "kwargs"]:
                        continue
                    arg_key = f"scheduler_{param_name}"
                    if arg_key not in args_dict and param_name in args_dict:
                        arg_key = param_name
                    if arg_key in args_dict and args_dict[arg_key] is not None:
                        value = args_dict[arg_key]
                        try:
                            scheduler_kwargs[param_name] = float(value)
                        except (ValueError, TypeError):
                            try:
                                scheduler_kwargs[param_name] = int(value)
                            except (ValueError, TypeError):
                                try:
                                    if isinstance(value, str):
                                        scheduler_kwargs[param_name] = value.lower() == "true"
                                    else:
                                        scheduler_kwargs[param_name] = bool(value)
                                except Exception:
                                    scheduler_kwargs[param_name] = value
                print(f"  Instantiating {scheduler_class.__name__} with args: {scheduler_kwargs}")
                scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            except Exception as e:
                print(f"ERROR: Failed custom scheduler '{scheduler_name}': {e}")
                scheduler = None

        if scheduler is None:
            scheduler_type = None
            is_cosine = getattr(args, "cosine", None)
            has_warmup = getattr(args, "warmup_steps", 0) > 0

            if is_cosine is True or scheduler_name in ["cosineannealinglr", "cosine"]:
                scheduler_type = "cosine"
            elif has_warmup and scheduler_name == "linearlr":
                scheduler_type = "warmup"
            elif scheduler_name in ["none", None]:
                scheduler_type = None
            elif scheduler_name not in schedulers:
                print(
                    f"Warning: Scheduler '{scheduler_name}' not found in custom SCHEDULERS or standard options. "
                    "No scheduler used."
                )

            if scheduler_type == "cosine":
                print("Using standard torch.optim.lr_scheduler.CosineAnnealingLR.")
                t_max_steps = getattr(args, "scheduler_t_max", args.max_train_steps)
                eta_min = getattr(args, "scheduler_eta_min", 0)
                print(f"  Setting T_max = {t_max_steps}, eta_min = {eta_min}")
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=int(t_max_steps), eta_min=float(eta_min)
                )
            elif scheduler_type == "warmup":
                print("Using standard torch.optim.lr_scheduler.LinearLR for warmup.")
                warmup_iters = int(args.warmup_steps)
                print(f"  Setting total_iters = {warmup_iters} for LinearLR.")
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_iters
                )
            elif scheduler is None:
                print("No matching standard or custom scheduler specified. Proceeding without scheduler.")

    else:
        print("Using a schedule-free optimizer, no scheduler will be used.")

    print("Optimizer and Scheduler setup complete.")
    return optimizer, scheduler, is_schedule_free
