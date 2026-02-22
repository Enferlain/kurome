"""Shared bootstrap helpers for training scripts.

This module centralizes training-script startup behavior:
- Optional SageAttention patching
- Torch runtime tuning
- Precision selection
- Optional Weights & Biases initialization
- Dynamic custom optimizer/scheduler registry loading
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
from safetensors.torch import load_file


def apply_sageattention_patch(functional_module: Any) -> None:
    """Apply SageAttention monkey-patch if the package is available."""
    try:
        from sageattention import sageattn

        functional_module.scaled_dot_product_attention = sageattn
        print("!!! Successfully applied SageAttention monkey-patch !!!")
    except ImportError:
        print("SageAttention not found or failed to import, using default F.scaled_dot_product_attention.")
    except Exception as exc:  # pragma: no cover - defensive logging path
        print(f"Error applying SageAttention monkey-patch: {exc}")


def configure_torch_runtime() -> str:
    """Configure torch runtime flags and return the selected target device."""
    target_dev = "cuda" if torch.cuda.is_available() else "cpu"
    if target_dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 support enabled for CUDA operations.")
    else:
        print("CUDA not available, TF32 settings not applied.")
    torch.backends.cudnn.benchmark = True
    return target_dev


def setup_precision(args: Any, target_dev: str) -> tuple[torch.dtype, bool]:
    """Set up training precision based on args."""
    precision_arg = getattr(args, "precision", "fp32").lower()
    enabled_amp = False
    amp_dtype = torch.float32

    if precision_arg == "fp16":
        if target_dev == "cuda":
            amp_dtype = torch.float16
            enabled_amp = True
            print("Using fp16 mixed precision.")
        else:
            print("Warning: fp16 requested but CUDA is not available. Falling back to fp32.")
    elif precision_arg == "bf16":
        if target_dev == "cuda" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            enabled_amp = True
            print("Using bf16 mixed precision.")
        else:
            if target_dev != "cuda":
                print("Warning: bf16 requested but CUDA is not available. Falling back to fp32.")
            else:
                print("Warning: bf16 requested but not supported by hardware. Falling back to fp32.")
    elif precision_arg == "fp64":
        amp_dtype = torch.float64
        enabled_amp = False
        print("Using fp64 (double precision). AMP disabled.")
    else:
        print("Using fp32 precision.")
        if precision_arg != "fp32":
            print(f"Warning: Unknown precision '{precision_arg}' specified. Using fp32.")

    return amp_dtype, enabled_amp


def setup_wandb(args: Any, wandb_module: Any | None) -> Any | None:
    """Initialize Weights & Biases if importable and configured."""
    if wandb_module is None or not hasattr(wandb_module, "init"):
        print("Wandb library not available.")
        return None
    try:
        if not hasattr(args, "name") or not args.name:
            args.name = f"{getattr(args, 'base', 'model')}-{getattr(args, 'rev', 'rev')}"

        wandb_run = wandb_module.init(
            project=getattr(args, "wandb_project", "city-classifiers"),
            name=args.name,
            config=vars(args),
        )
        print("Weights & Biases initialized successfully.")
        return wandb_run
    except Exception as exc:
        print(f"Could not initialize Weights & Biases: {exc}. Training without wandb logging.")
        return None


def load_optimizer_registry(project_root: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load custom optimizer and scheduler registries from local optimizer package."""
    optimizer_dir_path = os.path.join(project_root, "optimizer")
    if os.path.isdir(optimizer_dir_path) and project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from optimizer import OPTIMIZERS, SCHEDULERS

        print(f"Successfully imported custom optimizers: {list(OPTIMIZERS.keys())}")
        print(f"Successfully imported custom schedulers: {list(SCHEDULERS.keys())}")
        return OPTIMIZERS, SCHEDULERS
    except ImportError as exc:
        print(
            f"Warning: Custom optimizer/scheduler import failed ({exc}). "
            "Check optimizer/__init__.py. Standard torch modules only."
        )
        return {}, {}


def load_checkpoint_state(
    args: Any,
    model: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    target_dev: str,
    load_optimizer_state_fn: Any,
    load_scheduler_state_fn: Any,
    load_scaler_state_fn: Any,
) -> tuple[int, int]:
    """Load model/optimizer/scheduler/scaler state from args.resume."""
    start_epoch = 0
    initial_global_step = 0

    if args.resume:
        print(f"Resuming from {args.resume}")
        if not os.path.isfile(args.resume):
            print(f"Error: Resume file not found: {args.resume}")
            raise SystemExit(1)

        try:
            print("Loading model state_dict...")
            model.load_state_dict(load_file(args.resume), strict=True)
            print("Model state loaded successfully.")
        except Exception as exc:
            print(f"Error loading model state_dict from {args.resume}: {exc}")
            print("Attempting to load with strict=False (may indicate model structure mismatch)...")
            try:
                model.load_state_dict(load_file(args.resume), strict=False)
                print("Model state loaded with strict=False. Check for missing/unexpected keys.")
            except Exception as nonstrict_exc:
                print(f"Error loading model state_dict even with strict=False: {nonstrict_exc}")
                raise SystemExit(1)

        base_path = os.path.splitext(args.resume)[0]
        optim_path = f"{base_path}.optim"
        sched_path = f"{base_path}.sched"
        scaler_path = f"{base_path}.scaler"
        state_path = f"{base_path}.state"

        load_optimizer_state_fn(optimizer, optim_path, target_dev)
        load_scheduler_state_fn(scheduler, sched_path, target_dev)
        load_scaler_state_fn(scaler, scaler_path, target_dev)

        if os.path.exists(state_path):
            try:
                train_state = torch.load(state_path, map_location="cpu")
                start_epoch = train_state.get("epoch", 0)
                initial_global_step = train_state.get("global_step", 0)
                print(
                    "Loaded training state: "
                    f"Resuming from start of Epoch {start_epoch + 1}, Global Step {initial_global_step}"
                )
            except Exception as exc:
                print(
                    f"Warning: Could not load training state file '{state_path}': {exc}. "
                    "Resuming from epoch 0, step 0."
                )
                start_epoch, initial_global_step = 0, 0
        else:
            print(f"Warning: Training state file '{state_path}' not found. Resuming from epoch 0, step 0.")
            start_epoch, initial_global_step = 0, 0
    else:
        print("Starting new training run.")

    return start_epoch, initial_global_step
