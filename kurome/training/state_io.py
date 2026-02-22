"""Checkpoint auxiliary-state loading helpers."""

from __future__ import annotations

import os

import torch


def load_optimizer_state(optimizer, path: str, device: str) -> bool:
    if os.path.exists(path):
        try:
            optimizer.load_state_dict(torch.load(path, map_location=device))
            print(f"Optimizer state loaded from {os.path.basename(path)}.")
            return True
        except Exception as e:
            print(f"Warning: Could not load optimizer state from {os.path.basename(path)}: {e}")
    else:
        print(f"Warning: Optimizer state file not found at {path}. Starting fresh.")
    return False


def load_scheduler_state(scheduler, path: str, device: str) -> bool:
    if scheduler is not None and os.path.exists(path):
        try:
            scheduler.load_state_dict(torch.load(path, map_location=device))
            print(f"Scheduler state loaded from {os.path.basename(path)}.")
            return True
        except Exception as e:
            print(f"Warning: Could not load scheduler state from {os.path.basename(path)}: {e}")
    return False


def load_scaler_state(scaler, path: str, device: str) -> bool:
    if scaler is not None and scaler.is_enabled() and os.path.exists(path):
        try:
            scaler.load_state_dict(torch.load(path, map_location=device))
            print(f"GradScaler state loaded from {os.path.basename(path)}.")
            return True
        except Exception as e:
            print(f"Warning: Could not load GradScaler state from {os.path.basename(path)}: {e}")
    return False
