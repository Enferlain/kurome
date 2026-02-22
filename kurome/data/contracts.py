"""Documented batch contracts consumed by training and inference code paths."""

from __future__ import annotations

from typing import TypedDict

import torch

# Embedding mode expects a dictionary batch with these keys.
EMBEDDING_BATCH_KEYS = ("emb", "val")

# Sequence mode expects a dictionary batch with these keys.
SEQUENCE_BATCH_KEYS = ("sequence", "mask", "label")

# Image mode expects a dictionary batch with these keys.
IMAGE_BATCH_KEYS = ("pixel_values", "label")


class EmbeddingBatch(TypedDict):
    """Embedding-mode batch item contract."""

    emb: torch.Tensor
    val: torch.Tensor


class SequenceBatch(TypedDict):
    """Sequence-mode batch item contract."""

    sequence: torch.Tensor
    mask: torch.Tensor
    label: torch.Tensor


class ImageBatch(TypedDict):
    """Image-mode batch item contract."""

    pixel_values: torch.Tensor
    label: torch.Tensor
