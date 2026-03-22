"""Lightweight registries for model classes."""

from __future__ import annotations

from .backbones import EarlyExtractAnatomyModel
from .heads import (
    ForensicConvMILHeadModel,
    ForensicMILHeadModel,
    ForensicMultiPoolHeadModel,
    HeadModel,
    HybridHeadModel,
    PredictorModel,
    TensorCNNModel,
)

MODEL_REGISTRY: dict[str, type] = {
    "predictor_model": PredictorModel,
    "head_model": HeadModel,
    "hybrid_head_model": HybridHeadModel,
    "tensor_cnn_model": TensorCNNModel,
    "forensic_mil_head_model": ForensicMILHeadModel,
    "forensic_multipool_head_model": ForensicMultiPoolHeadModel,
    "forensic_convmil_head_model": ForensicConvMILHeadModel,
    "early_extract_model": EarlyExtractAnatomyModel,
}


def get_model_class(model_id: str) -> type:
    """Get a registered model class by id."""
    key = model_id.lower()
    if key not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model id '{model_id}'. Known ids: {known}")
    return MODEL_REGISTRY[key]
