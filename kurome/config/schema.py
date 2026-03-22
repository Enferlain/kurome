"""Typed configuration views used by package-level orchestration code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Normalized model configuration."""

    base: str
    rev: str
    arch: str
    base_vision_model: str
    embed_ver: str | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class DataConfig:
    """Normalized data configuration."""

    mode: str
    data_root: str | None
    feature_dir_name: str | None
    manifest_path: str | None
    artifact_key: str | None
    class_names: tuple[str, ...]
    val_split_count: int


@dataclass(frozen=True)
class TrainConfig:
    """Normalized training configuration subset used by orchestration."""

    optimizer: str
    loss_function: str | None
    lr: float
    batch: int
    precision: str
    seed: int
    num_workers: int
    max_train_steps: int | None
    max_train_epochs: int | None


@dataclass(frozen=True)
class LoggingConfig:
    """Normalized logging configuration."""

    wandb_project: str


@dataclass(frozen=True)
class PredictorParamsConfig:
    """Normalized predictor/head parameters used for embeddings mode."""

    features: int | None
    hidden_dim: int
    use_attention: bool
    num_attn_heads: int
    attn_dropout: float
    num_res_blocks: int
    dropout_rate: float
    output_mode: str | None
    rms_norm_eps: float
    pooling_strategy: str | None
    attn_pool_heads: int | None
    attn_pool_dropout: float | None


@dataclass(frozen=True)
class HeadParamsConfig:
    """Normalized head parameters used for sequence/E2E heads."""

    features: int | None
    hidden_dim: int
    pooling_strategy: str
    num_res_blocks: int
    dropout_rate: float
    output_mode: str | None
    attn_pool_heads: int
    attn_pool_dropout: float


@dataclass(frozen=True)
class E2EParamsConfig:
    """Normalized end-to-end extraction configuration."""

    is_end_to_end: bool
    extract_layer: int
    pooling_strategy: str
    freeze_base_model: bool


@dataclass(frozen=True)
class ExperimentConfig:
    """Normalized config view with access to compatibility runtime args."""

    config_path: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    logging: LoggingConfig
    predictor_params: PredictorParamsConfig | None
    head_params: HeadParamsConfig | None
    e2e_params: E2EParamsConfig
    run_name: str
    runtime_args: Any = field(repr=False, compare=False)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
