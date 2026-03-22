"""Configuration loading bridge.

This module provides package-level config entrypoints while preserving legacy
behavior through a package-local runtime-args compatibility parser.
"""

from __future__ import annotations

import copy
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    DataConfig,
    E2EParamsConfig,
    ExperimentConfig,
    HeadParamsConfig,
    LoggingConfig,
    ModelConfig,
    PredictorParamsConfig,
    TrainConfig,
)


def _as_mapping(value: Any, *, key: str, config_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Config key '{key}' in '{config_path}' must be a mapping.")
    return value


def _as_str(value: Any, *, key: str, config_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config key '{key}' in '{config_path}' must be a non-empty string.")
    return value


def _as_int(value: Any, *, key: str, config_path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Config key '{key}' in '{config_path}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError as exc:
            raise ValueError(f"Config key '{key}' in '{config_path}' must be an integer.") from exc
    raise ValueError(f"Config key '{key}' in '{config_path}' must be an integer.")


def _as_number(value: Any, *, key: str, config_path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Config key '{key}' in '{config_path}' must be a number.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise ValueError(f"Config key '{key}' in '{config_path}' must be a number.") from exc
    raise ValueError(f"Config key '{key}' in '{config_path}' must be a number.")


def _as_bool(value: Any, *, key: str, config_path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Config key '{key}' in '{config_path}' must be a boolean.")


def _optional_int(value: Any, *, key: str, config_path: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, key=key, config_path=config_path)


def _optional_number(value: Any, *, key: str, config_path: str) -> float | None:
    if value is None:
        return None
    return _as_number(value, key=key, config_path=config_path)


def _infer_mode_from_raw(raw: dict[str, Any]) -> str:
    data_raw = raw.get("data", {}) if isinstance(raw.get("data", {}), dict) else {}
    mode = data_raw.get("mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip().lower()

    model_raw = raw.get("model", {}) if isinstance(raw.get("model", {}), dict) else {}
    if isinstance(data_raw.get("feature_dir_name"), str) and data_raw.get("feature_dir_name"):
        return "features"
    is_e2e = model_raw.get("is_end_to_end")
    if isinstance(is_e2e, bool) and is_e2e:
        return "images"
    if isinstance(model_raw.get("embed_ver"), str) and model_raw.get("embed_ver"):
        return "embeddings"
    if "e2e_params" in raw:
        return "images"
    return "embeddings"


def _build_legacy_compatible_raw(raw: dict[str, Any], *, config_path: str) -> dict[str, Any]:
    prepared = copy.deepcopy(raw)

    data_raw = prepared.get("data")
    if not isinstance(data_raw, dict):
        data_raw = {}
    prepared["data"] = data_raw

    mode = _infer_mode_from_raw(prepared)
    data_raw.setdefault("mode", mode)

    model_raw = prepared.get("model", {})
    if not isinstance(model_raw, dict):
        model_raw = {}
    prepared["model"] = model_raw

    if mode == "features" and not data_raw.get("feature_dir_name"):
        embed_ver = model_raw.get("embed_ver")
        if isinstance(embed_ver, str) and embed_ver:
            data_raw["feature_dir_name"] = embed_ver

    if "val_split_count" not in data_raw:
        train_raw = prepared.get("train", {})
        if isinstance(train_raw, dict) and "val_split_count" in train_raw:
            data_raw["val_split_count"] = train_raw.get("val_split_count")
        else:
            data_raw["val_split_count"] = 0

    model_raw.setdefault("arch", "class")

    return prepared


def load_raw_config(config_path: str) -> dict[str, Any]:
    """Load raw YAML config from disk."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{config_path}' must contain a YAML mapping at top level.")
    return data


def parse_and_load_args(config_path: str):
    """Compatibility wrapper around the existing legacy parser."""
    from kurome.config.runtime_args import parse_and_load_args as legacy_parse_and_load_args

    raw = load_raw_config(config_path=config_path)
    prepared = _build_legacy_compatible_raw(raw, config_path=config_path)
    if prepared == raw:
        return legacy_parse_and_load_args(config_path=config_path)

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
            yaml.safe_dump(prepared, tmp, sort_keys=False)
            tmp_path = tmp.name
        return legacy_parse_and_load_args(config_path=tmp_path)
    finally:
        if tmp_path:
            with suppress(OSError):
                Path(tmp_path).unlink(missing_ok=True)


def normalize_experiment_config(
    raw: dict[str, Any],
    *,
    config_path: str,
    runtime_args: Any,
) -> ExperimentConfig:
    """Normalize/validate raw YAML into a typed experiment config."""
    model_raw = _as_mapping(raw.get("model"), key="model", config_path=config_path)
    data_val = raw.get("data", {})
    if data_val is None:
        data_val = {}
    data_raw = _as_mapping(data_val, key="data", config_path=config_path)
    train_raw = _as_mapping(raw.get("train"), key="train", config_path=config_path)
    predictor_raw = _as_mapping(raw.get("predictor_params", {}), key="predictor_params", config_path=config_path)
    head_raw = _as_mapping(raw.get("head_params", {}), key="head_params", config_path=config_path)
    attn_pool_raw = _as_mapping(
        raw.get("attn_pool_params", {}),
        key="attn_pool_params",
        config_path=config_path,
    )
    e2e_raw = _as_mapping(raw.get("e2e_params", {}), key="e2e_params", config_path=config_path)

    model_id_raw = model_raw.get("model_id")
    if model_id_raw is not None and (not isinstance(model_id_raw, str) or not model_id_raw.strip()):
        raise ValueError(f"Config key 'model.model_id' in '{config_path}' must be a non-empty string.")

    model = ModelConfig(
        base=_as_str(model_raw.get("base"), key="model.base", config_path=config_path),
        rev=_as_str(model_raw.get("rev"), key="model.rev", config_path=config_path),
        arch=_as_str(model_raw.get("arch", "class"), key="model.arch", config_path=config_path),
        base_vision_model=_as_str(
            model_raw.get("base_vision_model"),
            key="model.base_vision_model",
            config_path=config_path,
        ),
        embed_ver=model_raw.get("embed_ver"),
        model_id=model_id_raw.strip().lower() if isinstance(model_id_raw, str) else None,
    )

    mode = _infer_mode_from_raw(raw)
    if mode not in {"embeddings", "features", "images", "tensors"}:
        raise ValueError(
            f"Config key 'data.mode' in '{config_path}' must be one of "
            "'embeddings', 'features', 'images', or 'tensors'."
        )

    feature_dir_name = data_raw.get("feature_dir_name")
    if feature_dir_name is not None and not isinstance(feature_dir_name, str):
        raise ValueError(f"Config key 'data.feature_dir_name' in '{config_path}' must be a string.")
    manifest_path = data_raw.get("manifest_path")
    if manifest_path is not None and (not isinstance(manifest_path, str) or not manifest_path.strip()):
        raise ValueError(f"Config key 'data.manifest_path' in '{config_path}' must be a non-empty string.")
    artifact_key = data_raw.get("artifact_key")
    if artifact_key is not None and (not isinstance(artifact_key, str) or not artifact_key.strip()):
        raise ValueError(f"Config key 'data.artifact_key' in '{config_path}' must be a non-empty string.")
    if mode == "features" and not feature_dir_name and not (manifest_path and artifact_key):
        raise ValueError(
            "Config for data.mode='features' must provide either 'data.feature_dir_name' "
            "or both 'data.manifest_path' and 'data.artifact_key'."
        )

    class_names_raw = data_raw.get("class_names")
    if class_names_raw is None:
        class_names: tuple[str, ...] = ()
    else:
        if not isinstance(class_names_raw, list) or not all(
            isinstance(item, str) and item.strip() for item in class_names_raw
        ):
            raise ValueError(
                f"Config key 'data.class_names' in '{config_path}' must be a list of non-empty strings."
            )
        class_names = tuple(item.strip() for item in class_names_raw)

    data_root_raw = data_raw.get("data_root", raw.get("data_root"))
    if data_root_raw is None:
        data_root: str | None = None
    elif isinstance(data_root_raw, str) and data_root_raw.strip():
        data_root = data_root_raw
    else:
        raise ValueError(f"Config key 'data_root' in '{config_path}' must be a non-empty string when provided.")

    if mode == "tensors":
        if not manifest_path:
            raise ValueError(f"Config key 'data.manifest_path' is required when data.mode='tensors'.")
        if not artifact_key:
            raise ValueError(f"Config key 'data.artifact_key' is required when data.mode='tensors'.")
    elif mode == "features" and manifest_path and artifact_key:
        pass
    elif mode != "images" and not data_root:
        raise ValueError(f"Config key 'data_root' in '{config_path}' is required when data.mode='{mode}'.")
    if mode == "images" and not data_root and not manifest_path:
        raise ValueError(
            f"Config in '{config_path}' must provide either 'data.data_root' or 'data.manifest_path' when data.mode='images'."
        )

    val_split_raw = data_raw.get("val_split_count", train_raw.get("val_split_count", 0))

    data = DataConfig(
        mode=mode,
        data_root=data_root,
        feature_dir_name=feature_dir_name,
        manifest_path=manifest_path.strip() if isinstance(manifest_path, str) else None,
        artifact_key=artifact_key.strip() if isinstance(artifact_key, str) else None,
        class_names=class_names,
        val_split_count=_as_int(
            val_split_raw,
            key="data.val_split_count",
            config_path=config_path,
        ),
    )

    train = TrainConfig(
        optimizer=str(train_raw.get("optimizer", "adamw")),
        loss_function=train_raw.get("loss_function"),
        lr=_as_number(train_raw.get("lr", 1e-4), key="train.lr", config_path=config_path),
        batch=_as_int(train_raw.get("batch", 1), key="train.batch", config_path=config_path),
        precision=str(train_raw.get("precision", "fp32")),
        seed=_as_int(train_raw.get("seed", 42), key="train.seed", config_path=config_path),
        num_workers=_as_int(train_raw.get("num_workers", 0), key="train.num_workers", config_path=config_path),
        max_train_steps=train_raw.get("max_train_steps"),
        max_train_epochs=train_raw.get("max_train_epochs"),
    )
    if train.max_train_steps is not None:
        _as_int(train.max_train_steps, key="train.max_train_steps", config_path=config_path)
    if train.max_train_epochs is not None:
        _as_int(train.max_train_epochs, key="train.max_train_epochs", config_path=config_path)

    logging = LoggingConfig(
        wandb_project=str(raw.get("wandb_project", "city-classifiers")),
    )

    model_is_e2e_raw = model_raw.get("is_end_to_end", False)
    model_is_e2e = _as_bool(model_is_e2e_raw, key="model.is_end_to_end", config_path=config_path) if model_is_e2e_raw is not None else False

    predictor_params = PredictorParamsConfig(
        features=_optional_int(predictor_raw.get("features"), key="predictor_params.features", config_path=config_path),
        hidden_dim=_as_int(
            predictor_raw.get("hidden_dim", 1280),
            key="predictor_params.hidden_dim",
            config_path=config_path,
        ),
        use_attention=_as_bool(
            predictor_raw.get("use_attention", True),
            key="predictor_params.use_attention",
            config_path=config_path,
        ),
        num_attn_heads=_as_int(
            predictor_raw.get("num_attn_heads", 16),
            key="predictor_params.num_attn_heads",
            config_path=config_path,
        ),
        attn_dropout=_as_number(
            predictor_raw.get("attn_dropout", 0.1),
            key="predictor_params.attn_dropout",
            config_path=config_path,
        ),
        num_res_blocks=_as_int(
            predictor_raw.get("num_res_blocks", 3),
            key="predictor_params.num_res_blocks",
            config_path=config_path,
        ),
        dropout_rate=_as_number(
            predictor_raw.get("dropout_rate", 0.1),
            key="predictor_params.dropout_rate",
            config_path=config_path,
        ),
        output_mode=(
            str(predictor_raw.get("output_mode")).lower()
            if isinstance(predictor_raw.get("output_mode"), str) and predictor_raw.get("output_mode")
            else None
        ),
        rms_norm_eps=_as_number(
            predictor_raw.get("rms_norm_eps", 1e-6),
            key="predictor_params.rms_norm_eps",
            config_path=config_path,
        ),
        pooling_strategy=(
            str(predictor_raw.get("pooling_strategy"))
            if isinstance(predictor_raw.get("pooling_strategy"), str)
            else None
        ),
        attn_pool_heads=_optional_int(
            predictor_raw.get("attn_pool_heads"),
            key="predictor_params.attn_pool_heads",
            config_path=config_path,
        ),
        attn_pool_dropout=_optional_number(
            predictor_raw.get("attn_pool_dropout"),
            key="predictor_params.attn_pool_dropout",
            config_path=config_path,
        ),
    )

    head_features_raw = head_raw.get("features", predictor_raw.get("features"))
    head_output_mode_raw = head_raw.get("output_mode", predictor_raw.get("output_mode"))
    head_attn_heads_raw = head_raw.get("attn_pool_heads", attn_pool_raw.get("attn_pool_heads", model_raw.get("attn_pool_heads", 16)))
    head_attn_dropout_raw = head_raw.get("attn_pool_dropout", attn_pool_raw.get("attn_pool_dropout", model_raw.get("attn_pool_dropout", 0.2)))
    head_pooling_raw = head_raw.get("pooling_strategy", model_raw.get("pooling_strategy", "attn"))
    head_params = HeadParamsConfig(
        features=_optional_int(head_features_raw, key="head_params.features", config_path=config_path),
        hidden_dim=_as_int(head_raw.get("hidden_dim", 1024), key="head_params.hidden_dim", config_path=config_path),
        pooling_strategy=_as_str(head_pooling_raw, key="head_params.pooling_strategy", config_path=config_path),
        num_res_blocks=_as_int(
            head_raw.get("num_res_blocks", 3),
            key="head_params.num_res_blocks",
            config_path=config_path,
        ),
        dropout_rate=_as_number(
            head_raw.get("dropout_rate", 0.2),
            key="head_params.dropout_rate",
            config_path=config_path,
        ),
        output_mode=(str(head_output_mode_raw).lower() if isinstance(head_output_mode_raw, str) and head_output_mode_raw else None),
        attn_pool_heads=_as_int(head_attn_heads_raw, key="head_params.attn_pool_heads", config_path=config_path),
        attn_pool_dropout=_as_number(
            head_attn_dropout_raw,
            key="head_params.attn_pool_dropout",
            config_path=config_path,
        ),
    )

    e2e_extract_raw = e2e_raw.get("extract_layer", model_raw.get("extract_layer", -1))
    e2e_pool_raw = e2e_raw.get("pooling_strategy", model_raw.get("pooling_strategy", "attn"))
    e2e_freeze_raw = e2e_raw.get("freeze_base_model", model_raw.get("freeze_base_model", True))
    e2e_params = E2EParamsConfig(
        is_end_to_end=model_is_e2e or mode == "images",
        extract_layer=_as_int(e2e_extract_raw, key="e2e_params.extract_layer", config_path=config_path),
        pooling_strategy=_as_str(e2e_pool_raw, key="e2e_params.pooling_strategy", config_path=config_path),
        freeze_base_model=_as_bool(
            e2e_freeze_raw,
            key="e2e_params.freeze_base_model",
            config_path=config_path,
        ),
    )

    if mode == "images" and head_params.output_mode is None:
        raise ValueError("Config key 'head_params.output_mode' is required when data.mode='images'.")
    if mode == "features" and head_params.output_mode is None:
        raise ValueError("Config key 'head_params.output_mode' is required when data.mode='features'.")
    if mode == "tensors" and head_params.output_mode is None:
        raise ValueError("Config key 'head_params.output_mode' is required when data.mode='tensors'.")
    if mode == "embeddings" and predictor_params.output_mode is None and head_params.output_mode is None:
        raise ValueError(
            "Config requires predictor_params.output_mode or head_params.output_mode when data.mode='embeddings'."
        )

    if getattr(runtime_args, "model_id", None) is None and isinstance(model.model_id, str):
        model_id = model.model_id.strip().lower()
        if model_id:
            runtime_args.model_id = model_id

    return ExperimentConfig(
        config_path=config_path,
        model=model,
        data=data,
        train=train,
        logging=logging,
        predictor_params=predictor_params,
        head_params=head_params,
        e2e_params=e2e_params,
        run_name=f"{model.base}-{model.rev}",
        runtime_args=runtime_args,
        raw=raw,
    )


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load config and expose a normalized typed view plus runtime args."""
    raw = load_raw_config(config_path=config_path)
    args = parse_and_load_args(config_path=config_path)
    return normalize_experiment_config(
        raw,
        config_path=config_path,
        runtime_args=args,
    )
