"""Legacy runtime-args compatibility helpers.

These helpers are kept package-local so CLI/runtime code no longer imports
root-level ``utils.py``.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os

import yaml

from .embed_params import get_embed_params

SAVE_FOLDER = "models"


def parse_and_load_args(config_path: str):
    """Load training arguments from YAML into an argparse namespace."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found.")
    try:
        with open(config_path, encoding="utf-8") as f:
            conf = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Error loading YAML config '{config_path}': {e}") from e

    args = argparse.Namespace()
    args.config_path = config_path

    def get_required_config(key_path, config_dict):
        keys = key_path.split(".")
        value = config_dict
        try:
            for key in keys:
                value = value[key]
            if value is None:
                raise KeyError
            return value
        except (KeyError, TypeError) as e:
            raise ValueError(f"Missing required config key: '{key_path}' in {config_path}") from e

    def get_optional_config(key_path, config_dict, default=None):
        keys = key_path.split(".")
        value = config_dict
        try:
            for key in keys:
                value = value[key]
            return default if value is None else value
        except (KeyError, TypeError):
            return default

    args.data_root = get_optional_config(
        "data.data_root",
        conf,
        default=get_optional_config("data_root", conf, default=None),
    )
    args.wandb_project = get_optional_config("wandb_project", conf, default="city-classifiers")
    args.resume = get_optional_config("resume", conf, default=None)
    args.base = get_required_config("model.base", conf)
    args.rev = get_required_config("model.rev", conf)
    args.arch = get_optional_config("model.arch", conf, default="class")
    args.name = f"{args.base}-{args.rev}"
    args.base_vision_model = get_required_config("model.base_vision_model", conf)
    args.embed_ver = get_optional_config("model.embed_ver", conf)

    args.data_mode = get_required_config("data.mode", conf)
    class_names_conf = get_optional_config("data.class_names", conf, default=None)
    if class_names_conf is not None:
        if not isinstance(class_names_conf, list) or not all(
            isinstance(name, str) and name.strip() for name in class_names_conf
        ):
            raise ValueError("Config Error: 'data.class_names' must be a list of non-empty strings.")
        args.class_names = [name.strip() for name in class_names_conf]
    else:
        args.class_names = None
    args.manifest_path = get_optional_config("data.manifest_path", conf, default=None)
    args.artifact_key = get_optional_config("data.artifact_key", conf, default=None)

    if args.data_mode == "embeddings":
        print("DEBUG: Loading config for EMBEDDINGS mode...")
        args.is_end_to_end = False
        if not args.embed_ver:
            raise ValueError("Config Error: 'model.embed_ver' must be specified.")
        try:
            embed_params = get_embed_params(args.embed_ver)
            args.features = embed_params["features"]
        except ValueError as e:
            raise ValueError(f"Config Error: {e}") from e
        args.preload_data = get_optional_config("train.preload_data", conf, default=True)

        predictor_conf = get_optional_config("predictor_params", conf, default={})
        if predictor_conf:
            print("DEBUG: Found predictor_params section.")
            default_hidden_pred = embed_params.get("hidden", 1280)
            args.hidden_dim = predictor_conf.get("hidden_dim", default_hidden_pred)
            args.use_attention = predictor_conf.get("use_attention", True)
            args.num_attn_heads = predictor_conf.get("num_attn_heads", 8)
            args.attn_dropout = predictor_conf.get("attn_dropout", 0.1)
            args.num_res_blocks = predictor_conf.get("num_res_blocks", 1)
            args.dropout_rate = predictor_conf.get("dropout_rate", 0.1)
            args.output_mode = predictor_conf.get("output_mode")

        head_conf = get_optional_config("head_params", conf, default={})
        if head_conf:
            print("DEBUG: Found head_params section.")
            args.head_features = head_conf.get("features")
            if args.head_features and args.head_features != args.features:
                args.features = args.head_features
            elif not hasattr(args, "features"):
                raise SystemExit("Error: Cannot determine input features for HeadModel.")
            args.head_hidden_dim = head_conf.get("hidden_dim", 1024)
            args.pooling_strategy = head_conf.get("pooling_strategy", "none")
            args.head_num_res_blocks = head_conf.get("num_res_blocks", 3)
            args.head_dropout_rate = head_conf.get("dropout_rate", 0.1)
            args.head_output_mode = head_conf.get("output_mode")
            attn_conf = get_optional_config("attn_pool_params", conf, default={})
            args.attn_pool_heads = attn_conf.get("attn_pool_heads", 16)
            args.attn_pool_dropout = attn_conf.get("attn_pool_dropout", 0.1)

        if not getattr(args, "output_mode", None) and not getattr(args, "head_output_mode", None):
            raise SystemExit(
                "Config Error: Missing 'output_mode' in 'predictor_params' OR 'head_params' section for embeddings mode."
            )

    elif args.data_mode == "features":
        print("DEBUG: Loading config for FEATURES mode...")
        args.is_end_to_end = False
        args.feature_dir_name = get_optional_config("data.feature_dir_name", conf, default=None)
        if not args.feature_dir_name and not (args.manifest_path and args.artifact_key):
            raise ValueError(
                "Config Error: features mode requires 'data.feature_dir_name' or both "
                "'data.manifest_path' and 'data.artifact_key'."
            )
        args.preload_data = False

        head_conf = get_optional_config("head_params", conf, default={})
        args.head_features = get_required_config("head_params.features", conf)
        args.head_hidden_dim = head_conf.get("hidden_dim", 1024)
        args.pooling_strategy = head_conf.get("pooling_strategy", "attn")
        args.head_num_res_blocks = head_conf.get("num_res_blocks", 3)
        args.head_dropout_rate = head_conf.get("dropout_rate", 0.2)
        args.head_output_mode = get_required_config("head_params.output_mode", conf)
        if args.pooling_strategy == "attn":
            attn_conf = get_optional_config("attn_pool_params", conf, default={})
            args.attn_pool_heads = attn_conf.get("attn_pool_heads", 16)
            args.attn_pool_dropout = attn_conf.get("attn_pool_dropout", 0.2)

    elif args.data_mode == "images":
        print("DEBUG: Loading config for E2E IMAGES mode...")
        args.is_end_to_end = True
        args.preload_data = False

        e2e_conf = get_optional_config("e2e_params", conf, default={})
        args.extract_layer = e2e_conf.get("extract_layer", -1)
        args.pooling_strategy = e2e_conf.get("pooling_strategy", "attn")
        args.freeze_base_model = e2e_conf.get("freeze_base_model", True)

        head_conf = get_optional_config("head_params", conf, default={})
        args.head_hidden_dim = head_conf.get("hidden_dim", 1024)
        args.head_num_res_blocks = head_conf.get("num_res_blocks", 2)
        args.head_dropout_rate = head_conf.get("dropout_rate", 0.2)
        args.head_output_mode = get_required_config("head_params.output_mode", conf)
        if args.pooling_strategy == "attn":
            attn_conf = get_optional_config("attn_pool_params", conf, default={})
            args.attn_pool_heads = attn_conf.get("attn_pool_heads", 8)
            args.attn_pool_dropout = attn_conf.get("attn_pool_dropout", 0.1)

    elif args.data_mode == "tensors":
        print("DEBUG: Loading config for TENSORS mode...")
        args.is_end_to_end = False
        if not args.manifest_path:
            raise ValueError("Config Error: 'data.manifest_path' must be specified for tensors mode.")
        if not args.artifact_key:
            raise ValueError("Config Error: 'data.artifact_key' must be specified for tensors mode.")
        args.preload_data = False

        head_conf = get_optional_config("head_params", conf, default={})
        args.head_hidden_dim = head_conf.get("hidden_dim", 256)
        args.head_num_res_blocks = head_conf.get("num_res_blocks", 3)
        args.head_dropout_rate = head_conf.get("dropout_rate", 0.1)
        args.head_output_mode = get_required_config("head_params.output_mode", conf)

    else:
        raise ValueError(
            f"Invalid data.mode '{args.data_mode}' in config. Must be 'embeddings', 'features', 'images', or 'tensors'."
        )

    train_conf = get_required_config("train", conf)
    args.lr = get_optional_config("lr", train_conf, default=1e-4)
    args.batch = get_optional_config("batch", train_conf, default=4)
    args.loss_function = get_optional_config("loss_function", train_conf)
    args.optimizer = get_optional_config("optimizer", train_conf, default="AdamW")
    args.betas = get_optional_config("betas", train_conf)
    args.eps = get_optional_config("eps", train_conf)
    args.weight_decay = get_optional_config("weight_decay", train_conf)
    args.max_train_epochs = get_optional_config("max_train_epochs", train_conf)
    args.max_train_steps = get_optional_config("max_train_steps", train_conf)
    args.precision = get_optional_config("precision", train_conf, default="fp32")
    args.nsave = get_optional_config("nsave", train_conf, default=0)
    args.val_split_count = get_optional_config("val_split_count", conf.get("data", {}), default=0)
    args.seed = get_optional_config("seed", train_conf, default=42)
    args.num_workers = get_optional_config("num_workers", train_conf, default=0)
    args.save_full_model = get_optional_config("save_full_model", train_conf, default=False)
    args.log_every_n = get_optional_config("log_every_n", train_conf, default=50)
    args.validate_every_n = get_optional_config("validate_every_n", train_conf, default=0)

    known_train_keys = {
        "lr",
        "batch",
        "loss_function",
        "optimizer",
        "betas",
        "eps",
        "weight_decay",
        "max_train_epochs",
        "max_train_steps",
        "precision",
        "nsave",
        "seed",
        "num_workers",
        "preload_data",
        "save_full_model",
        "log_every_n",
        "validate_every_n",
        "gradient_accumulation_steps",
        "train_drop_last",
        "prefetch_factor",
    }
    for key, value in train_conf.items():
        if isinstance(value, list):
            value = tuple(value)
        if key not in known_train_keys and not hasattr(args, key) and value is not None:
            setattr(args, key, value)

    if args.arch == "class":
        labels_conf = get_optional_config("labels", conf, default={})
        args.labels = None
        args.weights = None
        args.num_labels = 0
        if labels_conf:
            normalized_labels_conf = {str(k): v for k, v in labels_conf.items()}
            digit_keys = [key for key in normalized_labels_conf if key.isdigit()]
            named_keys = [key for key in normalized_labels_conf if not key.isdigit()]

            if named_keys and args.class_names is None:
                args.class_names = list(normalized_labels_conf.keys())

            if named_keys:
                if not args.class_names:
                    raise ValueError(
                        "Config Error: named labels require 'data.class_names' or label keys in desired order."
                    )
                missing = [name for name in args.class_names if name not in normalized_labels_conf]
                if missing:
                    raise ValueError(
                        f"Config Error: labels missing entries for class_names: {missing}"
                    )
                args.num_labels = len(args.class_names)
                args.labels = {
                    str(index): normalized_labels_conf[class_name].get("name", class_name)
                    for index, class_name in enumerate(args.class_names)
                }
                weights = [1.0] * args.num_labels
                for index, class_name in enumerate(args.class_names):
                    label_conf = normalized_labels_conf[class_name]
                    with suppress(ValueError, TypeError):
                        weights[index] = float(label_conf.get("loss", 1.0))
                args.weights = weights
            elif digit_keys:
                valid_labels = {key: normalized_labels_conf[key] for key in digit_keys}
                args.labels = {k: v.get("name", k) for k, v in valid_labels.items()}
                try:
                    args.num_labels = max(int(k) for k in args.labels) + 1
                except ValueError:
                    args.num_labels = 0
                if args.num_labels > 0:
                    weights = [1.0] * args.num_labels
                    for k_str, label_conf in valid_labels.items():
                        with suppress(ValueError, IndexError, TypeError):
                            weights[int(k_str)] = float(label_conf.get("loss", 1.0))
                    args.weights = weights
        if args.num_labels == 0:
            args.num_labels = 2
    else:
        args.labels = None
        args.weights = None
        args.num_labels = 1

    if args.arch == "score" and args.loss_function is None:
        args.loss_function = "l1"
    if args.arch == "class" and args.loss_function is None:
        args.loss_function = "focal"
    if args.max_train_epochs is None and args.max_train_steps is None:
        args.max_train_steps = 10000

    if args.data_mode == "embeddings" and not getattr(args, "output_mode", None) and not getattr(args, "head_output_mode", None):
        raise SystemExit(
            "Config Error: Missing 'output_mode' in EITHER 'predictor_params' OR 'head_params' section for embeddings mode."
        )
    if args.data_mode == "features" and not getattr(args, "head_output_mode", None):
        raise SystemExit("Config Error: Missing 'head_output_mode' in 'head_params' for features mode.")
    if args.data_mode == "images" and not getattr(args, "head_output_mode", None):
        raise SystemExit("Config Error: Missing 'head_output_mode' in 'head_params' for images mode.")
    if args.data_mode == "tensors" and not getattr(args, "head_output_mode", None):
        raise SystemExit("Config Error: Missing 'head_output_mode' in 'head_params' for tensors mode.")

    return args


def write_config(args):
    """Write effective runtime args to ``models/<name>.config.json``."""
    conf_to_save = vars(args).copy()
    conf_to_save.pop("config_path", None)

    os.makedirs(SAVE_FOLDER, exist_ok=True)
    config_path_out = f"{SAVE_FOLDER}/{getattr(args, 'name', 'config')}.config.json"
    try:
        with open(config_path_out, "w", encoding="utf-8") as f:
            f.write(json.dumps(conf_to_save, indent=2, default=repr))
        print(f"Saved final effective training config to {config_path_out}")
    except Exception as e:
        print(f"Error saving config file {config_path_out}: {e}")
