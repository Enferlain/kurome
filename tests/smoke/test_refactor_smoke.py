import py_compile
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_root_wrappers_removed() -> None:
    for rel_path in ["train.py", "train_features.py", "inference.py"]:
        assert not (REPO_ROOT / rel_path).exists()


def test_legacy_root_model_and_dataset_modules_removed() -> None:
    for rel_path in [
        "head_model.py",
        "hybrid_model.py",
        "model_early_extract.py",
        "dataset.py",
        "sequence_dataset.py",
        "image_dataset.py",
    ]:
        assert not (REPO_ROOT / rel_path).exists()


def test_legacy_root_helper_modules_removed() -> None:
    for rel_path in [
        "model.py",
        "losses.py",
        "utils.py",
    ]:
        assert not (REPO_ROOT / rel_path).exists()


def test_cli_modules_define_main() -> None:
    for rel_path in [
        "kurome/cli/train_embeddings.py",
        "kurome/cli/train_features.py",
    ]:
        module = ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))
        fn_names = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        assert "main" in fn_names, f"main() missing in {rel_path}"


def test_inference_pipeline_uses_package_head_adapters() -> None:
    text = (REPO_ROOT / "kurome" / "inference" / "pipeline.py").read_text(encoding="utf-8")
    assert "from kurome.models.heads import (" in text
    assert "from kurome.inference.postprocess import (" in text
    assert "from model import PredictorModel" not in text
    assert "from head_model import HeadModel" not in text
    assert "from hybrid_model import HybridHeadModel" not in text


def test_package_model_modules_do_not_import_legacy_root_models() -> None:
    sequence_text = (
        REPO_ROOT / "kurome" / "models" / "heads" / "sequence_head.py"
    ).read_text(encoding="utf-8")
    hybrid_text = (
        REPO_ROOT / "kurome" / "models" / "heads" / "hybrid_head.py"
    ).read_text(encoding="utf-8")
    early_text = (
        REPO_ROOT / "kurome" / "models" / "backbones" / "early_extract.py"
    ).read_text(encoding="utf-8")

    assert "from head_model import HeadModel" not in sequence_text
    assert "from hybrid_model import HybridHeadModel" not in hybrid_text
    assert "from model_early_extract import EarlyExtractAnatomyModel" not in early_text


def test_quality_gate_script_exists() -> None:
    script_path = REPO_ROOT / "scripts" / "quality" / "run_quality.sh"
    text = script_path.read_text(encoding="utf-8")
    assert "ruff" in text
    assert "check_root_surface.py" in text
    assert "check_wrapper_references.py" in text
    assert "pytest tests/unit" in text
    assert "pytest tests/integration" in text
    assert "scripts/smoke/run_smoke.sh" in text


def test_extension_docs_exist() -> None:
    for rel_path in [
        "docs/how-to-add-model.md",
        "docs/how-to-add-dataset.md",
        "docs/how-it-works-now.md",
        "docs/deprecations.md",
        "docs/root-surface.md",
        "docs/wrapper-removal-checklist.md",
    ]:
        assert (REPO_ROOT / rel_path).is_file()


def test_wrapper_reference_checker_exists() -> None:
    checker = REPO_ROOT / "scripts" / "quality" / "check_wrapper_references.py"
    allowlist = REPO_ROOT / "scripts" / "quality" / "wrapper_reference_allowlist.txt"
    assert checker.is_file()
    assert allowlist.is_file()


def test_config_loader_reads_yaml_mapping() -> None:
    from kurome.config.loader import load_experiment_config, load_raw_config

    config_path = REPO_ROOT / "config" / "anatomy_dinov3_7b_bnb.yaml"
    config = load_raw_config(str(config_path))
    assert isinstance(config, dict)
    assert "model" in config
    assert "data" in config
    assert "train" in config
    experiment = load_experiment_config(str(config_path))
    assert experiment.data.mode == "embeddings"
    assert experiment.model.base
    assert experiment.run_name == experiment.runtime_args.name
    assert experiment.predictor_params is not None
    assert experiment.head_params is not None
    assert experiment.e2e_params.is_end_to_end is False


def test_config_loader_supports_legacy_yaml_shape() -> None:
    from kurome.config.loader import load_experiment_config

    experiment = load_experiment_config(str(REPO_ROOT / "config" / "anatomy_dinov2.yaml"))
    assert experiment.data.mode == "embeddings"
    assert experiment.runtime_args.data_mode == "embeddings"
    assert experiment.predictor_params.output_mode is not None
    assert experiment.head_params.output_mode is not None


def test_config_normalization_rejects_invalid_mode() -> None:
    from kurome.config.loader import normalize_experiment_config

    bad = {
        "model": {
            "base": "x",
            "rev": "y",
            "arch": "class",
            "base_vision_model": "facebook/dinov2-giant",
        },
        "data": {"mode": "invalid"},
        "train": {"lr": 1e-4, "batch": 1, "seed": 1, "num_workers": 0},
    }
    with pytest.raises(ValueError):
        normalize_experiment_config(bad, config_path="bad.yaml", runtime_args=object())


def test_data_contract_constants() -> None:
    from kurome.data.contracts import EMBEDDING_BATCH_KEYS, IMAGE_BATCH_KEYS, SEQUENCE_BATCH_KEYS

    assert EMBEDDING_BATCH_KEYS == ("emb", "val")
    assert SEQUENCE_BATCH_KEYS == ("sequence", "mask", "label")
    assert IMAGE_BATCH_KEYS == ("pixel_values", "label")


def test_model_factory_registry_basics() -> None:
    from kurome.models.factory import (
        build_criterion,
        build_model_with_filtered_kwargs,
        resolve_num_classes,
    )
    from kurome.models.registry import get_model_class

    model_cls = get_model_class("head_model")
    assert model_cls.__name__ == "HeadModel"

    resolved_loss, num_classes, warnings = resolve_num_classes(
        arch="class",
        loss_function=None,
        dataset_num_labels=2,
    )
    assert resolved_loss == "focal"
    assert num_classes == 2
    assert warnings

    criterion = build_criterion(
        loss_name="crossentropy",
        num_classes=2,
        output_mode="linear",
        class_weights_tensor=None,
        args=object(),
    )
    assert isinstance(criterion, nn.CrossEntropyLoss)

    criterion_bce_multi = build_criterion(
        loss_name="bce",
        num_classes=3,
        output_mode="sigmoid",
        class_weights_tensor=None,
        args=object(),
        allow_bce_multiclass=True,
        require_linear_logits_losses=False,
    )
    assert isinstance(criterion_bce_multi, nn.BCEWithLogitsLoss)

    predictor = build_model_with_filtered_kwargs(
        "predictor_model",
        {"features": 8, "num_classes": 2, "output_mode": "linear", "hidden_dim": 16},
    )
    assert predictor.__class__.__name__ == "PredictorModel"
    with pytest.raises(ValueError):
        build_model_with_filtered_kwargs("early_extract_model", {"hidden_dim": 16})

    registry_text = (REPO_ROOT / "kurome" / "models" / "registry.py").read_text(encoding="utf-8")
    factory_text = (REPO_ROOT / "kurome" / "models" / "factory.py").read_text(encoding="utf-8")
    engine_text = (REPO_ROOT / "kurome" / "training" / "engine.py").read_text(encoding="utf-8")

    assert "from .heads import HeadModel, HybridHeadModel, PredictorModel" in registry_text
    assert "from .backbones import EarlyExtractAnatomyModel" in registry_text
    assert "from head_model import HeadModel" not in registry_text
    assert "from model import PredictorModel" not in registry_text

    assert "from kurome.models.tasks import FocalLoss, GHMC_Loss" in factory_text
    assert "from losses import FocalLoss, GHMC_Loss" not in factory_text
    assert "from kurome.models.tasks import FocalLoss, GHMC_Loss" in engine_text
    assert "from losses import FocalLoss, GHMC_Loss" not in engine_text

    tasks_losses_text = (
        REPO_ROOT / "kurome" / "models" / "tasks" / "losses.py"
    ).read_text(encoding="utf-8")
    predictor_text = (
        REPO_ROOT / "kurome" / "models" / "heads" / "predictor.py"
    ).read_text(encoding="utf-8")
    assert "from losses import FocalLoss, GHMC_Loss" not in tasks_losses_text
    assert "from model import PredictorModel" not in predictor_text


def test_config_normalization_propagates_model_id() -> None:
    from kurome.config.loader import normalize_experiment_config

    runtime_args = SimpleNamespace()
    raw = {
        "model": {
            "base": "base-x",
            "rev": "r1",
            "arch": "class",
            "base_vision_model": "facebook/dinov2-giant",
            "model_id": "predictor_model",
        },
        "data": {"mode": "embeddings"},
        "predictor_params": {"output_mode": "linear"},
        "train": {"lr": 1e-4, "batch": 1, "seed": 1, "num_workers": 0},
    }
    experiment = normalize_experiment_config(raw, config_path="test.yaml", runtime_args=runtime_args)
    assert experiment.model.model_id == "predictor_model"
    assert runtime_args.model_id == "predictor_model"


def test_config_normalization_uses_raw_data_keys_over_runtime_fallbacks() -> None:
    from kurome.config.loader import normalize_experiment_config

    runtime_args = SimpleNamespace(data_root="runtime-root", val_split_count=999)
    raw = {
        "model": {
            "base": "base-x",
            "rev": "r1",
            "arch": "class",
            "base_vision_model": "facebook/dinov2-giant",
        },
        "data": {"mode": "embeddings", "data_root": "yaml-root", "val_split_count": 12},
        "predictor_params": {"output_mode": "linear"},
        "train": {"lr": 1e-4, "batch": 1, "seed": 1, "num_workers": 0},
    }
    experiment = normalize_experiment_config(raw, config_path="test.yaml", runtime_args=runtime_args)
    assert experiment.data.data_root == "yaml-root"
    assert experiment.data.val_split_count == 12


def test_config_normalization_infers_images_mode_from_model_flag() -> None:
    from kurome.config.loader import normalize_experiment_config

    raw = {
        "model": {
            "base": "base-x",
            "rev": "r1",
            "arch": "class",
            "base_vision_model": "apple/aimv2-large-patch14-native",
            "is_end_to_end": True,
        },
        "head_params": {"output_mode": "linear"},
        "train": {"lr": 1e-4, "batch": 1, "seed": 1, "num_workers": 0},
    }
    experiment = normalize_experiment_config(raw, config_path="test.yaml", runtime_args=SimpleNamespace())
    assert experiment.data.mode == "images"
    assert experiment.e2e_params.is_end_to_end is True


def test_train_embeddings_uses_factory_registry_path() -> None:
    text = (REPO_ROOT / "kurome" / "cli" / "train_embeddings.py").read_text(encoding="utf-8")
    assert "from kurome.models.factory import (" in text
    assert "from model_early_extract import EarlyExtractAnatomyModel" not in text
    assert "from utils import (" not in text
    assert 'embedding_model_id = str(experiment.model.model_id or "hybrid_head_model").strip().lower()' in text


def test_training_clis_delegate_dataloaders_to_data_layer() -> None:
    embeddings_text = (REPO_ROOT / "kurome" / "cli" / "train_embeddings.py").read_text(
        encoding="utf-8"
    )
    features_text = (REPO_ROOT / "kurome" / "cli" / "train_features.py").read_text(
        encoding="utf-8"
    )

    assert "from kurome.data.embeddings import build_embedding_training_dataloaders" in embeddings_text
    assert "from kurome.data.images import build_image_training_dataloaders" in embeddings_text
    assert "from kurome.data.transforms import load_image_processor" in embeddings_text
    assert "return build_image_training_dataloaders(args, image_processor=image_processor)" in embeddings_text
    assert "return build_embedding_training_dataloaders(args, image_processor=image_processor)" in embeddings_text

    assert "from kurome.data.sequences import build_feature_sequence_dataloaders" in features_text
    assert "return build_feature_sequence_dataloaders(args)" in features_text
    assert "from utils import (" not in features_text


def test_config_loader_avoids_root_utils_dependency() -> None:
    text = (REPO_ROOT / "kurome" / "config" / "loader.py").read_text(encoding="utf-8")
    assert "from utils import parse_and_load_args" not in text
    assert "from kurome.config.runtime_args import parse_and_load_args as legacy_parse_and_load_args" in text


def test_data_adapters_use_shared_dataloader_helper() -> None:
    embeddings_text = (REPO_ROOT / "kurome" / "data" / "embeddings.py").read_text(encoding="utf-8")
    images_text = (REPO_ROOT / "kurome" / "data" / "images.py").read_text(encoding="utf-8")
    sequences_text = (REPO_ROOT / "kurome" / "data" / "sequences.py").read_text(encoding="utf-8")

    assert "from kurome.data.dataloaders import (" in embeddings_text
    assert "build_training_dataloader(" in embeddings_text
    assert "build_validation_dataloader(" in embeddings_text
    assert "log_train_val_loader_summary(" in embeddings_text

    assert "from kurome.data.dataloaders import (" in images_text
    assert "build_training_dataloader(" in images_text
    assert "build_validation_dataloader(" in images_text
    assert "log_train_val_loader_summary(" in images_text

    assert "from kurome.data.dataloaders import (" in sequences_text
    assert "build_training_dataloader(" in sequences_text
    assert "build_validation_dataloader(" in sequences_text
    assert "log_train_val_loader_summary(" in sequences_text

    assert "from dataset import EmbeddingDataset" not in embeddings_text
    assert "from image_dataset import ImageFolderDataset, collate_group_by_size" not in images_text
    assert "from sequence_dataset import FeatureSequenceDataset, collate_sequences" not in sequences_text


def test_train_embeddings_avoids_direct_autoprocessor_import() -> None:
    text = (REPO_ROOT / "kurome" / "cli" / "train_embeddings.py").read_text(encoding="utf-8")

    assert "from transformers import AutoProcessor" not in text
    assert "image_processor = load_image_processor(experiment.model.base_vision_model)" in text


def test_training_clis_delegate_training_loops() -> None:
    embeddings_text = (REPO_ROOT / "kurome" / "cli" / "train_embeddings.py").read_text(
        encoding="utf-8"
    )
    features_text = (REPO_ROOT / "kurome" / "cli" / "train_features.py").read_text(
        encoding="utf-8"
    )

    assert "from kurome.training.loops import run_embedding_training_loop" in embeddings_text
    assert "run_embedding_training_loop(" in embeddings_text
    assert 'is_e2e=getattr(args, "is_end_to_end", False)' in embeddings_text

    assert "from kurome.training.loops import run_feature_sequence_training_loop" in features_text
    assert "run_feature_sequence_training_loop(" in features_text


def test_training_loops_use_shared_engine_helpers() -> None:
    loops_text = (REPO_ROOT / "kurome" / "training" / "loops.py").read_text(encoding="utf-8")

    assert "from kurome.training.engine import (" in loops_text
    assert "ensure_training_mode(" in loops_text
    assert "maybe_step_scheduler(" in loops_text
    assert "update_progress_postfix(" in loops_text
    assert "prepare_prediction_for_loss(" in loops_text
    assert "compute_loss(" in loops_text
    assert "prepare_embedding_sub_batch(" in loops_text
    assert "prepare_sequence_micro_batch(" in loops_text
    assert "normalize_embedding_batch_data(" in loops_text
    assert "advance_global_step(" in loops_text
    assert "has_reached_total_steps(" in loops_text
    assert "should_run_interval(" in loops_text
    assert "restore_training_mode_if_needed(" in loops_text
    assert "run_optimizer_step_and_zero_grad(" in loops_text
    assert "accumulate_scalar_loss(" in loops_text
    assert "average_and_reset_loss_window(" in loops_text


def test_training_clis_use_shared_optim_and_checkpoint_modules() -> None:
    embeddings_text = (REPO_ROOT / "kurome" / "cli" / "train_embeddings.py").read_text(
        encoding="utf-8"
    )
    features_text = (REPO_ROOT / "kurome" / "cli" / "train_features.py").read_text(
        encoding="utf-8"
    )

    assert "from kurome.training.optim import setup_optimizer_scheduler as setup_optimizer_scheduler_common" in embeddings_text
    assert "from kurome.training.checkpoint import load_checkpoint as load_checkpoint_common" in embeddings_text
    assert "return setup_optimizer_scheduler_common(" in embeddings_text
    assert "return load_checkpoint_common(" in embeddings_text

    assert "from kurome.training.optim import setup_optimizer_scheduler as setup_optimizer_scheduler_common" in features_text
    assert "from kurome.training.checkpoint import load_checkpoint as load_checkpoint_common" in features_text
    assert "return setup_optimizer_scheduler_common(" in features_text
    assert "return load_checkpoint_common(" in features_text


def test_training_clis_use_shared_metrics_and_checkpoint_save_helpers() -> None:
    loops_text = (REPO_ROOT / "kurome" / "training" / "loops.py").read_text(encoding="utf-8")

    assert "from kurome.training.metrics import (" in loops_text
    assert "log_eval_loss(" in loops_text
    assert "append_and_average_validation_loss(" in loops_text
    assert "log_eval_and_average(" in loops_text
    assert "update_last_eval_loss(" in loops_text
    assert "update_best_and_maybe_save(" in loops_text
    assert "maybe_save_periodic_checkpoint(" in loops_text


def test_engine_loss_helpers_basics() -> None:
    import torch
    import torch.nn as nn

    from kurome.training.engine import (
        advance_global_step,
        accumulate_scalar_loss,
        average_and_reset_loss_window,
        compute_loss,
        has_reached_total_steps,
        normalize_embedding_batch_data,
        run_optimizer_step_and_zero_grad,
        prepare_embedding_sub_batch,
        prepare_prediction_for_loss,
        prepare_sequence_micro_batch,
        restore_training_mode_if_needed,
        should_run_interval,
    )

    y_pred = torch.randn(4, 1)
    target = torch.randn(4)
    criterion = nn.MSELoss()
    loss_input = prepare_prediction_for_loss(y_pred, criterion, num_classes=1)
    assert loss_input.shape == (4,)
    loss = compute_loss(
        criterion=criterion,
        loss_input=loss_input,
        target=target,
        output_mode="linear",
        cast_target_by_criterion=True,
    )
    assert torch.isfinite(loss)

    prepared_embedding = prepare_embedding_sub_batch(
        {"emb": torch.randn(4, 8), "val": torch.tensor([0, 1, 0, 1])},
        device="cpu",
        num_classes=2,
    )
    assert prepared_embedding is not None
    emb_input, emb_target, emb_size = prepared_embedding
    assert emb_input.shape == (4, 8)
    assert emb_target.shape == (4,)
    assert emb_size == 4

    prepared_sequence = prepare_sequence_micro_batch(
        {
            "sequence": torch.randn(3, 5, 7),
            "mask": torch.ones(3, 5, dtype=torch.bool),
            "label": torch.tensor([0, 1, 0]),
        },
        device="cpu",
        num_classes=2,
    )
    assert prepared_sequence is not None
    seq_batch, mask_batch, seq_target, seq_size = prepared_sequence
    assert seq_batch.shape == (3, 5, 7)
    assert mask_batch.shape == (3, 5)
    assert seq_target.shape == (3,)
    assert seq_size == 3

    normalized_dict = normalize_embedding_batch_data({"emb": torch.randn(2, 4), "val": torch.tensor([0, 1])})
    assert normalized_dict is not None
    assert len(normalized_dict) == 1

    normalized_list = normalize_embedding_batch_data(
        [{"emb": torch.randn(2, 4), "val": torch.tensor([0, 1])}]
    )
    assert normalized_list is not None
    assert len(normalized_list) == 1
    assert normalize_embedding_batch_data("bad-batch") is None

    class DummyProgress:
        def __init__(self) -> None:
            self.updated = 0

        def update(self, amount: int) -> None:
            self.updated += amount

    class DummyWrapper:
        def __init__(self) -> None:
            self.step = None

        def update_step(self, step: int) -> None:
            self.step = step

    progress = DummyProgress()
    wrapper = DummyWrapper()
    next_step = advance_global_step(7, progress_bar=progress, wrapper=wrapper)
    assert next_step == 8
    assert progress.updated == 1
    assert wrapper.step == 8
    assert has_reached_total_steps(5, 5) is True
    assert has_reached_total_steps(4, 5) is False
    assert should_run_interval(10, 5) is True
    assert should_run_interval(10, 5, min_step_exclusive=10) is False
    assert should_run_interval(10, 0) is False

    class DummyModel:
        def __init__(self) -> None:
            self.training = False
            self.train_called = False

        def train(self) -> None:
            self.training = True
            self.train_called = True

    class DummyTrainOptimizer:
        def __init__(self) -> None:
            self.train_called = False

        def train(self) -> None:
            self.train_called = True

    model = DummyModel()
    train_optimizer = DummyTrainOptimizer()
    restore_training_mode_if_needed(model, train_optimizer)
    assert model.train_called is True
    assert train_optimizer.train_called is True

    loss_sum = torch.tensor(0.0)
    steps = 0
    steps = accumulate_scalar_loss(loss_sum, steps, 1.25)
    steps = accumulate_scalar_loss(loss_sum, steps, float("nan"))
    assert steps == 1
    assert loss_sum.item() == pytest.approx(1.25)
    avg, steps = average_and_reset_loss_window(loss_sum, steps)
    assert avg == pytest.approx(1.25)
    assert steps == 0
    assert loss_sum.item() == pytest.approx(0.0)

    class DummyScaler:
        def __init__(self) -> None:
            self.step_called = False
            self.update_called = False

        def step(self, _optimizer) -> None:
            self.step_called = True

        def update(self) -> None:
            self.update_called = True

    class DummyOptimizer:
        def __init__(self) -> None:
            self.zero_called = False

        def zero_grad(self, set_to_none: bool = True) -> None:
            self.zero_called = set_to_none

    scaler = DummyScaler()
    optimizer = DummyOptimizer()
    stepped = run_optimizer_step_and_zero_grad(scaler=scaler, optimizer=optimizer)
    assert stepped is True
    assert scaler.step_called is True
    assert scaler.update_called is True
    assert optimizer.zero_called is True


def test_metrics_helpers_basics() -> None:
    from kurome.training.metrics import update_last_eval_loss

    assert update_last_eval_loss(1.0, 0.5) == pytest.approx(0.5)
    assert update_last_eval_loss(1.0, float("nan")) == pytest.approx(1.0)


def test_refactor_core_modules_compile() -> None:
    paths = [
        "kurome/data/__init__.py",
        "kurome/data/contracts.py",
        "kurome/data/dataloaders.py",
        "kurome/data/datasets/__init__.py",
        "kurome/data/datasets/embedding_dataset.py",
        "kurome/data/datasets/sequence_dataset.py",
        "kurome/data/datasets/image_dataset.py",
        "kurome/data/embeddings.py",
        "kurome/data/images.py",
        "kurome/data/sequences.py",
        "kurome/data/transforms.py",
        "kurome/config/__init__.py",
        "kurome/config/embed_params.py",
        "kurome/config/runtime_args.py",
        "kurome/config/schema.py",
        "kurome/config/loader.py",
        "kurome/models/__init__.py",
        "kurome/models/backbones/__init__.py",
        "kurome/models/backbones/early_extract.py",
        "kurome/models/heads/__init__.py",
        "kurome/models/heads/predictor.py",
        "kurome/models/heads/sequence_head.py",
        "kurome/models/heads/hybrid_head.py",
        "kurome/models/tasks/__init__.py",
        "kurome/models/tasks/losses.py",
        "kurome/models/registry.py",
        "kurome/models/factory.py",
        "kurome/cli/train_embeddings.py",
        "kurome/cli/train_features.py",
        "kurome/cli/infer.py",
        "kurome/inference/pipeline.py",
        "kurome/inference/postprocess.py",
        "kurome/training/checkpoint.py",
        "kurome/training/engine.py",
        "kurome/training/loops.py",
        "kurome/training/metrics.py",
        "kurome/training/state_io.py",
        "kurome/training/validation.py",
        "kurome/training/wrapper.py",
        "kurome/training/optim.py",
        "kurome/training/bootstrap.py",
    ]
    for rel_path in paths:
        py_compile.compile(str(REPO_ROOT / rel_path), doraise=True)
