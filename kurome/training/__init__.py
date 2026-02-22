"""Training utilities shared by training entrypoints."""

from .checkpoint import (
    load_checkpoint,
    maybe_save_periodic_checkpoint,
    update_best_and_maybe_save,
)
from .engine import (
    advance_global_step,
    accumulate_scalar_loss,
    average_and_reset_loss_window,
    compute_loss,
    ensure_training_mode,
    has_reached_total_steps,
    maybe_step_scheduler,
    mean_tensor_losses,
    normalize_embedding_batch_data,
    prepare_embedding_sub_batch,
    prepare_prediction_for_loss,
    prepare_sequence_micro_batch,
    restore_training_mode_if_needed,
    run_optimizer_step_and_zero_grad,
    should_run_interval,
    update_progress_postfix,
)
from .metrics import (
    append_and_average_validation_loss,
    log_eval_and_average,
    log_eval_loss,
    update_last_eval_loss,
)
from .state_io import load_optimizer_state, load_scaler_state, load_scheduler_state
from .validation import run_validation_embeddings, run_validation_sequences
from .wrapper import LOSS_MEMORY, SAVE_FOLDER, ModelWrapper
from .loops import run_embedding_training_loop, run_feature_sequence_training_loop
from .optim import setup_optimizer_scheduler

__all__ = [
    "LOSS_MEMORY",
    "SAVE_FOLDER",
    "ModelWrapper",
    "load_optimizer_state",
    "load_scheduler_state",
    "load_scaler_state",
    "run_validation_embeddings",
    "run_validation_sequences",
    "setup_optimizer_scheduler",
    "load_checkpoint",
    "update_best_and_maybe_save",
    "maybe_save_periodic_checkpoint",
    "ensure_training_mode",
    "restore_training_mode_if_needed",
    "maybe_step_scheduler",
    "update_progress_postfix",
    "run_optimizer_step_and_zero_grad",
    "accumulate_scalar_loss",
    "average_and_reset_loss_window",
    "mean_tensor_losses",
    "normalize_embedding_batch_data",
    "advance_global_step",
    "has_reached_total_steps",
    "should_run_interval",
    "prepare_prediction_for_loss",
    "compute_loss",
    "prepare_embedding_sub_batch",
    "prepare_sequence_micro_batch",
    "log_eval_loss",
    "append_and_average_validation_loss",
    "update_last_eval_loss",
    "log_eval_and_average",
    "run_embedding_training_loop",
    "run_feature_sequence_training_loop",
]
