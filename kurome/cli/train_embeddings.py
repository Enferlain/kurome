# Version: 2.3.0 (Handles E2E Dataset Loading)
import argparse
import importlib
import os
import traceback
from typing import Any

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# --- Local Imports ---
from kurome.training.bootstrap import (
    apply_sageattention_patch,
    configure_torch_runtime,
    load_checkpoint_state,
    load_optimizer_registry,
    setup_precision,
    setup_wandb,
)
from kurome.training.checkpoint import load_checkpoint as load_checkpoint_common
from kurome.training.loops import run_embedding_training_loop
from kurome.training.optim import setup_optimizer_scheduler as setup_optimizer_scheduler_common
from kurome.config.loader import load_experiment_config
from kurome.config.schema import ExperimentConfig
from kurome.data.embeddings import build_embedding_training_dataloaders
from kurome.data.images import build_image_training_dataloaders
from kurome.data.tensors import build_tensor_training_dataloaders
from kurome.data.transforms import load_image_processor
from kurome.models.factory import (
    build_criterion,
    build_model,
    build_model_with_filtered_kwargs,
    resolve_num_classes,
)
from kurome.config.embed_params import get_embed_params
from kurome.config.runtime_args import write_config
from kurome.training.state_io import (
    load_optimizer_state,
    load_scaler_state,
    load_scheduler_state,
)
from kurome.training.validation import run_validation_embeddings
from kurome.training.wrapper import ModelWrapper


def _load_optional_wandb() -> Any | None:
    """Return wandb module when installed, otherwise None."""
    try:
        return importlib.import_module("wandb")
    except ImportError:
        return None


wandb = _load_optional_wandb()

apply_sageattention_patch(F)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- Global Settings ---
# LOG_EVERY_N = 1 # How often to log metrics and check validation
# VALIDATE_EVERY_N = 50 # Run validation less often
OPTIMIZERS, SCHEDULERS = load_optimizer_registry(project_root=PROJECT_ROOT)
TARGET_DEV = configure_torch_runtime()
# Seed setting moved to main after args parsing
# --- End Global Settings ---

# Version 2.3.0: Updated setup_dataloaders function
def setup_dataloaders(args, image_processor=None):
    """
    Sets up the dataset and dataloaders based on args.
    Uses image data adapter in E2E mode, otherwise embedding data adapter.
    """
    try:
        if getattr(args, "data_mode", None) == "tensors":
            return build_tensor_training_dataloaders(args)
        if getattr(args, "is_end_to_end", False):
            return build_image_training_dataloaders(args, image_processor=image_processor)
        return build_embedding_training_dataloaders(args, image_processor=image_processor)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"Error setting up dataloaders: {e}")
        exit(1)
# --- End setup_dataloaders ---

# Version 2.4.0: Handles E2E, PredictorModel, or HeadModel based on config
def setup_model_criterion(args, dataset, experiment: ExperimentConfig):
    """
    Sets up the model and criterion based on args.
    Instantiates EarlyExtractAnatomyModel (E2E), HeadModel (features/embeddings),
    or PredictorModel (embeddings) based on config structure.
    Determines criterion based on args.loss_function.
    """
    print("DEBUG setup_model_criterion v2.4.0: Setting up model and criterion...")

    config_loss_function = experiment.train.loss_function
    config_arch = experiment.model.arch
    predictor_cfg = experiment.predictor_params
    head_cfg = experiment.head_params
    e2e_cfg = experiment.e2e_params
    if head_cfg is None:
        exit("Config Error: head_params must be set.")
    if getattr(args, "data_mode", None) not in {"images", "tensors"} and not e2e_cfg.is_end_to_end and predictor_cfg is None:
        exit("Config Error: predictor_params must be set for embeddings mode.")
    head = head_cfg
    predictor = predictor_cfg

    num_labels_from_dataset = getattr(dataset, 'num_labels', 0)
    print(f"DEBUG: Labels found by dataset: {num_labels_from_dataset}")

    try:
        resolved_loss_name, final_num_classes, resolution_warnings = resolve_num_classes(
            arch=config_arch,
            loss_function=config_loss_function,
            dataset_num_labels=num_labels_from_dataset,
            default_class_loss='focal',
        )
    except ValueError as e:
        exit(f"Config Error: {e}")
    for warning in resolution_warnings:
        print(f"Warning: {warning}")

    args.num_classes = final_num_classes
    args.loss_function = resolved_loss_name
    print(f"DEBUG: Final validated num_classes to be used: {args.num_classes}")
    print(f"DEBUG: Resolved loss_function: {resolved_loss_name}")

    class_weights_tensor = None
    if config_arch == "class" and hasattr(args, "weights") and args.weights:
         if len(args.weights) == final_num_classes:
              try:
                  class_weights_tensor = torch.tensor(args.weights, device=TARGET_DEV, dtype=torch.float32)
                  print(f"DEBUG: Using class weights: {args.weights}")
              except Exception as e:
                   print(f"Warning: Failed to create tensor from class weights: {e}. Ignoring weights.")
         else:
              print(f"Warning: Length of weights in config ({len(args.weights)}) != num_classes ({final_num_classes}). Ignoring weights.")

    model_output_mode = None
    if getattr(args, "data_mode", None) == "tensors":
        model_output_mode = head.output_mode
        if model_output_mode is None:
            exit("Error: 'head_output_mode' missing in config for tensors mode.")
    elif e2e_cfg.is_end_to_end:
        model_output_mode = head.output_mode
        if model_output_mode is None:
            exit("Error: 'head_output_mode' missing in config for E2E mode.")
    else:
        if predictor is None:
            exit("Error: predictor_params missing.")
        model_output_mode = predictor.output_mode or head.output_mode
        if model_output_mode is None:
            exit("Error: 'output_mode' missing in predictor_params (or head_params).")
    model_output_mode = model_output_mode.lower()
    print(f"DEBUG: Determined model_output_mode for loss validation: '{model_output_mode}'")

    try:
        criterion = build_criterion(
            loss_name=resolved_loss_name,
            num_classes=final_num_classes,
            output_mode=model_output_mode,
            class_weights_tensor=class_weights_tensor,
            args=args,
            allow_bce_multiclass=False,
            strict_regression_num_classes=True,
            require_linear_logits_losses=True,
        )
    except ValueError as e:
        exit(f"Config Error: {e}")

    if resolved_loss_name in {"l1", "mse"} and final_num_classes != 1:
        print(
            f"Warning: {resolved_loss_name.upper()} is usually used with num_classes=1, "
            f"but found {final_num_classes}."
        )
    if resolved_loss_name in {"focal", "crossentropy", "bce"} and model_output_mode != "linear":
        print(
            f"Warning: loss_function='{resolved_loss_name}' usually expects output_mode='linear', "
            f"but got '{model_output_mode}'. Training with non-logit input."
        )
    if resolved_loss_name == "ghm" and class_weights_tensor is not None:
        print("Warning: Class weights specified but GHMC_Loss does not use them directly.")

    model = None
    amp_dtype, enabled_amp = setup_precision(args, TARGET_DEV) # Get compute dtype

    if getattr(args, "data_mode", None) == "tensors":
        tensor_model_id = str(experiment.model.model_id or "tensor_cnn_model").strip().lower()
        print(f"DEBUG: Instantiating tensor model '{tensor_model_id}' ...")
        try:
            model = build_model_with_filtered_kwargs(
                tensor_model_id,
                {
                    "input_channels": getattr(args, "input_channels", None),
                    "hidden_dim": head.hidden_dim,
                    "num_classes": final_num_classes,
                    "num_res_blocks": head.num_res_blocks,
                    "dropout_rate": head.dropout_rate,
                    "output_mode": head.output_mode,
                },
            )
        except Exception as e:
            print(f"Error details during tensor model instantiation: {e}")
            traceback.print_exc()
            exit(f"Error instantiating tensor model '{tensor_model_id}'.")
    elif e2e_cfg.is_end_to_end:
        print("DEBUG: Instantiating EarlyExtractAnatomyModel ...")
        try:
            model = build_model(
                "early_extract_model",
                base_model_name=experiment.model.base_vision_model,
                device=TARGET_DEV,
                extract_layer=e2e_cfg.extract_layer,
                pooling_strategy=e2e_cfg.pooling_strategy,
                head_features=None,
                head_hidden_dim=head.hidden_dim,
                head_num_classes=final_num_classes,
                head_num_res_blocks=head.num_res_blocks,
                head_dropout_rate=head.dropout_rate,
                head_output_mode=head.output_mode,
                attn_pool_heads=head.attn_pool_heads,
                attn_pool_dropout=head.attn_pool_dropout,
                freeze_base_model=e2e_cfg.freeze_base_model,
                compute_dtype=amp_dtype,
            )
        except Exception as e:
            print(f"Error details during EarlyExtractAnatomyModel instantiation: {e}")
            traceback.print_exc()
            exit("Error instantiating EarlyExtractAnatomyModel.")
    else:
        if predictor is None:
            exit("Error: predictor_params missing.")
        features = predictor.features if predictor.features is not None else getattr(args, "features", None)
        if features is None:
            exit("Error: Features missing.")

        embedding_model_id = str(experiment.model.model_id or "hybrid_head_model").strip().lower()
        print(f"DEBUG: Embedding mode. Instantiating model_id='{embedding_model_id}'...")
        output_mode = predictor.output_mode or head.output_mode
        if output_mode is None:
            exit("Error: 'output_mode' missing in predictor_params (or head_params).")
        num_res_blocks = head.num_res_blocks
        dropout_rate = head.dropout_rate
        pooling_strategy = predictor.pooling_strategy or head.pooling_strategy
        attn_pool_heads = predictor.attn_pool_heads if predictor.attn_pool_heads is not None else head.attn_pool_heads
        attn_pool_dropout = (
            predictor.attn_pool_dropout
            if predictor.attn_pool_dropout is not None
            else head.attn_pool_dropout
        )
        try:
            model = build_model_with_filtered_kwargs(
                embedding_model_id,
                {
                    "features": features,
                    "hidden_dim": predictor.hidden_dim,
                    "num_classes": args.num_classes,
                    "use_attention": predictor.use_attention,
                    "num_attn_heads": predictor.num_attn_heads,
                    "attn_dropout": predictor.attn_dropout,
                    "num_res_blocks": num_res_blocks,
                    "dropout_rate": dropout_rate,
                    "rms_norm_eps": predictor.rms_norm_eps,
                    "pooling_strategy": pooling_strategy,
                    "attn_pool_heads": attn_pool_heads,
                    "attn_pool_dropout": attn_pool_dropout,
                    "output_mode": output_mode,
                },
            )
        except Exception as e:
            print(f"Error details during embedding model instantiation: {e}")
            exit(f"Error instantiating embedding model '{embedding_model_id}': {e}")

    if model is None:
        exit("Model instantiation failed.")
    model.to(TARGET_DEV)
    print(f"Model ({type(model).__name__}) and criterion setup complete.")

    return model, criterion

# Version 2.6.0: fixes
# Version 2.5.1: Improved float conversion for optimizer/scheduler args
def setup_optimizer_scheduler(args, model):
    return setup_optimizer_scheduler_common(
        args,
        model,
        optimizers=OPTIMIZERS,
        schedulers=SCHEDULERS,
    )

# ================================================
#        Checkpoint Loading (Epoch-Aware)
# ================================================
# Version 2.0.0: Loads epoch and global step
def load_checkpoint(args, model, optimizer, scheduler, scaler):
    """Loads state from checkpoint if args.resume is provided."""
    return load_checkpoint_common(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        target_dev=TARGET_DEV,
        load_checkpoint_state_fn=load_checkpoint_state,
        load_optimizer_state_fn=load_optimizer_state,
        load_scheduler_state_fn=load_scheduler_state,
        load_scaler_state_fn=load_scaler_state,
    )
# ================================================

# ================================================
#        Main Training Loop (Epoch-Based)
# ================================================
# Version 3.6.0: Delegates training loop implementation to kurome.training.loops
def train_loop(args, model, criterion, optimizer, scheduler, scaler,
               train_loader, val_loader, wrapper, start_epoch, initial_global_step,
               enabled_amp, amp_dtype, is_schedule_free):
    """Compatibility wrapper that delegates to package training loop."""
    run_embedding_training_loop(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        train_loader=train_loader,
        val_loader=val_loader,
        wrapper=wrapper,
        start_epoch=start_epoch,
        initial_global_step=initial_global_step,
        enabled_amp=enabled_amp,
        amp_dtype=amp_dtype,
        is_schedule_free=is_schedule_free,
        target_dev=TARGET_DEV,
        run_validation_embeddings_fn=run_validation_embeddings,
        is_e2e=getattr(args, "is_end_to_end", False),
    )

# ================================================
#        Main Execution Block (Updated)
# ================================================
# Version 2.4.0: Updated main execution flow
def main():
    """Main function to run the training process."""

    # <<< NEW: Parse command-line arguments FIRST >>>
    parser = argparse.ArgumentParser(description="Train PredictorModel on embeddings.")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file.')
    # Add any other command-line overrides if needed (e.g., --resume, --data_root)
    # parser.add_argument('--resume', type=str, default=None, help='Override resume path.')
    cmd_args = parser.parse_args()
    # <<< END NEW >>>

    # 1. Load config using the path from command line
    # <<< Pass the config path to the function >>>
    try:
        experiment = load_experiment_config(config_path=cmd_args.config)
        args = experiment.runtime_args
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}")
        exit(1)
    # <<< END Pass >>>

    # --- Optional: Merge command-line overrides if you added any ---
    # Example:
    # if cmd_args.resume is not None:
    #     print(f"Overriding resume path with command-line arg: {cmd_args.resume}")
    #     args.resume = cmd_args.resume
    # --- End Overrides ---

    print(f"Target device: {TARGET_DEV}")

    # Set seed AFTER parsing args
    seed = experiment.train.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Seed all GPUs
    print(f"Set random seed to: {seed}")

    # 2. Setup basic components (precision, logging)
    amp_dtype, enabled_amp = setup_precision(args, TARGET_DEV)
    wandb_run = setup_wandb(args, wandb) # Needs args.name to be set

    # <<< NEW: Load Processor BEFORE Dataset if End-to-End >>>
    image_processor = None
    if experiment.data.mode == "images" and experiment.e2e_params.is_end_to_end:
        print("DEBUG Main: Loading processor for end-to-end model...")
        if not experiment.model.base_vision_model:
             exit("Error: base_vision_model needed for end-to-end processor loading.")
        try:
             image_processor = load_image_processor(experiment.model.base_vision_model)
             print(f"Loaded processor: {image_processor.__class__.__name__}")
        except Exception as e:
             exit(f"Error loading image processor: {e}")

    # 3. Setup Dataloaders (Pass processor)
    # This function now updates args.num_labels based on the dataset
    dataset, train_loader, val_loader = setup_dataloaders(args, image_processor)

    # 4. Calculate final training duration (after dataset length is known)
    if not train_loader: exit("Error: Train loader is empty or failed to initialize.")
    try:
         # If drop_last=True, len(train_loader) is floor(num_samples / batch_size)
         # If drop_last=False, it's ceil(num_samples / batch_size)
         # Using len(train_loader) directly is simpler.
         args.steps_per_epoch = len(train_loader)
         if args.steps_per_epoch == 0: raise ValueError("Train loader length is zero.")
    except (TypeError, ValueError) as e:
         exit(f"Error: Could not determine train loader length (steps_per_epoch): {e}.")

    if args.max_train_steps is not None and args.max_train_steps > 0:
        args.num_train_epochs = math.ceil(args.max_train_steps / args.steps_per_epoch)
    elif args.max_train_epochs is not None and args.max_train_epochs > 0:
        args.num_train_epochs = args.max_train_epochs
        args.max_train_steps = args.num_train_epochs * args.steps_per_epoch
    else:
        # Should have been caught by parse_and_load_args default/validation
        exit("Error: Training duration (max_train_steps or max_train_epochs) not properly set.")

    # 5. Setup Model, Criterion (Needs implementation)
    # setup_model_criterion uses args (including num_labels updated by dataset)
    model, criterion = setup_model_criterion(args, dataset, experiment)
    if model is None or criterion is None: exit("Error: Model or Criterion setup failed.")

    # 6. Setup Optimizer, Scheduler (Needs implementation)
    optimizer, scheduler, is_schedule_free = setup_optimizer_scheduler(args, model)
    if optimizer is None: exit("Error: Optimizer setup failed (using placeholders).")

    scaler = torch.amp.GradScaler(device=TARGET_DEV, enabled=(enabled_amp and TARGET_DEV == 'cuda'))

    # 7. Load Checkpoint state (Needs implementation)
    start_epoch, initial_global_step = load_checkpoint(args, model, optimizer, scheduler, scaler)

    # 8. Write Final Config (now includes calculated steps/epochs & dataset labels)
    print("\n--- Final Calculated Args ---")
    for k, v in sorted(vars(args).items()): print(f"  {k}: {v}")
    print("--------------------------\n")
    write_config(args)

    # 9. Setup Wrapper
    wrapper = ModelWrapper(
        name=args.name, model=model, device=TARGET_DEV,
        num_labels=getattr(args, 'num_labels', 1), criterion=criterion,
        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        wandb_run=wandb_run
    )
    # Optional: Restore best_val_loss here if loaded from state

    # 10. Run Training Loop (Needs implementation)
    try:
        train_loop(args, model, criterion, optimizer, scheduler, scaler,
                   train_loader, val_loader, wrapper,
                   start_epoch, initial_global_step,
                   enabled_amp, amp_dtype,
                   is_schedule_free)
    finally:
        # 11. Final Save & Cleanup
        print(f"Saving final model...")
        final_step = wrapper.get_current_step()
        # Save aux=True for final to allow easier resuming/analysis
        wrapper.save_model(step=final_step, epoch="final", suffix="_final", save_aux=False, args=args)
        wrapper.close()
        if wandb_run: wandb_run.finish()
        print("Training script finished.")

if __name__ == "__main__":
    main()
