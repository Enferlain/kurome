# Script for training on pre-computed feature sequences with bucketing
import argparse
import importlib
import os
from typing import Any
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import traceback # Keep traceback
from transformers import AutoProcessor # <<< Added AutoProcessor import

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
from kurome.training.loops import run_feature_sequence_training_loop
from kurome.training.optim import setup_optimizer_scheduler as setup_optimizer_scheduler_common
from kurome.config.loader import load_experiment_config
from kurome.config.schema import ExperimentConfig
from kurome.config.runtime_args import write_config
from kurome.data.sequences import build_feature_sequence_dataloaders
from kurome.models.factory import build_criterion, build_model_with_filtered_kwargs, resolve_num_classes
from kurome.training.state_io import (
    load_optimizer_state,
    load_scaler_state,
    load_scheduler_state,
)
from kurome.training.validation import run_validation_sequences
from kurome.training.wrapper import ModelWrapper, SAVE_FOLDER


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
OPTIMIZERS, SCHEDULERS = load_optimizer_registry(project_root=PROJECT_ROOT)
TARGET_DEV = configure_torch_runtime()
# Seed setting moved to main after args parsing
# --- End Global Settings ---

def setup_dataloaders(args):
    """Sets up FeatureSequenceDataset and DataLoader.
       v1.1.0: Uses standard DataLoader with shuffle=True, pooling done in dataset.
    """
    try:
        return build_feature_sequence_dataloaders(args)
    except Exception as e:
        print(f"Error creating FeatureSequenceDataset: {e}")
        exit(1)

# <<< REWRITTEN setup_model_criterion v1.1.0 >>>
def setup_model_criterion(args, dataset, experiment: ExperimentConfig):
    """
    Sets up the HeadModel and criterion based on args config,
    validating against the dataset's discovered labels.
    """
    print("DEBUG setup_model_criterion v1.1.0: Setting up HeadModel and criterion...")

    config_loss_function = experiment.train.loss_function
    config_arch = experiment.model.arch
    head_cfg = experiment.head_params
    if head_cfg is None:
        exit("Config Error: head_params must be set.")
    head = head_cfg

    # --- 2. Get num_labels from Dataset ---
    num_labels_from_dataset = getattr(dataset, 'num_labels', 0)
    print(f"DEBUG: Labels found by dataset: {num_labels_from_dataset}")

    # --- 3. Validate Intended vs. Dataset Labels ---
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

    # Store the final validated number back to args for other parts of the script
    args.num_classes = final_num_classes
    args.loss_function = resolved_loss_name
    print(f"DEBUG: Final validated num_classes to be used: {args.num_classes}")
    print(f"DEBUG: Resolved loss_function: {resolved_loss_name}")

    # --- 4. Instantiate Criterion (Using Validated Classes) ---
    class_weights_tensor = None
    # Get weights from args if they exist and match final_num_classes
    config_weights = getattr(args, "weights", None)
    if config_weights and len(config_weights) == final_num_classes:
         try:
              class_weights_tensor = torch.tensor(config_weights, device=TARGET_DEV, dtype=torch.float32)
              print(f"DEBUG: Using class weights: {config_weights}")
         except Exception as e: print(f"Warning: Failed to make tensor from weights: {e}.")
    elif config_weights: print(f"Warning: Config weights len ({len(config_weights)}) != final num_classes ({final_num_classes}). Ignoring weights.")

    try:
        criterion = build_criterion(
            loss_name=resolved_loss_name,
            num_classes=final_num_classes,
            output_mode=head.output_mode or "linear",
            class_weights_tensor=class_weights_tensor,
            args=args,
        )
    except ValueError as e:
        exit(f"Config Error: {e}")

    print(f"DEBUG: Criterion set to: {type(criterion).__name__}")

    # --- 5. Instantiate the sequence model (Using Validated Classes) ---
    model_id = str(experiment.model.model_id or "head_model").strip().lower()
    print(f"DEBUG: Instantiating sequence model '{model_id}'...")
    try:
        head_features = head.features
        if head_features is None: exit("Error: 'head_features' not specified.")

        model = build_model_with_filtered_kwargs(
            model_id,
            {
                "features": head_features,
                "num_classes": args.num_classes,
                "pooling_strategy": head.pooling_strategy,
                "hidden_dim": head.hidden_dim,
                "num_res_blocks": head.num_res_blocks,
                "dropout_rate": head.dropout_rate,
                "output_mode": head.output_mode or "linear",
                "attn_pool_heads": head.attn_pool_heads,
                "attn_pool_dropout": head.attn_pool_dropout,
                "num_attn_heads": head.attn_pool_heads,
                "attn_dropout": head.attn_pool_dropout,
                "rms_norm_eps": getattr(args, "rms_norm_eps", None),
                "topk_ratio": getattr(args, "topk_ratio", None),
                "min_topk": getattr(args, "min_topk", None),
                "conv_kernel_size": getattr(args, "conv_kernel_size", None),
            },
        )
    except Exception as e:
        print(f"Error details during sequence model instantiation: {e}")
        traceback.print_exc()
        exit(f"Error instantiating sequence model '{model_id}'.")

    model.to(TARGET_DEV)
    print(f"Sequence model '{model_id}' (Output Classes: {args.num_classes}) and criterion setup complete.")
    return model, criterion

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
#        Main Training Loop (for Feature Sequences)
# ================================================
# Version 1.3.0: Delegates training loop implementation to kurome.training.loops
def train_loop(args, model, criterion, optimizer, scheduler, scaler,
               train_loader, val_loader, wrapper, start_epoch, initial_global_step,
               enabled_amp, amp_dtype, is_schedule_free):
    """Compatibility wrapper that delegates to package training loop."""
    run_feature_sequence_training_loop(
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
        run_validation_sequences_fn=run_validation_sequences,
    )


# ================================================
#        Main Execution Block
# ================================================
# Version 1.0.0 (for train_features.py)
def main():
    # 1. Load Config / Args
    # <<< Add argparse for --config argument >>>
    parser = argparse.ArgumentParser(description="Train HeadModel on precomputed features.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    # Add --resume here too if needed as override? Or handle via config only?
    # Let's keep resume simple, require it in YAML if needed.
    # parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from (overrides config).')
    script_args = parser.parse_args()
    # <<< End add >>>

    # 1. Load Config / Args using the provided path
    # <<< Pass the config path from script_args >>>
    try:
        experiment = load_experiment_config(config_path=script_args.config)
        args = experiment.runtime_args
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading config: {e}")
        exit(1)
    # <<< Optionally merge resume override if we added it >>>
    # if script_args.resume: args.resume = script_args.resume

    print(f"Target device: {TARGET_DEV}")


    # 2. Setup Seed, Precision, WandB
    seed = experiment.train.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(f"Set random seed to: {seed}")
    amp_dtype, enabled_amp = setup_precision(args, TARGET_DEV)
    wandb_run = setup_wandb(args, wandb)

    # 3. Setup Dataloaders (uses FeatureSequenceDataset, BucketBatchSampler)
    dataset, train_loader, val_loader = setup_dataloaders(args)

    # 4. Calculate final training duration (Corrected for Grad Acc)
    if args.steps_per_epoch == 0: exit("Error: steps_per_epoch is zero.")

    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
    # <<< Calculate GLOBAL steps per epoch >>>
    global_steps_per_epoch = math.ceil(args.steps_per_epoch / gradient_accumulation_steps)
    if global_steps_per_epoch == 0: global_steps_per_epoch = 1 # Avoid division by zero if epoch is too short
    print(f"DEBUG Main: Micro-batches/epoch = {args.steps_per_epoch}, Grad Acc = {gradient_accumulation_steps}, Global Steps/epoch = {global_steps_per_epoch}")

    # <<< Use global_steps_per_epoch for calculations >>>
    if args.max_train_steps is not None and args.max_train_steps > 0:
        # Calculate epochs needed based on GLOBAL steps
        args.num_train_epochs = math.ceil(args.max_train_steps / global_steps_per_epoch)
        print(f"DEBUG Main: Calculated num_train_epochs = {args.num_train_epochs} from max_train_steps={args.max_train_steps}")
    elif args.max_train_epochs is not None and args.max_train_epochs > 0:
        args.num_train_epochs = args.max_train_epochs
        # Calculate total GLOBAL steps based on epochs
        args.max_train_steps = args.num_train_epochs * global_steps_per_epoch
        print(f"DEBUG Main: Calculated max_train_steps = {args.max_train_steps} from max_train_epochs={args.num_train_epochs}")
    else: # Default if neither specified
        args.max_train_steps = 10000 # Default GLOBAL steps
        args.num_train_epochs = math.ceil(args.max_train_steps / global_steps_per_epoch)
        print(f"DEBUG Main: Defaulting to max_train_steps={args.max_train_steps}, calculated num_train_epochs={args.num_train_epochs}")

    # 5. Setup Model & Criterion
    model, criterion = setup_model_criterion(args, dataset, experiment)

    # 6. Setup Optimizer, Scheduler (passing HeadModel)
    optimizer, scheduler, is_schedule_free = setup_optimizer_scheduler(args, model)
    scaler = torch.amp.GradScaler(device=TARGET_DEV, enabled=(enabled_amp and TARGET_DEV == 'cuda'))

    # 7. Load Checkpoint (loading HeadModel state)
    start_epoch, initial_global_step = load_checkpoint(args, model, optimizer, scheduler, scaler)

    # 8. Write Final Config
    print("\n--- Final Calculated Args ---")
    for k, v in sorted(vars(args).items()): print(f"  {k}: {v}")
    print("--------------------------\n")
    write_config(args)

    # 9. Instantiate Wrapper (passing HeadModel)
    wrapper = ModelWrapper(
        name=args.name, model=model, optimizer=optimizer,
        criterion=criterion, # Pass criterion for reference if needed? Maybe not.
        scheduler=scheduler, device=TARGET_DEV, scaler=scaler,
        wandb_run=wandb_run, num_labels=args.num_labels
    )
    # Restore best loss if loaded from checkpoint state
    if initial_global_step > 0 and os.path.exists(f"./{SAVE_FOLDER}/{args.name}_s{initial_global_step}.state"): # Check state file from step
         try:
              train_state = torch.load(f"./{SAVE_FOLDER}/{args.name}_s{initial_global_step}.state", map_location='cpu')
              wrapper.best_val_loss = train_state.get('best_val_loss', float('inf'))
              print(f"Restored best_val_loss from checkpoint state: {wrapper.best_val_loss}")
         except Exception as e_state: print(f"Could not load best_val_loss from state: {e_state}")


    # 10. Run Training Loop (Feature Sequence version)
    try:
        train_loop(args, model, criterion, optimizer, scheduler, scaler,
                   train_loader, val_loader, wrapper,
                   start_epoch, initial_global_step,
                   enabled_amp, amp_dtype, is_schedule_free)
    finally:
        # 11. Final Save & Cleanup
        print(f"Saving final model...")
        final_step = wrapper.get_current_step()
        wrapper.save_model(step=final_step, epoch="final", suffix="_final", save_aux=False, args=args)
        wrapper.close()
        if wandb_run: wandb_run.finish()
        print("Training script finished.")

if __name__ == "__main__":
    main()
