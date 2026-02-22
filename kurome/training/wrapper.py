"""Training wrapper and checkpoint-saving utilities."""

from __future__ import annotations

from contextlib import suppress
import math
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

LOSS_MEMORY = 500
SAVE_FOLDER = "models"


class ModelWrapper:
    """Stateful training wrapper used by CLI loops and checkpoint helpers."""

    def __init__(
        self,
        name,
        model,
        optimizer,
        criterion,
        scheduler=None,
        device="cpu",
        stdout=True,
        scaler=None,
        wandb_run=None,
        num_labels=None,
    ):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.scaler = scaler
        self.wandb_run = wandb_run
        self.stdout = stdout

        self.num_labels = num_labels if num_labels is not None else getattr(model, "num_classes", "?")
        self.losses = []
        self.current_epoch = 0
        self.current_global_step = 0
        self.best_val_loss = float("inf")
        self.current_best_val_model_path = None

        self._find_initial_best_model()
        print(f"ModelWrapper initialized for '{name}'.")

    def _find_initial_best_model(self):
        try:
            os.makedirs(SAVE_FOLDER, exist_ok=True)
            potential_best = None
            latest_mtime = 0
            for fname in os.listdir(SAVE_FOLDER):
                if fname.startswith(self.name) and fname.endswith("_best_val.safetensors"):
                    fpath = os.path.join(SAVE_FOLDER, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                        if mtime > latest_mtime:
                            latest_mtime = mtime
                            potential_best = fpath
                    except OSError:
                        continue
            if potential_best:
                self.current_best_val_model_path = potential_best
                print(f"Found existing best model checkpoint: {os.path.basename(potential_best)}")
            else:
                print("No existing best model checkpoint found.")
        except Exception as e:
            print(f"Warning: Error finding initial best model: {e}")

    def update_step(self, global_step):
        self.current_global_step = global_step

    def get_current_step(self):
        return self.current_global_step

    def log_step(self, loss):
        if not math.isnan(loss):
            self.losses.append(loss)
            if len(self.losses) > LOSS_MEMORY:
                self.losses.pop(0)

    def log_main(self, step, current_train_loss, current_val_loss):
        self.update_step(step)
        lr = float(self.optimizer.param_groups[0]["lr"]) if self.optimizer.param_groups else 0.0

        if self.wandb_run:
            log_data = {
                "loss/current": current_train_loss,
                "train/learning_rate": lr,
            }
            if current_val_loss is not None and not math.isnan(current_val_loss):
                log_data["loss/current_val_loss"] = current_val_loss
            try:
                self.wandb_run.log(log_data, step=step)
            except Exception as e_wandb:
                print(f"Warning: Failed to log to WandB at step {step}: {e_wandb}")

    def save_model(self, step=None, epoch=None, suffix="", save_aux=False, args=None):
        if args is None:
            print("Warning: 'args' object not provided to save_model. Defaulting to saving trainable parts only.")
            should_save_full = False
        else:
            should_save_full = getattr(args, "save_full_model", False)

        state_dict_to_save = {}

        if should_save_full:
            print("DEBUG save_model: Configured to save FULL model state_dict.")
            try:
                state_dict_to_save = self.model.state_dict()
            except Exception as e:
                print(f"Error getting full model state_dict: {e}")
                return
        else:
            print("DEBUG save_model: Configured to save TRAINABLE parts only. Manually collecting parameters...")
            collected_count = 0
            is_e2e = getattr(args, "is_end_to_end", False)
            if is_e2e and hasattr(self.model, "head") and isinstance(self.model.head, nn.Module):
                print("  - Collecting head parameters/buffers...")
                for name, param in self.model.head.named_parameters():
                    state_dict_to_save[f"head.{name}"] = param.detach().clone().cpu()
                    collected_count += 1
                for name, buf in self.model.head.named_buffers():
                    state_dict_to_save[f"head.{name}"] = buf.detach().clone().cpu()
                    collected_count += 1

                pool_strat = getattr(args, "pooling_strategy", None)
                if pool_strat == "attn" and hasattr(self.model, "pooler") and self.model.pooler is not None:
                    print("  - Collecting pooler parameters/buffers...")
                    for name, param in self.model.pooler.named_parameters():
                        state_dict_to_save[f"pooler.{name}"] = param.detach().clone().cpu()
                        collected_count += 1
                    for name, buf in self.model.pooler.named_buffers():
                        state_dict_to_save[f"pooler.{name}"] = buf.detach().clone().cpu()
                        collected_count += 1
            elif not is_e2e and isinstance(self.model, nn.Module):
                print("  - Model seems to be head-only (Embedding Mode). Collecting all its parameters/buffers...")
                for name, param in self.model.named_parameters():
                    state_dict_to_save[name] = param.detach().clone().cpu()
                    collected_count += 1
                for name, buf in self.model.named_buffers():
                    state_dict_to_save[name] = buf.detach().clone().cpu()
                    collected_count += 1
            else:
                print("Warning: Could not determine model structure for partial save.")

            if collected_count == 0 and not should_save_full:
                print("Error: No parameters collected for partial save! Saving full model as fallback.")
                try:
                    state_dict_to_save = self.model.state_dict()
                    should_save_full = True
                except Exception as e:
                    print(f"Error getting full model state_dict during fallback: {e}")
                    return
            elif collected_count > 0:
                print(f"  Manually collected {collected_count} parameters/buffers for partial save.")

        if not state_dict_to_save:
            print("Error: state_dict_to_save is empty or invalid after collection/fallback.")
            return

        current_global_step = step if step is not None else self.current_global_step
        current_epoch_to_save = epoch if epoch is not None else self.current_epoch
        step_str = ""
        epoch_str = ""
        if current_global_step is not None:
            if current_global_step >= 1_000_000:
                step_str = f"_s{round(current_global_step / 1_000_000, 1)}M"
            elif current_global_step >= 1_000:
                step_str = f"_s{round(current_global_step / 1_000)}K"
            else:
                step_str = f"_s{current_global_step}"
        if isinstance(epoch, str) and epoch.lower() == "final":
            epoch_str = "_efinal"

        base_output_name = f"./{SAVE_FOLDER}/{self.name}{step_str}{epoch_str}{suffix}"
        model_output_path = f"{base_output_name}.safetensors"
        optim_output_path = f"{base_output_name}.optim"
        sched_output_path = f"{base_output_name}.sched"
        scaler_output_path = f"{base_output_name}.scaler"
        state_output_path = f"{base_output_name}.state"

        is_best = "_best_val" in suffix
        save_type_str = "Full Model" if should_save_full else "Trainable Parts Only"
        print(f"\nSaving checkpoint: {os.path.basename(base_output_name)} ... [{save_type_str}]")
        estimated_size_mb = (
            sum(p.numel() * p.element_size() for p in state_dict_to_save.values() if hasattr(p, "numel"))
            / (1024 * 1024)
        )
        print(f"DEBUG: Keys in FINAL state_dict: {len(state_dict_to_save)}")
        print(f"DEBUG: Estimated size of FINAL state_dict: {estimated_size_mb:.2f} MB")

        try:
            os.makedirs(SAVE_FOLDER, exist_ok=True)
            if is_best:
                old_best_model_path = self.current_best_val_model_path
                if old_best_model_path and os.path.normpath(old_best_model_path) != os.path.normpath(model_output_path):
                    old_base = os.path.splitext(old_best_model_path)[0]
                    files_to_remove = [old_best_model_path] + [old_base + ext for ext in [".optim", ".sched", ".scaler", ".state"]]
                    print(f"  New best found. Removing previous best files starting with: {os.path.basename(old_base)}")
                    removed_count = 0
                    for f_path in files_to_remove:
                        if os.path.exists(f_path):
                            try:
                                os.remove(f_path)
                                removed_count += 1
                            except OSError as e:
                                print(f"  Warning: Could not remove '{os.path.basename(f_path)}': {e}")
                    if removed_count > 0:
                        print(f"  Removed {removed_count} previous best file(s).")
                    self.current_best_val_model_path = None

            save_file(state_dict_to_save, model_output_path)
            actual_size_mb = os.path.getsize(model_output_path) / (1024 * 1024)
            print(f"  Model saved successfully. Actual size: {actual_size_mb:.2f} MB")

            if save_aux:
                print("  Saving auxiliary files (optim, sched, scaler, state)...")
                torch.save(self.optimizer.state_dict(), optim_output_path)
                if self.scheduler is not None:
                    try:
                        torch.save(self.scheduler.state_dict(), sched_output_path)
                    except Exception as e_sched:
                        print(f"  Warning: Failed to save scheduler state: {e_sched}")
                if self.scaler is not None and self.scaler.is_enabled():
                    torch.save(self.scaler.state_dict(), scaler_output_path)
                train_state = {
                    "epoch": current_epoch_to_save,
                    "global_step": current_global_step,
                    "best_val_loss": self.best_val_loss,
                }
                torch.save(train_state, state_output_path)
            else:
                for ext in [".optim", ".sched", ".scaler", ".state"]:
                    path_to_check = base_output_name + ext
                    if os.path.exists(path_to_check):
                        with suppress(OSError):
                            os.remove(path_to_check)

            if is_best:
                self.current_best_val_model_path = model_output_path
                print(f"  Marked {os.path.basename(model_output_path)} as new best model path.")

            print(f"  Checkpoint saving process complete for base: {os.path.basename(base_output_name)}")

        except Exception as e:
            print(f"Error saving checkpoint {base_output_name}: {e}")
            import traceback

            traceback.print_exc()

    def close(self):
        print("ModelWrapper closed.")
