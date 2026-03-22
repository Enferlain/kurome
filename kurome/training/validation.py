"""Validation-loop helpers extracted from legacy utils runtime."""

from __future__ import annotations

from contextlib import suppress
import math
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from kurome.models.tasks import FocalLoss, GHMC_Loss


@torch.no_grad()
def run_validation_embeddings(model, val_loader, criterion, device, scaler, best_eval_loss_so_far: float):
    """Run validation loop for embedding-based models."""
    if val_loader is None:
        return float("nan"), False

    model.eval()
    autocast_enabled = scaler is not None and scaler.is_enabled()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    total_loss = 0.0
    total_samples = 0
    should_save_best = False

    val_iterator = tqdm(val_loader, desc="Validation (Embeddings)", leave=False, dynamic_ncols=True)
    for batch_data in val_iterator:
        emb_input = None
        target_val = None
        target = None
        y_pred = None
        y_pred_for_loss = None
        loss = None
        if batch_data is None or not batch_data:
            continue
        try:
            emb_input = batch_data.get("emb")
            target_val = batch_data.get("val")
            if emb_input is None or target_val is None:
                emb_input = batch_data.get("pixel_values")
                target_val = batch_data.get("label")
            if emb_input is None or target_val is None:
                continue

            emb_input = emb_input.to(device)
            current_batch_size = emb_input.size(0)
            num_classes = getattr(model, "num_classes", 1)

            if num_classes == 1:
                target = target_val.to(device=device, dtype=torch.float32).view(current_batch_size, -1).squeeze(-1)
            else:
                target = target_val.to(device=device, dtype=torch.long).view(current_batch_size)

            with torch.amp.autocast(device_type=device_type, enabled=autocast_enabled, dtype=amp_dtype):
                y_pred = model(emb_input)
                y_pred_for_loss = y_pred
                if (
                    isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss))
                    and num_classes == 1
                    and y_pred.ndim > 1
                    and y_pred.shape[1] == 1
                ):
                    y_pred_for_loss = y_pred.squeeze(-1)

                y_pred_final = y_pred_for_loss.to(torch.float32)
                target_for_loss = target.to(y_pred_final.device)

                loss = torch.tensor(float("nan"), device=device)
                if isinstance(criterion, nn.NLLLoss):
                    loss = criterion(F.log_softmax(y_pred_final, dim=-1), target_for_loss.long())
                elif isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, GHMC_Loss)):
                    loss = criterion(y_pred_final, target_for_loss.long())
                elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)):
                    loss = criterion(y_pred_final, target_for_loss.float())

            if not math.isnan(loss.item()):
                total_loss += loss.item() * current_batch_size
                total_samples += current_batch_size

            if total_samples > 0:
                val_iterator.set_postfix({"AvgLoss": f"{(total_loss / total_samples):.4e}"})
        except Exception as e_val:
            print(f"Error during embedding validation step: {e_val}")
            traceback.print_exc()
            continue
        finally:
            with suppress(NameError):
                del emb_input, target_val, target, y_pred, y_pred_for_loss, loss

    val_iterator.close()
    model.train()
    if total_samples == 0:
        return float("nan"), False

    avg_loss = total_loss / total_samples
    print(f"Validation (Embeddings) finished. Avg Loss: {avg_loss:.4e} ({total_samples} samples)")

    if not math.isnan(avg_loss) and avg_loss < best_eval_loss_so_far:
        should_save_best = True
        print(
            f"  New best validation loss candidate: {avg_loss:.4e} "
            f"(previous best: {best_eval_loss_so_far:.4e})"
        )
    elif not math.isnan(avg_loss):
        print(f"  Validation loss {avg_loss:.4e} did not improve on best {best_eval_loss_so_far:.4e}")

    return avg_loss, should_save_best


@torch.no_grad()
def run_validation_sequences(model, val_loader, criterion, device, scaler, num_labels, best_eval_loss_so_far: float):
    """Run validation loop for sequence models (with masks)."""
    if val_loader is None:
        print("run_validation_sequences: Validation loader missing.")
        return float("nan"), False

    model.eval()
    autocast_enabled = scaler is not None and scaler.is_enabled()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    total_loss = 0.0
    total_samples = 0
    should_save_best = False

    val_iterator = tqdm(val_loader, desc="Validation", leave=False, dynamic_ncols=True)
    for batch_data in val_iterator:
        sequence_batch = None
        mask_batch = None
        label_batch = None
        target_for_loss = None
        y_pred = None
        y_pred_for_loss = None
        loss = None

        if batch_data is None or not batch_data:
            continue
        try:
            sequence_batch = batch_data.get("sequence")
            mask_batch = batch_data.get("mask")
            label_batch = batch_data.get("label")
            if sequence_batch is None or mask_batch is None or label_batch is None:
                continue

            sequence_batch = sequence_batch.to(device)
            mask_batch = mask_batch.to(device)
            if not torch.isfinite(sequence_batch).all():
                print("Warning: Non-finite values in validation sequence. Skipping.")
                continue

            batch_size = sequence_batch.size(0)
            y_pred_final = None
            try:
                with torch.amp.autocast(device_type=device_type, enabled=autocast_enabled, dtype=amp_dtype):
                    y_pred = model(sequence_batch, attention_mask=mask_batch)
                    y_pred_for_loss = y_pred
                    if (
                        num_labels == 1
                        and isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss))
                        and y_pred.ndim == 2
                        and y_pred.shape[1] == 1
                    ):
                        y_pred_for_loss = y_pred.squeeze(1)
                    y_pred_final = y_pred_for_loss.to(torch.float32)
            except Exception as e_pred:
                print(f"Error during validation prediction: {e_pred}")
                continue

            try:
                if isinstance(criterion, (nn.CrossEntropyLoss, FocalLoss, nn.NLLLoss, GHMC_Loss)):
                    target_for_loss = label_batch.squeeze().to(device=device, dtype=torch.long)
                    if target_for_loss.shape[0] != batch_size:
                        raise ValueError("Target shape mismatch (Long)")
                    if isinstance(criterion, nn.NLLLoss):
                        loss = criterion(F.log_softmax(y_pred_final, dim=-1), target_for_loss)
                    else:
                        loss = criterion(y_pred_final, target_for_loss)
                elif isinstance(criterion, (nn.BCEWithLogitsLoss, nn.L1Loss, nn.MSELoss)):
                    target_for_loss = label_batch.squeeze().to(device=device, dtype=torch.float32)
                    if target_for_loss.shape[0] != batch_size:
                        raise ValueError("Target shape mismatch (Float)")
                    loss = criterion(y_pred_final, target_for_loss)
                else:
                    loss = torch.tensor(float("nan"), device=device)
            except Exception as e_val_step:
                print(f"Error during validation step calculation: {e_val_step}")
                print(
                    f"  Pred shape: {y_pred_final.shape if y_pred_final is not None else 'N/A'}, "
                    f"Target shape: {target_for_loss.shape if target_for_loss is not None else 'N/A'}, "
                    f"Target dtype: {target_for_loss.dtype if target_for_loss is not None else 'N/A'}"
                )
                loss = torch.tensor(float("nan"), device=device)

            if not math.isnan(loss.item()):
                total_loss += loss.item() * batch_size
                total_samples += batch_size
            else:
                print("Warning: NaN loss encountered during validation.")

            if total_samples > 0:
                val_iterator.set_postfix({"AvgLoss": f"{(total_loss / total_samples):.4e}"})
        except Exception as e_batch:
            print(f"Error processing validation batch: {e_batch}")
            traceback.print_exc()
        finally:
            with suppress(NameError):
                del sequence_batch, mask_batch, label_batch, target_for_loss, y_pred, y_pred_for_loss, loss

    val_iterator.close()
    model.train()
    if total_samples == 0:
        print("Warning: No valid samples processed during validation.")
        return float("nan"), False

    avg_loss = total_loss / total_samples
    print(f"Validation finished. Avg Loss: {avg_loss:.4e} ({total_samples} samples)")

    if not math.isnan(avg_loss) and avg_loss < best_eval_loss_so_far:
        should_save_best = True
        print(
            f"  New best validation loss candidate: {avg_loss:.4e} "
            f"(previous best: {best_eval_loss_so_far:.4e})"
        )
    elif not math.isnan(avg_loss):
        print(f"  Validation loss {avg_loss:.4e} did not improve on best {best_eval_loss_so_far:.4e}")

    return avg_loss, should_save_best
