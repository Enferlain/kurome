"""Training-loop implementations extracted from CLI entrypoints."""

from __future__ import annotations

import math
import traceback
from collections.abc import Callable
from typing import Any

import torch
from tqdm import tqdm

from kurome.training.checkpoint import (
    maybe_save_periodic_checkpoint,
    update_best_and_maybe_save,
)
from kurome.training.engine import (
    advance_global_step,
    accumulate_scalar_loss,
    average_and_reset_loss_window,
    compute_loss,
    ensure_training_mode,
    has_reached_total_steps,
    mean_tensor_losses,
    maybe_step_scheduler,
    normalize_embedding_batch_data,
    prepare_embedding_sub_batch,
    prepare_prediction_for_loss,
    prepare_sequence_micro_batch,
    restore_training_mode_if_needed,
    run_optimizer_step_and_zero_grad,
    should_run_interval,
    update_progress_postfix,
)
from kurome.training.metrics import (
    append_and_average_validation_loss,
    log_eval_and_average,
    log_eval_loss,
    update_last_eval_loss,
)


def run_embedding_training_loop(
    *,
    args: Any,
    model: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    train_loader: Any,
    val_loader: Any,
    wrapper: Any,
    start_epoch: int,
    initial_global_step: int,
    enabled_amp: bool,
    amp_dtype: torch.dtype,
    is_schedule_free: bool,
    target_dev: str,
    run_validation_embeddings_fn: Callable[..., tuple[float, bool]],
    is_e2e: bool = False,
) -> None:
    """Run embeddings-mode training loop."""
    if (
        not hasattr(args, "num_train_epochs")
        or not hasattr(args, "steps_per_epoch")
        or not hasattr(args, "max_train_steps")
    ):
        print("ERROR: num_train_epochs, steps_per_epoch, or max_train_steps missing from args.")
        return

    ensure_training_mode(model, optimizer)

    total_steps_to_run = args.max_train_steps
    total_epochs_to_run = args.num_train_epochs
    steps_per_epoch = args.steps_per_epoch
    log_every_n = getattr(args, "log_every_n", 10)
    validate_every_n = getattr(args, "validate_every_n", 50)

    print(
        f"Starting training loop. Target Epochs: {total_epochs_to_run}, "
        f"Target Steps: {total_steps_to_run}"
    )
    print(
        f"Steps per epoch: {steps_per_epoch}. End-to-End Mode: {is_e2e}. "
        f"Logging every {log_every_n} steps."
    )

    global_step = initial_global_step
    best_eval_loss = float("inf")
    if hasattr(wrapper, "best_val_loss") and wrapper.best_val_loss != float("inf"):
        best_eval_loss = wrapper.best_val_loss
        print(f"Retrieved best validation loss from wrapper: {best_eval_loss:.4e}")

    last_eval_loss_val = float("nan")

    progress_bar = tqdm(
        initial=initial_global_step,
        total=total_steps_to_run,
        desc="Overall Training",
        unit="step",
        dynamic_ncols=True,
    )

    accumulated_loss_for_log = torch.tensor(0.0, device=target_dev)
    accumulation_steps_for_log = 0

    print("Initializing DataLoader iterator...")
    train_iterator = iter(train_loader)

    try:
        for epoch in range(start_epoch, total_epochs_to_run):
            if has_reached_total_steps(global_step, total_steps_to_run):
                break

            if epoch == start_epoch or (epoch + 1) % 50 == 0 or epoch == total_epochs_to_run - 1:
                print(
                    f"\n--- Starting Epoch {epoch + 1} / {total_epochs_to_run} "
                    f"(Global Step: {global_step}) ---"
                )
            wrapper.current_epoch = epoch

            ensure_training_mode(model, optimizer)

            step_in_epoch = 0
            while step_in_epoch < steps_per_epoch:
                if has_reached_total_steps(global_step, total_steps_to_run):
                    break

                try:
                    batch_data = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    try:
                        batch_data = next(train_iterator)
                    except StopIteration:
                        print("ERROR: DataLoader empty after restart.")
                        global_step = total_steps_to_run
                        break
                except Exception as e_iter:
                    print(f"\nError getting batch data: {e_iter}")
                    global_step = total_steps_to_run
                    break

                if global_step < initial_global_step:
                    step_in_epoch += 1
                    global_step = advance_global_step(global_step, progress_bar=progress_bar)
                    continue

                if batch_data is None or not batch_data:
                    print(f"Warning: Skipping step {global_step} due to invalid batch_data.")
                    step_in_epoch += 1
                    global_step = advance_global_step(global_step, progress_bar=progress_bar)
                    continue

                batch_data_list = normalize_embedding_batch_data(batch_data, global_step=global_step)
                if batch_data_list is None:
                    step_in_epoch += 1
                    global_step = advance_global_step(global_step, progress_bar=progress_bar)
                    continue

                processed_samples_in_step = 0
                minibatch_losses = []
                num_sub_batches = len(batch_data_list)

                for sub_batch in batch_data_list:
                    loss = torch.tensor(float("nan"), device=target_dev)
                    try:
                        if not model.training:
                            ensure_training_mode(model, optimizer)

                        prepared = prepare_embedding_sub_batch(
                            sub_batch,
                            device=target_dev,
                            num_classes=args.num_classes,
                            global_step=global_step,
                        )
                        if prepared is None:
                            continue
                        model_input, target, current_sub_batch_size = prepared

                        with torch.amp.autocast(
                            device_type=target_dev,
                            enabled=enabled_amp,
                            dtype=amp_dtype,
                        ):
                            y_pred = model(model_input)
                            loss_input = prepare_prediction_for_loss(y_pred, criterion, args.num_classes)
                            loss = compute_loss(
                                criterion=criterion,
                                loss_input=loss_input,
                                target=target,
                                output_mode=getattr(args, "output_mode", "linear"),
                                cast_target_by_criterion=False,
                            )

                        if torch.isnan(loss) or torch.isinf(loss):
                            print(
                                f"Warning: NaN/Inf loss detected in sub-batch at step {global_step}. "
                                "Skipping backward."
                            )
                            loss = None
                        else:
                            minibatch_losses.append(loss.detach())
                            processed_samples_in_step += current_sub_batch_size
                            scaler.scale(loss / num_sub_batches).backward()

                    except Exception as exc:
                        print(f"\nError processing sub-batch step {global_step}: {exc}")
                        traceback.print_exc()
                        continue

                optimizer_stepped = False
                if processed_samples_in_step > 0:
                    optimizer_stepped = run_optimizer_step_and_zero_grad(
                        scaler=scaler,
                        optimizer=optimizer,
                        step_label=global_step,
                        error_prefix="Error optimizer step",
                    )
                else:
                    print(f"Warning: No samples processed step {global_step}. Skipping optimizer step.")
                if processed_samples_in_step == 0:
                    optimizer.zero_grad(set_to_none=True)

                maybe_step_scheduler(
                    scheduler,
                    is_schedule_free=is_schedule_free,
                    optimizer_stepped=optimizer_stepped,
                )

                if minibatch_losses:
                    avg_step_loss = mean_tensor_losses(minibatch_losses)
                    accumulation_steps_for_log = accumulate_scalar_loss(
                        accumulated_loss_for_log,
                        accumulation_steps_for_log,
                        avg_step_loss,
                    )

                step_in_epoch += 1
                global_step = advance_global_step(
                    global_step,
                    progress_bar=progress_bar,
                    wrapper=wrapper,
                )

                if should_run_interval(global_step, log_every_n, min_step_exclusive=initial_global_step):
                    avg_loss_value_log, accumulation_steps_for_log = average_and_reset_loss_window(
                        accumulated_loss_for_log,
                        accumulation_steps_for_log,
                    )
                    wrapper.log_main(
                        step=global_step,
                        current_train_loss=avg_loss_value_log,
                        current_val_loss=last_eval_loss_val,
                    )

                    lr = optimizer.param_groups[0]["lr"]
                    postfix_dict = {
                        "Epoch": epoch + 1,
                        "AvgLoss": f"{avg_loss_value_log:.3e}"
                        if not math.isnan(avg_loss_value_log)
                        else "N/A",
                        "LR": f"{lr:.1e}",
                    }
                    if not math.isnan(last_eval_loss_val):
                        postfix_dict["LastEval"] = f"{last_eval_loss_val:.3e}"
                    update_progress_postfix(progress_bar, postfix_dict)

                if should_run_interval(
                    global_step,
                    validate_every_n,
                    min_step_exclusive=initial_global_step,
                ):
                    print(
                        f"\n--- Running Validation @ Step {global_step} "
                        f"(Current best_eval_loss: {best_eval_loss:.4e}) ---"
                    )
                    eval_loss_this_run = float("nan")
                    should_save_new_best = False

                    if val_loader:
                        if is_e2e:
                            print("Placeholder: E2E Validation called. Ensure it's updated.")
                            eval_loss_this_run = (
                                last_eval_loss_val
                                if not math.isnan(last_eval_loss_val)
                                else float("inf")
                            )
                        else:
                            eval_loss_this_run, should_save_new_best = run_validation_embeddings_fn(
                                model,
                                val_loader,
                                criterion,
                                target_dev,
                                scaler,
                                best_eval_loss,
                            )

                        last_eval_loss_val = update_last_eval_loss(last_eval_loss_val, eval_loss_this_run)
                        restore_training_mode_if_needed(model, optimizer)

                    log_eval_loss(
                        wrapper=wrapper,
                        step=global_step,
                        eval_loss=eval_loss_this_run,
                        key="eval/loss",
                        error_prefix="Wandb eval log error",
                    )

                    if math.isnan(eval_loss_this_run):
                        print(f"Warning: Eval loss is NaN at Step {global_step}.")
                    else:
                        print(
                            f"--- Validation Complete @ Step {global_step}: "
                            f"Eval Loss = {eval_loss_this_run:.4e} ---"
                        )

                    best_eval_loss = update_best_and_maybe_save(
                        wrapper=wrapper,
                        args=args,
                        global_step=global_step,
                        epoch=epoch,
                        eval_loss=eval_loss_this_run,
                        best_eval_loss=best_eval_loss,
                        should_save_new_best=should_save_new_best,
                        new_best_message=(
                            "  Validation loss {eval_loss:.4e} IS a new best "
                            "(old best: {best_eval_loss:.4e}). Saving best model..."
                        ),
                        not_improved_message=(
                            "  Validation loss {eval_loss:.4e} did not improve on best "
                            "{best_eval_loss:.4e}. Not saving as best."
                        ),
                        nan_message="  Eval loss was NaN. Not saving as best.",
                    )

                maybe_save_periodic_checkpoint(
                    args=args,
                    wrapper=wrapper,
                    global_step=global_step,
                    epoch=epoch,
                    initial_global_step=initial_global_step,
                )

            if has_reached_total_steps(global_step, total_steps_to_run):
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        progress_bar.close()
        print(f"\nTraining loop finished. Reached Global Step: {global_step}")


def run_feature_sequence_training_loop(
    *,
    args: Any,
    model: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    train_loader: Any,
    val_loader: Any,
    wrapper: Any,
    start_epoch: int,
    initial_global_step: int,
    enabled_amp: bool,
    amp_dtype: torch.dtype,
    is_schedule_free: bool,
    target_dev: str,
    run_validation_sequences_fn: Callable[..., tuple[float, bool]],
) -> None:
    """Run feature-sequence training loop."""
    if not all(
        hasattr(args, attr)
        for attr in ["num_train_epochs", "steps_per_epoch", "max_train_steps", "num_labels"]
    ):
        print(
            "ERROR: Missing required args attributes (num_train_epochs, "
            "steps_per_epoch, max_train_steps, num_labels)."
        )
        return

    ensure_training_mode(model, optimizer)

    gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
    effective_batch_size = args.batch * gradient_accumulation_steps
    print(
        f"Using Gradient Accumulation: {gradient_accumulation_steps} steps "
        f"(Micro Bsz: {args.batch}, Eff Bsz: {effective_batch_size})"
    )

    total_steps_to_run = args.max_train_steps
    total_epochs_to_run = args.num_train_epochs
    steps_per_epoch = getattr(args, "steps_per_epoch", len(train_loader))

    log_every_n = getattr(args, "log_every_n", 100)
    validate_every_n = getattr(args, "validate_every_n", 0)
    if validate_every_n <= 0:
        global_steps_per_epoch = (
            math.ceil(steps_per_epoch / gradient_accumulation_steps)
            if gradient_accumulation_steps > 0
            else steps_per_epoch
        )
        validate_every_n = global_steps_per_epoch if global_steps_per_epoch > 0 else 1
        print(
            "Validation frequency not set or invalid, defaulting to once per epoch "
            f"({validate_every_n} steps)."
        )
    if log_every_n <= 0:
        log_every_n = 1

    print(f"Starting Training Loop. Target Epochs: {total_epochs_to_run}, Target Steps: {total_steps_to_run}")
    print(
        f"Micro-batches per epoch: {steps_per_epoch}. Logging every {log_every_n} steps. "
        f"Validating every {validate_every_n} steps."
    )

    global_step = initial_global_step
    best_eval_loss = (
        wrapper.best_val_loss
        if wrapper.best_val_loss is not None and not math.isnan(wrapper.best_val_loss)
        else float("inf")
    )
    last_eval_loss_val = float("nan")
    current_train_loss_avg = float("nan")

    current_window_loss_sum = torch.tensor(0.0, device=target_dev)
    current_window_steps = 0
    epoch_loss_sum = torch.tensor(0.0, device=target_dev)
    epoch_steps = 0
    all_validation_losses = []

    progress_bar = tqdm(
        initial=initial_global_step,
        total=total_steps_to_run,
        desc=f"Training (Eff Bsz {effective_batch_size})",
        unit="it",
        dynamic_ncols=True,
    )

    try:
        for epoch in range(start_epoch, total_epochs_to_run):
            if has_reached_total_steps(global_step, total_steps_to_run):
                break

            epoch_loss_sum.zero_()
            epoch_steps = 0

            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                print(f"\n--- Starting Epoch {epoch + 1}/{total_epochs_to_run} (Sampler Epoch Set) ---")
                train_loader.sampler.set_epoch(epoch)
            elif hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
                print(
                    f"\n--- Starting Epoch {epoch + 1}/{total_epochs_to_run} "
                    "(Batch Sampler Epoch Set) ---"
                )
                train_loader.batch_sampler.set_epoch(epoch)
            else:
                print(f"\n--- Starting Epoch {epoch + 1}/{total_epochs_to_run} ---")

            wrapper.current_epoch = epoch
            ensure_training_mode(model, optimizer)
            optimizer.zero_grad(set_to_none=True)

            for i, batch_data in enumerate(train_loader):
                if has_reached_total_steps(global_step, total_steps_to_run):
                    break

                if batch_data is None or not batch_data:
                    print(f"Warning: Skipping micro-batch {i} due to invalid batch_data.")
                    continue

                loss_this_step: float = float("nan")
                try:
                    prepared = prepare_sequence_micro_batch(
                        batch_data,
                        device=target_dev,
                        num_classes=args.num_classes,
                        micro_batch_index=i,
                    )
                    if prepared is None:
                        continue
                    sequence_batch, mask_batch, target, _ = prepared

                    with torch.amp.autocast(device_type=target_dev, enabled=enabled_amp, dtype=amp_dtype):
                        y_pred = model(sequence_batch, attention_mask=mask_batch)
                        loss_input = prepare_prediction_for_loss(y_pred, criterion, args.num_classes)
                        loss = compute_loss(
                            criterion=criterion,
                            loss_input=loss_input,
                            target=target,
                            output_mode="linear",
                            cast_target_by_criterion=True,
                        )
                        loss_this_step = float(loss.detach().item())
                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss detected at micro-batch {i}. Skipping backward.")
                        loss = None
                    else:
                        scaler.scale(loss).backward()

                except Exception as exc:
                    print(f"\nError processing micro-batch {i} (Global Step approx {global_step}): {exc}")
                    traceback.print_exc()
                    loss = None

                if loss is not None:
                    current_window_steps = accumulate_scalar_loss(
                        current_window_loss_sum,
                        current_window_steps,
                        loss_this_step,
                    )
                    epoch_steps = accumulate_scalar_loss(
                        epoch_loss_sum,
                        epoch_steps,
                        loss_this_step,
                    )

                if (i + 1) % gradient_accumulation_steps == 0:
                    if global_step < initial_global_step:
                        if gradient_accumulation_steps > 1:
                            optimizer.zero_grad(set_to_none=True)
                        global_step = advance_global_step(global_step, progress_bar=progress_bar)
                        continue

                    optimizer_stepped = run_optimizer_step_and_zero_grad(
                        scaler=scaler,
                        optimizer=optimizer,
                        step_label=global_step,
                        error_prefix="Error during optimizer step",
                    )

                    maybe_step_scheduler(
                        scheduler,
                        is_schedule_free=is_schedule_free,
                        optimizer_stepped=optimizer_stepped,
                    )

                    if optimizer_stepped:
                        global_step = advance_global_step(
                            global_step,
                            progress_bar=progress_bar,
                            wrapper=wrapper,
                        )

                        if should_run_interval(global_step, log_every_n):
                            current_train_loss_avg, current_window_steps = average_and_reset_loss_window(
                                current_window_loss_sum,
                                current_window_steps,
                            )

                            wrapper.log_main(
                                step=global_step,
                                current_train_loss=current_train_loss_avg,
                                current_val_loss=last_eval_loss_val,
                            )
                            lr = optimizer.param_groups[0]["lr"]
                            postfix_dict = {
                                "Epoch": epoch + 1,
                                "Loss(curr)": f"{current_train_loss_avg:.3e}"
                                if not math.isnan(current_train_loss_avg)
                                else "N/A",
                                "LR": f"{lr:.1e}",
                            }
                            if not math.isnan(last_eval_loss_val):
                                postfix_dict["Loss(val)"] = f"{last_eval_loss_val:.3e}"
                            update_progress_postfix(progress_bar, postfix_dict)

                        if should_run_interval(global_step, validate_every_n):
                            print(f"\n--- Running Validation @ Step {global_step} ---")
                            eval_loss_val, should_save_new_best = run_validation_sequences_fn(
                                model,
                                val_loader,
                                criterion,
                                target_dev,
                                scaler,
                                args.num_labels,
                                best_eval_loss,
                            )
                            restore_training_mode_if_needed(model, optimizer)
                            last_eval_loss_val = update_last_eval_loss(last_eval_loss_val, eval_loss_val)

                            if not math.isnan(eval_loss_val):
                                avg_val_loss = append_and_average_validation_loss(
                                    all_validation_losses,
                                    eval_loss_val,
                                )
                                print(f"--- Validation Complete: Eval Loss = {eval_loss_val:.4e} ---")
                                print(f"--- Average Validation Loss So Far: {avg_val_loss:.4e} ---")
                                log_eval_and_average(
                                    wrapper=wrapper,
                                    step=global_step,
                                    eval_loss=eval_loss_val,
                                    avg_eval_loss=avg_val_loss,
                                    eval_key="loss/eval_run_result",
                                    avg_key="loss/average_val_loss",
                                    error_prefix="Wandb val log error",
                                )

                                best_eval_loss = update_best_and_maybe_save(
                                    wrapper=wrapper,
                                    args=args,
                                    global_step=global_step,
                                    epoch=epoch,
                                    eval_loss=eval_loss_val,
                                    best_eval_loss=best_eval_loss,
                                    should_save_new_best=should_save_new_best,
                                    new_best_message=(
                                        "New best val loss: {eval_loss:.4e}. Saving best model..."
                                    ),
                                    not_improved_message=(
                                        "Validation loss {eval_loss:.4e} did not improve on best "
                                        "{best_eval_loss:.4e}"
                                    ),
                                )

                                lr = optimizer.param_groups[0]["lr"]
                                postfix_dict = {
                                    "Epoch": epoch + 1,
                                    "Loss(curr)": f"{current_train_loss_avg:.3e}"
                                    if not math.isnan(current_train_loss_avg)
                                    else "N/A",
                                    "Loss(val)": f"{eval_loss_val:.3e}",
                                    "Loss(avg_val)": f"{avg_val_loss:.3e}",
                                    "LR": f"{lr:.1e}",
                                }
                                update_progress_postfix(progress_bar, postfix_dict)

                            else:
                                print(
                                    f"Warning: Eval loss is NaN at Step {global_step}. Cannot update "
                                    "averages or save best."
                                )

                        maybe_save_periodic_checkpoint(
                            args=args,
                            wrapper=wrapper,
                            global_step=global_step,
                            epoch=epoch,
                            initial_global_step=initial_global_step,
                        )

                        if has_reached_total_steps(global_step, total_steps_to_run):
                            break

                if has_reached_total_steps(global_step, total_steps_to_run):
                    break

            if epoch_steps > 0:
                epoch_avg_loss = (epoch_loss_sum / epoch_steps).item()
                print(f"--- Epoch {epoch + 1} Finished. Average Train Loss: {epoch_avg_loss:.4e} ---")
                if wrapper.wandb_run:
                    try:
                        wrapper.wandb_run.log({"loss/epoch": epoch_avg_loss}, step=global_step)
                    except Exception as exc:
                        print(f"Wandb epoch log error: {exc}")

            if has_reached_total_steps(global_step, total_steps_to_run):
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        progress_bar.close()
        print(f"\nTraining loop finished. Reached Global Step: {global_step}")
