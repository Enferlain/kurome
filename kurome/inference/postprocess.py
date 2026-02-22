"""Postprocessing helpers for inference outputs."""

from __future__ import annotations

import torch


def format_classifier_prediction(
    pred: torch.Tensor,
    *,
    labels: dict,
    num_classes: int,
    output_mode: str,
    drop_default: bool = False,
    tile_strategy: str = "mean",
) -> dict:
    """Format single-model classifier outputs into a label->score dictionary."""
    pred_to_format = _reduce_tiled_prediction(pred, num_classes=num_classes, tile_strategy=tile_strategy)
    model_output_mode = (output_mode or "linear").lower()

    out = {}
    probabilities = pred_to_format
    if num_classes == 1:
        scalar_value = probabilities.item() if probabilities.ndim == 0 else probabilities[0].item()
        final_score = scalar_value
        if model_output_mode == "linear":
            final_score = torch.sigmoid(torch.tensor(scalar_value)).item()
        positive_label_name = labels.get("1", "1")
        negative_label_name = labels.get("0", "0")
        if not (drop_default and negative_label_name == labels.get("0")):
            out[negative_label_name] = float(1.0 - final_score)
        out[positive_label_name] = float(final_score)
    elif num_classes > 1:
        if model_output_mode == "linear":
            probabilities = torch.softmax(pred_to_format, dim=-1)
        for k in range(num_classes):
            if k == 0 and drop_default:
                continue
            key = labels.get(str(k), str(k))
            out[key] = float(probabilities[k].item())
    else:
        print(f"ERROR: Invalid num_classes ({num_classes}).")
    return out


def format_multi_model_prediction_raw(
    pred: torch.Tensor,
    *,
    labels: dict,
    num_classes: int,
    drop_default: bool = False,
    tile_strategy: str = "mean",
) -> dict:
    """Format multi-model classifier outputs, preserving existing raw-score behavior."""
    num_tiles = pred.shape[0] if pred.ndim == 2 else 1
    if num_tiles > 1 and tile_strategy != "raw":
        pred_to_format = _reduce_tiled_prediction(pred, num_classes=num_classes, tile_strategy=tile_strategy)
    else:
        pred_to_format = pred[0].detach().cpu() if pred.ndim > 1 else pred.detach().cpu()

    out = {}
    for k in range(num_classes):
        if k == 0 and drop_default:
            continue
        label_index_str = str(k)
        key = labels.get(label_index_str, label_index_str)
        out[key] = float(pred_to_format[k].item())
    return out


def format_sequence_prediction(
    pred: torch.Tensor,
    *,
    labels: dict,
    num_labels: int,
    output_mode: str,
) -> dict:
    """Format head-sequence prediction outputs into a label->score dictionary."""
    output = {}
    try:
        if num_labels == 1:
            scalar_value = pred.item()
            final_score = scalar_value
            if output_mode == "linear":
                final_score = torch.sigmoid(torch.tensor(scalar_value)).item()
            pos_label_name = labels.get("1", "1")
            neg_label_name = labels.get("0", "0")
            output[neg_label_name] = float(1.0 - final_score)
            output[pos_label_name] = float(final_score)
        elif num_labels > 1:
            probabilities = pred.squeeze(0)
            if output_mode == "linear":
                probabilities = torch.softmax(probabilities, dim=-1)
            for k in range(num_labels):
                label_index_str = str(k)
                key = labels.get(label_index_str, label_index_str)
                output[key] = float(probabilities[k].item())
        else:
            output = {"error": f"Invalid num_labels: {num_labels}"}
    except Exception as exc:
        print(f"Error formatting prediction: {exc}")
        return {"error": "Formatting failed"}
    return output


def _reduce_tiled_prediction(pred: torch.Tensor, *, num_classes: int, tile_strategy: str) -> torch.Tensor:
    """Reduce tile predictions to one vector according to aggregation strategy."""
    if pred.ndim >= 2 and tile_strategy != "raw":
        combined_pred = torch.zeros(num_classes, device="cpu")
        for k in range(num_classes):
            tile_scores = pred[:, k].detach().cpu()
            val = 0.0
            try:
                if tile_strategy == "mean":
                    val = torch.mean(tile_scores).item()
                elif tile_strategy == "median":
                    val = torch.median(tile_scores).item()
                elif tile_strategy == "max":
                    val = torch.max(tile_scores).item()
                elif tile_strategy == "min":
                    val = torch.min(tile_scores).item()
                else:
                    raise NotImplementedError(f"Invalid strategy '{tile_strategy}'")
            except Exception as exc:
                print(f"Error calculating tile strategy '{tile_strategy}' for class {k}: {exc}")
                val = 0.0
            combined_pred[k] = val
        return combined_pred

    if pred.ndim > 1:
        return pred.detach().cpu().squeeze(0)
    return pred.detach().cpu()
