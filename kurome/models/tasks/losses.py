"""Task losses used by training factory and engines."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GHMC_Loss(nn.Module):
    """Gradient Harmonizing Mechanism for Classification (GHM-C)."""

    def __init__(self, bins: int = 10, momentum: float = 0.75, reduction: str = "mean") -> None:
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.reduction = reduction
        self.register_buffer("acc_sum", torch.zeros(bins))
        self.is_initialized = False

        print(f"DEBUG GHMC_Loss Init: bins={bins}, momentum={momentum}, reduction='{reduction}'")

    def _get_grad_norm_approx(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits) if logits.shape[-1] == 1 else F.softmax(logits, dim=-1)
        if probs.shape[-1] == 1:
            p_target = probs * targets + (1 - probs) * (1 - targets)
        else:
            p_target = probs.gather(1, targets.view(-1, 1)).view(-1)
        return torch.abs(p_target - 1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[-1]
        if num_classes == 1 and self.acc_sum.device != logits.device:
            print("Warning: GHMC_Loss might work best with num_classes >= 2 and CE-style targets.")

        if num_classes <= 1 and logits.ndim > 1:
            logits = logits.squeeze(-1)

        if targets.dtype != torch.long:
            try:
                targets_long = targets.long()
            except Exception as exc:
                raise TypeError(
                    f"GHMC_Loss requires Long targets for gather, got {targets.dtype}"
                ) from exc
        else:
            targets_long = targets

        if self.edges.device != logits.device:
            self.edges = self.edges.to(logits.device)
            self.acc_sum = self.acc_sum.to(logits.device)

        g = self._get_grad_norm_approx(logits, targets_long)
        bin_indices = torch.floor(g * self.bins).long()
        bin_indices = torch.clamp(bin_indices, 0, self.bins - 1)

        bin_counts = torch.zeros(self.bins, device=logits.device)
        bin_indices_unique, counts = torch.unique(bin_indices, return_counts=True)
        bin_counts[bin_indices_unique] = counts.float()

        if not self.is_initialized:
            self.acc_sum = bin_counts
            self.is_initialized = True
        else:
            self.acc_sum = self.momentum * self.acc_sum + (1 - self.momentum) * bin_counts

        num_examples = logits.shape[0]
        safe_acc_sum = self.acc_sum.clamp(min=1e-6)
        beta = num_examples / safe_acc_sum
        weights = beta[bin_indices]

        if num_classes == 1:
            print("Warning/TODO: GHMC_Loss with single logit output needs review.")
            raise NotImplementedError("GHMC_Loss currently expects CE-style input (logits shape [N, C] where C>=2).")
        if num_classes >= 2:
            ce_loss = F.cross_entropy(logits, targets_long, reduction="none")
            weighted_loss = ce_loss * weights
        else:
            raise ValueError(f"Invalid num_classes ({num_classes})")

        if self.reduction == "mean":
            return weighted_loss.mean()
        if self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"Invalid gamma: {gamma}")
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

        if self.weight is not None:
            print(f"DEBUG FocalLoss Init: Using class weights: {self.weight.tolist()}")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs.float(), targets.long(), reduction="none", weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss_unreduced = (1 - pt).pow(self.gamma) * ce_loss

        if self.reduction == "mean":
            return torch.mean(focal_loss_unreduced)
        if self.reduction == "sum":
            return torch.sum(focal_loss_unreduced)
        if self.reduction == "none":
            return focal_loss_unreduced
        raise ValueError(f"Invalid reduction: {self.reduction}")


__all__ = ["FocalLoss", "GHMC_Loss"]
