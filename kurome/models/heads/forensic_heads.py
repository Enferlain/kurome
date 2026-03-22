"""Task-specific sequence heads for forensic signal classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_output_mode(logits: torch.Tensor, output_mode: str) -> torch.Tensor:
    """Apply the configured output activation to logits."""
    mode = output_mode.lower()
    if mode == "linear":
        return logits
    if mode == "sigmoid":
        return torch.sigmoid(logits)
    if mode == "softmax":
        return F.softmax(logits, dim=-1)
    if mode == "tanh_scaled":
        return (torch.tanh(logits) + 1.0) / 2.0
    raise RuntimeError(f"Invalid output_mode '{output_mode}'.")


def masked_mean_pool(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Pool a sequence with an optional boolean mask."""
    if mask is None:
        return sequence.mean(dim=1)
    mask_f = mask.unsqueeze(-1).float()
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (sequence * mask_f).sum(dim=1) / denom


def masked_max_pool(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Max-pool a sequence with an optional boolean mask."""
    if mask is None:
        return sequence.max(dim=1).values
    masked = sequence.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    pooled = masked.max(dim=1).values
    return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def masked_std_pool(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Compute per-channel standard deviation with an optional mask."""
    mean = masked_mean_pool(sequence, mask)
    if mask is None:
        centered = sequence - mean.unsqueeze(1)
        return centered.square().mean(dim=1).clamp(min=1e-12).sqrt()
    mask_f = mask.unsqueeze(-1).float()
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    centered = (sequence - mean.unsqueeze(1)) * mask_f
    var = centered.square().sum(dim=1) / denom
    return var.clamp(min=1e-12).sqrt()


class SwiGLUFFN(nn.Module):
    """Compact SwiGLU block used across forensic heads."""

    def __init__(self, dim: int, hidden_mult: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = max(dim, int(dim * hidden_mult))
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.dropout(x)
        return self.out(x)


class TokenResBlock(nn.Module):
    """Token-wise residual block."""

    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = SwiGLUFFN(dim=dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.ffn(self.norm(x)))


class AttentionPool(nn.Module):
    """Single-query attention pooling over a token sequence."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        actual_heads = max(1, num_heads)
        if embed_dim % actual_heads != 0:
            divisors = [h for h in [1, 2, 4, 8, 16, 32] if embed_dim % h == 0]
            actual_heads = divisors[-1] if divisors else 1
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=actual_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = sequence.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        inverted_mask = ~mask if mask is not None else None
        pooled, _ = self.attn(query=query, key=sequence, value=sequence, key_padding_mask=inverted_mask)
        return self.norm(pooled.squeeze(1))


class TopKTokenPool(nn.Module):
    """Pool the highest-scoring tokens to emphasize localized forensic evidence."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        topk_ratio: float = 0.1,
        min_topk: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(sequence).squeeze(-1)
        batch_pooled: list[torch.Tensor] = []
        batch_stats: list[torch.Tensor] = []

        for batch_idx in range(sequence.shape[0]):
            valid_mask = mask[batch_idx] if mask is not None else None
            tokens = sequence[batch_idx]
            token_scores = scores[batch_idx]
            if valid_mask is not None:
                tokens = tokens[valid_mask]
                token_scores = token_scores[valid_mask]

            if tokens.shape[0] == 0:
                batch_pooled.append(sequence.new_zeros(sequence.shape[-1]))
                batch_stats.append(sequence.new_zeros(3))
                continue

            k = max(min_topk, math.ceil(tokens.shape[0] * topk_ratio))
            k = min(k, tokens.shape[0])
            top_scores, top_indices = torch.topk(token_scores, k=k, dim=0)
            top_tokens = tokens[top_indices]
            weights = torch.softmax(top_scores, dim=0)
            pooled = (top_tokens * weights.unsqueeze(-1)).sum(dim=0)
            stats = torch.stack(
                [
                    token_scores.mean(),
                    token_scores.max(),
                    top_scores.mean(),
                ]
            )
            batch_pooled.append(pooled)
            batch_stats.append(stats)

        return torch.stack(batch_pooled, dim=0), torch.stack(batch_stats, dim=0)


class ConvContextBlock(nn.Module):
    """Lightweight local-context mixer over token sequences."""

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        y = F.gelu(self.pointwise(y))
        y = self.dropout(y.transpose(1, 2))
        return residual + y


class ForensicMILHeadModel(nn.Module):
    """Sequence model that fuses global context with top-k suspicious tokens."""

    def __init__(
        self,
        features: int,
        num_classes: int,
        pooling_strategy: str = "attn",
        hidden_dim: int = 512,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.2,
        output_mode: str = "linear",
        attn_pool_heads: int = 8,
        attn_pool_dropout: float = 0.1,
        topk_ratio: float = 0.1,
        min_topk: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.output_mode = output_mode.lower()
        self.topk_ratio = topk_ratio
        self.min_topk = min_topk

        self.input_proj = nn.Linear(features, hidden_dim)
        self.blocks = nn.Sequential(*[TokenResBlock(hidden_dim, dropout=dropout_rate) for _ in range(num_res_blocks)])
        self.global_pool = AttentionPool(hidden_dim, num_heads=attn_pool_heads, dropout=attn_pool_dropout)
        self.local_pool = TopKTokenPool(hidden_dim, hidden_dim=max(64, hidden_dim // 2), dropout=dropout_rate)

        fusion_dim = hidden_dim * 5 + 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.blocks(self.input_proj(sequence.float()))
        global_attn = self.global_pool(x, mask=attention_mask)
        global_mean = masked_mean_pool(x, attention_mask)
        local_topk, score_stats = self.local_pool(
            x,
            attention_mask,
            topk_ratio=self.topk_ratio,
            min_topk=self.min_topk,
        )
        fused = torch.cat(
            [
                global_attn,
                global_mean,
                local_topk,
                torch.abs(global_attn - local_topk),
                global_attn * local_topk,
                score_stats,
            ],
            dim=-1,
        )
        logits = self.classifier(fused)
        output = apply_output_mode(logits, self.output_mode)
        if self.num_classes == 1 and output.ndim == 2 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output


class ForensicMultiPoolHeadModel(nn.Module):
    """Sequence model that fuses several complementary global pooling strategies."""

    def __init__(
        self,
        features: int,
        num_classes: int,
        pooling_strategy: str = "attn",
        hidden_dim: int = 512,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.2,
        output_mode: str = "linear",
        attn_pool_heads: int = 8,
        attn_pool_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.output_mode = output_mode.lower()

        self.input_proj = nn.Linear(features, hidden_dim)
        self.blocks = nn.Sequential(*[TokenResBlock(hidden_dim, dropout=dropout_rate) for _ in range(num_res_blocks)])
        self.attn_pool = AttentionPool(hidden_dim, num_heads=attn_pool_heads, dropout=attn_pool_dropout)

        fusion_dim = hidden_dim * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.blocks(self.input_proj(sequence.float()))
        pooled = torch.cat(
            [
                self.attn_pool(x, mask=attention_mask),
                masked_mean_pool(x, attention_mask),
                masked_max_pool(x, attention_mask),
                masked_std_pool(x, attention_mask),
            ],
            dim=-1,
        )
        logits = self.classifier(pooled)
        output = apply_output_mode(logits, self.output_mode)
        if self.num_classes == 1 and output.ndim == 2 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output


class ForensicConvMILHeadModel(nn.Module):
    """Sequence model that adds local token mixing before suspicious-token pooling."""

    def __init__(
        self,
        features: int,
        num_classes: int,
        pooling_strategy: str = "attn",
        hidden_dim: int = 512,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.2,
        output_mode: str = "linear",
        attn_pool_heads: int = 8,
        attn_pool_dropout: float = 0.1,
        topk_ratio: float = 0.1,
        min_topk: int = 8,
        conv_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.output_mode = output_mode.lower()
        self.topk_ratio = topk_ratio
        self.min_topk = min_topk

        self.input_proj = nn.Linear(features, hidden_dim)
        self.conv_blocks = nn.Sequential(
            *[
                ConvContextBlock(hidden_dim, kernel_size=conv_kernel_size, dropout=dropout_rate)
                for _ in range(num_res_blocks)
            ]
        )
        self.post_blocks = nn.Sequential(*[TokenResBlock(hidden_dim, dropout=dropout_rate) for _ in range(max(1, num_res_blocks // 2))])
        self.global_pool = AttentionPool(hidden_dim, num_heads=attn_pool_heads, dropout=attn_pool_dropout)
        self.local_pool = TopKTokenPool(hidden_dim, hidden_dim=max(64, hidden_dim // 2), dropout=dropout_rate)

        fusion_dim = hidden_dim * 5 + 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(sequence.float())
        x = self.conv_blocks(x)
        x = self.post_blocks(x)
        global_attn = self.global_pool(x, mask=attention_mask)
        global_max = masked_max_pool(x, attention_mask)
        local_topk, score_stats = self.local_pool(
            x,
            attention_mask,
            topk_ratio=self.topk_ratio,
            min_topk=self.min_topk,
        )
        fused = torch.cat(
            [
                global_attn,
                global_max,
                local_topk,
                torch.abs(global_attn - local_topk),
                global_attn * local_topk,
                score_stats,
            ],
            dim=-1,
        )
        logits = self.classifier(fused)
        output = apply_output_mode(logits, self.output_mode)
        if self.num_classes == 1 and output.ndim == 2 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output


__all__ = [
    "ForensicMILHeadModel",
    "ForensicMultiPoolHeadModel",
    "ForensicConvMILHeadModel",
]
