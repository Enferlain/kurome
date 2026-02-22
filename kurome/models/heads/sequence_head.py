"""Sequence head model for feature-sequence training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep current default behavior for existing configs/checkpoints.
RMSNorm = nn.RMSNorm
USE_RMSNORM = True
NormLayer = nn.RMSNorm if USE_RMSNORM else nn.LayerNorm
print(f"DEBUG HeadModel: Using {'RMSNorm' if USE_RMSNORM else 'LayerNorm'}")


class SwiGLUFFNHead(nn.Module):
    """SwiGLU feed-forward block used by the sequence head."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.SiLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, hidden_features * 2, bias=False)
        self.act = act_layer()
        self.dropout1 = nn.Dropout(dropout)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.w12(x).chunk(2, dim=-1)
        x = self.dropout1(self.act(gate) * value)
        x = self.dropout2(self.w3(x))
        return x


class ResBlock(nn.Module):
    """Residual block with configurable normalization/FFN flavor."""

    def __init__(self, ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = NormLayer(ch)
        if USE_RMSNORM:
            self.ffn = SwiGLUFFNHead(in_features=ch, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(ch, ch * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ch * 4, ch),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class AttentionPool(nn.Module):
    """Pools a sequence using a learnable query token + MHA."""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.query_token.expand(batch_size, -1, -1)
        # torch.nn.MultiheadAttention expects True=masked(padded); our masks are True=real.
        inverted_mask = ~key_padding_mask if key_padding_mask is not None else None
        attn_output, _ = self.attention(
            query=query,
            key=x,
            value=x,
            key_padding_mask=inverted_mask,
        )
        pooled_output = attn_output.squeeze(1)
        pooled_output = self.norm(pooled_output)
        return pooled_output


class HeadModel(nn.Module):
    """Sequence head: optional pooling followed by an MLP classifier/regressor."""

    def __init__(
        self,
        features: int,
        num_classes: int,
        pooling_strategy: str = "attn",
        hidden_dim: int = 1024,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.2,
        output_mode: str = "linear",
        attn_pool_heads: int = 16,
        attn_pool_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy.lower()
        self.output_mode = output_mode.lower()
        self.num_classes = num_classes

        print("Initializing HeadModel v1.7.0 (Pool + MLP):")
        print(f"  Input Features: {features}, Pooling: {self.pooling_strategy}")
        print(
            "  MLP Hidden: "
            f"{hidden_dim}, ResBlocks: {num_res_blocks}, Dropout: {dropout_rate}, "
            f"Norm: {'RMSNorm' if USE_RMSNORM else 'LayerNorm'}"
        )
        print(f"  Output Mode: {self.output_mode}, Classes: {self.num_classes}")

        self.pooler: AttentionPool | None = None
        if self.pooling_strategy == "attn":
            actual_attn_heads = attn_pool_heads
            if features % attn_pool_heads != 0:
                possible_heads = [h for h in [1, 2, 4, 8, 16, 32] if features % h == 0]
                if not possible_heads:
                    raise ValueError(f"Features ({features}) not divisible by any standard head count.")
                actual_attn_heads = min(possible_heads, key=lambda x: abs(x - attn_pool_heads))
                print(
                    "  Warning: Adjusting pooling attn heads from "
                    f"{attn_pool_heads} to {actual_attn_heads} for features={features}."
                )
            self.pooler = AttentionPool(
                embed_dim=features,
                num_heads=actual_attn_heads,
                dropout=attn_pool_dropout,
            )
            print(f"  Using Attention Pooling on input sequence (Heads: {actual_attn_heads}).")
        elif self.pooling_strategy not in ["avg", "none"]:
            print(f"Warning: Unknown pooling strategy '{self.pooling_strategy}'. Defaulting to average pooling.")
            self.pooling_strategy = "avg"

        mlp_layers: list[nn.Module] = []
        mlp_layers.append(NormLayer(features))
        mlp_layers.append(nn.Linear(features, hidden_dim))
        if not USE_RMSNORM:
            mlp_layers.append(nn.GELU())
        mlp_layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_res_blocks):
            mlp_layers.append(ResBlock(ch=hidden_dim, dropout=dropout_rate))

        if USE_RMSNORM:
            mlp_layers.extend(
                [
                    RMSNorm(hidden_dim),
                    SwiGLUFFNHead(
                        hidden_dim,
                        hidden_features=hidden_dim // 2,
                        out_features=hidden_dim // 2,
                        dropout=dropout_rate / 2,
                    ),
                    RMSNorm(hidden_dim // 2),
                    nn.Linear(hidden_dim // 2, self.num_classes),
                ]
            )
        else:
            mlp_layers.extend(
                [
                    NormLayer(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout_rate / 2),
                    NormLayer(hidden_dim // 2),
                    nn.Linear(hidden_dim // 2, self.num_classes),
                ]
            )
        self.mlp_head = nn.Sequential(*mlp_layers)

        valid_modes = ["linear", "sigmoid", "softmax", "tanh_scaled"]
        if self.output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{self.output_mode}'. Must be one of {valid_modes}")
        if self.output_mode == "softmax" and self.num_classes <= 1:
            print(f"  Warning: output_mode='softmax' usually used with num_classes > 1 (got {self.num_classes}).")
        if self.output_mode in ["sigmoid", "tanh_scaled"] and self.num_classes != 1:
            print(f"  Warning: output_mode='{self.output_mode}' usually used with num_classes=1 (got {self.num_classes}).")

    def forward(self, sequence: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pooling_strategy == "attn":
            if self.pooler is None:
                raise ValueError("Attn pooling selected but pooler not initialized.")
            features = self.pooler(sequence, key_padding_mask=attention_mask)
        elif self.pooling_strategy == "avg":
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(sequence)
                masked_sum = torch.sum(sequence * mask_expanded.float(), dim=1)
                num_real_tokens = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
                features = masked_sum / num_real_tokens
            else:
                features = torch.mean(sequence, dim=1)
        elif self.pooling_strategy == "none":
            features = sequence
            if features.ndim != 2:
                raise ValueError(
                    f"Pooling strategy is 'none' but input sequence has {features.ndim} dims (expected 2)."
                )
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        logits = self.mlp_head(features)
        if self.output_mode == "linear":
            output = logits
        elif self.output_mode == "sigmoid":
            output = torch.sigmoid(logits)
        elif self.output_mode == "softmax":
            output = F.softmax(logits, dim=-1)
        elif self.output_mode == "tanh_scaled":
            output = (torch.tanh(logits) + 1.0) / 2.0
        else:
            raise RuntimeError(f"Invalid output_mode '{self.output_mode}'.")

        if self.num_classes == 1 and output.ndim == 2 and output.shape[-1] == 1:
            output = output.squeeze(-1)
        return output


__all__ = ["HeadModel"]
