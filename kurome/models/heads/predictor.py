"""Predictor head implementation for embedding-mode training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Standard residual block with LayerNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.long = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.long(self.norm(x))


class PredictorModel(nn.Module):
    """Flexible predictor/classifier head for single-vector embeddings."""

    def __init__(
        self,
        features: int = 1152,
        hidden_dim: int = 1280,
        num_classes: int = 2,
        use_attention: bool = True,
        num_attn_heads: int = 8,
        attn_dropout: float = 0.1,
        num_res_blocks: int = 1,
        dropout_rate: float = 0.1,
        output_mode: str = "linear",
    ) -> None:
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.output_mode = output_mode.lower()

        print("DEBUG PredictorModel v2.0.0 Init:")
        print(f"  features={features}, hidden_dim={hidden_dim}, num_classes={num_classes}")
        print(f"  use_attention={use_attention}, heads={num_attn_heads}, attn_drop={attn_dropout}")
        print(f"  num_res_blocks={num_res_blocks}, dropout_rate={dropout_rate}, output_mode='{output_mode}'")

        self.attention: nn.MultiheadAttention | None = None
        self.norm_attn: nn.LayerNorm | None = None
        if self.use_attention:
            actual_num_heads = num_attn_heads
            if features % num_attn_heads != 0:
                possible_heads = [h for h in [1, 2, 4, 6, 8, 12, 16] if features % h == 0]
                if not possible_heads:
                    raise ValueError(f"Features ({features}) not divisible by any standard head count.")
                actual_num_heads = min(possible_heads, key=lambda x: abs(x - num_attn_heads))
                print(
                    f"  Warning: Adjusting attn heads from {num_attn_heads} "
                    f"to {actual_num_heads} for features={features}."
                )
            self.attention = nn.MultiheadAttention(
                embed_dim=self.features,
                num_heads=actual_num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            self.norm_attn = nn.LayerNorm(self.features)

        self.initial_proj = nn.Linear(self.features, self.hidden_dim)
        self.norm_initial = nn.LayerNorm(self.hidden_dim)
        self.activation_initial = nn.GELU()
        self.res_blocks = nn.Sequential(*[ResBlock(channels=self.hidden_dim) for _ in range(num_res_blocks)])

        self.down = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Dropout(p=dropout_rate),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.final_layer = nn.Linear(32, self.num_classes)

        valid_modes = ["linear", "sigmoid", "softmax", "tanh_scaled"]
        if self.output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{self.output_mode}'. Must be one of {valid_modes}")
        if self.output_mode == "softmax" and self.num_classes <= 1:
            print(f"  Warning: output_mode='softmax' usually used with num_classes > 1 (got {self.num_classes}).")
        if self.output_mode in ["sigmoid", "tanh_scaled"] and self.num_classes != 1:
            print(f"  Warning: output_mode='{self.output_mode}' usually used with num_classes=1 (got {self.num_classes}).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention and self.attention is not None and self.norm_attn is not None:
            x_seq = x.unsqueeze(1)
            attn_output, _ = self.attention(x_seq, x_seq, x_seq)
            x = self.norm_attn(x + attn_output.squeeze(1))

        x = self.initial_proj(x)
        x = self.norm_initial(x)
        x = self.activation_initial(x)
        x = self.res_blocks(x)
        x = self.down(x)
        logits = self.final_layer(x)

        if self.output_mode == "linear":
            output = logits
        elif self.output_mode == "sigmoid":
            output = torch.sigmoid(logits)
        elif self.output_mode == "softmax":
            output = F.softmax(logits, dim=-1)
        elif self.output_mode == "tanh_scaled":
            output = (torch.tanh(logits) + 1.0) / 2.0
        else:
            raise RuntimeError(f"Invalid output_mode '{self.output_mode}' in forward pass.")

        if self.num_classes == 1 and output.ndim == 2 and output.shape[1] == 1:
            output = output.squeeze(-1)

        return output


__all__ = ["PredictorModel"]
