"""Hybrid head for embedding-mode training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return x / (rms + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block."""

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
        hidden_features = hidden_features or int(in_features * 8 / 3 / 2 * 2)
        hidden_features = (hidden_features + 1) // 2 * 2

        self.w12 = nn.Linear(in_features, hidden_features * 2, bias=False)
        self.act = act_layer()
        self.dropout1 = nn.Dropout(dropout)
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_val, up_val = self.w12(x).chunk(2, dim=-1)
        x = self.dropout1(self.act(gate_val) * up_val)
        x = self.dropout2(self.w3(x))
        return x


class ResBlockRMS(nn.Module):
    """Residual block based on RMSNorm + SwiGLU."""

    def __init__(self, ch: int, dropout: float = 0.0, rms_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = RMSNorm(ch, eps=rms_norm_eps)
        self.ffn = SwiGLUFFN(in_features=ch, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class HybridHeadModel(nn.Module):
    """Embedding head combining optional self-attention with RMSNorm/SwiGLU MLP blocks."""

    def __init__(
        self,
        features: int,
        hidden_dim: int = 1280,
        num_classes: int = 2,
        use_attention: bool = True,
        num_attn_heads: int = 16,
        attn_dropout: float = 0.1,
        num_res_blocks: int = 3,
        dropout_rate: float = 0.1,
        rms_norm_eps: float = 1e-6,
        output_mode: str = "linear",
    ) -> None:
        super().__init__()
        self.features = features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.output_mode = output_mode.lower()

        print("DEBUG HybridHeadModel Init:")
        print(f"  features={features}, hidden_dim={hidden_dim}, num_classes={num_classes}")
        print(f"  use_attention={use_attention}, heads={num_attn_heads}, attn_drop={attn_dropout}")
        print(
            f"  num_res_blocks={num_res_blocks}, dropout_rate={dropout_rate}, "
            f"RMSNorm(eps={rms_norm_eps}), output_mode='{output_mode}'"
        )

        self.attention: nn.MultiheadAttention | None = None
        self.norm_attn: RMSNorm | None = None
        if self.use_attention:
            actual_num_heads = num_attn_heads
            if features % num_attn_heads != 0:
                possible_heads = [h for h in [1, 2, 4, 8, 16] if features % h == 0]
                if not possible_heads:
                    raise ValueError(
                        f"Attention Error: Features ({features}) not divisible by any standard head count."
                    )
                actual_num_heads = min(possible_heads, key=lambda x: abs(x - num_attn_heads))
                print(
                    f"  Warning: Adjusting self-attention heads from {num_attn_heads} "
                    f"to {actual_num_heads} for features={features}."
                )

            self.attention = nn.MultiheadAttention(
                embed_dim=self.features,
                num_heads=actual_num_heads,
                dropout=attn_dropout,
                batch_first=True,
                bias=True,
            )
            self.norm_attn = RMSNorm(self.features, eps=rms_norm_eps)
            print(f"  Using initial self-attention layer (Heads: {actual_num_heads}).")

        mlp_layers: list[nn.Module] = []
        mlp_layers.append(nn.Linear(self.features, self.hidden_dim))
        mlp_layers.append(RMSNorm(self.hidden_dim, eps=rms_norm_eps))
        for _ in range(num_res_blocks):
            mlp_layers.append(ResBlockRMS(ch=self.hidden_dim, dropout=dropout_rate, rms_norm_eps=rms_norm_eps))

        mlp_layers.append(RMSNorm(self.hidden_dim, eps=rms_norm_eps))
        down_proj_hidden = self.hidden_dim // 2
        mlp_layers.append(
            SwiGLUFFN(
                in_features=self.hidden_dim,
                hidden_features=down_proj_hidden,
                out_features=down_proj_hidden,
                dropout=dropout_rate,
            )
        )
        mlp_layers.append(RMSNorm(down_proj_hidden, eps=rms_norm_eps))
        mlp_layers.append(nn.Linear(down_proj_hidden, self.num_classes))
        self.mlp_head = nn.Sequential(*mlp_layers)

        valid_modes = ["linear", "sigmoid", "softmax", "tanh_scaled"]
        if self.output_mode not in valid_modes:
            raise ValueError(f"Invalid output_mode '{self.output_mode}'. Must be one of {valid_modes}")
        if self.output_mode == "softmax" and self.num_classes <= 1:
            print("  Warning: output_mode='softmax' usually used with num_classes > 1.")
        if self.output_mode in ["sigmoid", "tanh_scaled"] and self.num_classes != 1:
            print(f"  Warning: output_mode='{self.output_mode}' usually used with num_classes=1.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention and self.attention is not None and self.norm_attn is not None:
            x_seq = x.unsqueeze(1)
            attn_output, _ = self.attention(x_seq, x_seq, x_seq)
            x = self.norm_attn(x + attn_output.squeeze(1))

        logits = self.mlp_head(x.float())
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

        if self.num_classes == 1 and output.ndim == 2 and output.shape[1] == 1:
            output = output.squeeze(-1)
        return output


__all__ = ["HybridHeadModel"]
