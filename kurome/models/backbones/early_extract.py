"""End-to-end image backbone + head model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from aim.v2.utils import load_pretrained

    AIMV2_AVAILABLE = True
except ImportError:
    print("Warning: Could not import 'load_pretrained' from 'aim.v2.utils'.")
    print("Ensure aim.v2 is installed: pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v2'")
    AIMV2_AVAILABLE = False


class ResBlock(nn.Module):
    """Standard residual block with LayerNorm."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(ch)
        self.long = nn.Sequential(
            nn.Linear(ch, ch),
            nn.GELU(),
            nn.Linear(ch, ch),
            nn.GELU(),
            nn.Linear(ch, ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.long(self.norm(x))


class AttentionPool(nn.Module):
    """Pools a sequence of features with a learnable query token."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.query_token.expand(batch_size, -1, -1)
        attn_output, _ = self.attention(query, x, x)
        pooled_output = attn_output.squeeze(1)
        pooled_output = self.norm(pooled_output)
        return pooled_output


class EarlyExtractAnatomyModel(nn.Module):
    """AIMv2-backed end-to-end model with configurable pooling + predictor head."""

    def __init__(
        self,
        base_model_name: str,
        device: str = "cpu",
        extract_layer: int = -1,
        pooling_strategy: str = "attn",
        head_features: int | None = None,
        head_hidden_dim: int = 1024,
        head_num_classes: int = 2,
        head_num_res_blocks: int = 2,
        head_dropout_rate: float = 0.2,
        head_output_mode: str = "linear",
        attn_pool_heads: int = 8,
        attn_pool_dropout: float = 0.1,
        freeze_base_model: bool = True,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.extract_layer = extract_layer
        self.pooling_strategy = pooling_strategy.lower()
        self.compute_dtype = compute_dtype
        self.device = device

        aimv2_short_name = base_model_name.split("/")[-1]
        print(f"Loading base vision model using aim.v2: {base_model_name} (using short name: {aimv2_short_name})")
        if not AIMV2_AVAILABLE:
            exit("Error: aim.v2 package not available. Cannot load AIMv2 model.")

        try:
            self.vision_model = load_pretrained(aimv2_short_name, backend="torch")
            self.vision_model = self.vision_model.to(device=self.device, dtype=self.compute_dtype)

            if hasattr(self.vision_model, "config"):
                self.vision_model.config.output_hidden_states = True
                print(
                    "  Set vision_model.config.output_hidden_states = "
                    f"{self.vision_model.config.output_hidden_states}"
                )
            else:
                print("  Warning: Cannot access vision_model.config to set output_hidden_states.")

            if hasattr(self.vision_model, "config") and hasattr(self.vision_model.config, "hidden_size"):
                self.base_hidden_dim = self.vision_model.config.hidden_size
            else:
                try:
                    self.base_hidden_dim = self.vision_model.trunk.post_trunk_norm.weight.shape[0]
                except Exception as dim_error:
                    raise ValueError(
                        "Could not determine base model hidden dimension "
                        f"from loaded AIMv2 model: {dim_error}"
                    ) from dim_error
            print(f"  Base model loaded via aim.v2. Hidden Dim: {self.base_hidden_dim}")
        except Exception as load_error:
            print(f"Details: {load_error}")
            raise RuntimeError(
                f"Failed to load base vision model {base_model_name} using aim.v2."
            ) from load_error

        if freeze_base_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
            print("  Base vision model frozen.")
        else:
            self.vision_model.train()

        if head_features is None:
            if self.pooling_strategy in ["cls", "avg", "attn"]:
                self.head_features = self.base_hidden_dim
            elif self.pooling_strategy == "pooler":
                print("Warning: Pooling strategy 'pooler' selected for AIMv2. Using 'cls' logic instead (first token).")
                self.head_features = self.base_hidden_dim
                self.pooling_strategy = "cls"
            else:
                raise ValueError(f"Cannot determine head features for pooling strategy '{self.pooling_strategy}'.")
            print(f"  Inferred head input features: {self.head_features}")
        else:
            self.head_features = head_features
            print(f"  Using specified head input features: {self.head_features}")

        self.pooler: AttentionPool | None = None
        if self.pooling_strategy == "attn":
            if self.head_features != self.base_hidden_dim:
                print("Warning: Attention pooling input dim mismatch? Using base_hidden_dim.")
            self.pooler = AttentionPool(
                embed_dim=self.base_hidden_dim,
                num_heads=attn_pool_heads,
                dropout=attn_pool_dropout,
            ).to(device=self.device)
            print("  Using Attention Pooling layer.")
        elif self.pooling_strategy not in ["cls", "avg", "pooler"]:
            raise ValueError(
                f"Invalid pooling_strategy: '{self.pooling_strategy}'. Choose 'cls', 'avg', or 'attn'."
            )

        print("Initializing Predictor Head...")
        self.head = nn.Sequential(
            nn.LayerNorm(self.head_features),
            nn.Linear(self.head_features, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout_rate),
            *[ResBlock(ch=head_hidden_dim) for _ in range(head_num_res_blocks)],
            nn.Linear(head_hidden_dim, head_hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(head_dropout_rate),
            nn.Linear(head_hidden_dim // 4, head_num_classes),
        ).to(device=self.device)

        self.head_output_mode = head_output_mode.lower()
        print(f"  Head initialized. Input: {self.head_features}, Output: {head_num_classes}, Mode: {self.head_output_mode}")
        self.needs_final_norm = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask, spatial_shapes
        input_pixels_tensor = pixel_values.to(dtype=self.compute_dtype)
        request_features = (self.pooling_strategy != "pooler") or (self.extract_layer != -1)
        outputs_tuple = self.vision_model(input_pixels_tensor, output_features=request_features)

        if request_features:
            if isinstance(outputs_tuple, tuple) and len(outputs_tuple) == 2:
                base_output_raw, hidden_states_tuple = outputs_tuple
            else:
                base_output_raw = outputs_tuple
                hidden_states_tuple = None

            if self.extract_layer == -1 and hidden_states_tuple:
                base_features_raw = hidden_states_tuple[-1]
            elif hidden_states_tuple and 0 <= self.extract_layer < len(hidden_states_tuple):
                base_features_raw = hidden_states_tuple[self.extract_layer]
            else:
                base_features_raw = base_output_raw
        else:
            base_output_raw = outputs_tuple
            base_features_raw = base_output_raw

        if base_features_raw is None:
            exit("Error: Could not determine base_features_raw.")

        features_to_pool_or_use = base_features_raw
        is_already_pooled = (self.pooling_strategy == "pooler") or (features_to_pool_or_use.ndim != 3)
        if is_already_pooled:
            pooled_features = features_to_pool_or_use
        elif self.pooling_strategy in {"cls", "avg"}:
            pooled_features = torch.mean(features_to_pool_or_use, dim=1)
        elif self.pooling_strategy == "attn":
            if self.pooler is not None:
                pooled_features = self.pooler(features_to_pool_or_use)
            else:
                pooled_features = torch.mean(features_to_pool_or_use, dim=1)
        else:
            pooled_features = torch.mean(features_to_pool_or_use, dim=1)

        if self.needs_final_norm:
            pooled_features = F.normalize(pooled_features.float(), p=2, dim=-1).to(dtype=self.compute_dtype)

        logits = self.head(pooled_features.to(torch.float32))
        if self.head_output_mode == "linear":
            output = logits
        elif self.head_output_mode == "sigmoid":
            output = torch.sigmoid(logits)
        elif self.head_output_mode == "softmax":
            output = F.softmax(logits, dim=-1)
        elif self.head_output_mode == "tanh_scaled":
            output = (torch.tanh(logits) + 1.0) / 2.0
        else:
            raise RuntimeError(f"Invalid head_output_mode '{self.head_output_mode}'.")
        if output.shape[-1] == 1 and output.ndim > 1:
            output = output.squeeze(-1)
        return output


__all__ = ["EarlyExtractAnatomyModel"]
