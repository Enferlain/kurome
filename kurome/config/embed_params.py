"""Embedding-version metadata lookup used by training and inference."""

from __future__ import annotations

EMBED_CONFIGS: dict[str, dict[str, int]] = {
    # SigLIP
    "CLIP": {"features": 768, "hidden": 1024},
    "siglip2_so400m_patch16_512_FitPad": {"features": 1152, "hidden": 1280},
    "siglip2_so400m_patch16_512_CenterCrop": {"features": 1152, "hidden": 1280},
    "siglip2_so400m_patch16_naflex_Naflex_Proc1024": {"features": 1152, "hidden": 1280},
    "siglip2_so400m_patch16_naflex_Naflex_Proc2048": {"features": 1152, "hidden": 1280},
    "apple_aimv2_large_patch14_native_AIMv2CLS": {"features": 1024, "hidden": 1280},
    # DINOv2
    "fb_dinov2_giant_FitPad": {"features": 1536, "hidden": 1280},
    "timm_vit_large_patch14_dinov2.lvd142m_FitPad": {"features": 1024, "hidden": 1280},
    # DINOv3
    "fb_dinov3_vit7b16_pretrain_lvd1689m_8bit_DINOv3_8bit_BnB": {"features": 4096, "hidden": 1280},
    # Other
    "META": {"features": 1024, "hidden": 1280},
}


def get_embed_params(ver: str) -> dict[str, int]:
    """Return features/hidden dimensions for a known embedding version key."""
    if ver in EMBED_CONFIGS:
        return EMBED_CONFIGS[ver]
    raise ValueError(
        f"Unknown/undefined embedding version key '{ver}' in get_embed_params. "
        "Please add it or check YAML config."
    )
