"""Transform and processor loading helpers for data adapters."""

from __future__ import annotations


def load_image_processor(base_vision_model: str):
    """Load a Hugging Face image processor for end-to-end image training."""
    if not base_vision_model:
        raise ValueError("base_vision_model is required to load an image processor.")

    try:
        from transformers import AutoProcessor
    except ImportError as exc:  # pragma: no cover - exercised in runtime env
        raise RuntimeError(
            "transformers is required for end-to-end image mode (AutoProcessor unavailable)."
        ) from exc

    try:
        return AutoProcessor.from_pretrained(base_vision_model, trust_remote_code=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load image processor for '{base_vision_model}': {exc}"
        ) from exc
