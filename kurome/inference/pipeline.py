# Version: 1.10.0 (Adds HybridHeadModel loading support)

import os
import json
import traceback

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Resampling
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModel, AutoImageProcessor # Use AutoImageProcessor
import math

# --- Import Models ---
import dinov3_7b_quant_bnb # <<< ADDED: Import for DINOv3 BnB loading >>>
try:
    from kurome.models.heads import (
        HeadModel,  # Sequence head (unused by these pipelines directly)
        HybridHeadModel,
        PredictorModel,
    )
    from kurome.inference.postprocess import (
        format_classifier_prediction,
        format_multi_model_prediction_raw,
        format_sequence_prediction,
    )
    from kurome.config.embed_params import get_embed_params
except ImportError as e:
    print(f"Error importing model classes or get_embed_params: {e}")
    raise

try:
    import timm
    import timm.data
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm library not found. Cannot use TIMM models for inference.")
    TIMM_AVAILABLE = False

# Define target_len, maybe make it configurable later
TARGET_LEN_INFERENCE = 4096 # Keep if needed by any preprocess func
AIMV2_PATCH_SIZE = 14
AIMV2_TARGET_MAX_PATCHES = 4096
# <<< ADDED: DINOv3 Specific Constants >>>
DINOV3_PATCH_SIZE = 16 # DINOv3 patch size
MAX_DINOV3_RESOLUTION = 4096 # Max resolution for DINOv3, adjust based on GPU memory
# <<< END ADDED DINOv3 Specific Constants >>>


# --- v4.0: Aims for target_patches, ensures dims multiple of patch_size (floor) ---
# Copied from generate_embeddings.py - simplified slightly for inference use
def preprocess_naflex_resize(img_pil, target_patches=1024, patch_size=16):
    """
    Resizes image preserving aspect ratio to have close to target_patches,
    ensuring dimensions are multiples of patch_size by flooring.
    (Inference version - doesn't return patch count)
    """
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None

    aspect_ratio = original_width / original_height

    ideal_patch_w_f = math.sqrt(target_patches * aspect_ratio)
    ideal_patch_h_f = math.sqrt(target_patches / aspect_ratio)

    ideal_width_f = ideal_patch_w_f * patch_size
    ideal_height_f = ideal_patch_h_f * patch_size

    new_width = math.floor(ideal_width_f / patch_size) * patch_size
    new_height = math.floor(ideal_height_f / patch_size) * patch_size

    if new_width == 0: new_width = patch_size
    if new_height == 0: new_height = patch_size

    # Calculate resulting patches (for debugging/verification)
    num_patches_w = new_width // patch_size
    num_patches_h = new_height // patch_size
    total_patches = num_patches_w * num_patches_h
    # print(f"  DEBUG Inf NaflexResize(v4): Original: {original_width}x{original_height}, TargetPatches: {target_patches}, New: {new_width}x{new_height}, Patches: {total_patches} ({num_patches_w}x{num_patches_h})")

    if total_patches > target_patches: # Should not happen with floor, but good sanity check
        print(f"  ERROR Inf: Calculated patches ({total_patches}) exceed target ({target_patches})!")
        return None # Return None if calculation seems wrong

    img_resized = img_pil.resize((int(new_width), int(new_height)), Image.Resampling.LANCZOS)
    return img_resized

# --- Preprocessing Function ---
# (Ensure this is IDENTICAL to the one in generate_embeddings.py)
def preprocess_fit_pad(img_pil, target_size=512, fill_color=(0, 0, 0)):
    """Resizes image to fit, pads to target size."""
    original_width, original_height = img_pil.size
    target_w, target_h = target_size, target_size
    scale = min(target_w / original_width, target_h / original_height)
    new_w = int(original_width * scale)
    new_h = int(original_height * scale)
    if new_w == 0 or new_h == 0: return None # Avoid error on zero-size image
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_padded = Image.new(img_pil.mode, (target_w, target_h), fill_color)
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    img_padded.paste(img_resized, (pad_left, pad_top))
    return img_padded
# --- End Preprocessing ---

# --- Utility: Load Config ---
def _load_config_helper(config_path):
    """Loads full config from JSON file."""
    if not config_path or not os.path.isfile(config_path):
        print(f"DEBUG: Config file not found or not provided: {config_path}")
        return None
    try:
        with open(config_path) as f: config_data = json.load(f)
        print(f"DEBUG: Loaded config from {config_path}")
        return config_data
    except Exception as e:
        print(f"Error reading or processing config file {config_path}: {e}")
        return None
# --- End Utility: Load Config ---

# --- Utility: Load Model State Dict ---
def _load_model_helper(model_path, expected_features, expected_outputs=None):
    """Loads state dict, verifies input features, checks output size."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        sd = load_file(model_path)

        # --- Verify input feature dimension (keep existing logic) ---
        # Check attention first, then MLP input
        first_layer_key_mlp = "initial_proj.weight" # New first linear layer
        first_attn_key = "attention.in_proj_weight" # Attention key
        found_features_from_key = None

        if first_attn_key in sd:
             # Input to attention should match feature dimension
             found_features_from_key = sd[first_attn_key].shape[1]
             if found_features_from_key != expected_features:
                  print(f"Warning: Model {model_path} attention input dim mismatch! Expected: {expected_features}, Found: {found_features_from_key}")
             # Check the MLP input dimension too for safety
             if first_layer_key_mlp in sd and sd[first_layer_key_mlp].shape[1] != expected_features:
                  print(f"Warning: Model {model_path} initial_proj input dim mismatch! Expected: {expected_features}, Found: {sd[first_layer_key_mlp].shape[1]}")

        elif first_layer_key_mlp in sd:
             # No attention, check input to initial_proj
             found_features_from_key = sd[first_layer_key_mlp].shape[1]
             if found_features_from_key != expected_features:
                  print(f"Warning: Model {model_path} initial_proj input dim mismatch! Expected: {expected_features}, Found: {found_features_from_key}")
        else:
             print(f"Warning: Cannot find expected first layer keys ('{first_attn_key}' or '{first_layer_key_mlp}') in {model_path}. Cannot verify input features.")
        # --- End Input Verification ---

        # --- Verify output dimension (Use NEW key name) ---
        # v1.7.14: Look for 'final_layer.bias' from PredictorModel v2
        final_bias_key = "final_layer.bias"
        actual_outputs = None
        if final_bias_key in sd:
             actual_outputs = sd[final_bias_key].shape[0]
        else:
             # Add fallback for old key? Or just error? Let's error for now.
             print(f"ERROR: Could not find final layer bias key '{final_bias_key}' in state dict {model_path}.")
             print(f"       Available keys start with: {[k for k in sd.keys()[:5]]}...") # Print first few keys for debug
             raise KeyError(f"Could not determine outputs from key '{final_bias_key}' in {model_path}.")

        if expected_outputs is not None and actual_outputs != expected_outputs:
            print(f"Warning: Output size mismatch! Expected {expected_outputs}, found {actual_outputs} in {model_path}.")
        # --- End Output Verification ---

        return sd, actual_outputs # Return state dict and actual output count
    except Exception as e:
        print(f"Error loading model state dict from {model_path}: {e}")
        raise
# --- End Utility: Load Model State Dict ---


# --- Utility: Load State Dict Helper (Checks features/outputs) ---
# v1.1.0: More robust output key finding, better feature checking
def _load_state_dict_helper(model_path, expected_features=None, expected_outputs=None):
    """Loads state dict, verifies input features, checks output size."""
    if not os.path.isfile(model_path): raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        sd = load_file(model_path)
        actual_outputs = None
        inferred_features = None

        # --- Infer/Verify input feature dimension ---
        # Check keys associated with different model types
        pred_initial_proj_key = "initial_proj.weight"
        hybrid_initial_proj_key = "mlp_head.0.weight" # Usually the first Linear in the Sequential MLP
        hybrid_attn_qkv_key = "attention.in_proj_weight" # Optional attention in Hybrid

        if pred_initial_proj_key in sd: inferred_features = sd[pred_initial_proj_key].shape[1]
        elif hybrid_initial_proj_key in sd: inferred_features = sd[hybrid_initial_proj_key].shape[1]
        elif hybrid_attn_qkv_key in sd: inferred_features = sd[hybrid_attn_qkv_key].shape[1] # Input to attention QKV
        # Add more checks if needed for other first layers

        if inferred_features is None: print(f"Warning: Cannot infer input features from known keys in {model_path}.")
        elif expected_features is not None and inferred_features != expected_features:
             print(f"Warning: Input feature mismatch! Expected {expected_features}, Inferred {inferred_features} in {model_path}.")
             # Decide whether to raise error or proceed? Let's proceed but warn.

        # --- Infer/Verify output dimension ---
        # Check possible final layer keys
        pred_final_bias = "final_layer.bias" # PredictorModel
        hybrid_final_bias = "mlp_head.7.bias" # Example if Hybrid MLP has 8 layers (0-7), last Linear bias
        hybrid_final_weight = "mlp_head.7.weight" # Example weight key

        # Find the bias key first, as it directly gives num_classes
        potential_bias_keys = [k for k in sd if k.endswith(".bias")]
        if potential_bias_keys:
             # Assume the numerically highest indexed layer in mlp_head is the final one, or use specific name
             final_bias_key = None
             if pred_final_bias in sd: final_bias_key = pred_final_bias
             else: # Try finding highest index in mlp_head
                  max_idx = -1; target_key = None
                  for k in potential_bias_keys:
                       if k.startswith("mlp_head."):
                            parts = k.split('.')
                            if len(parts) == 3 and parts[1].isdigit():
                                 idx = int(parts[1])
                                 if idx > max_idx: max_idx = idx; target_key = k
                  if target_key: final_bias_key = target_key

             if final_bias_key: actual_outputs = sd[final_bias_key].shape[0]
             else: print(f"Warning: Could not reliably identify final bias key in {model_path}.")

        # Fallback: Check weight keys if bias not found/identified
        if actual_outputs is None:
             potential_weight_keys = [k for k in sd if k.endswith(".weight")]
             if potential_weight_keys:
                  # Similar logic to find final weight key...
                  # ... (implementation depends on naming convention) ...
                  # For now, let's assume bias is sufficient for output inference.
                  pass

        if actual_outputs is None: print(f"Warning: Could not infer output size from state dict {model_path}.")
        elif expected_outputs is not None and actual_outputs != expected_outputs:
            print(f"Warning: Output size mismatch! Expected {expected_outputs}, found {actual_outputs} in {model_path}.")

        return sd, inferred_features, actual_outputs

    except Exception as e: print(f"Error loading state dict helper {model_path}: {e}"); raise


# ================================================
#        Base Pipeline Class
# ================================================
class BasePipeline:
    # v1.2.0: Uses _load_model_head, more robust vision init
    def __init__(self, model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        self.device = device
        self.clip_dtype = clip_dtype
        self.model_path = model_path

        # --- Config Loading ---
        if config_path is None or not os.path.isfile(config_path):
             config_path = self._infer_config_path(model_path)
             if config_path is None: raise FileNotFoundError(f"Could not find/infer config for {model_path}")
        self.config_path = config_path
        self.config = _load_config_helper(self.config_path)
        if self.config is None: raise ValueError("Failed to load configuration.")
        # print(f"DEBUG BasePipeline: Config keys: {list(self.config.keys())}") # Optional

        # --- Get Base Vision Model Info from Config ---
        # These are now assumed to be top-level keys saved by write_config
        self.base_vision_model_name = self.config.get("base_vision_model")
        self.embed_ver = self.config.get("embed_ver") # Can be None if E2E mode
        if not self.base_vision_model_name: raise ValueError("Missing 'base_vision_model' in config.")
        print(f"DEBUG BasePipeline: Base Vision='{self.base_vision_model_name}', EmbedVer='{self.embed_ver}'")

        # --- Initialize vision model attributes ---
        self.vision_model_type = "unknown"; self.vision_model = None; self.hf_processor = None;
        self.timm_transforms = None; self.proc_size = 512
        self._init_vision_model() # Initialize the vision backbone

        # --- Preprocessing function selection (Based on embed_ver or model type) ---
        self.preprocess_func = None
        # <<< Use more robust checking based on model name/embed_ver >>>
        if "dinov3" in self.base_vision_model_name.lower():
            print("DEBUG BasePipeline: Selecting DINOv3 variable resolution logic.")
            self.preprocess_func = None # Signal internal handling
        elif "naflex" in self.base_vision_model_name.lower() or ("Naflex" in (self.embed_ver or "")):
            print("DEBUG BasePipeline: Selecting NaFlex processor logic (via get_clip_emb).")
            # NaFlex uses the processor directly, no manual func needed here
            self.preprocess_func = None # Signal direct processor use
        elif "dinov2" in self.base_vision_model_name.lower() or ("DINOv2" in (self.embed_ver or "")):
             print("DEBUG BasePipeline: Selecting FitPad preprocessing (likely DINOv2).")
             self.preprocess_func = preprocess_fit_pad # Default FitPad for DINO
        elif "aimv2" in self.base_vision_model_name.lower():
             print("DEBUG BasePipeline: Selecting AIMv2 Native logic (controlled resize + processor).")
             # Logic handled inside get_clip_emb, no single preprocess func here
             self.preprocess_func = None # Signal internal handling
        elif "siglip" in self.base_vision_model_name.lower(): # Non-Naflex Siglip
             print("DEBUG BasePipeline: Selecting FitPad preprocessing (likely SigLIP).")
             self.preprocess_func = preprocess_fit_pad # Default FitPad
        else: # Fallback
            print(f"DEBUG BasePipeline: Unknown model type '{self.base_vision_model_name}'. Defaulting to FitPad preprocessing.")
            self.preprocess_func = preprocess_fit_pad

        # --- Model Head Setup (Loaded in subclasses using _load_model_head) ---
        self.model = None # This will be the loaded head (PredictorModel or HybridHeadModel)
        self.num_labels = 0
        self.labels = {} # Labels specific to the loaded head

    def _infer_config_path(self, model_path: str) -> str | None:
        """Tries to infer the base config path from a checkpoint path."""
        model_base_name = os.path.basename(model_path)
        # Remove .safetensors suffix first
        if model_base_name.endswith(".safetensors"):
             model_base_name = model_base_name[:-len(".safetensors")]

        # Remove common suffixes iteratively
        suffixes_to_remove = ["_best_val", "_efinal"]
        for suffix in suffixes_to_remove:
             if model_base_name.endswith(suffix):
                  model_base_name = model_base_name[:-len(suffix)]
                  break # Stop after removing one suffix type

        # Remove step suffixes like _s28K, _s1M etc.
        if "_s" in model_base_name:
             parts = model_base_name.split("_s")
             # Check if the part after _s looks like a step count (e.g., digits + K/M)
             if len(parts) > 1 and parts[-1] and (parts[-1][0].isdigit() or parts[-1][-1] in ['K', 'M']):
                  model_base_name = parts[0]

        inferred_path = os.path.join(os.path.dirname(model_path), f"{model_base_name}.config.json")
        if os.path.isfile(inferred_path):
            print(f"DEBUG: Inferred config path: {inferred_path}")
            return inferred_path
        else:
            print(f"DEBUG: Could not infer config path from {model_path}, tried {inferred_path}")
            return None

    def _init_vision_model(self):
        """Initializes the vision model and processor/transforms based on config."""
        model_name = self.base_vision_model_name
        print(f"Initializing Vision Model: {model_name} on {self.device} with dtype {self.clip_dtype}")

        # --- Check if TIMM Model ---
        if model_name.startswith("timm/") and TIMM_AVAILABLE:
            self.vision_model_type = "timm"
            print("  Detected TIMM model type.")
            try:
                timm_model_name = model_name.split('/', 1)[1] # Get actual model name for timm
                # Load TIMM model without classifier head
                self.vision_model = timm.create_model(
                    timm_model_name,
                    pretrained=True,
                    num_classes=0 # <<< Crucial: Get pooled features
                ).to(self.device).eval()

                # Apply clip_dtype AFTER loading pretrained weights (usually FP32)
                self.vision_model = self.vision_model.to(dtype=self.clip_dtype)

                # Get TIMM transforms
                data_config = timm.data.resolve_model_data_config(self.vision_model)
                self.timm_transforms = timm.data.create_transform(**data_config, is_training=False)
                print(f"  Loaded TIMM model: {timm_model_name}")
                print(f"  Loaded TIMM transforms: {self.timm_transforms}")

                # Store input size if needed (less critical for TIMM as transforms handle it)
                timm_input_size = data_config.get('input_size')
                if timm_input_size: self.proc_size = timm_input_size[-1]

            except Exception as e:
                print(f"Error initializing TIMM vision model {timm_model_name}: {e}")
                raise

        # --- Hugging Face Model Path ---
        else:
            if model_name.startswith("timm/"): print("Warning: TIMM model specified but 'timm' library not available.")
            self.vision_model_type = "hf"
            print("  Detected Hugging Face model type.")
            try:
                self.hf_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                
                # <<< MODIFIED: DINOv3 8-bit BnB Model Loading >>>
                if model_name == "dinov3-vit7b16-pretrain-lvd1689m-8bit":
                    print(f"  Loading DINOv3 8-bit BnB model from saved path: {model_name}")
                    self.vision_model, self.hf_processor = dinov3_7b_quant_bnb.load_from_saved_bnb(save_path=f"./{model_name}")
                    # For 8-bit models, device_map="auto" already handles device placement.
                    # Do NOT call .to(device) again, as it's not supported and causes an error.
                    self.vision_model.eval() # Ensure model is in eval mode
                    print("  INFO: Skipping .to(device) for 8-bit DINOv3 model.")
                else:
                    self.hf_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                    self.vision_model = AutoModel.from_pretrained(
                        model_name, torch_dtype=self.clip_dtype, trust_remote_code=True
                    ).to(self.device).eval()
                # <<< END MODIFIED >>>

                print(f"  Loaded HF model: {self.vision_model.__class__.__name__}")
                print(f"  Loaded HF processor: {self.hf_processor.__class__.__name__}")

                # Disable processor's built-in transforms (remains same)
                if hasattr(self.hf_processor, 'image_processor'):
                     image_processor = self.hf_processor.image_processor
                     # This check might be the one failing to disable for BitImageProcessor
                     print(f"  DEBUG Inference: Original processor config: {image_processor}")
                     if hasattr(image_processor, 'do_center_crop'): image_processor.do_center_crop = False
                     else: print("  Warning: Cannot access 'do_center_crop' on image_processor.")
                     if hasattr(image_processor, 'do_normalize'): image_processor.do_normalize = True # Keep normalization ON
                     else: print("  Warning: Cannot access 'do_normalize' on image_processor.")
                     print(f"  DEBUG Inference: Attempted modified processor config: {image_processor}")
                else:
                     print("  Warning: Cannot access image_processor to disable auto-preprocessing.")


                # --- Determine Processor Size (Attempt automatic first) ---
                self.proc_size = 512 # Start with default
                if hasattr(self.hf_processor, 'image_processor') and hasattr(self.hf_processor.image_processor, 'size'):
                    proc_sz = self.hf_processor.image_processor.size
                    if isinstance(proc_sz, dict): self.proc_size = int(proc_sz.get("height", 512))
                    elif isinstance(proc_sz, int): self.proc_size = int(proc_sz)
                # Also check model config directly for size (like in generate_embeddings)
                model_config = getattr(self.vision_model, 'config', None)
                if model_config and hasattr(model_config, 'image_size'):
                     config_size = int(model_config.image_size)
                     if config_size != self.proc_size:
                          print(f"  DEBUG: Overriding proc size ({self.proc_size}) with model config size ({config_size}).")
                          self.proc_size = config_size

                # <<< ADD EXPLICIT OVERRIDE FOR DINOv2 GIANT >>>
                if "dinov2" in model_name.lower() and "giant" in model_name.lower():
                    if self.proc_size != 518:
                        print(f"  INFO: Explicitly setting processor target size to 518 for DINOv2 Giant (was {self.proc_size}).")
                        self.proc_size = 518
                # <<< END OVERRIDE >>>

                print(f"  Determined processor target size (for FitPad/CenterCrop): {self.proc_size}") # This log should now show 518

            except Exception as e:
                print(f"Error initializing Hugging Face vision model {model_name}: {e}")
                raise


    # --- _preprocess_images (Simplified - Only applies manual funcs if set) ---
    def _preprocess_images(self, img_list: list[Image.Image]) -> list[Image.Image] | None:
        """Applies manual preprocessing function IF one is selected."""
        if self.preprocess_func is None:
            # print("DEBUG _preprocess_images: No manual func set, returning raw images.")
            return img_list # Return original list if no manual func needed

        processed_imgs = []
        for img_pil in img_list:
            try: # Apply the function (e.g., preprocess_fit_pad)
                target_s = self.proc_size # Use determined size for manual funcs
                if "dinov2" in self.base_vision_model_name.lower() and "giant" in self.base_vision_model_name.lower(): target_s = 518
                # Call the function stored in self.preprocess_func
                img_processed = self.preprocess_func(img_pil, target_size=target_s)
                if img_processed: processed_imgs.append(img_processed)
                else: print("Warning: Manual preprocessing returned None.")
            except Exception as e_prep: print(f"Error manual preprocess: {e_prep}"); continue
        return processed_imgs if processed_imgs else None

    # v1.9.0: Includes AIMv2 CLS logic with pre-resizing
    def get_clip_emb(self, img_list: list[Image.Image]) -> torch.Tensor | None:
        """
        Generates embeddings for a list of PIL images using the loaded vision model.
        Correctly handles TIMM-native, FB DINOv2 (Manual), SigLIP NaFlex, SigLIP Manual.
        Input img_list expected to be RAW PIL images for NaFlex/TIMM-native.
        Input img_list expected to be PREPROCESSED PIL images for Manual modes if called after _preprocess_images.
        """
        if not isinstance(img_list, list): img_list = [img_list]
        if not img_list: print("Error: Empty image list provided."); return None

        final_emb = None
        do_l2_normalize = False

        try:
            # --- Path 1: TIMM Model using TIMM Transforms ---
            if self.vision_model_type == "timm":
                if self.vision_model is None or self.timm_transforms is None: raise ValueError("TIMM model or transforms not initialized.")
                # Expects RAW PIL images in img_list here
                processed_tensors = [self.timm_transforms(img) for img in img_list]
                input_batch = torch.stack(processed_tensors).to(device=self.device, dtype=self.clip_dtype)

                with torch.no_grad():
                    emb = self.vision_model(input_batch) # Pooled features
                # Check if base model name indicates DINOv2 for normalization
                do_l2_normalize = "dinov2" in self.base_vision_model_name.lower()

            # --- Path 2: Hugging Face Models ---
            elif self.vision_model_type == "hf":
                processor = self.hf_processor; model = self.vision_model
                if model is None or processor is None: raise ValueError("HF model/processor missing.")
                is_naflex_mode = "Naflex" in (self.embed_ver or "") or "naflex" in self.base_vision_model_name.lower()
                is_aimv2_model = "aimv2" in self.base_vision_model_name.lower()
                is_siglip_model = "siglip" in self.base_vision_model_name.lower() and not is_aimv2_model # Avoid conflict
                is_dinov2_model = "dinov2" in self.base_vision_model_name.lower() and not is_aimv2_model
                # <<< MODIFIED: Check for DINOv3ViTModel name >>>
                is_dinov3_model = "dinov3" in self.base_vision_model_name.lower() and not is_aimv2_model # Avoid conflict

                # --- SubPath 2a: SigLIP NaFlex ---
                if is_siglip_model and is_naflex_mode:
                    # Expects RAW PIL images in img_list
                    inputs = self.hf_processor(images=img_list, return_tensors="pt", max_num_patches=2048)
                    pixel_values = inputs.get("pixel_values"); attention_mask = inputs.get("pixel_attention_mask"); spatial_shapes = inputs.get("spatial_shapes")
                    if pixel_values is None or attention_mask is None or spatial_shapes is None: raise ValueError("Missing tensors from HF NaFlex processor.")

                    model_call_kwargs = {
                        "pixel_values": pixel_values.to(device=self.device, dtype=self.clip_dtype),
                        "attention_mask": attention_mask.to(device=self.device),
                        "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.long).to(device=self.device)
                    }
                    # Call SigLIP model
                    with torch.no_grad():
                        vision_model_component = getattr(self.vision_model, 'vision_model', None)
                        if vision_model_component: emb = vision_model_component(**model_call_kwargs).pooler_output
                        elif hasattr(self.vision_model, 'get_image_features'):
                            kwargs_for_get = {k: v for k, v in model_call_kwargs.items() if k in ['pixel_values', 'attention_mask', 'spatial_shapes']}
                            emb = self.vision_model.get_image_features(**kwargs_for_get)
                        else: raise AttributeError("SigLIP Model missing expected methods.")
                    do_l2_normalize = False # SigLIP internal norm

                # --- SubPath 2b: AIMv2 Native CLS ---
                elif is_aimv2_model: # Assuming CLS mode is intended for inference
                     # Expects RAW PIL images, performs controlled pre-resize THEN processor
                     processed_tensors = []
                     for raw_img_pil in img_list:
                          img_to_process = raw_img_pil
                          # <<< Apply pre-resizing logic >>>
                          try: # Identical logic to generate_embeddings v4.4.0
                               original_width, original_height = raw_img_pil.size
                               if original_width > 0 and original_height > 0:
                                    patches_w = math.floor(original_width / AIMV2_PATCH_SIZE); patches_h = math.floor(original_height / AIMV2_PATCH_SIZE)
                                    if patches_w * patches_h > AIMV2_TARGET_MAX_PATCHES:
                                         # (iterative resizing logic to fit AIMV2_TARGET_MAX_PATCHES)
                                         scale_factor = math.sqrt(AIMV2_TARGET_MAX_PATCHES / (patches_w * patches_h))
                                         # find target_w, target_h that are multiples of 14
                                         target_w = int(original_width * scale_factor + 0.5)
                                         target_h = int(original_height * scale_factor + 0.5)
                                         if target_w < 1: target_w = 1
                                         if target_h < 1: target_h = 1
                                         target_w = math.floor(target_w / AIMV2_PATCH_SIZE) * AIMV2_PATCH_SIZE
                                         target_h = math.floor(target_h / AIMV2_PATCH_SIZE) * AIMV2_PATCH_SIZE
                                         if target_w == 0: target_w = AIMV2_PATCH_SIZE
                                         if target_h == 0: target_h = AIMV2_PATCH_SIZE
                                         img_to_process = raw_img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                                    else: # Ensure dims are multiple of patch size even if not resizing
                                         current_w, current_h = img_to_process.size
                                         new_w = math.floor(current_w / AIMV2_PATCH_SIZE) * AIMV2_PATCH_SIZE
                                         new_h = math.floor(current_h / AIMV2_PATCH_SIZE) * AIMV2_PATCH_SIZE
                                         if new_w == 0: new_w = AIMV2_PATCH_SIZE
                                         if new_h == 0: new_h = AIMV2_PATCH_SIZE
                                         if new_w != current_w or new_h != current_h:
                                              img_to_process = img_to_process.resize((new_w, new_h), Image.Resampling.LANCZOS)
                          except Exception as e_resize: print(f"Resize error AIMv2 Inf: {e_resize}"); continue
                          # <<< End pre-resizing >>>
                          # Now process the (maybe resized) image
                          inputs = processor(images=[img_to_process], return_tensors="pt")
                          processed_tensors.append(inputs.pixel_values)

                     if not processed_tensors: raise ValueError("AIMv2 processing failed for all images.")
                     pixel_values = torch.cat(processed_tensors, dim=0).to(device=self.device, dtype=self.clip_dtype)
                     model_call_kwargs = {"pixel_values": pixel_values}
                     with torch.no_grad(): outputs = model(**model_call_kwargs)
                     last_hidden_state = getattr(outputs, 'last_hidden_state', None)
                     if last_hidden_state is None: raise ValueError("AIMv2 missing LHS.")
                     emb = last_hidden_state[:, 0, :] # CLS token
                     do_l2_normalize = True

                # <<< ADDED: DINOv3 7B 8-bit BnB Mode (Variable Resolution + Mean Pooling) >>>
                elif is_dinov3_model:
                    processed_tensors = []
                    for raw_img_pil in img_list:
                        current_w, current_h = raw_img_pil.size
                        img_to_process = raw_img_pil

                        # Optional: Add max resolution limit for memory protection
                        if max(current_w, current_h) > MAX_DINOV3_RESOLUTION:
                            scale = MAX_DINOV3_RESOLUTION / max(current_w, current_h)
                            current_w = int(current_w * scale)
                            current_h = int(current_h * scale)
                            img_to_process = raw_img_pil.resize((current_w, current_h), Image.Resampling.LANCZOS)
                            print(f"  - INFO: Scaling down image to fit within {MAX_DINOV3_RESOLUTION}px limit.")

                        # Validate reasonable image sizes after potential downscaling
                        if current_w < DINOV3_PATCH_SIZE or current_h < DINOV3_PATCH_SIZE:
                            raise ValueError(f"Image too small: {current_w}x{current_h}. Minimum size is {DINOV3_PATCH_SIZE}x{DINOV3_PATCH_SIZE} pixels.")
                        if current_w > 4096 or current_h > 4096: # Max observed, but warn if exceeding common limits
                            print(f"  - WARNING: Very large image size {current_w}x{current_h} may cause memory issues.")

                        # Ensure image dimensions are multiples of 16 (DINOV3_PATCH_SIZE)
                        # This logic rounds up to the nearest multiple of 16, as praised in feedback.
                        new_w = ((current_w + DINOV3_PATCH_SIZE - 1) // DINOV3_PATCH_SIZE) * DINOV3_PATCH_SIZE
                        new_h = ((current_h + DINOV3_PATCH_SIZE - 1) // DINOV3_PATCH_SIZE) * DINOV3_PATCH_SIZE

                        if new_w != current_w or new_h != current_h:
                            print(f"  - INFO: Adjusting dims ({current_w}x{current_h}) -> ({new_w}x{new_h}) to be multiples of {DINOV3_PATCH_SIZE} for DINOv3.")
                            img_to_process = img_to_process.resize((new_w, new_h), Image.Resampling.LANCZOS)

                        # Use processor for ToTensor/Normalize
                        inputs = processor(images=[img_to_process], return_tensors="pt")
                        processed_tensors.append(inputs.pixel_values)

                    if not processed_tensors: raise ValueError("DINOv3 processing failed for all images.")
                    pixel_values = torch.cat(processed_tensors, dim=0).to(device=self.device, dtype=self.clip_dtype)
                    model_call_kwargs = {"pixel_values": pixel_values}
                    # Call DINOv3 model
                    with torch.no_grad(): outputs = model(**model_call_kwargs)
                    last_hidden_state = getattr(outputs, 'last_hidden_state', None)
                    if last_hidden_state is None: raise ValueError("DINOv3 model did not return last_hidden_state.")
                    # Use mean pooling of patch tokens (exclude CLS and register tokens)
                    # DINOv3-7B has 4 register tokens. We skip CLS (pos 0) and registers (pos 1-4).
                    nreg = getattr(model.config, 'num_register_tokens', 0)
                    patch_embeddings = last_hidden_state[:, 1 + nreg:] # Skip CLS and Registers
                    emb = torch.mean(patch_embeddings, dim=1)  # [B, embed_dim]
                    do_l2_normalize = True # DINOv3 typically requires L2 normalization

                # --- SubPath 2c: Manual Preprocessing (FB DINOv2 or SigLIP FitPad/CenterCrop) ---
                elif not is_naflex_mode:
                    # Expects PREPROCESSED PIL images in img_list (processed by _preprocess_images)
                    # Use processor ONLY for ToTensor + Normalize
                    inputs = self.hf_processor(images=img_list, return_tensors="pt")
                    pixel_values = inputs.get("pixel_values")
                    if pixel_values is None: raise ValueError("HF Processor didn't return 'pixel_values'.")
                    model_call_kwargs = {"pixel_values": pixel_values.to(device=self.device, dtype=self.clip_dtype)}
                    attention_mask = inputs.get("pixel_attention_mask")
                    if attention_mask is not None: model_call_kwargs["attention_mask"] = attention_mask.to(device=self.device)

                    # Call appropriate model
                    with torch.no_grad():
                        if is_dinov2_model:
                            outputs = self.vision_model(**model_call_kwargs)
                            emb = outputs.last_hidden_state[:, 0] # CLS token
                        elif is_siglip_model:
                             vision_model_component = getattr(self.vision_model, 'vision_model', None)
                             if vision_model_component: emb = vision_model_component(**model_call_kwargs).pooler_output
                             elif hasattr(self.vision_model, 'get_image_features'):
                                  kwargs_for_get = {k: v for k, v in model_call_kwargs.items() if k in ['pixel_values', 'attention_mask', 'spatial_shapes']}
                                  emb = self.vision_model.get_image_features(**kwargs_for_get)
                             else: raise AttributeError("SigLIP Model missing expected methods.")
                        else: # Should not happen if model loaded correctly
                            raise TypeError(f"Model type mismatch in HF manual preproc path: {self.vision_model.__class__.__name__}")

                    # Set normalization flag based on model type
                    do_l2_normalize = is_dinov2_model

                else: # Should not happen: is_naflex_mode is True but model isn't SigLIP?
                    raise ValueError(f"Unsupported NaFlex mode for model type: {self.vision_model.__class__.__name__}")

            else: # Unknown vision_model_type
                raise ValueError(f"Unknown vision_model_type '{self.vision_model_type}'")

            # --- Check if emb was obtained ---
            if emb is None: raise ValueError("Failed to get embedding from model call.")

            # --- Final Normalization & Conversion ---
            if do_l2_normalize:
                norm = torch.linalg.norm(emb.float(), dim=-1, keepdim=True).clamp(min=1e-8)
                emb = emb / norm.to(emb.dtype)
            final_emb = emb.detach().to(device='cpu', dtype=torch.float32)

        except Exception as e:
            print(f"Error during get_clip_emb (v1.8.6): {e}")
            import traceback; traceback.print_exc(); return None

        return final_emb

    # --- v1.8.6: Adjusted comments to reflect where preprocessing happens ---
    def get_clip_emb_tiled(self, raw_pil_image: Image.Image, tiling: bool = False) -> torch.Tensor | None:
        """Generates embeddings, potentially using 5-crop tiling."""
        img_list = []
        is_naflex_mode = "Naflex" in self.embed_ver

        # --- Prepare img_list (PIL images) ---
        if is_naflex_mode:
             # NaFlex always uses single view RAW image
             if tiling: print("DEBUG: Tiling requested with NaFlex, processing single view instead.")
             img_list = [raw_pil_image]
        elif tiling:
             # Manual Modes (FitPad/CenterCrop) Tiling:
             # Apply manual preprocessing FIRST to get base image for cropping
             base_processed_img_list = self._preprocess_images([raw_pil_image]) # Uses correct size (e.g., 518)
             if not base_processed_img_list: print("Error: Preprocessing failed for tiling base."); return None
             base_processed_img = base_processed_img_list[0]

             # Determine crop size (should match final input size)
             crop_size = [self.proc_size, self.proc_size] # self.proc_size should be correct now (e.g., 518)

             # Perform Tiling on the *Preprocessed* Base Image
             base_processed_img_tensor = TF.to_tensor(base_processed_img)
             try:
                  if base_processed_img_tensor.shape[1] < crop_size[0] or base_processed_img_tensor.shape[2] < crop_size[1]:
                        print(f"Warning: Preprocessed image ({base_processed_img_tensor.shape}) smaller than crop size ({crop_size}). Using single view.")
                        img_list = [base_processed_img] # Pass the single PREPROCESSED image
                  else:
                        tiled_crops_tensors = TF.five_crop(base_processed_img_tensor, size=crop_size)
                        # Pass PREPROCESSED crops (as PIL) to get_clip_emb
                        # get_clip_emb for manual modes expects preprocessed PIL
                        img_list = [TF.to_pil_image(crop) for crop in tiled_crops_tensors]
             except Exception as e_tile:
                  print(f"Error during five_crop tiling: {e_tile}. Falling back to single view.")
                  img_list = [base_processed_img] # Fallback to single PREPROCESSED image
        else:
             # Manual Modes (FitPad/CenterCrop) Non-Tiled:
             # Apply manual preprocessing first.
             processed_single_view = self._preprocess_images([raw_pil_image])
             if not processed_single_view: print("Error: Preprocessing failed for single view."); return None
             img_list = processed_single_view # List containing one PREPROCESSED PIL image

        if not all(isinstance(img, Image.Image) for img in img_list): print("Error: img_list invalid."); return None

        # Call get_clip_emb. It expects RAW images for NaFlex/TIMM-native,
        # and PREPROCESSED images for manual modes (FitPad/CenterCrop).
        return self.get_clip_emb(img_list)

    # v1.9.4: More robust output inference - handle nested keys better
    def _infer_outputs_from_state_dict(self, state_dict, model_path):
        """Infers the number of output classes from a state dictionary."""
        print(f"DEBUG: Inferring outputs for {os.path.basename(model_path)}...")

        potential_final_layer_key = None

        # --- Strategy 1: Look for explicit 'final_layer' name ---
        for key in state_dict:
            if key.endswith("final_layer.bias"): potential_final_layer_key = key; break
            if key.endswith("final_layer.weight") and not potential_final_layer_key: potential_final_layer_key = key;
        if potential_final_layer_key:
             print(f"  Found explicit 'final_layer' key: {potential_final_layer_key}")

        # --- Strategy 2: Find highest indexed layer within 'head.' or 'mlp_head.' ---
        if not potential_final_layer_key:
            prefixes_to_check = ["mlp_head.", "head."]
            found_prefix = ""
            max_layer_idx = -1
            potential_target_base = None

            for prefix in prefixes_to_check:
                for key in state_dict:
                     if key.startswith(prefix):
                          found_prefix = prefix # Found keys with this prefix
                          parts = key[len(prefix):].split('.') # Get parts after prefix
                          if len(parts) > 0 and parts[0].isdigit():
                               try:
                                    idx = int(parts[0])
                                    if idx > max_layer_idx:
                                         max_layer_idx = idx
                                         potential_target_base = f"{prefix}{idx}" # Store base like "head.11"
                               except ValueError: continue
                if potential_target_base: break # Stop if found in first prefix

            if potential_target_base:
                # Now look for .weight or .bias associated with this base index
                bias_key = f"{potential_target_base}.bias"
                weight_key = f"{potential_target_base}.weight"
                if bias_key in state_dict:
                    potential_final_layer_key = bias_key
                    print(f"  Found layer by max index '{potential_target_base}' using bias key: {bias_key}")
                elif weight_key in state_dict:
                    potential_final_layer_key = weight_key
                    print(f"  Found layer by max index '{potential_target_base}' using weight key: {weight_key}")
                else:
                     print(f"  Found max index '{potential_target_base}' but no corresponding weight/bias key.")
            else:
                 print("  Could not find indexed layers starting with 'mlp_head.' or 'head.'.")

        # --- Strategy 3: Fallback (e.g., smallest output dim - optional/less reliable) ---
        # Add only if needed, might still be unreliable.
        # if not potential_final_layer_key:
        #     print("  Fallback: Searching for layer with smallest output dimension...")
        #     min_out_dim = float('inf')
        #     target_base = None
        #     # ... (logic to find smallest dim weight/bias not belonging to norm) ...
        #     if target_base: potential_final_layer_key = f"{target_base}.bias" # or .weight

        # --- Determine Output Size ---
        if not potential_final_layer_key:
            raise ValueError(f"Could not identify final classification layer key in {model_path}.")

        if potential_final_layer_key.endswith(".bias"):
            if potential_final_layer_key in state_dict:
                 inferred_outputs = state_dict[potential_final_layer_key].shape[0]
                 print(f"  Inferred outputs from bias key '{potential_final_layer_key}': {inferred_outputs}")
                 return inferred_outputs
            else: raise ValueError(f"Logic error: Bias key '{potential_final_layer_key}' not found.")
        elif potential_final_layer_key.endswith(".weight"):
             if potential_final_layer_key in state_dict:
                  inferred_outputs = state_dict[potential_final_layer_key].shape[0]
                  print(f"  Inferred outputs from weight key '{potential_final_layer_key}': {inferred_outputs}")
                  return inferred_outputs
             else: raise ValueError(f"Logic error: Weight key '{potential_final_layer_key}' not found.")
        else:
            raise ValueError(f"Identified key '{potential_final_layer_key}' is not a bias or weight key.")

    # --- <<< NEW: Unified Head Loading Function >>> ---
    # v1.0.0: Loads PredictorModel or HybridHeadModel
    def _load_model_head(self):
        """Loads the appropriate head model (PredictorModel or HybridHeadModel) based on config/state_dict."""
        print(f"DEBUG _load_model_head: Loading head from {os.path.basename(self.model_path)}...")
        if not self.config: raise ValueError("Pipeline config not loaded.")

        # --- Load State Dict and Config ---
        # Determine expected features from base vision model or config
        expected_features = self.config.get("features") # Check if saved directly
        if not expected_features and self.vision_model and hasattr(self.vision_model, 'config'):
             expected_features = getattr(self.vision_model.config, 'hidden_size', None)
        if not expected_features and self.embed_ver: # Fallback to embed_ver lookup
             try: expected_features = get_embed_params(self.embed_ver)["features"]
             except: pass
        if not expected_features: raise ValueError("Cannot determine input features for head.")

        # Load state dict and try to infer outputs
        sd, inferred_features, actual_outputs = _load_state_dict_helper(self.model_path, expected_features)
        if actual_outputs is None: raise ValueError("Could not infer model output size from state dict.")
        self.num_labels = actual_outputs # Use inferred outputs as truth

        # --- Determine Model Type ---
        # Check for keys unique to HybridHeadModel (e.g., from RMSNorm/SwiGLU in MLP)
        # Example: "mlp_head.1.weight" might be RMSNorm weight, "mlp_head.2.ffn.w12.weight" might be SwiGLU
        is_hybrid = any(k.startswith("mlp_head.") and ("norm.weight" in k or "ffn.w" in k) for k in sd)
        # Crude check, might need refinement based on exact layer names in HybridHeadModel

        model = None
        model_type_str = "HybridHeadModel" if is_hybrid else "PredictorModel"
        print(f"DEBUG: Detected head type: {model_type_str}")

        # --- Get Hyperparameters from Config ---
        # Config saved by train.py should have top-level args
        # Prioritize specific args if available, fallback to defaults
        output_mode = self.config.get("output_mode", self.config.get("head_output_mode", "linear")) # Get output mode
        hidden_dim = self.config.get("hidden_dim", self.config.get("head_hidden_dim", 1280))
        num_res_blocks = self.config.get("num_res_blocks", self.config.get("head_num_res_blocks", 1))
        dropout_rate = self.config.get("dropout_rate", self.config.get("head_dropout_rate", 0.1))
        # Attention params (usually only needed if PredictorModel or Hybrid with use_attention=True)
        use_attention = self.config.get("use_attention", True if not is_hybrid else False) # Default ON for Predictor, OFF for Hybrid unless specified
        num_attn_heads = self.config.get("num_attn_heads", 16)
        attn_dropout = self.config.get("attn_dropout", 0.1)
        # Hybrid specific
        rms_norm_eps = self.config.get("rms_norm_eps", 1e-6)


        # --- Instantiate Correct Model ---
        try:
            if is_hybrid:
                model = HybridHeadModel(
                    features=inferred_features or expected_features, # Prioritize inferred
                    hidden_dim=hidden_dim,
                    num_classes=self.num_labels, # Use inferred from state dict
                    use_attention=use_attention, # Allow enabling attention
                    num_attn_heads=num_attn_heads,
                    attn_dropout=attn_dropout,
                    num_res_blocks=num_res_blocks,
                    dropout_rate=dropout_rate,
                    rms_norm_eps=rms_norm_eps,
                    output_mode=output_mode
                )
            else: # Assume PredictorModel
                model = PredictorModel(
                    features=inferred_features or expected_features,
                    hidden_dim=hidden_dim,
                    num_classes=self.num_labels,
                    use_attention=use_attention,
                    num_attn_heads=num_attn_heads,
                    attn_dropout=attn_dropout,
                    num_res_blocks=num_res_blocks,
                    dropout_rate=dropout_rate,
                    output_mode=output_mode
                )
        except Exception as e: raise TypeError(f"Failed to instantiate {model_type_str}: {e}") from e

        # --- Load State Dict ---
        model.eval()
        try:
            missing, unexpected = model.load_state_dict(sd, strict=True) # Try strict first
            if missing or unexpected: print(f" Load warnings: Missing={missing}, Unexpected={unexpected}")
        except RuntimeError as e:
            print(f"Warning: Strict load failed for {model_type_str}. Trying non-strict. Error: {e}")
            missing, unexpected = model.load_state_dict(sd, strict=False) # Fallback non-strict
            if missing or unexpected: print(f" Load warnings (non-strict): Missing={missing}, Unexpected={unexpected}")


        model.to(self.device)
        print(f"Successfully loaded head '{os.path.basename(self.model_path)}' ({model_type_str}, Outputs: {self.num_labels}, Mode: '{output_mode}').")
        self.model = model # Assign to instance variable

        # --- Load Labels ---
        # Use labels saved in the config file
        self.labels = self.config.get("labels", {})
        if not self.labels: # Fallback to default numeric labels
            self.labels = {str(i): str(i) for i in range(self.num_labels)}
            print(f"DEBUG: Using default numeric labels (0 to {self.num_labels-1})")
        elif len(self.labels) != self.num_labels:
             print(f"Warning: Config labels count ({len(self.labels)}) != model outputs ({self.num_labels}).")


    def get_model_pred_generic(self, emb: torch.Tensor) -> torch.Tensor:
        """Runs prediction using the loaded predictor head."""
        if self.model is None: raise ValueError("Predictor head model not loaded.")
        if emb is None: raise ValueError("Input embedding is None.")
        # Model head expects FP32 on its device
        with torch.no_grad():
             # Move embedding to the correct device right before prediction
             pred = self.model(emb.to(self.device, dtype=torch.float32))
        # Return predictions on CPU
        return pred.detach().cpu()

    # v1.9.4: Corrected structure for loading HeadModel, using inferred outputs robustly
    def _load_head_model(self, head_model_path: str):
        """Loads the HeadModel state dict using config."""
        print(f"DEBUG _load_head_model: Loading head from {os.path.basename(head_model_path)}...")
        if not self.config: raise ValueError("Pipeline configuration not loaded.")
        if not os.path.isfile(head_model_path): raise FileNotFoundError(f"HeadModel file not found: {head_model_path}")

        # --- 1. Load State Dict Once ---
        sd = None
        try:
            sd = load_file(head_model_path)
        except Exception as e_load:
            raise ValueError(f"Error loading state dict from {head_model_path}: {e_load}")

        # --- 2. Determine Input Features ---
        expected_features = self.config.get("head_features")
        if expected_features is None:
            if self.vision_model and hasattr(self.vision_model, 'config') and hasattr(self.vision_model.config, 'hidden_size'):
                expected_features = self.vision_model.config.hidden_size
                print(f"DEBUG: Inferred head input features from vision model config: {expected_features}")
            else:
                try: expected_features = get_embed_params(self.embed_ver)["features"]
                except Exception as e: raise ValueError(f"Could not determine input features: {e}")

        # --- 3. Determine Number of Classes (Config vs State Dict) ---
        num_classes_config = self.config.get("num_classes", self.config.get("num_labels"))
        outputs_in_file = None
        try:
            outputs_in_file = self._infer_outputs_from_state_dict(sd, head_model_path) # Use helper
        except Exception as e_infer:
             raise ValueError(f"Failed to infer outputs from state dict for {head_model_path}: {e_infer}")

        # Decide final num_classes
        final_num_classes = None
        if num_classes_config is not None:
             num_classes_config = int(num_classes_config)
             if num_classes_config != outputs_in_file:
                  print(f"Warning: Config num_classes ({num_classes_config}) != state dict outputs ({outputs_in_file}). Using state dict value.")
                  final_num_classes = outputs_in_file
             else:
                  final_num_classes = num_classes_config # Config matches state dict
        else:
             print(f"Inferred num_classes from state dict: {outputs_in_file}")
             final_num_classes = outputs_in_file # Use inferred value if config missing

        if final_num_classes is None or final_num_classes <= 0:
             raise ValueError("Failed to determine a valid number of classes for the model.")

        # --- 4. Load Other Model Parameters from Config ---
        # MLP Head Params
        hidden_dim = self.config.get('head_hidden_dim', 1024)
        num_res_blocks = self.config.get('head_num_res_blocks', 3)
        dropout_rate = self.config.get('head_dropout_rate', 0.2)
        output_mode = self.config.get('head_output_mode', 'linear')

        # Attn Pool Params (check both sub-dict and top-level, needed if pooling_after='attn')
        attn_conf = self.config.get('attn_pool_params', self.config)
        attn_pool_heads = attn_conf.get('attn_pool_heads', 16)
        attn_pool_dropout = attn_conf.get('attn_pool_dropout', 0.2)

        # --- 5. Instantiate HeadModel (with correct final_num_classes) ---
        try:
            # Assuming HeadModel v1.5.0+ signature
            model = HeadModel(
                features=expected_features,
                num_classes=final_num_classes, # <<< Use the verified number of classes >>>
                # MLP params
                hidden_dim=hidden_dim,
                num_res_blocks=num_res_blocks,
                dropout_rate=dropout_rate,
                output_mode=output_mode,
                # Attn Pool params (passed even if pooling_after != 'attn')
                attn_pool_heads=attn_pool_heads,
                attn_pool_dropout=attn_pool_dropout
            )
        except Exception as e:
            raise TypeError(f"Failed to instantiate HeadModel: {e}") from e

        # --- 6. Load State Dict into Instantiated Model ---
        model.eval()
        try:
             missing_keys, unexpected_keys = model.load_state_dict(sd, strict=True) # Use the sd loaded earlier
             if unexpected_keys: print(f"Warning: Unexpected keys found loading HeadModel: {unexpected_keys}")
             if missing_keys: print(f"Warning: Missing keys loading HeadModel: {missing_keys}")
        except RuntimeError as e:
            print(f"ERROR: State dict mismatch loading into HeadModel for {head_model_path}.")
            raise e

        model.to(self.device)
        # <<< Update Print Statement >>>
        # print(f"Successfully loaded HeadModel '{os.path.basename(head_model_path)}' with {model.num_classes} outputs (pooling_after_transformer='{model.pooling_after_transformer}', output_mode='{model.output_mode}').")
        self.model = model # Assign to instance variable


# ================================================
#        Aesthetics Pipeline (Scorer)
# ================================================
class CityAestheticsPipeline(BasePipeline):
    """Pipeline for single-output score prediction models."""
    def __init__(self, model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        super().__init__(model_path=model_path, config_path=config_path, device=device, clip_dtype=clip_dtype)
        # <<< Use new loading function >>>
        self._load_model_head()
        # <<< Add check for single output >>>
        if self.num_labels != 1:
             print(f"Warning: CityAestheticsPipeline expects num_labels=1, but loaded model has {self.num_labels}. Output might be incorrect.")
        print("CityAestheticsPipeline: Init OK.")

    def __call__(self, raw_pil_image: Image.Image) -> float:
        emb = self.get_clip_emb_tiled(raw_pil_image, tiling=False) # Assumes non-tiled for scoring
        if emb is None: return 0.0
        pred = self.get_model_pred_generic(emb)
        return float(pred.squeeze().item()) # Assumes single output


# ================================================
#        Classifier Pipeline (Single Model)
# ================================================
class CityClassifierPipeline(BasePipeline):
    """Pipeline for multi-class classification models."""
    # v1.2: Added explicit config_path, default tiling=False
    def __init__(self, model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        # Base class handles config loading
        super().__init__(model_path=model_path, config_path=config_path, device=device, clip_dtype=clip_dtype)

        # Load labels from config
        self.labels = self.config.get("labels", {})

        # Load the predictor head, output count inferred from config/state_dict
        self._load_model_head()
        self.num_labels = self.model.num_classes

        # Populate default labels if needed (based on actual outputs)
        if not self.labels:
             self.labels = {str(i): str(i) for i in range(self.num_labels)}
             print(f"DEBUG: Using default numeric labels (0 to {self.num_labels-1})")
        elif len(self.labels) != self.num_labels:
             print(f"Warning: Config labels count ({len(self.labels)}) != model outputs ({self.num_labels}). Check config.")
             # Optionally reconcile or prioritize model outputs? For now, just warn.

        print(f"CityClassifierPipeline: Init OK (Labels: {self.labels}, Num Outputs: {self.num_labels})")

    # --- __call__ and format_pred remain largely the same ---
    def __call__(self, raw_pil_image: Image.Image, default: bool = True, tiling: bool = False, tile_strat: str = "mean") -> dict:
        emb = self.get_clip_emb_tiled(raw_pil_image, tiling=tiling)
        if emb is None: return {"error": "Failed embedding"}
        pred = self.get_model_pred_generic(emb)
        num_tiles_pred = pred.shape[0] if pred.ndim == 2 else 1
        formatted_output = self.format_pred(
            pred, labels=self.labels, drop_default=(not default),
            tile_strategy=tile_strat if tiling and num_tiles_pred > 1 else "raw"
        )
        return formatted_output

    def format_pred(self, pred: torch.Tensor, labels: dict, drop_default: bool = False, tile_strategy: str = "mean") -> dict:
        return format_classifier_prediction(
            pred,
            labels=labels,
            num_classes=self.num_labels,
            output_mode=getattr(self.model, "output_mode", "linear"),
            drop_default=drop_default,
            tile_strategy=tile_strategy,
        )

# ================================================
#        Multi-Model Classifier Pipeline
# ================================================
class CityClassifierMultiModelPipeline(BasePipeline):
    """Pipeline for running multiple classification models on one image."""
    # v1.2: Updated to load PredictorModel v2 params
    def __init__(self, model_paths: list[str], config_paths: list[str] = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        # Init common things (vision model, processor) using the *first* model's config
        # Assumes all models use the SAME vision backbone.
        first_config_path = None
        if config_paths and config_paths[0]: first_config_path = config_paths[0]
        # BasePipeline.__init__ handles loading first config, vision model, selecting preprocess_func
        super().__init__(model_path=model_paths[0], config_path=first_config_path, device=device, clip_dtype=clip_dtype)

        # Store paths
        self.model_paths = model_paths
        self.config_paths = config_paths if config_paths else [None] * len(model_paths)
        if len(self.model_paths) != len(self.config_paths):
             raise ValueError("Mismatch between number of model paths and config paths.")

        # Load individual predictor heads
        self.models = {} # Stores name -> loaded PredictorModel instance
        self.labels = {} # Stores name -> labels dict
        print(f"Initializing MultiModel Classifier Pipeline for {len(self.model_paths)} models...")

        for i, m_path in enumerate(self.model_paths):
            if not os.path.isfile(m_path):
                 print(f"Warning: Model path not found, skipping: {m_path}")
                 continue
            name = os.path.splitext(os.path.basename(m_path))[0]
            c_path = self.config_paths[i]

            # Infer config if needed (using BasePipeline's helper)
            if c_path is None or not os.path.isfile(c_path):
                 inferred_c_path = self._infer_config_path(m_path)
                 if inferred_c_path:
                     c_path = inferred_c_path
                     print(f"DEBUG MultiModel: Inferred config path {c_path} for {name}")
                 else:
                     print(f"Warning: Could not load or infer config for model: {name}. Skipping.")
                     continue

            try:
                 # --- Load THIS model's config ---
                 current_config = _load_config_helper(c_path)
                 if not current_config:
                     raise ValueError(f"Failed to load model config from {c_path}")

                 # --- Get Parameters using logic similar to _load_predictor_head ---
                 current_embed_ver = current_config.get("model", {}).get("embed_ver", self.embed_ver) # Use specific or default

                 pred_params_conf = current_config.get("predictor_params", {})
                 if not pred_params_conf: # Fallback for older configs
                     print(f"Warning: 'predictor_params' not found in {c_path}. Trying 'model_params'.")
                     pred_params_conf = current_config.get("model_params", {})
                     if not pred_params_conf: raise ValueError("Missing predictor/model params.")

                 # Load features, hidden_dim (with fallbacks)
                 expected_features = pred_params_conf.get("features")
                 if expected_features is None: expected_features = get_embed_params(current_embed_ver)["features"]

                 hidden_dim = pred_params_conf.get("hidden_dim", pred_params_conf.get("hidden")) # Try new then old name
                 if hidden_dim is None: hidden_dim = get_embed_params(current_embed_ver)["hidden"]

                 # Load num_classes (infer from state dict as fallback)
                 num_classes = pred_params_conf.get("num_classes", pred_params_conf.get("outputs"))
                 if num_classes is None: # Infer if missing
                      sd_temp, outputs_in_file = _load_model_helper(m_path, expected_features, None)
                      num_classes = outputs_in_file
                      del sd_temp
                 num_classes = int(num_classes)

                 # Load other PredictorModel v2 params (with defaults)
                 use_attention = pred_params_conf.get("use_attention", True)
                 num_attn_heads = pred_params_conf.get("num_attn_heads", 8)
                 attn_dropout = pred_params_conf.get("attn_dropout", 0.1)
                 num_res_blocks = pred_params_conf.get("num_res_blocks", 1)
                 dropout_rate = pred_params_conf.get("dropout_rate", 0.1)
                 output_mode = pred_params_conf.get("output_mode", 'linear')
                 # --- End Get Parameters ---

                 # --- Load State Dict & Verify Outputs ---
                 sd, outputs_in_file = _load_model_helper(m_path, expected_features, None)
                 if num_classes != outputs_in_file:
                      print(f"Warning: Config num_classes ({num_classes}) != state dict outputs ({outputs_in_file}) for {name}. Using state dict value.")
                      num_classes = outputs_in_file
                 # --- End Load State Dict ---

                 # --- Instantiate PredictorModel v2 ---
                 current_model = PredictorModel(
                     features=expected_features,
                     hidden_dim=hidden_dim,
                     num_classes=num_classes,
                     use_attention=use_attention,
                     num_attn_heads=num_attn_heads,
                     attn_dropout=attn_dropout,
                     num_res_blocks=num_res_blocks,
                     dropout_rate=dropout_rate,
                     output_mode=output_mode
                 )
                 # --- End Instantiate ---

                 # --- Load Weights ---
                 current_model.load_state_dict(sd, strict=True) # Keep strict=True
                 current_model.to(self.device).eval()
                 # --- End Load Weights ---

                 self.models[name] = current_model

                 # Store labels from this model's config
                 current_labels_from_config = current_config.get("labels", {})
                 model_outputs = current_model.num_classes
                 if not current_labels_from_config: self.labels[name] = {str(j): str(j) for j in range(model_outputs)}
                 elif len(current_labels_from_config) != model_outputs:
                      print(f"  Warning: Label count mismatch in {c_path} for {name}. Using defaults.")
                      self.labels[name] = {str(j): str(j) for j in range(model_outputs)}
                 else: self.labels[name] = current_labels_from_config
                 print(f"  Loaded model: {name} (Outputs: {model_outputs})") # Use variable here

            except Exception as e:
                print(f"Error loading model/config for {name} from {m_path}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for multi-model issues

        if not self.models:
            raise ValueError("No valid models loaded for MultiModel pipeline.")
        print(f"CityClassifier MultiModel: Pipeline init ok ({len(self.models)} models loaded)")

    # v1.6: Default tiling=False
    def __call__(self, raw_pil_image: Image.Image, default: bool = True, tiling: bool = False, tile_strat: str = "mean") -> list[dict]:
        """Processes image with all models, returns list of results."""
        # Get embedding once (fit-pad happens in get_clip_emb)
        emb = self.get_clip_emb_tiled(raw_pil_image, tiling=tiling)
        if emb is None: return [{"error": "Failed to get embedding"}] * len(self.models)

        out_list = []
        # Loop through each loaded model head
        for name, model_head in self.models.items():
             # Get predictions using the shared embedding
             # Need generic predictor logic here, moving emb inside loop temporarily
             # pred = self.get_model_pred_generic(model_head, emb) # Incorrectly uses self.model
             with torch.no_grad():
                  pred = model_head(emb.to(self.device, dtype=torch.float32)).detach().cpu()

             # Format the prediction (applies tile combination strategy)
             # Use the _format_single_pred helper method
             num_tiles_pred = pred.shape[0] if pred.ndim == 2 else 1
             formatted_pred = self._format_single_pred(
                 pred,
                 labels=self.labels[name],
                 drop_default=(not default),
                 tile_strategy=tile_strat if tiling and num_tiles_pred > 1 else "raw",
                 num_classes=model_head.outputs
             )
             out_list.append(formatted_pred)

        return out_list

    # v1.1: Corrected median/max/min, final formatting
    def _format_single_pred(self, pred: torch.Tensor, labels: dict, drop_default: bool, tile_strategy: str, num_classes: int) -> dict:
        """Helper to format predictions for one model in the multi-pipeline."""
        return format_multi_model_prediction_raw(
            pred,
            labels=labels,
            num_classes=num_classes,
            drop_default=drop_default,
            tile_strategy=tile_strategy,
        )


# ================================================
#        Head Sequence Pipeline
# ================================================
class HeadSequencePipeline(BasePipeline):
    """Pipeline for HeadModels trained on pre-computed sequences with padding/masking."""
    # v1.1.0: Updated for padding/masking inference
    def __init__(self, head_model_path: str, config_path: str = None, device: str = "cpu", clip_dtype: torch.dtype = torch.float32):
        super().__init__(model_path=head_model_path, config_path=config_path, device=device, clip_dtype=clip_dtype)

        self.labels = self.config.get("labels", {})
        if not self.labels:
             num_classes = self.config.get("num_classes", self.config.get("num_labels", 0))
             self.labels = {str(i): str(i) for i in range(num_classes)}

        # <<< --- Get target_len used during training (if saved, else use default) --- >>>
        # Check if 'fixed_len' or similar was saved in config, otherwise use default
        self.target_len = self.config.get("fixed_len", TARGET_LEN_INFERENCE)
        print(f"HeadSequencePipeline: Using target_len={self.target_len} for padding/masking.")

        # Load the HeadModel (expects pooling_strategy='attn' in config now)
        self._load_head_model(head_model_path)
        if not hasattr(self.model, 'pooling_strategy') or self.model.pooling_strategy != 'attn':
             print(f"Warning: Loaded HeadModel pooling is '{getattr(self.model, 'pooling_strategy', 'N/A')}', but padding/masking pipeline expects 'attn'.")

        self.num_labels = self.model.num_classes

        print(f"HeadSequencePipeline (Padding/Masking Mode): Init OK (Labels: {self.labels}, TargetLen: {self.target_len}, Outputs: {self.num_labels})")

    # --- Helper: Extract and Normalize Sequence ---
    # v1.1.0: Renamed and simplified: only extracts and normalizes
    @torch.no_grad()
    def _extract_and_normalize_sequence(self, pil_image: Image.Image) -> torch.Tensor | None:
        """Extracts sequence using vision model and normalizes it."""
        if self.vision_model is None or self.hf_processor is None:
             print("Error: Vision model/processor not initialized."); return None

        # 1. Preprocess Image (Handle >4096 patches resize)
        img_to_process = pil_image
        try:
            original_width, original_height = pil_image.size
            if original_width <= 0 or original_height <= 0: raise ValueError("Invalid image dimensions")

            TARGET_MAX_PATCHES = self.target_len # Use pipeline's target_len
            PATCH_SIZE = 14

            patches_w_initial = math.floor(original_width / PATCH_SIZE)
            patches_h_initial = math.floor(original_height / PATCH_SIZE)
            total_patches_initial = patches_w_initial * patches_h_initial

            if total_patches_initial > TARGET_MAX_PATCHES:
                # Replicate the resizing logic from generate_features
                scale_factor = math.sqrt(TARGET_MAX_PATCHES / total_patches_initial)
                resize_needed = True
                max_iterations = 10
                iterations = 0
                while iterations < max_iterations:
                    target_w = int(original_width * scale_factor + 0.5)
                    target_h = int(original_height * scale_factor + 0.5)
                    if target_w < 1: target_w = 1;
                    if target_h < 1: target_h = 1;
                    new_patches_w = math.floor(target_w / PATCH_SIZE)
                    new_patches_h = math.floor(target_h / PATCH_SIZE)
                    if new_patches_w * new_patches_h <= TARGET_MAX_PATCHES:
                        # print(f"  - Resizing ({original_width}x{original_height}, {total_patches_initial}p) -> ({target_w}x{target_h}, {new_patches_w * new_patches_h}p)")
                        img_to_process = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                        resize_needed = False
                        break
                    scale_factor *= 0.995
                    iterations += 1
                if resize_needed: print(
                    f"Warning: Could not find suitable resize. Using original."); img_to_process = pil_image;

        except Exception as e_resize: print(f"Error during inference resize check: {e_resize}"); return None

        # 2. Extract Sequence (last_hidden_state)
        feature_sequence = None
        try:
            with torch.amp.autocast(device_type=self.device, enabled=(self.clip_dtype != torch.float32), dtype=self.clip_dtype):
                 inputs = self.hf_processor(images=[img_to_process], return_tensors="pt").to(self.device)
                 outputs = self.vision_model(**inputs)
                 if hasattr(outputs, 'last_hidden_state'):
                     feature_sequence = outputs.last_hidden_state # Shape [1, N, F]
                 else: raise ValueError("Model output missing 'last_hidden_state'")
        except Exception as e_extract: print(f"Error extracting features: {e_extract}"); return None

        if feature_sequence is None or feature_sequence.ndim != 3 or feature_sequence.shape[0] != 1:
             print(f"Error: Invalid feature sequence shape {feature_sequence.shape if feature_sequence is not None else 'None'}."); return None

        # 3. Normalize Sequence
        try:
             feature_sequence = F.normalize(feature_sequence, p=2, dim=-1) # Normalize [1, N, F] along F dim
             if torch.isnan(feature_sequence).any(): raise ValueError("NaN detected after normalization")
        except Exception as e_norm: print(f"Error during sequence normalization: {e_norm}"); return None

        # <<< --- Return the normalized sequence on GPU --- >>>
        # Keep on GPU, convert to Float32 for subsequent processing
        return feature_sequence.squeeze(0).float() # Shape [N, F], Float32

    # --- Main Call Method ---
    # v1.1.0: Implements padding/masking before calling HeadModel
    def __call__(self, raw_pil_image: Image.Image) -> dict:
        """Processes image, extracts+normalizes sequence, pads/masks, runs HeadModel."""

        # 1. Extract and Normalize Sequence
        # Returns shape [N, F] on GPU, Float32
        norm_sequence = self._extract_and_normalize_sequence(raw_pil_image)

        if norm_sequence is None:
            return {"error": "Failed to extract or normalize sequence"}

        # 2. Pad/Truncate Sequence and Create Mask
        try:
            current_len = norm_sequence.shape[0]
            features_dim = norm_sequence.shape[1]
            # Create tensors on the same device as the sequence
            final_sequence_gpu = torch.zeros((self.target_len, features_dim), dtype=torch.float32, device=norm_sequence.device)
            attention_mask_gpu = torch.zeros(self.target_len, dtype=torch.bool, device=norm_sequence.device)

            if current_len == 0:
                 print("Warning: Empty normalized sequence. Using zero padding.")
            elif current_len > self.target_len:
                 # This shouldn't happen if generation capped length, but handle defensively
                 print(f"Warning: Inference sequence length {current_len} > target {self.target_len}. Truncating.")
                 final_sequence_gpu = norm_sequence[:self.target_len, :]
                 attention_mask_gpu[:] = True # All real tokens
            else: # Pad
                 final_sequence_gpu[:current_len, :] = norm_sequence
                 attention_mask_gpu[:current_len] = True # Mark real tokens

            # Add batch dimension for the model
            final_sequence_batch = final_sequence_gpu.unsqueeze(0) # Shape [1, target_len, F]
            attention_mask_batch = attention_mask_gpu.unsqueeze(0) # Shape [1, target_len]

        except Exception as e_pad:
             print(f"Error during padding/masking: {e_pad}"); return {"error": "Padding/masking failed"}

        # 3. Run Head Model Prediction
        # self.model is the loaded HeadModel, expects Float32 input [B, Seq, F] and mask [B, Seq]
        try:
            with torch.no_grad():
                 # Pass sequence AND mask
                 pred = self.model(final_sequence_batch, attention_mask=attention_mask_batch)
        except Exception as e_pred:
             print(f"Error during HeadModel prediction: {e_pred}"); traceback.print_exc(); return {"error": "Prediction failed"}

        # 4. Format Output
        pred_cpu = pred.detach().cpu()
        return format_sequence_prediction(
            pred_cpu,
            labels=self.labels,
            num_labels=self.num_labels,
            output_mode=getattr(self.model, "output_mode", "linear"),
        )


# --- get_model_path (Utility) ---
# v1.1: Corrected path joining for local dir
def get_model_path(name: str, repo: str, token: str | bool | None = None, extension: str = "safetensors", local_dir: str = "models", use_hf: bool = True) -> str:
    """Gets model path, checking local folder first, then Hugging Face Hub."""
    fname = f"{name}.{extension}"
    # Keep local model resolution anchored at repo root for compatibility.
    project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))
    local_path = os.path.join(project_root, local_dir, fname)

    if os.path.isfile(local_path):
        print(f"Using local model: '{local_path}'")
        return local_path

    if not use_hf:
        raise FileNotFoundError(f"Local model '{local_path}' not found and Hugging Face download disabled.")

    # HF download logic
    hf_token_arg = None
    if isinstance(token, str) and token: hf_token_arg = token
    elif token is True: hf_token_arg = True # Use env var or cached login
    # token=False or None means no token explicitly passed

    print(f"Local model not found. Downloading from HF Hub: '{repo}/{fname}'")
    try:
        downloaded_path = hf_hub_download(repo_id=repo, filename=fname, token=hf_token_arg)
        return str(downloaded_path) # Ensure it's a string path
    except Exception as e:
        print(f"Error downloading model '{fname}' from repo '{repo}': {e}")
        raise

# --- End get_model_path ---
