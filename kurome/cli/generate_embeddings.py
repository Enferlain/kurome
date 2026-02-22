# generate_embeddings.py
# Version: 4.2.0 (Added TIMM DINOv2 support)

import os
from concurrent.futures import ThreadPoolExecutor, Future
import time
from typing import List, Optional # <<< MODIFIED: Added Optional for type hinting >>>

import torch.nn.functional as F
import torch
import torchvision.transforms.functional as TF # Renamed from F to avoid conflict
import torch.nn.functional as torch_func

# <<< ADD TIMM IMPORT >>>
# Note: 'timm' is now imported on-demand where needed to satisfy static analysis
# and avoid 'unbound' errors if the library is not present.
# <<< END ADD >>>

from PIL import Image, UnidentifiedImageError # <<< Added UnidentifiedImageError >>>
from tqdm import tqdm
import numpy as np
from transformers import AutoProcessor, AutoModel, AutoImageProcessor, Dinov2Model
import argparse
import math
import shutil
import traceback

# --- Constants ---
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_PRECISION_MAP = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
# Defaulting to standard SigLIP here, will be overridden by args
DEFAULT_MODEL_ID = "google/siglip-so400m-patch14-384" # Changed default for clarity
# <<< Define NaFlex patch size >>>
SIGLIP_NAFLEX_PATCH_SIZE = 16
# <<< ADDED: AIMv2 Specific Constants >>>
AIMV2_PATCH_SIZE = 14
AIMV2_TARGET_MAX_PATCHES = 4096 # Target patch count for resizing, adjust if needed
# <<< ADDED: Threading config >>>
NUM_LOAD_WORKERS = 64 # Number of threads for loading images
LOAD_QUEUE_SIZE = 32  # How many images to keep loaded ahead of time
# <<< ADDED: DINOv3 Specific Constants >>>
DINOV3_PATCH_SIZE = 16 # DINOv3 patch size
MAX_DINOV3_RESOLUTION = 4096 # Max resolution for DINOv3, adjust based on GPU memory
# <<< END ADDED DINOv3 Specific Constants >>>


# --- Argument Parsing ---
# v4.2.0: Added dinov2_large_timm_fitpad choice
def parse_gen_args():
    parser = argparse.ArgumentParser(description="Generate vision model embeddings for image folders.")
    parser.add_argument('--image_dir', required=True, help="Directory containing subfolders (e.g., '0', '1') with images.")
    parser.add_argument('--output_dir_root', required=True, help="Root directory to save embeddings (e.g., 'data').")
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_ID, help=f"Vision model name from Hugging Face or TIMM. Default: {DEFAULT_MODEL_ID}")
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'], help="Precision for model computation (default: fp32).")
    parser.add_argument('--preprocess_mode', type=str, default='fit_pad',
                        choices=[
                            'fit_pad',              # Manual FitPad (SigLIP HF)
                            'center_crop',          # Manual CenterCrop (SigLIP HF)
                            # 'avg_crop',             # Manual AvgCrop (NYI)
                            'naflex_resize',        # NaFlex HF (Proc Logic @ 1024)
                            'naflex_resize2',  # NaFlex HF (Proc Logic @ 2048)
                            'dinov2_large_timm_fitpad',  # DINOv2 TIMM (TIMM Transforms)
                            'dinov2_giant_fb_fitpad',  # DINOv2 Facebook (Manual FitPad 518)
                            'aimv2_native_cls',  # <<< ADDED: AIMv2 Native CLS Token (HF Processor) >>>
                            'dinov3_7b_8bit_bnb' # <<< ADDED: DINOv3 7B 8-bit BnB >>>
                        ],
                        help="Image preprocessing method before embedding.")
    # parser.add_argument('--resize_factor_avg_crop', type=float, default=2.0,
    #                     help="Factor to multiply model's native size by for resizing ONLY for avg_crop mode (default: 2.0).")
    parser.add_argument('--naflex_max_patches', type=int, default=1024, help="Value for max_num_patches for SigLIP NaFlex processor (default: 1024).")
    parser.add_argument('--output_dir_suffix', type=str, default="", help="Optional suffix for the output directory name.")

    args = parser.parse_args()

    # --- Validation ---
    if args.preprocess_mode == 'dinov2_large_timm_fitpad':
        try:
            import timm
        except ImportError:
            parser.error("preprocess_mode 'dinov2_large_timm_fitpad' requires 'timm' to be installed.")
    if args.preprocess_mode == 'dinov2_large_timm_fitpad' and not args.model_name.startswith('timm/'):
         print(f"Warning: Mode 'dinov2_large_timm_fitpad' expects a TIMM model (timm/...), got '{args.model_name}'.")
    if args.preprocess_mode == 'dinov2_giant_fb_fitpad' and not args.model_name.startswith('facebook/dinov2'):
         print(f"Warning: Mode 'dinov2_giant_fb_fitpad' expects a Facebook DINOv2 model (facebook/dinov2-...), got '{args.model_name}'.")
    # <<< ADDED: Validation for aimv2_native_cls >>>
    if args.preprocess_mode == 'aimv2_native_cls' and not args.model_name.startswith('apple/aimv2'):
         print(f"Warning: Mode 'aimv2_native_cls' expects an Apple AIMv2 model (apple/aimv2-...), got '{args.model_name}'.")
    # <<< ADDED: Validation for dinov3_7b_8bit_bnb >>>
    if args.preprocess_mode == 'dinov3_7b_8bit_bnb' and not args.model_name.startswith('dinov3-vit7b16-pretrain-lvd1689m-8bit'):
         print(f"Warning: Mode 'dinov3_7b_8bit_bnb' expects model_name 'dinov3-vit7b16-pretrain-lvd1689m-8bit', got '{args.model_name}'.")

    return args

# --- Preprocessing Functions ---

def preprocess_fit_pad(img_pil, target_size=512, fill_color=(0, 0, 0)):
    """Resizes image to fit, pads to target size."""
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None
    target_w, target_h = target_size, target_size
    scale = min(target_w / original_width, target_h / original_height)
    new_w = int(original_width * scale)
    new_h = int(original_height * scale)
    if new_w == 0: new_w = 1 # Prevent zero size
    if new_h == 0: new_h = 1
    try:
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        img_padded = Image.new(img_pil.mode, (target_w, target_h), fill_color)
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        img_padded.paste(img_resized, (pad_left, pad_top))
        return img_padded
    except Exception as e:
        print(f"Error during preprocess_fit_pad: {e}")
        return None

def preprocess_center_crop(img_pil, target_size=512):
    """Resizes shortest side to target_size, then center crops."""
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None
    try:
        short, long = (original_width, original_height) if original_width <= original_height else (original_height, original_width)
        if short == 0: return None # Avoid division by zero
        scale = target_size / short
        new_w = int(original_width * scale)
        new_h = int(original_height * scale)
        if new_w == 0: new_w = 1
        if new_h == 0: new_h = 1

        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

        crop_w, crop_h = target_size, target_size
        left = (new_w - crop_w) // 2
        top = (new_h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h

        img_cropped = img_resized.crop((left, top, right, bottom))
        return img_cropped
    except Exception as e:
        print(f"Error during preprocess_center_crop: {e}")
        return None

# v4.1: Targets patches, floors dims, ensures multiple-of-16, returns (img, patch_count)
def preprocess_naflex_resize(img_pil, target_patches=1024, patch_size=16):
    """
    Resizes image preserving aspect ratio for target_patches, ensuring dimensions
    are multiples of patch_size (flooring). Returns tuple: (resized_image, total_patches).
    """
    original_width, original_height = img_pil.size
    if original_width <= 0 or original_height <= 0: return None, 0
    try:
        aspect_ratio = original_width / original_height
        ideal_patch_w_f = math.sqrt(target_patches * aspect_ratio)
        ideal_patch_h_f = math.sqrt(target_patches / aspect_ratio)
        ideal_width_f = ideal_patch_w_f * patch_size
        ideal_height_f = ideal_patch_h_f * patch_size
        new_width = math.floor(ideal_width_f / patch_size) * patch_size
        new_height = math.floor(ideal_height_f / patch_size) * patch_size
        
        # Assign to new_w and new_h at the beginning
        new_w = new_width # <<< FIXED: Assign new_width to new_w >>>
        new_h = new_height # <<< FIXED: Assign new_height to new_h >>>

        if new_w == 0: new_w = patch_size
        if new_h == 0: new_h = patch_size
        num_patches_w = new_w // patch_size # <<< FIXED: Use new_w and new_h >>>
        num_patches_h = new_h // patch_size # <<< FIXED: Use new_w and new_h >>>
        total_patches = num_patches_w * num_patches_h

        print(f"  DEBUG Preprocess NaflexResize(v4.1): Original: {original_width}x{original_height}, TargetPatches: {target_patches}, New: {new_w}x{new_h}, Patches: {total_patches} ({num_patches_w}x{num_patches_h})") # <<< FIXED: Use new_w and new_h >>>

        if total_patches > target_patches * 1.05: # Allow small overshoot? Or strict?
            print(f"  ERROR: Calculated patches ({total_patches}) exceed target ({target_patches})!")
            return None, 0

        img_resized = img_pil.resize((int(new_w), int(new_h)), Image.Resampling.LANCZOS) # <<< FIXED: Use new_w and new_h >>>
        return img_resized, total_patches
    except Exception as e:
        print(f"Error during preprocess_naflex_resize: {e}")
        return None, 0


# --- Model Initialization (Now uses AutoImageProcessor) ---
# v4.3.0: Use AutoImageProcessor
import dinov3_7b_quant_bnb # <<< ADDED: Import for DINOv3 BnB loading >>>

def init_vision_model(model_name, device, dtype):
    """Initializes vision model and image processor."""
    print(f"Initializing Vision Model: {model_name} on {device} with dtype {dtype}")
    try:
        # <<< ADDED: DINOv3 8-bit BnB Model Loading >>>
        if model_name == "dinov3-vit7b16-pretrain-lvd1689m-8bit":
            print(f"  Loading DINOv3 8-bit BnB model from saved path: {model_name}")
            model, processor = dinov3_7b_quant_bnb.load_from_saved_bnb(save_path=f"./{model_name}")
            # For 8-bit models, device_map="auto" already handles device placement.
            # Do NOT call .to(device) again, as it's not supported and causes an error.
            model.eval() # Ensure model is in eval mode
            
            print(f"  Loaded DINOv3 8-bit BnB model class: {model.__class__.__name__}")
            print(f"  Loaded DINOv3 8-bit BnB processor class: {processor.__class__.__name__}")
        else:
            # <<< Use AutoImageProcessor explicitly >>>
            # trust_remote_code might be needed for AIMv2's processor config
            processor = AutoImageProcessor.from_pretrained(model_name, attn_implementation="sdpa", trust_remote_code=True)
            model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="sdpa", trust_remote_code=True).to(device).eval()

            print(f"  Loaded model class: {model.__class__.__name__}")
            print(f"  Loaded processor class: {processor.__class__.__name__}")

        # --- Disable Processor Auto-Preprocessing for manual modes ONLY ---
        # <<< This logic is now less critical if we use processor directly for AIMv2 CLS >>>
        # Keep it for SigLIP/DINOv2 FitPad/CenterCrop modes
        print(f"  DEBUG: Original processor config: {processor}")
        # Check if these attributes exist before trying to set them
        if hasattr(processor, 'do_resize'): processor.do_resize = True
        if hasattr(processor, 'do_center_crop'): processor.do_center_crop = False
        if hasattr(processor, 'do_normalize'): processor.do_normalize = True # Keep normalize ON for processor use

        print(f"  DEBUG: Modified processor config (for manual modes): {processor}")

        # --- Determine Model Input Size ---
        # This is mostly relevant for manual modes (FitPad/CenterCrop)
        image_size = 384  # Default guess
        try:
            model_config = getattr(model, 'config', None)
            if model_config:
                 # Check standard vision_config first (like SigLIP)
                 vision_config = getattr(model_config, 'vision_config', None)
                 if vision_config and hasattr(vision_config, 'image_size'): image_size = int(vision_config.image_size)
                 # Check DINOv2/AIMv2 specific config location
                 elif hasattr(model_config, 'image_size'): image_size = int(model_config.image_size)
                 # Fallback to processor config
                 elif hasattr(processor, 'size'):
                     proc_sz_config = processor.size
                     if isinstance(proc_sz_config, dict): image_size = int(proc_sz_config.get("height", image_size))
                     elif isinstance(proc_sz_config, int): image_size = int(proc_sz_config)
                 # print(f"  DEBUG: Found size via config/processor: {image_size}") # Keep debug optional
            else: print(f"  DEBUG: Model has no 'config' attribute. Using default size: {image_size}")

            print(f"  Determined Model Standard Input Size (for FitPad/CenterCrop modes): {image_size}x{image_size}")
            # <<< Add explicit override check for DINOv2 Giant >>>
            if "dinov2" in model_name.lower() and "giant" in model_name.lower() and image_size != 518:
                print(f"  INFO: Explicitly setting target size to 518 for DINOv2 Giant (was {image_size}).")
                image_size = 518

        except Exception as e_size: print(f"  Warning: Could not determine model input size: {e_size}. Using default {image_size}.")

        return processor, model, image_size

    except Exception as e: print(f"Error initializing vision model {model_name}: {e}"); raise


# --- Unified Embedding Function ---
# Version 4.5.1: Corrects SigLIP NaFlex mask creation based on spatial_shapes
@torch.no_grad()
def get_embedding(
    raw_img_pil: Image.Image,
    preprocess_mode: str,
    model_info: dict,
    device,
    dtype,
    model_image_size: int, # Needed for manual modes like FitPad
    filename: Optional[str] = None, # <<< MODIFIED: Use Optional for type hinting >>>
    naflex_max_patches: int = 1024 # Default for SigLIP NaFlex
) -> np.ndarray | None:
    """
    Generates embedding for a single raw PIL image using the specified mode
    and the provided model_info bundle.
    Returns normalized float32 numpy embedding or None on error.
    """
    img_name = filename if filename else getattr(raw_img_pil, 'filename', 'UNKNOWN') # <<< UPDATED line
    model_type = model_info.get("type", "hf")

    try:
        emb = None
        do_l2_normalize = False
        img_to_process = raw_img_pil # Start with the raw image

        # --- TIMM Model Handling (Unchanged) ---
        if model_type == "timm" and preprocess_mode == 'dinov2_large_timm_fitpad':
            timm_model = model_info.get("model"); timm_transforms = model_info.get("transforms")
            if timm_model is None or timm_transforms is None: raise ValueError("TIMM model/transforms missing.")
            input_tensor = timm_transforms(raw_img_pil).unsqueeze(0).to(device=device, dtype=dtype)
            output = timm_model(input_tensor); emb = output; do_l2_normalize = True

        # --- Hugging Face Model Handling (SigLIP, FB DINOv2, AIMv2) ---
        elif model_type == "hf":
            processor = model_info.get("processor") # Should be AutoImageProcessor now
            model = model_info.get("model")
            if processor is None or model is None: raise ValueError("HF processor or model missing.")

            model_call_kwargs = {}
            is_siglip_model = "Siglip" in model.__class__.__name__
            is_dinov2_model = "Dinov2Model" in model.__class__.__name__
            # <<< ADDED: Check for AIMv2Model name >>>
            is_aimv2_model = "AIMv2Model" in model.__class__.__name__
            # <<< MODIFIED: Check for DINOv3ViTModel name >>>
            is_dinov3_model = "DINOv3ViTModel" in model.__class__.__name__

            # --- HF NaFlex Mode (SigLIP Processor Logic @ 1024) ---
            if is_siglip_model and preprocess_mode == 'naflex_resize':
                inputs = processor(images=[raw_img_pil], return_tensors="pt", max_num_patches=1024)
                pixel_values = inputs.get("pixel_values"); attention_mask = inputs.get("pixel_attention_mask"); spatial_shapes = inputs.get("spatial_shapes")
                if pixel_values is None or attention_mask is None or spatial_shapes is None: raise ValueError("Missing tensors from HF NaFlex processor.")
                model_call_kwargs = {"pixel_values": pixel_values.to(device=device, dtype=dtype), "attention_mask": attention_mask.to(device=device), "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.long).to(device=device)}
                # Call SigLIP model (logic unchanged)
                vision_model_component = getattr(model, 'vision_model', None)
                if vision_model_component: emb = vision_model_component(**model_call_kwargs).pooler_output
                elif hasattr(model, 'get_image_features'): emb = model.get_image_features(**model_call_kwargs)
                else: raise AttributeError("SigLIP Model missing expected methods.")
                do_l2_normalize = False # SigLIP internal norm

            # --- HF NaFlex Mode (SigLIP Processor Logic @ 2048) ---
            elif is_siglip_model and preprocess_mode == 'naflex_resize2':
                inputs = processor(images=[raw_img_pil], return_tensors="pt", max_num_patches=2048)
                pixel_values = inputs.get("pixel_values"); attention_mask = inputs.get("pixel_attention_mask"); spatial_shapes = inputs.get("spatial_shapes")
                if pixel_values is None or attention_mask is None or spatial_shapes is None: raise ValueError("Missing tensors from HF NaFlex processor.")
                model_call_kwargs = {"pixel_values": pixel_values.to(device=device, dtype=dtype), "attention_mask": attention_mask.to(device=device), "spatial_shapes": torch.tensor(spatial_shapes, dtype=torch.long).to(device=device)}
                # Call SigLIP model (logic unchanged)
                vision_model_component = getattr(model, 'vision_model', None)
                if vision_model_component: emb = vision_model_component(**model_call_kwargs).pooler_output
                elif hasattr(model, 'get_image_features'): emb = model.get_image_features(**model_call_kwargs)
                else: raise AttributeError("SigLIP Model missing expected methods.")
                do_l2_normalize = False # SigLIP internal norm

            # --- HF DINOv2 FB Mode (Manual FitPad 518 + CLS Token) ---
            elif is_dinov2_model and preprocess_mode == 'dinov2_giant_fb_fitpad':
                processed_img_pil = preprocess_fit_pad(raw_img_pil, target_size=model_image_size) # Uses 518
                if processed_img_pil is None: return None
                # Use processor only for ToTensor/Normalize
                inputs = processor(images=[processed_img_pil], return_tensors="pt")
                pixel_values = inputs.get("pixel_values")
                if pixel_values is None: raise ValueError("HF DINOv2 Processor didn't return 'pixel_values'.")
                model_call_kwargs = {"pixel_values": pixel_values.to(device=device, dtype=dtype)}
                # Call DINOv2 model
                outputs = model(**model_call_kwargs)
                last_hidden_state = getattr(outputs, 'last_hidden_state', None)
                if last_hidden_state is None: raise ValueError("DINOv2 model did not return last_hidden_state.")
                emb = last_hidden_state[:, 0, :] # Extract CLS token
                do_l2_normalize = True

            # --- HF AIMv2 Native CLS Mode (WITH PRE-RESIZING) ---
            elif is_aimv2_model and preprocess_mode == 'aimv2_native_cls':
                # print(f"DEBUG get_embedding [HF/AIMv2 CLS]: Applying pre-resize if needed.")
                # <<< START PRE-RESIZING LOGIC (Adapated from generate_feature_sequences) >>>
                try:
                    original_width, original_height = raw_img_pil.size
                    if original_width <= 0 or original_height <= 0: raise ValueError("Invalid image dimensions")

                    # Check initial patch count
                    patches_w_initial = math.floor(original_width / AIMV2_PATCH_SIZE)
                    patches_h_initial = math.floor(original_height / AIMV2_PATCH_SIZE)
                    total_patches_initial = patches_w_initial * patches_h_initial

                    if total_patches_initial > AIMV2_TARGET_MAX_PATCHES:
                        # Calculate scale factor and resize iteratively
                        scale_factor = math.sqrt(AIMV2_TARGET_MAX_PATCHES / total_patches_initial)
                        resize_needed = True; max_iterations = 10; iterations = 0
                        while iterations < max_iterations:
                             target_w = int(original_width * scale_factor + 0.5); target_h = int(original_height * scale_factor + 0.5)
                             if target_w < 1: target_w = 1;
                             if target_h < 1: target_h = 1
                             # Ensure new dims are multiples of patch size
                             target_w = math.floor(target_w / AIMV2_PATCH_SIZE) * AIMV2_PATCH_SIZE
                             target_h = math.floor(target_h / AIMV2_PATCH_SIZE) * AIMV2_PATCH_SIZE
                             if target_w == 0: target_w = AIMV2_PATCH_SIZE
                             if target_h == 0: target_h = AIMV2_PATCH_SIZE

                             new_patches_w = target_w // AIMV2_PATCH_SIZE
                             new_patches_h = target_h // AIMV2_PATCH_SIZE

                             if new_patches_w * new_patches_h <= AIMV2_TARGET_MAX_PATCHES:
                                 print(f"  - INFO: Resizing {img_name} ({original_width}x{original_height}, {total_patches_initial}p) -> ({target_w}x{target_h}, {new_patches_w * new_patches_h}p) for AIMv2 CLS.")
                                 img_to_process = raw_img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                                 resize_needed = False; break
                             scale_factor *= 0.995; iterations += 1
                        if resize_needed: # If loop finished without success
                            print(f"  Warning: Could not find suitable resize dimensions for {img_name}. Using original (might OOM or error).")
                            img_to_process = raw_img_pil # Use original as fallback
                    else:
                        # Ensure original dimensions are multiples of patch size if not resizing
                        current_w, current_h = img_to_process.size
                        new_w = math.floor(current_w / AIMV2_PATCH_SIZE) * AIMV2_PATCH_SIZE
                        new_h = math.floor(current_h / AIMV2_PATCH_SIZE) * AIMV2_PATCH_SIZE
                        if new_w == 0: new_w = AIMV2_PATCH_SIZE
                        if new_h == 0: new_h = AIMV2_PATCH_SIZE
                        if new_w != current_w or new_h != current_h:
                             print(f"  - INFO: Adjusting {img_name} dims ({current_w}x{current_h}) -> ({new_w}x{new_h}) to be multiples of {AIMV2_PATCH_SIZE}.")
                             img_to_process = img_to_process.resize((new_w, new_h), Image.Resampling.LANCZOS)
                except Exception as e_resize:
                     print(f"\nError during AIMv2 pre-resizing for {img_name}: {e_resize}")
                     traceback.print_exc(); return None
                # <<< END PRE-RESIZING LOGIC >>>

                # Now, use the processor on the (potentially resized) image
                try: inputs = processor(images=[img_to_process], return_tensors="pt")
                except Exception as e_proc: print(f"\nError processor call: {e_proc}"); return None
                pixel_values = inputs.get("pixel_values")
                if pixel_values is None: raise ValueError("HF AIMv2 Processor didn't return 'pixel_values'.")
                model_call_kwargs = {"pixel_values": pixel_values.to(device=device, dtype=dtype)}

                # Call model, extract CLS token
                outputs = model(**model_call_kwargs)
                last_hidden_state = getattr(outputs, 'last_hidden_state', None)
                if last_hidden_state is None: raise ValueError("AIMv2 model did not return last_hidden_state.")
                emb = last_hidden_state[:, 0, :]; do_l2_normalize = True

            # <<< ADDED: DINOv3 7B 8-bit BnB Mode (Variable Resolution + Mean Pooling) >>>
            elif is_dinov3_model and preprocess_mode == 'dinov3_7b_8bit_bnb':
                current_w, current_h = raw_img_pil.size
                img_to_process = raw_img_pil

                # Optional: Add max resolution limit for memory protection
                if max(current_w, current_h) > MAX_DINOV3_RESOLUTION:
                    scale = MAX_DINOV3_RESOLUTION / max(current_w, current_h)
                    current_w = int(current_w * scale)
                    current_h = int(current_h * scale)
                    img_to_process = raw_img_pil.resize((current_w, current_h), Image.Resampling.LANCZOS)
                    print(f"  - INFO: Scaling down {img_name} to fit within {MAX_DINOV3_RESOLUTION}px limit.")

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
                    print(f"  - INFO: Adjusting {img_name} dims ({current_w}x{current_h}) -> ({new_w}x{new_h}) to be multiples of {DINOV3_PATCH_SIZE} for DINOv3.")
                    img_to_process = img_to_process.resize((new_w, new_h), Image.Resampling.LANCZOS)

                # Use processor for ToTensor/Normalize
                inputs = processor(images=[img_to_process], return_tensors="pt")
                pixel_values = inputs.get("pixel_values")
                if pixel_values is None: raise ValueError("HF DINOv3 Processor didn't return 'pixel_values'.")
                model_call_kwargs = {"pixel_values": pixel_values.to(device=device, dtype=dtype)}
                # Call DINOv3 model
                outputs = model(**model_call_kwargs)
                last_hidden_state = getattr(outputs, 'last_hidden_state', None)
                if last_hidden_state is None: raise ValueError("DINOv3 model did not return last_hidden_state.")
                # Use mean pooling of patch tokens (exclude CLS and register tokens)
                # DINOv3-7B has 4 register tokens. We skip CLS (pos 0) and registers (pos 1-4).
                nreg = getattr(model.config, 'num_register_tokens', 0)
                patch_embeddings = last_hidden_state[:, 1 + nreg:] # Skip CLS and Registers
                emb = torch.mean(patch_embeddings, dim=1)  # [B, embed_dim]
                do_l2_normalize = True # DINOv3 typically requires L2 normalization

            # --- HF Manual Modes (FitPad / CenterCrop for SigLIP/non-FB DINOv2) ---
            elif preprocess_mode in ['fit_pad', 'center_crop']:
                processed_img_pil = None
                if preprocess_mode == 'fit_pad': processed_img_pil = preprocess_fit_pad(raw_img_pil, target_size=model_image_size)
                elif preprocess_mode == 'center_crop': processed_img_pil = preprocess_center_crop(raw_img_pil, target_size=model_image_size)
                if processed_img_pil is None: return None # Preprocessing failed
                # Use processor ONLY for ToTensor + Normalize
                inputs = processor(images=[processed_img_pil], return_tensors="pt")
                pixel_values = inputs.get("pixel_values")
                if pixel_values is None: raise ValueError("Processor didn't return 'pixel_values'.")
                model_call_kwargs = {"pixel_values": pixel_values.to(device=device, dtype=dtype)}
                # Call appropriate model (logic unchanged)
                if is_dinov2_model: outputs = model(**model_call_kwargs); emb = outputs.last_hidden_state[:, 0]
                elif is_siglip_model:
                     vision_model_component = getattr(model, 'vision_model', None)
                     if vision_model_component: emb = vision_model_component(**model_call_kwargs).pooler_output
                     elif hasattr(model, 'get_image_features'): emb = model.get_image_features(**model_call_kwargs)
                     else: raise AttributeError("SigLIP Model missing expected methods.")
                else: raise TypeError(f"Model type mismatch in HF manual preproc path: {model.__class__.__name__}")
                do_l2_normalize = is_dinov2_model # Normalize only DINOv2 here

            # --- Invalid HF Mode ---
            else:
                print(f"ERROR get_embedding [HF]: Invalid/unsupported preprocess_mode '{preprocess_mode}' for detected HF model type.")
                return None

            # --- HF Model Call (already happened within specific mode blocks) ---
            if emb is None: raise ValueError("Failed to get embedding from HF model call.")

        # --- Invalid Mode ---
        else:
            print(f"ERROR get_embedding: Unknown model_type '{model_type}' or unhandled preprocess_mode '{preprocess_mode}'.")
            return None

        # --- Final Normalization & Conversion ---
        if emb is None: print(f"ERROR: Embedding is None before final conversion for {img_name}."); return None
        if not isinstance(emb, torch.Tensor): raise TypeError(f"Embedding is not a Tensor ({type(emb)}).")

        if do_l2_normalize:
            norm = torch.linalg.norm(emb.float(), dim=-1, keepdim=True).clamp(min=1e-8)
            normalized_emb = emb / norm.to(emb.dtype); emb = normalized_emb

        # <<< Ensure output is 1D numpy array >>>
        # .squeeze() should remove the batch dimension if present (e.g., [1, F] -> [F])
        embedding_result_np = emb.cpu().to(torch.float32).numpy().squeeze()
        if embedding_result_np.ndim != 1: # Add check just in case
            print(f"Warning: Final embedding shape is {embedding_result_np.shape} (expected 1D). Attempting reshape.")
            embedding_result_np = embedding_result_np.reshape(-1) # Flatten if necessary
            if embedding_result_np.ndim != 1:
                 print(f"Error: Could not reshape embedding to 1D for {img_name}. Shape: {embedding_result_np.shape}")
                 return None

        return embedding_result_np

    except Exception as e:
        print(f"\nError during get_embedding (v4.3.0) for {img_name} (Mode: {preprocess_mode}, Type: {model_type}):")
        traceback.print_exc()
        return None
# --- End Unified Embedding Function ---


# <<< ADDED: Helper function to load image in thread >>>
def load_image_job(img_path):
    """Loads a single image in a separate thread."""
    try:
        img_pil = Image.open(img_path).convert("RGB")
        # Basic check for valid image dimensions
        w, h = img_pil.size
        if w <= 0 or h <= 0: raise ValueError(f"Invalid image dimensions ({w}x{h})")
        return img_pil, img_path # Return PIL image and its original path
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as img_e:
        # Print error but return None for image, keep path for tracking
        print(f"\nError opening image {os.path.basename(img_path)} in thread: {img_e}. Returning None.")
        return None, img_path
    except Exception as e: # Catch any other unexpected errors
        print(f"\nUnexpected error loading image {os.path.basename(img_path)} in thread: {e}")
        traceback.print_exc()
        return None, img_path


# v4.4.0: Added ThreadPoolExecutor for image loading
# --- Main Execution Logic ---
if __name__ == "__main__":
    args = parse_gen_args()

    # --- Setup Compute Precision ---
    COMPUTE_DTYPE = DEFAULT_PRECISION_MAP.get(args.precision, torch.float32)
    if COMPUTE_DTYPE != torch.float32 and TARGET_DEV == 'cpu':
        print(f"Warning: {args.precision} requested on CPU. Using float32.")
        COMPUTE_DTYPE = torch.float32
    if COMPUTE_DTYPE == torch.bfloat16 and (TARGET_DEV != 'cuda' or not torch.cuda.is_bf16_supported()):
         print("Warning: bf16 not supported. Using float32.")
         COMPUTE_DTYPE = torch.float32

    # --- Initialize Model & Preprocessing Info ---
    model_info = {} # Bundle to hold model, processor, transforms, type
    model_image_size = 0 # Needed for manual preprocessing

    # --- TIMM Model Handling ---
    if args.model_name.startswith("timm/"):
         print("Detected TIMM model name.")
         if args.preprocess_mode != 'dinov2_large_timm_fitpad':
             exit("Error: TIMM model requires 'dinov2_large_timm_fitpad' preprocess_mode.")
         try:
              import timm
              import timm.data
              print(f"Initializing TIMM Model: {args.model_name}...")
              timm_model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
              data_config = timm.data.resolve_model_data_config(timm_model)
              timm_input_size = data_config.get('input_size')
              if timm_input_size: model_image_size = timm_input_size[-1]
              else: model_image_size = 224 # Fallback
              print(f"  TIMM Input Size: {model_image_size}")
              timm_transforms = timm.data.create_transform(**data_config, is_training=False)
              timm_model = timm_model.to(device=TARGET_DEV, dtype=COMPUTE_DTYPE).eval()

              model_info = {"type": "timm", "model": timm_model, "transforms": timm_transforms}
         except ImportError:
             exit("Error: TIMM model requested but timm library is not available. Please install it via 'pip install timm'.")
         except Exception as e:
              print(f"Error initializing TIMM model {args.model_name}: {e}")
              exit(1)

    # --- Hugging Face Model Handling (SigLIP or FB DINOv2) ---
    else:
         print("Assuming Hugging Face model name.")
         try:
              # init_vision_model loads processor/model and determines image_size
              processor, vision_model, hf_model_image_size = init_vision_model(args.model_name, TARGET_DEV, COMPUTE_DTYPE)
              model_info = {"type": "hf", "processor": processor, "model": vision_model}
              model_image_size = hf_model_image_size # Use size from HF config/processor

              # Add specific check for DINOv2 FB size
              if args.preprocess_mode == 'dinov2_giant_fb_fitpad' and model_image_size != 518:
                   print(f"Warning: FB DINOv2 mode selected, but detected image size is {model_image_size} (expected 518). Using detected size.")
                   # Or force it? Let's use detected for now.
                   # model_image_size = 518

         except Exception as e:
             print(f"Error initializing Hugging Face model/processor: {e}")
             exit(1)

    # --- Determine Output Directory ---
    model_name_safe = args.model_name.split('/')[-1].replace('-', '_')
    # Updated Naming Map
    mode_suffix_map = {
        'fit_pad': "FitPad",              # HF SigLIP Manual FitPad
        'center_crop': "CenterCrop",      # HF SigLIP Manual CenterCrop
        # 'avg_crop': "AvgCrop",            # NYI
        'naflex_resize': "Naflex_Proc1024",
        'naflex_resize2': "Naflex_Proc2048",
        'dinov2_large_timm_fitpad': "FitPad",
        'dinov2_giant_fb_fitpad': "FitPad518", # Make FB DINOv2 distinct
        'aimv2_native_cls': "AIMv2CLS",       # <<< New Suffix >>>
        'dinov3_7b_8bit_bnb': "DINOv3_8bit_BnB" # <<< New Suffix for DINOv3 8-bit >>>
    }
    mode_suffix = mode_suffix_map.get(args.preprocess_mode, "UnknownMode")

    # Add prefix based on model source (remains same)
    model_prefix = ""
    if model_info["type"] == "timm": model_prefix = "timm_"
    elif model_info["type"] == "hf" and "dinov2" in args.model_name.lower(): model_prefix = "fb_" # DINOv2 models are typically FB
    elif model_info["type"] == "hf" and "aimv2" in args.model_name.lower(): model_prefix = "apple_" # Prefix for Apple AIMv2
    # <<< ADDED: Prefix for DINOv3 8-bit >>>
    elif model_info["type"] == "hf" and "dinov3" in args.model_name.lower(): model_prefix = "fb_" # DINOv3 models are typically FB

    output_subdir_name = f"{model_prefix}{model_name_safe}_{mode_suffix}{args.output_dir_suffix}"
    final_output_dir = os.path.join(args.output_dir_root, output_subdir_name)

    print(f"\nSelected Model: {args.model_name} ({model_info['type']} type)")
    print(f"Selected Preprocessing Mode: {args.preprocess_mode}")
    # <<< Updated Print Statements >>>
    if args.preprocess_mode == 'naflex_resize': print("  (Using HF Processor logic with target max_num_patches=1024)")
    if args.preprocess_mode == 'naflex_resize2': print("  (Using HF Processor logic with target max_num_patches=2048)")
    if args.preprocess_mode == 'dinov2_large_timm_fitpad': print(f"  (Using TIMM transforms with input size {model_image_size})")
    if args.preprocess_mode == 'dinov2_giant_fb_fitpad': print(f"  (Using Manual FitPad to size {model_image_size})")
    if args.preprocess_mode == 'aimv2_native_cls': print(f"  (Using HF Processor for native transforms, extracting CLS token)")
    if args.preprocess_mode == 'dinov3_7b_8bit_bnb': print(f"  (Using DINOv3 7B 8-bit BnB model, variable resolution (multiples of 16), mean pooling)") # <<< UPDATED Print Statement >>>
    print(f"Embeddings will be saved in: {final_output_dir}")

    # --- Process Source Folders (Main Loop with Threading) ---
    source_subfolders = sorted([d for d in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, d)) and not d.startswith('.')])
    if not source_subfolders: exit(f"Error: No valid subfolders found in {args.image_dir}")
    print(f"Found source subfolders: {source_subfolders}")

    total_processed_count = 0
    total_skipped_count = 0
    total_error_count = 0

    # <<< START THREADING LOGIC >>>
    with ThreadPoolExecutor(max_workers=NUM_LOAD_WORKERS) as executor:
        for src_folder in source_subfolders:
            current_image_dir = os.path.join(args.image_dir, src_folder)
            target_label = src_folder # Assuming folder name is the label/subfolder name
            current_output_subdir = os.path.join(final_output_dir, target_label)
            os.makedirs(current_output_subdir, exist_ok=True)
            print(f"\nProcessing Folder: {current_image_dir} -> {current_output_subdir}")

            # --- Collect all tasks for this folder ---
            tasks = [] # List of tuples: (output_path, input_path)
            skipped_in_folder = 0
            try:
                image_files = [f for f in os.listdir(current_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')) and not f.startswith('.')]
                for fname in image_files:
                    base_fname = os.path.splitext(fname)[0]
                    output_path = os.path.join(current_output_subdir, f"{base_fname}.npy")
                    img_path = os.path.join(current_image_dir, fname)
                    # Skip if output already exists
                    if os.path.exists(output_path):
                        skipped_in_folder += 1
                        continue
                    tasks.append((output_path, img_path))
            except OSError as e:
                print(f"  Error listing files in {current_image_dir}: {e}. Skipping folder.")
                continue

            total_skipped_count += skipped_in_folder
            if not tasks:
                 print(f"  All {len(image_files)} images already processed/skipped in this folder."); continue
            # --- End Task Collection ---

            processed_in_folder = 0
            error_in_folder = 0
            futures: List[Future] = []
            task_map = {} # Map future to output path for saving

            # Wrap folder processing in tqdm
            with tqdm(total=len(tasks), desc=f"Folder '{src_folder}'", unit="image", dynamic_ncols=True) as pbar:
                 # Submit initial batch of load jobs
                 for i in range(min(LOAD_QUEUE_SIZE, len(tasks))):
                     out_p, img_p = tasks[i]
                     future = executor.submit(load_image_job, img_p)
                     futures.append(future)
                     task_map[future] = out_p # Store output path with future

                 current_task_index = 0
                 # Process futures as they complete
                 while current_task_index < len(tasks):
                     # Wait for the oldest future to complete
                     # Using futures.pop(0) makes this FIFO
                     completed_future = futures.pop(0)
                     output_path_for_task = task_map.pop(completed_future) # Get corresponding output path

                     # Submit a new load job if there are more tasks
                     next_job_index = current_task_index + LOAD_QUEUE_SIZE
                     if next_job_index < len(tasks):
                         next_out_p, next_img_p = tasks[next_job_index]
                         new_future = executor.submit(load_image_job, next_img_p)
                         futures.append(new_future)
                         task_map[new_future] = next_out_p

                     # --- Process the completed image load job ---
                     loaded_path: Optional[str] = None # <<< MODIFIED: Initialize loaded_path with Optional type hint >>>
                     try:
                         loaded_image, loaded_path = completed_future.result() # Get result (PIL image, path)

                         if loaded_image is None: # Loading failed in thread
                             error_in_folder += 1
                         else: # Image loaded successfully, generate embedding
                             # Pass relevant args from command line and initialized models
                             embedding_result = get_embedding(
                                 raw_img_pil=loaded_image,
                                 preprocess_mode=args.preprocess_mode,
                                 model_info=model_info,
                                 device=TARGET_DEV,
                                 dtype=COMPUTE_DTYPE,
                                 model_image_size=model_image_size,
                                 filename=os.path.basename(loaded_path) if loaded_path else 'UNKNOWN', # Pass basename for logging
                                 naflex_max_patches=args.naflex_max_patches # Pass limit
                             )

                             # --- Save result ---
                             if embedding_result is not None and embedding_result.ndim == 1:
                                 try:
                                      np.save(output_path_for_task, embedding_result)
                                      processed_in_folder += 1
                                 except Exception as e_save:
                                      print(f"\nError saving {output_path_for_task}: {e_save}")
                                      error_in_folder += 1
                             elif embedding_result is not None: # Wrong shape
                                  print(f"Error: Embedding for {os.path.basename(loaded_path if loaded_path else 'UNKNOWN')} not 1D. Shape: {embedding_result.shape}. Skipping save.") # <<< MODIFIED: Handle None for loaded_path >>>
                                  error_in_folder += 1
                             else: # Generation failed (error printed in get_embedding)
                                  error_in_folder += 1
                             # Explicitly delete to free memory sooner?
                             del loaded_image, embedding_result

                     except Exception as e_proc: # Catch errors during result processing/embedding gen
                         print(f"\nError processing result for task {current_task_index} (path: {loaded_path if loaded_path else 'unknown'}): {e_proc}") # <<< FIXED: Check if loaded_path is not None >>>
                         traceback.print_exc()
                         error_in_folder += 1
                     finally:
                          pbar.update(1) # Update progress bar for the completed task index
                          current_task_index += 1
                          # Give GPU tiny breather? Unlikely needed.
                          # time.sleep(0.001)

            # --- End Folder Task Loop (tqdm) ---
            total_processed_count += processed_in_folder
            total_error_count += error_in_folder
            print(f"  Folder '{src_folder}' Summary: Processed: {processed_in_folder}, Skipped: {skipped_in_folder}, Errors: {error_in_folder}")
    # <<< END THREADING LOGIC WRAPPER >>>

    print("\n--- Overall Summary ---")
    print(f"Total Images Processed: {total_processed_count}")
    print(f"Total Images Skipped:   {total_skipped_count}")
    print(f"Total Errors:           {total_error_count}")
    print(f"Embeddings saved to:    {final_output_dir}")
    print("\nEmbedding generation complete!")
