# Version 1.1.0: Added skipping existing files and basic threading for I/O.
import math
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import traceback
# <<< Add threading imports >>>
from concurrent.futures import ThreadPoolExecutor, Future
import time

try:
    from transformers import AutoProcessor, AutoModel, PretrainedConfig
    HF_AVAILABLE = True
except ImportError:
    print("Error: transformers library not found. Please install it.")
    HF_AVAILABLE = False
    exit(1)

# # --- Add AIMv2 import ---
# try:
#     from aim.v2.utils import load_pretrained
#     # We also need the specific transforms for AIMv2 Native
#     from aim.v1.torch.data import val_transforms as aim_val_transforms
#     AIM_AVAILABLE = True
# except ImportError:
#     print("Warning: Could not import AIMv1/AIMv2 utils/transforms.")
#     print("Ensure packages are installed: ")
#     print("  pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v1'")
#     print("  pip install 'git+https://github.com/apple/ml-aim.git#subdirectory=aim-v2'")
#     AIM_AVAILABLE = False

# --- HF Import ---
try:
    from transformers import AutoProcessor, AutoModel, PretrainedConfig
    HF_AVAILABLE = True
except ImportError:
    print("Error: transformers library not found. Please install it.")
    HF_AVAILABLE = False
    # Exit or handle gracefully if transformers is essential
    exit(1)


# --- Constants ---
TARGET_DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp') # Use tuple
DEFAULT_PRECISION_MAP = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
SAVE_PRECISION_MAP = {'fp32': torch.float32, 'fp16': torch.float16}
# <<< Threading config >>>
NUM_LOAD_WORKERS = 4 # Number of threads for loading images
LOAD_QUEUE_SIZE = 8  # How many images to keep loaded ahead of time

# --- Argument Parsing (remains same) ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate feature sequences from a vision model.")
    parser.add_argument('--image_dir', required=True, help="Directory containing class subfolders (e.g., '0', '1') with images.")
    parser.add_argument('--output_dir_root', required=True, help="Root directory to save feature sequences (e.g., 'data').")
    parser.add_argument('--model_name', required=True, type=str, help="Base vision model name (Hugging Face ID).")
    parser.add_argument('--compute_precision', type=str, default='bf16', choices=['fp32', 'bf16', 'fp16'], help="Precision for model computation (default: bf16).")
    parser.add_argument('--save_precision', type=str, default='fp16', choices=['fp32', 'fp16'], help="Precision for saving features (default: fp16).")
    parser.add_argument('--output_subdir', type=str, default=None, help="Optional specific subdirectory name under output_dir_root.")
    args = parser.parse_args()
    return args

# --- Model and Processor Loading (Simplified to use AutoModel/AutoProcessor always) ---
def load_model_and_processor(model_name, device, compute_dtype):
    model = None
    processor = None
    hidden_size = None
    is_aim_native = "native" in model_name # Keep flag based on name for potential logic differences

    print(f"Loading model: {model_name} on {device} with dtype {compute_dtype} using AutoClasses...")

    try:
        # Load Processor using AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        print(f"Loaded Processor: {type(processor).__name__}")
        if hasattr(processor, 'image_processor'):
             print(f"  Processor Config: resize={getattr(processor.image_processor, 'do_resize', 'N/A')}, crop={getattr(processor.image_processor, 'do_center_crop', 'N/A')}")

        # Load Model using AutoModel
        # For AIMv2, trust_remote_code=True is essential for now
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            # Request hidden states if we ever need intermediate layers
            # For last_hidden_state, output_hidden_states=False is usually sufficient
            output_hidden_states=False
        ).to(device).eval()
        print(f"Loaded Model: {model.__class__.__name__}")

        # Get hidden dim from config
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        else: # Fallback attempts
             config = PretrainedConfig.from_pretrained(model_name, trust_remote_code=True)
             hidden_size = getattr(config, 'hidden_size', None)
             if hidden_size is None and hasattr(config, 'vision_config'): hidden_size = getattr(config.vision_config, 'hidden_size', None)
             if hidden_size is None: print("Warning: Could not automatically determine hidden size.")

    except Exception as e:
        print(f"Error loading model/processor {model_name} via AutoClasses: {e}")
        import traceback; traceback.print_exc()
        raise # Re-raise after printing

    if model is None or processor is None:
        exit(f"Failed to load model or processor for {model_name}.")

    if hidden_size: print(f"Detected hidden size: {hidden_size}")
    return model, processor, hidden_size, is_aim_native

# <<< Helper function to load image in thread >>>
def load_image_job(img_path):
    try:
        img_pil = Image.open(img_path).convert("RGB")
        # Perform basic check early
        w, h = img_pil.size
        if w <= 0 or h <= 0: raise ValueError("Invalid image dimensions")
        return img_pil, img_path # Return path too for matching
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as img_e:
        print(f"\nError opening image {os.path.basename(img_path)} in thread: {img_e}. Returning None.")
        return None, img_path # Return None on error, but keep path

# --- Main Processing Function ---
@torch.no_grad()
def generate_features(args, model, processor, device, compute_dtype, save_dtype, is_aim_native):

    # --- Determine Output Directory ---
    if args.output_subdir:
        output_subdir_name = args.output_subdir
    else:
        model_name_safe = args.model_name.split('/')[-1].replace('-', '_')
        save_prec_str = "fp16" if save_dtype == torch.float16 else "fp32"
        output_subdir_name = f"{model_name_safe}_SeqFeatures_{save_prec_str}"

    final_output_dir = os.path.join(args.output_dir_root, output_subdir_name)
    print(f"\nFeatures will be saved in: {final_output_dir}")
    print(f"Compute Precision: {compute_dtype}, Save Precision: {save_dtype}")

    # --- Find Image Files and Create Task List (Scan ONCE) ---
    all_image_tasks = [] # List of tuples: (output_path, class_label, fname, img_path)
    print("Scanning image directory and creating task list...")
    try:
        source_subfolders = sorted([d for d in os.listdir(args.image_dir) if os.path.isdir(os.path.join(args.image_dir, d)) and not d.startswith('.')])
        if not source_subfolders: exit(f"Error: No valid subfolders found in: {args.image_dir}")
        print(f"Found source subfolders: {source_subfolders}")

        for class_label in source_subfolders: # Use class_label directly
            current_image_dir = os.path.join(args.image_dir, class_label)
            current_output_subdir = os.path.join(final_output_dir, class_label)
            os.makedirs(current_output_subdir, exist_ok=True) # Create output dir

            try:
                image_files = [f for f in os.listdir(current_image_dir) if f.lower().endswith(IMAGE_EXTS) and not f.startswith('.')]
                for fname in sorted(image_files):
                    img_path = os.path.join(current_image_dir, fname)
                    base_fname = os.path.splitext(fname)[0]
                    output_path = os.path.join(current_output_subdir, f"{base_fname}.npz")
                    all_image_tasks.append((output_path, class_label, fname, img_path))
            except OSError as e:
                print(f"Warning: Could not access files in {current_image_dir}: {e}")

    except Exception as e:
         exit(f"Error scanning image directory {args.image_dir}: {e}")

    total_images = len(all_image_tasks)
    if total_images == 0: exit("Error: No images found in any accessible subfolder.")
    # <<< Print the CORRECT total count ONCE >>>
    print(f"Found {total_images} total images to process.")

    # --- Process Images with ThreadPoolExecutor ---
    processed_count = 0
    skipped_count = 0
    error_count = 0
    futures = []

    with ThreadPoolExecutor(max_workers=NUM_LOAD_WORKERS) as executor, \
         tqdm(total=total_images, desc="Generating Features", unit="image") as pbar:

        # Submit initial batch of load jobs
        for i in range(min(LOAD_QUEUE_SIZE, total_images)):
             output_path, _, _, img_path = all_image_tasks[i]
             if os.path.exists(output_path):
                  skipped_count += 1; pbar.update(1); futures.append(None)
             else:
                  futures.append(executor.submit(load_image_job, img_path))

        current_task_index = 0
        while current_task_index < total_images:
            # Get result from the oldest future
            future = futures.pop(0)
            output_path, class_label, fname, img_path = all_image_tasks[current_task_index]

             # Submit a new load job if needed (checking for existing files)
            # ... (same as before) ...
            next_job_index = current_task_index + LOAD_QUEUE_SIZE
            if next_job_index < total_images:
                 next_output_path, _, _, next_img_path = all_image_tasks[next_job_index]
                 if os.path.exists(next_output_path):
                      skipped_count += 1; pbar.update(1); futures.append(None)
                 else:
                      futures.append(executor.submit(load_image_job, next_img_path))


            # Process the current image if it wasn't skipped and loaded correctly
            if future is not None:
                 try:
                      img_pil, _ = future.result() # Wait for load

                      if img_pil is None: # Loading failed
                           error_count += 1
                      else: # Image loaded successfully
                           original_width, original_height = img_pil.size

                           # <<< --- START RESIZING LOGIC --- >>>
                           TARGET_MAX_PATCHES = 4096
                           PATCH_SIZE = 14
                           img_to_process = img_pil # Default to original

                           # Calculate initial patch count only if needed
                           patches_w_initial = math.floor(original_width / PATCH_SIZE)
                           patches_h_initial = math.floor(original_height / PATCH_SIZE)
                           total_patches_initial = patches_w_initial * patches_h_initial

                           if total_patches_initial > TARGET_MAX_PATCHES:
                               scale_factor = math.sqrt(TARGET_MAX_PATCHES / total_patches_initial)
                               resize_needed = True
                               # Iterate shrinking scale factor slightly to ensure fit
                               max_iterations = 10; iterations = 0
                               while iterations < max_iterations:
                                   target_w = int(original_width * scale_factor + 0.5)
                                   target_h = int(original_height * scale_factor + 0.5)
                                   if target_w < 1: target_w = 1
                                   if target_h < 1: target_h = 1

                                   new_patches_w = math.floor(target_w / PATCH_SIZE)
                                   new_patches_h = math.floor(target_h / PATCH_SIZE)

                                   if new_patches_w * new_patches_h <= TARGET_MAX_PATCHES:
                                        print(f"  - Resizing {fname} ({original_width}x{original_height}, {total_patches_initial}p) -> ({target_w}x{target_h}, {new_patches_w * new_patches_h}p)")
                                        img_to_process = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                                        resize_needed = False # Flag that resize succeeded
                                        break # Exit loop

                                   scale_factor *= 0.995 # Shrink scale factor
                                   iterations += 1
                               # End while loop

                               if resize_needed: # If loop finished without success
                                    print(f"\nWarning: Could not find suitable resize dimensions for {fname}. Skipping.")
                                    error_count += 1; pbar.update(1); current_task_index += 1; continue # Skip to next image index
                           # <<< --- END RESIZING LOGIC --- >>>


                           # 2. Preprocess Image (using potentially resized img_to_process)
                           inputs = processor(images=[img_to_process], return_tensors="pt").to(device)

                           # 3. Run Model Forward Pass
                           with torch.amp.autocast(device_type=device, enabled=(compute_dtype != torch.float32), dtype=compute_dtype):
                               inputs = processor(images=[img_to_process], return_tensors="pt").to(device) # Make sure inputs are created correctly
                               outputs = model(**inputs) # Call the model

                               # <<< --- DEBUG PRINT --- >>>
                               print(f"\nDEBUG: Output type for {fname}: {type(outputs)}")
                               output_attributes = {}
                               if hasattr(outputs, 'keys'): # For dict-like outputs (e.g., BaseModelOutputWithPooling)
                                   print(f"DEBUG: Output keys: {outputs.keys()}")
                                   for key in outputs.keys():
                                       value = outputs[key]
                                       if isinstance(value, torch.Tensor):
                                           output_attributes[key] = value.shape
                                       else:
                                           output_attributes[key] = type(value)
                               elif isinstance(outputs, torch.Tensor): # If output is just a tensor
                                    output_attributes['output_tensor'] = outputs.shape
                               elif isinstance(outputs, tuple): # If output is a tuple
                                    for i, item in enumerate(outputs):
                                         if isinstance(item, torch.Tensor):
                                              output_attributes[f'tuple_item_{i}'] = item.shape
                                         else:
                                              output_attributes[f'tuple_item_{i}'] = type(item)
                               else: # Fallback: Use dir()
                                   print(f"DEBUG: dir(outputs): {dir(outputs)}")
                                   # Try checking common attributes specifically
                                   if hasattr(outputs, 'pooler_output') and isinstance(outputs.pooler_output, torch.Tensor):
                                        output_attributes['pooler_output'] = outputs.pooler_output.shape
                                   if hasattr(outputs, 'last_hidden_state') and isinstance(outputs.last_hidden_state, torch.Tensor):
                                        output_attributes['last_hidden_state'] = outputs.last_hidden_state.shape

                               print(f"DEBUG: Output Attributes & Shapes: {output_attributes}")
                               # <<< --- END DEBUG PRINT --- >>>

                               # Determine feature_sequence based on availability (keep using last_hidden_state for now)
                               feature_sequence = None
                               if hasattr(outputs, 'last_hidden_state') and isinstance(outputs.last_hidden_state, torch.Tensor):
                                    feature_sequence = outputs.last_hidden_state
                                    if 'pooler_output' in output_attributes:
                                         print(f"INFO: pooler_output exists (Shape: {output_attributes['pooler_output']}), but using last_hidden_state for now.")
                                    else:
                                         print(f"INFO: Using last_hidden_state (Shape: {output_attributes.get('last_hidden_state', 'N/A')}). No pooler_output found.")
                               else:
                                    print(f"\nError: Could not find 'last_hidden_state' in model outputs for {fname}. Skipping.")
                                    error_count += 1; pbar.update(1); current_task_index += 1; continue

                           # 4. Process Output
                           if feature_sequence is None or feature_sequence.ndim != 3 or feature_sequence.shape[0] != 1:
                                print(f"\nError: Invalid feature sequence output for {fname}. Skipping.")
                                error_count += 1; pbar.update(1); current_task_index += 1; continue # Make sure to continue loop correctly
                           else:
                                # --- Normalize the features BEFORE saving ---
                                # Input feature_sequence shape: [1, NumPatches, Features], on device, compute_dtype
                                try:
                                    feature_sequence = F.normalize(feature_sequence, p=2, dim=-1) # Normalize along the feature dimension (-1)
                                    # Optional check for NaNs after normalization
                                    if torch.isnan(feature_sequence).any():
                                         print(f"\nWarning: NaN detected in features for {fname} *after* normalization. Skipping save.")
                                         error_count += 1; pbar.update(1); current_task_index += 1; continue
                                except Exception as e_norm:
                                     print(f"\nError during normalization for {fname}: {e_norm}. Skipping.")
                                     error_count += 1; pbar.update(1); current_task_index += 1; continue
                                # --- End Normalization ---

                                # Now proceed with the original steps on the *normalized* tensor
                                feature_sequence = feature_sequence.squeeze(0).detach().to(dtype=save_dtype).cpu() # Convert normalized tensor to save_dtype (fp16) and move to CPU

                                # 5. Save Normalized Features to NPZ
                                feature_array_np = feature_sequence.numpy() # Convert normalized CPU tensor to numpy
                                original_shape_np = np.array([original_height, original_width], dtype=np.int32) # Keep original shape info
                                np.savez_compressed(output_path, sequence=feature_array_np, original_shape=original_shape_np) # Save normalized sequence
                                processed_count += 1
                                # Successfully processed and saved normalized features

                 except Exception as e:
                      print(f"\nError processing file {fname} during GPU/Save stage: {e}")
                      traceback.print_exc()
                      error_count += 1
                 finally:
                      try: del img_pil, img_to_process, inputs, outputs, feature_sequence, feature_array_np
                      except NameError: pass
                      # if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Update progress bar for the completed task index
            pbar.update(1)
            current_task_index += 1

            # Give GPU a tiny breather? Unlikely needed.
            # time.sleep(0.001)

    # --- Summary ---
    print("\n--- Feature Generation Summary ---")
    print(f"Found existing/skipped: {skipped_count} images")
    print(f"Successfully generated:   {processed_count} images")
    print(f"Failed during process:  {error_count} images")
    print(f"Total images checked:   {skipped_count + processed_count + error_count} / {total_images}")
    print(f"Features saved to: {final_output_dir}")


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()

    # --- Setup Compute Precision ---
    COMPUTE_DTYPE = DEFAULT_PRECISION_MAP.get(args.compute_precision, torch.float32)
    if COMPUTE_DTYPE != torch.float32 and TARGET_DEV == 'cpu':
        print(f"Warning: {args.compute_precision} requested on CPU. Using float32.")
        COMPUTE_DTYPE = torch.float32
    if COMPUTE_DTYPE == torch.bfloat16 and (TARGET_DEV != 'cuda' or not torch.cuda.is_bf16_supported()):
         print(f"Warning: {args.compute_precision} (bf16) not supported. Using float32.")
         COMPUTE_DTYPE = torch.float32
    if COMPUTE_DTYPE == torch.float16 and TARGET_DEV != 'cuda':
         # Allow fp16 on CPU for testing? Might be slow. Let's default to float32.
         print(f"Warning: {args.compute_precision} (fp16) requested on CPU. Using float32.")
         COMPUTE_DTYPE = torch.float32

    # --- Setup Save Precision ---
    SAVE_DTYPE = SAVE_PRECISION_MAP.get(args.save_precision, torch.float16) # Default fp16

    # --- Load Model & Processor ---
    model, processor, _, is_aim_native = load_model_and_processor(args.model_name, TARGET_DEV, COMPUTE_DTYPE)

    # --- Generate Features ---
    generate_features(args, model, processor, TARGET_DEV, COMPUTE_DTYPE, SAVE_DTYPE, is_aim_native)

    print("\nFeature sequence generation complete!")