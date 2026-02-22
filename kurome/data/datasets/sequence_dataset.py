# ruff: noqa
# sequence_dataset.py
# Version 1.1.0: Improved validation loader, __getitem__ robustness, clarified comments.

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler # Added Sampler
from collections import defaultdict
import random
from tqdm import tqdm
import math
import traceback # Added traceback

TARGET_LEN = 4096 # Define the fixed length globally or pass via init


# v1.3.0: Update collate function to handle masks
def collate_sequences(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    try:
        # Default collate handles dicts, will stack sequences and masks
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e_coll:
         print(f"\nError during collation: {e_coll}")
         traceback.print_exc()
         # Add shape printing for debugging
         for i, item in enumerate(batch):
             if isinstance(item, dict):
                 print(f" Item {i}:")
                 if 'sequence' in item and hasattr(item['sequence'], 'shape'): print(f"  seq shape: {item['sequence'].shape}")
                 if 'mask' in item and hasattr(item['mask'], 'shape'): print(f"  mask shape: {item['mask'].shape}")
                 if 'label' in item and hasattr(item['label'], 'shape'): print(f"  label shape: {item['label'].shape}")
         return None


# --- Main Dataset Class (Modified for Fixed Length) ---
class FeatureSequenceDataset(Dataset):
    """
    Dataset for loading pre-computed feature sequences saved as .npz files.
    ASSUMES sequences are already generated at a fixed length (TARGET_LEN).
    Handles train/val split and provides fixed-length sequences with masks.
    (Removed bucketing logic).
    """
    def __init__(self, feature_root_dir: str, validation_split_count: int = 0, seed: int = 42, preload: bool = False, preload_limit_gb: float = 10.0):
        """
        Args:
            feature_root_dir (str): Path to dir with class subfolders of .npz files.
            validation_split_count (int): Number of samples per class for validation.
            seed (int): Random seed for shuffling and splitting.
            preload (bool): Whether to preload features into RAM.
            preload_limit_gb (float): Approx RAM limit in GB for preloading.
        """
        print(f"Initializing FeatureSequenceDataset v1.2.0 (Fixed Length Mode) from: {feature_root_dir}")
        if not os.path.isdir(feature_root_dir):
             raise FileNotFoundError(f"Feature root directory not found: {feature_root_dir}")

        # <<< Store TARGET_LEN from constant >>>
        self.target_len = TARGET_LEN
        print(f"  Expecting sequences of fixed length: {self.target_len}")

        self.feature_root_dir = feature_root_dir
        self.validation_split_count = validation_split_count
        self.seed = seed
        self.preload = preload

        # Data storage
        # <<< Simplified metadata: No seq_len needed >>>
        self.metadata = [] # List of dicts: {'path': str, 'label': int}
        self.preloaded_data = None
        self.label_map = {}
        self.idx_to_label = {}
        self.num_labels = 0

        # <<< REMOVED Bucketing info >>>
        # self.buckets = defaultdict(list)
        # self.bucket_lengths = []

        # Train/Validation split (stores indices into self.metadata)
        self.train_indices = []
        self.val_indices = []

        # --- Initialization Steps ---
        self._scan_and_load_metadata() # Modified to skip length loading
        if not self.metadata: exit("Initialization failed: No valid feature files found.")

        if self.preload:
            self._preload_features(preload_limit_gb) # Modified to not need seq_len

        # <<< REMOVED _make_buckets() call >>>
        self._split_data() # Splits indices directly
        # --- End Initialization ---

        # Final print summary (Removed bucket info)
        print(f"Dataset Initialized.")
        print(f"  Total sequences found: {len(self.metadata)}")
        print(f"  Classes found: {self.num_labels} ({list(self.label_map.keys())})")
        print(f"  Training samples: {len(self.train_indices)}")
        print(f"  Validation samples: {len(self.val_indices)}")
        print(f"  Preloading enabled: {self.preload} ({len(self.preloaded_data) if self.preloaded_data else 0} items preloaded)")

    # <<< Modified _scan_and_load_metadata >>>
    def _scan_and_load_metadata(self):
        """Scans directories, finds .npz files, assigns labels."""
        print("Scanning for .npz feature files...")
        found_classes = sorted([d for d in os.listdir(self.feature_root_dir) if os.path.isdir(os.path.join(self.feature_root_dir, d)) and not d.startswith('.')])

        for class_folder in tqdm(found_classes, desc="Scanning Classes"):
            if not class_folder.isdigit():
                print(f"Warning: Skipping non-digit folder '{class_folder}'."); continue
            if class_folder not in self.label_map:
                 label_idx = self.num_labels; self.label_map[class_folder] = label_idx
                 self.idx_to_label[label_idx] = class_folder; self.num_labels += 1
            else: label_idx = self.label_map[class_folder]

            class_dir = os.path.join(self.feature_root_dir, class_folder)
            try: files_in_dir = os.listdir(class_dir)
            except OSError as e: print(f"Warning: Could not list files in {class_dir}: {e}. Skipping."); continue

            for filename in tqdm(files_in_dir, desc=f"Class {class_folder}", leave=False):
                if filename.lower().endswith(".npz") and not filename.startswith('.'):
                    filepath = os.path.join(class_dir, filename)
                    # <<< Store path and label ONLY >>>
                    # No need to open the file here just for length
                    self.metadata.append({'path': filepath, 'label': label_idx})

        print(f"Scan complete. Found {len(self.metadata)} candidate sequences.")

    # <<< Modified _preload_features >>>
    def _preload_features(self, limit_gb):
         """Attempts to preload feature sequences into RAM if enabled and within limit."""
         if not self.metadata: return
         print(f"Preloading requested. Estimating memory usage...")
         # Estimate assuming fixed length TARGET_LEN and fp16 save
         hidden_dim = 1024 # Assume this, or try to get from config if available
         bytes_per_elem = 2 # fp16
         estimated_size = len(self.metadata) * self.target_len * hidden_dim * bytes_per_elem
         estimated_gb = estimated_size / (1024**3)
         print(f"  Estimated RAM needed: {estimated_gb:.2f} GB (Based on {self.target_len} length, {hidden_dim} dim, fp16)")
         print(f"  Limit: {limit_gb:.2f} GB")

         if estimated_gb > limit_gb:
              print(f"  Estimated size exceeds limit. Preloading DISABLED.")
              self.preload = False; return

         print(f"  Preloading features into RAM...")
         self.preloaded_data = [None] * len(self.metadata)
         load_errors = 0
         loaded_bytes = 0
         for i, meta in enumerate(tqdm(self.metadata, desc="Preloading")):
              # Check limit dynamically
              if loaded_bytes / (1024**3) > limit_gb * 0.95: # Stop if near limit
                   print(f"\nApproaching RAM limit ({limit_gb} GB). Stopping preload.")
                   # Clear remaining Nones? Or just leave them? Leave them.
                   self.preloaded_data = self.preloaded_data[:i] # Trim list
                   break
              try:
                   with np.load(meta['path']) as data:
                        if 'sequence' not in data: raise KeyError("'sequence' key missing")
                        sequence_np = data['sequence']
                        # <<< Verify shape BEFORE storing >>>
                        if sequence_np.shape[0] != self.target_len:
                             # Maybe allow slight variations if padding handles it? For now, strict check.
                             print(f"Warning: Preload shape mismatch for {meta['path']}. Expected length {self.target_len}, got {sequence_np.shape[0]}. Skipping preload for this item.")
                             self.preloaded_data[i] = None # Don't preload wrong shape
                             continue
                        # Store as fp16 tensor on CPU
                        tensor_data = torch.from_numpy(sequence_np).to(torch.float16)
                        self.preloaded_data[i] = tensor_data
                        loaded_bytes += tensor_data.nelement() * tensor_data.element_size()
              except Exception as e:
                   print(f"Error preloading {meta['path']}: {e}")
                   self.preloaded_data[i] = None; load_errors += 1

         # Clean up None entries if stopped early or errors occurred
         loaded_count = sum(1 for item in self.preloaded_data if item is not None)
         self.preloaded_data = [item for item in self.preloaded_data if item is not None] # Filter Nones
         # We need a way to map original metadata index to new preloaded_data index if we filter...
         # This is getting complicated. Maybe just mark as None and handle in __getitem__? Yes.
         # Keep self.preloaded_data as [None] * len(self.metadata) and handle None check.

         if load_errors > 0: print(f"Warning: Failed to preload {load_errors} feature files.")
         print(f"Preloading complete. Loaded {loaded_count} sequences.")

    def _split_data(self):
        """Splits metadata indices into train/validation stratified by class."""
        print(f"Splitting data (Validation count per class: {self.validation_split_count})...")
        if self.validation_split_count <= 0:
            print("No validation split requested. Using all data for training.")
            self.train_indices = list(range(len(self.metadata)))
            self.val_indices = []
            random.seed(self.seed) # Seed shuffle even if no validation
            random.shuffle(self.train_indices)
            return

        random.seed(self.seed)
        metadata_by_class = defaultdict(list)
        for idx, meta in enumerate(self.metadata):
            metadata_by_class[meta['label']].append(idx) # Group indices by label

        self.train_indices = []
        self.val_indices = []

        print("Stratifying split by class:")
        for label_idx, indices in metadata_by_class.items():
            class_name = self.idx_to_label.get(label_idx, str(label_idx))
            num_items = len(indices)
            val_count = self.validation_split_count
            random.shuffle(indices) # Shuffle indices for this class

            if num_items <= val_count:
                print(f"  Class '{class_name}': {num_items} samples <= {val_count} validation count. Using all {num_items} for training.")
                self.train_indices.extend(indices)
            else:
                self.val_indices.extend(indices[:val_count])
                self.train_indices.extend(indices[val_count:])
                print(f"  Class '{class_name}': {num_items} samples -> {len(indices[val_count:])} train, {val_count} val.")

        # Final shuffle of training indices
        random.shuffle(self.train_indices)
        # Validation indices are kept in class-grouped order from split for potential analysis
        print(f"Split complete. Train indices: {len(self.train_indices)}, Val indices: {len(self.val_indices)}")

    def __len__(self):
        """Returns the number of training samples."""
        return len(self.train_indices) # Length based on train split

    def get_metadata(self, index):
         """Safely retrieves metadata for a given index."""
         if 0 <= index < len(self.metadata):
              return self.metadata[index]
         return None

    def get_item_by_metadata_index(self, metadata_index):
        """Loads data given an index into the main self.metadata list."""
        try:
            if not (0 <= metadata_index < len(self.metadata)):
                raise IndexError(f"Metadata index {metadata_index} out of bounds.")
            meta = self.metadata[metadata_index]

            # Check if preloaded
            if self.preload and self.preloaded_data is not None and metadata_index < len(self.preloaded_data) and self.preloaded_data[metadata_index] is not None:
                 sequence_tensor = self.preloaded_data[metadata_index].to(torch.float32) # Convert to fp32 for model input? Or keep fp16? Let's keep fp16.
                 sequence_tensor = self.preloaded_data[metadata_index]
            else: # Load from disk
                 filepath = meta['path']
                 try:
                     data = np.load(filepath)
                     sequence_np = data['sequence']
                     sequence_tensor = torch.from_numpy(sequence_np).to(torch.float16) # Load as fp16
                 except Exception as e_load:
                      print(f"Error loading npz file {filepath}: {e_load}")
                      return None

            label = meta['label']
            return {'sequence': sequence_tensor, 'label': torch.tensor(label, dtype=torch.long)}

        except IndexError as e_index: print(f"Error in get_item_by_metadata_index: {e_index}"); return None
        except Exception as e:
             meta_info = self.metadata[metadata_index] if 'metadata_index' in locals() and 0 <= metadata_index < len(self.metadata) else {'path': 'unknown'}
             print(f"Error processing metadata index {metadata_index} (path: {meta_info.get('path', 'unknown')}): {e}"); traceback.print_exc(); return None

    # <<< RESTORED Padding/Masking in __getitem__ >>>
    def __getitem__(self, index):
        metadata_index = -1
        try:
            # Get metadata index from training split
            if index >= len(self.train_indices): raise IndexError(f"Training index {index} out of bounds.")
            metadata_index = self.train_indices[index]
            if metadata_index >= len(self.metadata): raise IndexError(f"Metadata index {metadata_index} out of bounds.")
            meta = self.metadata[metadata_index]

            # --- Load sequence tensor (fp16) ---
            sequence_tensor = None
            # Try preloaded data first
            if self.preload and self.preloaded_data is not None and \
               metadata_index < len(self.preloaded_data) and self.preloaded_data[metadata_index] is not None:
                 sequence_tensor = self.preloaded_data[metadata_index]
            # Load from disk if not preloaded or preload failed
            if sequence_tensor is None:
                 filepath = meta['path']
                 try:
                     with np.load(filepath) as data:
                         if 'sequence' not in data: raise KeyError("'sequence' key missing")
                         sequence_np = data['sequence']
                         # Shape check is now less critical here, as we pad/truncate next
                         sequence_tensor = torch.from_numpy(sequence_np).to(torch.float16)
                 except Exception as e_load:
                      print(f"Error loading npz file {filepath}: {e_load}")
                      return None # Skip this item

            # --- Pad/Truncate and Create Mask ---
            current_len = sequence_tensor.shape[0]
            features_dim = sequence_tensor.shape[1] # Assume consistent feature dim
            # Initialize tensors for final fixed length
            final_sequence = torch.zeros((self.target_len, features_dim), dtype=torch.float16)
            attention_mask = torch.zeros(self.target_len, dtype=torch.bool) # False = padding

            if current_len == 0:
                print(f"Warning: Empty sequence tensor for {meta.get('path', 'unknown')}. Returning padded zeros.")
                # final_sequence and attention_mask are already zeros
            elif current_len > self.target_len:
                # Truncate if longer than target
                print(f"Warning: Sequence length {current_len} > target {self.target_len} for {meta.get('path', 'unknown')}. Truncating.")
                final_sequence = sequence_tensor[:self.target_len, :]
                attention_mask[:] = True # All are real tokens up to target_len
            else: # current_len <= self.target_len (Pad)
                final_sequence[:current_len, :] = sequence_tensor # Copy sequence data
                attention_mask[:current_len] = True # Mark real tokens as True

            label = meta['label']
            # Return sequence [target_len, F], mask [target_len], label
            return {
                'sequence': final_sequence.float(), # Convert final seq to Float32 for model input
                'mask': attention_mask,
                'label': torch.tensor(label, dtype=torch.long)
            }

        except IndexError as e_index: print(f"Error in __getitem__ with index {index}: {e_index}"); return None
        except Exception as e:
             path_info = self.metadata[metadata_index].get('path', 'unknown') if metadata_index != -1 and metadata_index < len(self.metadata) else 'unknown'
             print(f"Error processing data index {index} (metadata index {metadata_index}, path: {path_info}): {e}"); traceback.print_exc(); return None

    # --- get_validation_loader (Modified to use Correct SubDataset) ---
    def get_validation_loader(self, batch_size, num_workers=0, prefetch_factor=2):
        print("Creating validation loader (Fixed Length Mode)...")
        if not self.val_indices: print("Validation set is empty."); return None

        # <<< Use ValidationSubDatasetFeaturesPadded >>>
        val_dataset = ValidationSubDatasetFeaturesPadded(
            self.metadata,           # Pass full metadata list
            self.val_indices,        # Pass only validation indices
            self.preload,            # Pass preload flag
            self.preloaded_data,     # Pass preloaded data (or None)
            self.target_len          # Pass the target length
        )
        if len(val_dataset) == 0: print("Warning: ValidationSubDataset is empty."); return None

        val_loader = DataLoader(
             val_dataset, batch_size=batch_size, shuffle=False,
             num_workers=num_workers, collate_fn=collate_sequences, # Use same collate
             persistent_workers=(num_workers > 0), prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        return val_loader


# --- Validation SubDataset (Keep the Padded version) ---
class ValidationSubDatasetFeaturesPadded(Dataset):
    def __init__(self, metadata_list, validation_indices, preload, preloaded_data, target_len):
        self.metadata = metadata_list;
        self.val_indices = validation_indices
        self.preload = preload;
        self.preloaded_data = preloaded_data
        self.target_len = target_len
        print(f"ValidationSubDatasetFeaturesPadded initialized with {len(self.val_indices)} indices.")

    def __len__(self): return len(self.val_indices)

    # <<< RESTORED Padding/Masking in __getitem__ >>>
    def __getitem__(self, index):
        metadata_index = -1
        try:
            # Get metadata index from validation list
            if index >= len(self.val_indices): raise IndexError(...)
            metadata_index = self.val_indices[index]
            if metadata_index >= len(self.metadata): raise IndexError(...)
            meta = self.metadata[metadata_index]

            # Load sequence tensor (fp16) - Reuse logic from main dataset
            sequence_tensor = None
            if self.preload and self.preloaded_data is not None and \
                    metadata_index < len(self.preloaded_data) and self.preloaded_data[metadata_index] is not None:
                sequence_tensor = self.preloaded_data[metadata_index]
            if sequence_tensor is None:
                filepath = meta['path']
                try:
                    with np.load(filepath) as data:
                        if 'sequence' not in data: raise KeyError(...)
                        sequence_np = data['sequence']
                        sequence_tensor = torch.from_numpy(sequence_np).to(torch.float16)
                except Exception as e_load:
                    print(f"Error loading VAL npz {filepath}: {e_load}"); return None

            # Pad/Truncate and Create Mask (Same logic as main dataset __getitem__)
            current_len = sequence_tensor.shape[0]
            features_dim = sequence_tensor.shape[1]
            final_sequence = torch.zeros((self.target_len, features_dim), dtype=torch.float16)
            attention_mask = torch.zeros(self.target_len, dtype=torch.bool)  # False = padding
            if current_len == 0:
                pass  # Already zeros
            elif current_len > self.target_len:
                final_sequence = sequence_tensor[:self.target_len, :]
                attention_mask[:] = True
            else:  # Pad
                final_sequence[:current_len, :] = sequence_tensor
                attention_mask[:current_len] = True

            label = meta['label']
            return {
                'sequence': final_sequence.float(),  # Convert final seq to Float32
                'mask': attention_mask,
                'label': torch.tensor(label, dtype=torch.long)
            }

        except IndexError as e_index:
            print(f"Error in validation __getitem__: {e_index}"); return None
        except Exception as e:
            path_info = self.metadata[metadata_index].get('path',
                                                          'unknown') if metadata_index != -1 and metadata_index < len(
                self.metadata) else 'unknown'
            print(
                f"Error processing validation data index {index} (metadata index {metadata_index}, path: {path_info}): {e}")
            traceback.print_exc()
            return None


# # --- New Validation SubDataset ---
# # Needs to replicate the pooling logic from __getitem__
# class ValidationSubDatasetFeaturesPooled(Dataset):
#      def __init__(self, metadata_list, validation_indices, preload, preloaded_data):
#           self.metadata = metadata_list
#           self.val_indices = validation_indices
#           self.preload = preload
#           self.preloaded_data = preloaded_data
#           print(f"ValidationSubDatasetFeaturesPooled initialized with {len(self.val_indices)} indices.")
#
#      def __len__(self):
#           return len(self.val_indices)
#
#      def __getitem__(self, index):
#           metadata_index = -1 # Initialize
#           try:
#                if index >= len(self.val_indices):
#                     raise IndexError(f"Validation index {index} out of bounds for val_indices (len {len(self.val_indices)})")
#                metadata_index = self.val_indices[index] # Get index into main metadata
#
#                if metadata_index >= len(self.metadata):
#                     raise IndexError(f"Validation metadata index {metadata_index} out of bounds for metadata (len {len(self.metadata)})")
#                meta = self.metadata[metadata_index]
#
#                # Load sequence tensor (fp16 on CPU or preloaded) - Reuse logic from main dataset
#                if self.preload and self.preloaded_data is not None and metadata_index < len(self.preloaded_data) and self.preloaded_data[metadata_index] is not None:
#                    sequence_tensor = self.preloaded_data[metadata_index]
#                else:
#                    filepath = meta['path']
#                    try:
#                        with np.load(filepath) as data: sequence_np = data['sequence']
#                        sequence_tensor = torch.from_numpy(sequence_np).to(torch.float16)
#                    except Exception as e_load:
#                        print(f"Error loading validation npz file {filepath}: {e_load}")
#                        return None
#
#                # Apply Mean Pooling
#                if sequence_tensor.numel() == 0:
#                     print(f"Warning: Empty sequence tensor encountered for validation metadata index {metadata_index}, path {meta.get('path', 'unknown')}. Returning None.")
#                     return None
#                pooled_features = torch.mean(sequence_tensor.float(), dim=0) # Pool, convert to float32
#
#                label = meta['label']
#                return {'sequence': pooled_features, 'label': torch.tensor(label, dtype=torch.long)}
#
#           except IndexError as e_index:
#                print(f"Error in validation __getitem__: {e_index}")
#                return None
#           except Exception as e:
#                path_info = 'unknown'
#                metadata_index_info = metadata_index if 'metadata_index' in locals() else 'unknown'
#                if isinstance(metadata_index_info, int) and metadata_index_info < len(self.metadata):
#                     path_info = self.metadata[metadata_index_info].get('path', 'unknown')
#                print(f"Error processing validation data for index {index} (metadata index {metadata_index_info}, path: {path_info}): {e}")
#                traceback.print_exc()
#                return None
#
#
# # --- Simple Validation Dataset Wrapper ---
# # (Assumes indices map to the main dataset's metadata)
# class ValidationSubDatasetFeatures(Dataset):
#      def __init__(self, metadata_list, validation_indices):
#           self.metadata = metadata_list
#           self.val_indices = validation_indices
#           print(f"ValidationSubDataset initialized with {len(self.val_indices)} indices.")
#
#      def __len__(self):
#           return len(self.val_indices)
#
#      def __getitem__(self, index):
#           metadata_index = -1 # Initialize
#           try:
#                if index >= len(self.val_indices):
#                     raise IndexError(f"Validation index {index} out of bounds for val_indices (len {len(self.val_indices)})")
#                metadata_index = self.val_indices[index] # Get index into main metadata
#                if metadata_index >= len(self.metadata):
#                     raise IndexError(f"Validation metadata index {metadata_index} out of bounds for metadata (len {len(self.metadata)})")
#                meta = self.metadata[metadata_index]
#                filepath = meta['path']
#                label = meta['label']
#
#                try:
#                    data = np.load(filepath)
#                    sequence_np = data['sequence']
#                except Exception as e_load:
#                     print(f"Error loading validation npz file {filepath}: {e_load}")
#                     return None
#
#                sequence_tensor = torch.from_numpy(sequence_np).to(torch.float16)
#                return {'sequence': sequence_tensor, 'label': torch.tensor(label, dtype=torch.long)}
#
#           except IndexError as e_index:
#                print(f"Error in validation __getitem__: {e_index}")
#                return None
#           except Exception as e:
#                path_info = 'unknown'
#                if metadata_index != -1 and metadata_index < len(self.metadata):
#                     path_info = self.metadata[metadata_index].get('path', 'unknown')
#                print(f"Error processing validation data for index {index} (metadata index {metadata_index}, path: {path_info}): {e}")
#                traceback.print_exc()
#                return None

# class BucketBatchSampler(Sampler[list[int]]):
#     """
#     Sampler that yields batches of indices, ensuring all indices in a batch
#     belong to the same bucket (same sequence length).
#
#     Args:
#         buckets (dict[int, list[int]]): Dictionary mapping sequence length to list of metadata indices.
#         batch_size (int): The desired batch size.
#         drop_last (bool): If True, drop the last incomplete batch from each bucket.
#         shuffle_buckets (bool): If True, shuffle the order of buckets each epoch.
#         shuffle_within_bucket (bool): If True, shuffle indices within each bucket each epoch.
#         seed (int): Random seed for shuffling.
#     """
#     def __init__(self, buckets: dict[int, list[int]], batch_size: int, drop_last: bool,
#                  shuffle_buckets: bool = True, shuffle_within_bucket: bool = True, seed: int = 42):
#         # Pass None as data_source since we don't use it directly here
#         super().__init__(data_source=None)
#
#         if not isinstance(batch_size, int) or batch_size <= 0:
#             raise ValueError("batch_size should be a positive integer value")
#         if not isinstance(drop_last, bool):
#             raise ValueError("drop_last should be a boolean value")
#
#         self.buckets = buckets
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.shuffle_buckets = shuffle_buckets
#         self.shuffle_within_bucket = shuffle_within_bucket
#         self.seed = seed
#         self.epoch = 0
#
#         # Pre-calculate batches for length calculation
#         self.batches_per_bucket = {}
#         self.num_batches = 0
#         self.bucket_keys = sorted(list(self.buckets.keys())) # Store sorted keys
#
#         for seq_len in self.bucket_keys:
#             bucket_indices = self.buckets[seq_len]
#             num_samples_in_bucket = len(bucket_indices)
#             if num_samples_in_bucket == 0:
#                  self.batches_per_bucket[seq_len] = 0
#                  continue
#
#             num_batches_for_bucket = num_samples_in_bucket // self.batch_size
#             if not self.drop_last and num_samples_in_bucket % self.batch_size != 0:
#                 num_batches_for_bucket += 1
#
#             self.batches_per_bucket[seq_len] = num_batches_for_bucket
#             self.num_batches += num_batches_for_bucket
#
#         print(f"BucketBatchSampler initialized. Total batches per epoch: {self.num_batches}")
#
#     def __iter__(self):
#         # Use epoch number and seed for deterministic shuffling across epochs/resumes
#         g = torch.Generator()
#         g.manual_seed(self.seed + self.epoch)
#
#         # Determine order of buckets
#         bucket_processing_order = self.bucket_keys
#         if self.shuffle_buckets:
#             # Shuffle the keys using torch generator for reproducibility
#             rand_indices = torch.randperm(len(bucket_processing_order), generator=g).tolist()
#             bucket_processing_order = [self.bucket_keys[i] for i in rand_indices]
#
#         # Generate batches for this epoch
#         all_batches = []
#         for seq_len in bucket_processing_order:
#             bucket_indices = self.buckets[seq_len][:] # Get a copy
#             if not bucket_indices: continue # Skip empty buckets
#
#             if self.shuffle_within_bucket:
#                 # Shuffle indices within the bucket using torch generator
#                 rand_indices = torch.randperm(len(bucket_indices), generator=g).tolist()
#                 bucket_indices = [bucket_indices[i] for i in rand_indices]
#
#             # Create mini-batches for this bucket
#             for i in range(0, len(bucket_indices), self.batch_size):
#                 batch = bucket_indices[i : i + self.batch_size]
#                 if len(batch) < self.batch_size and self.drop_last:
#                     continue # Skip last incomplete batch if drop_last is True
#                 all_batches.append(batch)
#
#         # Shuffle the final list of batches (optional, batches are already from shuffled buckets)
#         # random.shuffle(all_batches) # Maybe not needed if buckets are shuffled
#
#         self.epoch += 1 # Increment epoch for next iteration's seed
#         return iter(all_batches)
#
#     def __len__(self) -> int:
#         return self.num_batches
#
#     def set_epoch(self, epoch: int) -> None:
#         # Allows DistributedSampler compatibility if needed later
#         self.epoch = epoch
