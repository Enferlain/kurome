# ruff: noqa
# Custom dataset to load CLIP embeddings from disk.
#  Files should contain embeddings as (1, E) or (E)

######## Folder Layout ########
#  Data                       #
#   |- CLIP                   #
#   |   |- test.npy <= eval   #
#   |   |- 0_test.npy <= cls  #
#   |   |- 01                 #
#   |   |   |- 000001.npy     #
#   |   |   |- 000002.npy     #
#   |   |   |   ...           #
#   |   |   |- 000999.npy     #
#   |   |   \- 001000.npy     #
#   |   |- 02_optional_name   #
#   |   |  ...                #
#   |   |- 09                 #
#   |   \- 10                 #
#   |- META <= other versions #
#     ...                     #
###############################

# dataset.py
# Version 2.0.1: Fixed preload error handling, mutable default, and getitem error path

import os
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import random
from collections import defaultdict # Added defaultdict

DEFAULT_ROOT = "data"
ALLOWED_EXTS = [".npy"] # Keep this as a module-level constant

class Shard:
    """
    Shard to store embedding:score pairs in
        path: path to embedding on disk
        value: score for the original image
    """
    def __init__(self, path, value):
        self.path = path
        self.value = value # Keep original value (e.g., 0 or 1 for class)
        self.normalized_value = None # Will hold normalized score if mode='score'
        self.data = None
        self._preload_ok = True # Flag to track preload success

    def exists(self):
        # Also check if value is not None for robustness
        return os.path.isfile(self.path) and self.value is not None

    def get_data(self):
        # If preload was successful and data exists, return it
        if self._preload_ok and self.data is not None:
            return deepcopy(self.data)
        # Otherwise, attempt to load now (handles failed preload or no preload)
        try:
            data_numpy = np.load(self.path)
            # Ensure data is at least 1D before squeeze
            if data_numpy.ndim > 1 and data_numpy.shape[0] == 1:
                 data_tensor = torch.from_numpy(data_numpy).squeeze(0)
            elif data_numpy.ndim == 1:
                 data_tensor = torch.from_numpy(data_numpy)
            else:
                raise ValueError(f"Unexpected numpy shape {data_numpy.shape}")

            if torch.isnan(data_tensor.float().sum()):
                 raise ValueError(f"NaN detected in embedding file")

            # Use normalized_value if available (for score mode), otherwise use original value
            target_value = self.normalized_value if self.normalized_value is not None else self.value
            loaded_data = {
                "emb": data_tensor,
                "raw": self.value, # Always return original value as raw
                "val": torch.tensor([target_value]), # Return normalized for score, original for class
            }
            # If preloading was originally requested but failed, maybe cache it now?
            # if not self._preload_ok and self.data is None:
            #     self.data = loaded_data # Optional: cache on successful first load
            return loaded_data
        except Exception as e:
             # Raise a specific error or return None/handle in collate_fn
             raise IOError(f"Failed to load data for shard {self.path}: {e}") from e


    def preload(self):
        # Preload needs to handle potential errors gracefully during batch loading
        try:
            # Attempt to load data into self.data
            self.data = self.get_data()
            self._preload_ok = True
        except Exception as e:
            print(f"Error preloading {self.path}: {e}. Marking as unloadable.")
            self.data = None # Ensure data is None on failure
            self._preload_ok = False # Mark preload as failed

    def is_preload_ok(self):
        return self._preload_ok

class EmbeddingDataset(Dataset):
    # v2.0.1: Refined preload handling and getitem
    def __init__(self, ver, root=DEFAULT_ROOT, mode="class", preload=False, validation_split_count=0, seed=42):
        """
        Main dataset that returns list of requested images as (C, E) embeddings
          ver: CLIP version (folder)
          root: Path to folder with sorted files
          mode: Model type. Class pads return val to length of labels. Score normalizes.
          preload: Load all files into memory on initialization (will skip failed preloads).
          validation_split_count: Number of samples per class to reserve for validation.
          seed: Random seed for the validation split shuffle.
        """
        print(f"Initializing EmbeddingDataset v2.0.1...")
        self.ver = ver
        self.root = f"{root}/{ver}"
        self.mode = mode
        self.shard_class = Shard
        self.validation_split_count = validation_split_count
        self.seed = seed
        self.num_labels = 0

        self.train_shards = []
        self.val_shards = []

        # Parse all shards first
        all_shards_by_class = self._parse_all_shards(
            vprep=(lambda x: int(x)) if self.mode == "class" else (lambda x: float(x))
        )

        # Perform splitting
        self._split_shards(all_shards_by_class)

        # Preload if requested - MUST happen before normalization/label parsing
        # so that failed preloads can be filtered out if needed (though currently we keep them)
        if preload:
            print("Dataset: Preloading training data to system RAM...")
            # We preload in place, the shard object marks itself if failed
            [shard.preload() for shard in tqdm(self.train_shards)]
            # Filter out failed preloads from training set? Optional, but safer.
            initial_train_count = len(self.train_shards)
            self.train_shards = [s for s in self.train_shards if s.is_preload_ok()]
            filtered_train_count = len(self.train_shards)
            if initial_train_count != filtered_train_count:
                 print(f"Dataset: Filtered out {initial_train_count - filtered_train_count} training shards that failed to preload.")

            if self.val_shards:
                print("Dataset: Preloading validation data to system RAM...")
                [shard.preload() for shard in tqdm(self.val_shards)]
                initial_val_count = len(self.val_shards)
                self.val_shards = [s for s in self.val_shards if s.is_preload_ok()]
                filtered_val_count = len(self.val_shards)
                if initial_val_count != filtered_val_count:
                     print(f"Dataset: Filtered out {initial_val_count - filtered_val_count} validation shards that failed to preload.")


        # Normalize scores only for training shards if mode is score
        if self.mode == "score":
            self._normalize_scores() # Normalize train and remaining val shards

        # Get number of labels from training data if class mode
        if self.mode == "class":
            self._parse_labels()

        print(f"Dataset: OK. Training items: {len(self.train_shards)}, Validation items: {len(self.val_shards)}")
        if self.mode == "class":
            print(f"Dataset: Found {self.num_labels} classes in training data.")

    def __len__(self):
        return len(self.train_shards)

    def __getitem__(self, index):
        # Get shard from the training set
        shard = self.train_shards[index]
        try:
            data = shard.get_data() # This might fail if loading on-the-fly fails
            data["index"] = index
            return data
        except Exception as e:
             # This shard failed loading, either during preload or just now.
             # Log the error and return None to be handled by collate_fn
             print(f"ERROR in __getitem__ for index {index}, shard path {shard.path}: {e}")
             return None # Signal error to DataLoader's collate_fn


    def get_validation_loader(self, batch_size, num_workers=0):
        if not self.val_shards:
            return None
        # ValidationSubDataset already uses the filtered self.val_shards
        val_dataset = ValidationSubDataset(self.val_shards, self.mode, self.num_labels)
        if len(val_dataset) == 0:
             print("Warning: ValidationSubDataset is empty after filtering.")
             return None

        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=collate_ignore_none # Use the collate function
        )

    # Fixed mutable default arg `exts`
    def _parse_all_shards(self, vprep, exts=None):
        if exts is None:
            exts = ALLOWED_EXTS # Use the module-level constant if None

        print("Dataset: Parsing all data from disk...")
        # Use defaultdict for cleaner insertion
        shards_by_class = defaultdict(list)
        root_dir = self.root
        if not os.path.isdir(root_dir):
             raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

        for cat_folder_name in tqdm(os.listdir(root_dir)):
            cat_dir = os.path.join(root_dir, cat_folder_name)
            if not os.path.isdir(cat_dir): continue

            try:
                class_label_str = cat_folder_name.split('_', 1)[0]
                class_label = vprep(class_label_str)
            except (ValueError, IndexError): # Catch potential errors splitting/converting
                print(f"Warning: Could not parse class label from folder name '{cat_folder_name}'. Skipping.")
                continue

            for item_name in os.listdir(cat_dir):
                fname, ext = os.path.splitext(item_name)
                # Ensure comparison is case-insensitive just in case
                if ext.lower() not in [e.lower() for e in exts]: continue
                shard_path = os.path.join(cat_dir, item_name)
                shard = self.shard_class(path=shard_path, value=class_label)
                # Check existence *before* adding to list
                if shard.exists():
                     shards_by_class[class_label].append(shard)
                # else: # Optional: Log skipped non-existent files
                #     print(f"Debug: Skipping non-existent shard file: {shard_path}")


        print("Dataset: Found shards per class:")
        if not shards_by_class:
             print("  No classes found!")
        else:
             # Sort by class label for consistent output
             for lbl in sorted(shards_by_class.keys()):
                  print(f"  Class {lbl}: {len(shards_by_class[lbl])} shards")

        return shards_by_class

    # _split_shards remains mostly the same, ensure it handles empty lists if a class had no valid shards
    def _split_shards(self, all_shards_by_class):
        print(f"Dataset: Splitting data (Validation count per class: {self.validation_split_count})...")
        random.seed(self.seed)
        self.train_shards = []
        self.val_shards = []

        if self.validation_split_count <= 0:
            print("Dataset: No validation split requested. Using all data for training.")
            for class_label, shards in all_shards_by_class.items():
                self.train_shards.extend(shards)
            if self.train_shards: random.shuffle(self.train_shards)
            return

        for class_label, shards in all_shards_by_class.items():
            if not shards: # Skip if class had no valid shards
                 print(f"Warning: Class {class_label} has no valid shards to split.")
                 continue
            num_shards = len(shards)
            val_count = self.validation_split_count

            if num_shards < val_count:
                print(f"Warning: Class {class_label} has only {num_shards} samples, less than requested validation count {val_count}. Using all for training.")
                self.train_shards.extend(shards)
            elif num_shards == val_count:
                print(f"Warning: Class {class_label} has exactly {num_shards} samples, equal to requested validation count {val_count}. Using all for validation, none for training this class.")
                self.val_shards.extend(shards)
            else:
                random.shuffle(shards) # Shuffle before splitting
                self.val_shards.extend(shards[:val_count])
                self.train_shards.extend(shards[val_count:])

        if self.train_shards: random.shuffle(self.train_shards) # Shuffle the final training list if it's not empty

    # _normalize_scores remains the same, operates on potentially filtered lists
    def _normalize_scores(self):
        if not self.train_shards:
            print("Warning: No training shards to calculate normalization range from.")
            return
        train_values = [s.value for s in self.train_shards]
        if not train_values:
            print("Warning: Training values list is empty after filtering, cannot normalize.")
            return

        shard_min = min(train_values)
        shard_max = max(train_values)
        print(f"Normalizing scores based on training range [{shard_min}, {shard_max}]")
        value_range = shard_max - shard_min
        if value_range == 0:
            print("Warning: Training score range is zero. Setting all normalized scores to 0.")
            norm_func = lambda x: 0.0
        else:
            norm_func = lambda x: (x - shard_min) / value_range

        for s in self.train_shards: s.normalized_value = norm_func(s.value)
        for s in self.val_shards: s.normalized_value = max(0.0, min(1.0, norm_func(s.value)))

    # _parse_labels remains the same, operates on potentially filtered list
    def _parse_labels(self):
        if not self.train_shards:
            print("Warning: No training shards to parse labels from.")
            self.num_labels = 0
            return
        labels = set(int(s.value) for s in self.train_shards)
        if not labels:
            print("Warning: No labels found in training shards.")
            self.num_labels = 0
            return
        # Determine num_labels reliably, even if non-sequential
        self.num_labels = max(labels) + 1 if labels else 0
        expected_labels = set(range(self.num_labels))
        if not expected_labels.issubset(labels):
             # It's okay if labels aren't sequential, just log it clearly
             print(f"Dataset: Training class labels found: {sorted(list(labels))}. Max label is {max(labels)}. Setting num_labels to {self.num_labels}.")


# ValidationSubDataset needs robust __getitem__ too
class ValidationSubDataset(Dataset):
    def __init__(self, shards, mode, num_labels):
        self.shards = shards # Assumes shards are already filtered/valid
        self.mode = mode
        self.num_labels = num_labels

    def __len__(self):
        return len(self.shards)

    # v2.0.2: Correct target format for classifier validation
    def __getitem__(self, index):
         shard = self.shards[index]
         try:
             data = shard.get_data() # Attempt to load data (returns dict: {'emb': ..., 'raw': ..., 'val': ...})

             # --- TARGET FORMAT CORRECTION ---
             if self.mode == 'class':
                  # The 'val' from shard.get_data() currently holds the original class label (0, 1, etc.)
                  # as a tensor like tensor([0.]) or tensor([1.]).
                  # CrossEntropyLoss in training loop expects LongTensor of shape [B] (indices).
                  # Let's ensure 'val' in the output dict IS the Long index.
                  try:
                      # Convert the single-element tensor from get_data() to a Long scalar, then back to a 0-dim Long tensor.
                      # Or maybe just ensure it's the right type and shape for collate_fn?
                      # Let's directly use the 'raw' value which should be the integer label.
                      label_index = int(data['raw']) # Get the integer label (0, 1, ...)
                      if not (0 <= label_index < self.num_labels):
                           # Handle case where label might be out of bounds if data is weird
                           print(f"Warning: Invalid label {label_index} encountered in validation set for num_labels {self.num_labels}. Using label 0.")
                           label_index = 0
                      # Put the correct LongTensor index into the 'val' field for the dataloader
                      data['val'] = torch.tensor(label_index, dtype=torch.long) # Store as 0-dim Long Tensor

                  except Exception as e_label:
                      print(f"ERROR converting validation label {data.get('raw')} to Long index: {e_label}")
                      # Handle error - maybe return None or a default? Returning None is safer.
                      return None

             elif self.mode == 'score':
                  # Ensure score 'val' is float and maybe shape [1] for consistency?
                  # get_data already returns 'val' as tensor([float_val]) which is fine.
                  data['val'] = data['val'].float() # Ensure float type

             # --- END TARGET FORMAT CORRECTION ---

             return data # Return the dictionary with corrected 'val'

         except Exception as e:
             # Log error and return None to be handled by collate_fn
             print(f"ERROR in ValidationSubDataset __getitem__ for index {index}, shard path {shard.path}: {e}")
             return None

# collate_ignore_none function remains the same
def collate_ignore_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    try:
         return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
         print(f"Error in collate_ignore_none: {e}. Batch contents might be inconsistent.")
         # Handle error, maybe return None or raise
         return None

# --- ImageDataset section removed for brevity, would need similar fixes ---
