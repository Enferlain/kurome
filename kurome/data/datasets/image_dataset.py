# ruff: noqa
# image_dataset.py
# Version 1.0.0: Dataset for loading images and applying transforms end-to-end.

import csv
import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import random
from collections import defaultdict
import traceback

# List of common image extensions
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]


# <<< NEW Function: collate_group_by_size >>>
def collate_group_by_size(batch):
    """
    Custom collate function that groups samples by the shape of their 'pixel_values'.

    Args:
        batch (list): A list of dictionaries, where each dict is a sample
                      from ImageFolderDataset (e.g., {'pixel_values': tensor, 'label': tensor}).

    Returns:
        list: A list where each element is a dictionary representing a mini-batch.
              Samples with the same pixel_values shape are stacked together.
              Returns None if the input batch is empty after filtering None items.
    """
    # 1. Filter out None items (e.g., from failed loads/transforms)
    batch = [item for item in batch if item is not None and 'pixel_values' in item and 'label' in item]
    if not batch:
        # print("Warning: collate_group_by_size received empty batch after filtering.")
        return None # Return None if the whole batch is invalid

    # 2. Group samples by the shape of their 'pixel_values'
    #    Using defaultdict makes grouping easy
    grouped_by_shape = defaultdict(list)
    for sample in batch:
        # Use tensor.shape as the key (it's hashable)
        shape_key = sample['pixel_values'].shape
        grouped_by_shape[shape_key].append(sample)

    # 3. Create the output list of mini-batches
    output_batch_list = []
    for shape, samples in grouped_by_shape.items():
        # Stack tensors for samples in this group
        # Use default_collate on the list of samples for this shape
        # This handles stacking 'pixel_values', 'label', and any other tensors properly
        try:
            # Use the standard default_collate for stacking within the group
            stacked_group = torch.utils.data.dataloader.default_collate(samples)
            output_batch_list.append(stacked_group)
        except Exception as e_stack:
            # This shouldn't happen if grouping worked, but handle just in case
            print(f"Error stacking group with shape {shape}: {e_stack}")
            # Optionally, add individual samples as fallback? For now, skip group on error.
            continue

    return output_batch_list
# <<< End NEW Function >>>

# --- Helper: Collate function to handle potential None from failed loads ---
def collate_skip_none(batch):
    """Collate function that filters out None items."""
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return an empty dictionary or signal error if batch becomes empty
        # Returning empty dict might work if train loop checks batch content
        # print("Warning: collate_skip_none resulted in empty batch.")
        return {} # Or maybe None? Needs handling in train loop. Let's try empty dict.
    try:
        # Use default collate on the filtered batch
        return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
        print(f"Error in collate_skip_none default_collate: {e}")
        print("Problematic batch contents (first item type):", type(batch[0]) if batch else "Empty")
        # Maybe print shapes if possible
        if isinstance(batch[0], dict):
            for key, value in batch[0].items():
                if hasattr(value, 'shape'): print(f"  Item 0, Key '{key}', Shape: {value.shape}")
        return {} # Return empty dict on error
# --- End Helper ---


class ImageFolderDataset(Dataset):
    """
    Dataset to load images directly from class-based folders and apply transforms.
    Handles train/validation splitting based on fixed count per class.
    """
    def __init__(
        self,
        root_dir,
        transform=None,
        validation_split_count=0,
        seed=42,
        class_names=None,
        manifest_path=None,
    ):
        """
        Args:
            root_dir (str): Path to the directory containing class subfolders (e.g., data/my_images/).
            transform (callable, optional): A function/transform to apply to the PIL image
                                           (e.g., the processor from Hugging Face).
            validation_split_count (int): Number of samples per class for validation.
            seed (int): Random seed for shuffling and splitting.
        """
        print(f"Initializing ImageFolderDataset v1.0.0...")
        if root_dir is None and manifest_path:
            root_dir = str(Path(manifest_path).resolve().parent)
        self.root_dir = root_dir
        self.transform = transform # This will be our AIMv2 processor
        self.validation_split_count = validation_split_count
        self.seed = seed
        self.class_names = [name.strip() for name in (class_names or []) if isinstance(name, str) and name.strip()]
        self.manifest_path = manifest_path
        self.num_labels = 0

        self.train_items = [] # List of tuples: (image_path, label_idx)
        self.val_items = []   # List of tuples: (image_path, label_idx)
        self.label_map = {}   # Maps class folder name (e.g., "0") to integer index (0)
        self.idx_to_label = {} # Maps integer index back to folder name (optional)

        if not self.root_dir or not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {self.root_dir}")

        # Parse, map labels, and split
        train_items_by_class_name, val_items_by_class_name = self._load_items()
        self._map_labels()
        self._map_labels_and_split(train_items_by_class_name, val_items_by_class_name)

        print(f"ImageFolderDataset: OK. Training items: {len(self.train_items)}, Validation items: {len(self.val_items)}")
        print(f"  Found {self.num_labels} classes. Label map: {self.label_map}")


    def _load_items(self):
        """Load items from a manifest or class-based folders."""
        if self.manifest_path:
            return self._parse_manifest()
        return self._parse_image_files()

    def _normalize_class_name(self, class_name):
        if class_name is None:
            return None
        normalized = str(class_name).strip()
        return normalized or None

    def _ordered_class_names(self, found_classes):
        if self.class_names:
            ordered = [name for name in self.class_names if name in found_classes]
            extras = [name for name in found_classes if name not in ordered]
            if extras:
                print(f"Warning: Found undeclared classes not in class_names: {sorted(extras)}. They will be skipped.")
            missing = [name for name in self.class_names if name not in found_classes]
            if missing:
                print(f"Warning: Declared classes with no samples found: {missing}")
            return ordered
        if all(name.isdigit() for name in found_classes):
            return sorted(found_classes, key=int)
        return sorted(found_classes)

    def _map_labels(self):
        class_names = self._ordered_class_names(set(self.class_names) if self.class_names else set())
        if not class_names:
            class_names = []
        self.num_labels = 0
        self.label_map = {}
        self.idx_to_label = {}
        for name in class_names:
            self.label_map[name] = self.num_labels
            self.idx_to_label[self.num_labels] = name
            self.num_labels += 1

    def _resolve_manifest_path(self, path_value):
        path_str = str(path_value).strip()
        if not path_str:
            raise ValueError("Manifest path entries must be non-empty.")
        candidate = Path(path_str)
        if not candidate.is_absolute():
            manifest_base = Path(self.manifest_path).resolve().parent
            candidate = manifest_base / candidate
            if not candidate.exists():
                candidate = Path(self.root_dir).resolve() / path_str
        return str(candidate.resolve())

    def _load_manifest_records(self):
        manifest = Path(self.manifest_path)
        if not manifest.is_file():
            raise FileNotFoundError(f"Image manifest not found: {self.manifest_path}")
        suffix = manifest.suffix.lower()
        if suffix == ".csv":
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                return list(csv.DictReader(handle))
        if suffix in {".jsonl", ".ndjson"}:
            records = []
            with manifest.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON on line {line_number} in manifest '{self.manifest_path}'.") from exc
            return records
        if suffix == ".json":
            with manifest.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return payload
            raise ValueError(f"JSON manifest '{self.manifest_path}' must contain a list of objects.")
        raise ValueError(
            f"Unsupported manifest format '{suffix}' for '{self.manifest_path}'. Use .csv, .jsonl, .ndjson, or .json."
        )

    def _parse_manifest(self):
        """Load image items from a manifest with optional explicit train/val splits."""
        print(f"Dataset: Loading image manifest from '{self.manifest_path}'...")
        train_items_by_class_name = defaultdict(list)
        val_items_by_class_name = defaultdict(list)
        found_classes = []

        records = self._load_manifest_records()
        if not isinstance(records, list):
            raise ValueError(f"Manifest '{self.manifest_path}' must decode into a list of records.")

        for idx, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Manifest record #{idx} must be an object/dict.")
            path_value = record.get("path") or record.get("image_path") or record.get("file")
            if path_value is None:
                raise ValueError(f"Manifest record #{idx} is missing a 'path' field.")
            image_path = self._resolve_manifest_path(path_value)
            if os.path.splitext(image_path)[1].lower() not in IMAGE_EXTS:
                print(f"Warning: Skipping non-image manifest entry '{image_path}'.")
                continue

            class_name = (
                record.get("label")
                or record.get("class_name")
                or record.get("class")
                or record.get("category")
            )
            if class_name is None:
                class_name = Path(image_path).parent.name
            class_name = self._normalize_class_name(class_name)
            if class_name is None:
                raise ValueError(f"Manifest record #{idx} has an empty class/label value.")

            if self.class_names and class_name not in self.class_names:
                print(f"Warning: Skipping manifest entry with unknown class '{class_name}': {image_path}")
                continue

            found_classes.append(class_name)
            split = str(record.get("split", "")).strip().lower()
            target_items = train_items_by_class_name
            if split in {"val", "valid", "validation"}:
                target_items = val_items_by_class_name
            elif split in {"test"}:
                continue
            elif split not in {"", "train"}:
                raise ValueError(
                    f"Manifest record #{idx} has unsupported split '{split}'. Use train/val/test or leave blank."
                )
            target_items[class_name].append(image_path)

        ordered_classes = self._ordered_class_names(set(found_classes))
        self.class_names = ordered_classes
        print(f"Dataset: Found manifest classes: {ordered_classes}")
        return train_items_by_class_name, val_items_by_class_name

    def _parse_image_files(self):
        """Scans the root directory for class folders and image files."""
        print("Dataset: Scanning for image files...")
        items_by_class_name = defaultdict(list)
        found_classes = set()

        for class_folder_name in tqdm(os.listdir(self.root_dir), desc="Parsing folders"):
            class_dir = os.path.join(self.root_dir, class_folder_name)
            if not os.path.isdir(class_dir):
                continue
            if class_folder_name.startswith("."):
                continue
            normalized_name = self._normalize_class_name(class_folder_name)
            if normalized_name is None:
                continue
            if self.class_names and normalized_name not in self.class_names:
                print(f"Warning: Skipping class folder not in class_names: '{normalized_name}'.")
                continue

            found_classes.add(normalized_name)
            for filename in os.listdir(class_dir):
                if os.path.splitext(filename)[1].lower() in IMAGE_EXTS:
                    image_path = os.path.join(class_dir, filename)
                    items_by_class_name[normalized_name].append(image_path)

        ordered_classes = self._ordered_class_names(found_classes)
        self.class_names = ordered_classes
        print(f"Dataset: Found class folders: {ordered_classes}")
        return items_by_class_name, defaultdict(list)

    def _map_labels_and_split(self, train_items_by_class_name, val_items_by_class_name):
        """Assigns integer labels and splits into train/validation sets."""
        print(f"Dataset: Splitting data (Validation count per class: {self.validation_split_count})...")
        random.seed(self.seed)
        self.train_items = []
        self.val_items = []

        for class_name in self.class_names:
            if class_name not in self.label_map:
                continue
            label_idx = self.label_map[class_name]
            train_paths = list(train_items_by_class_name.get(class_name, []))
            explicit_val_paths = list(val_items_by_class_name.get(class_name, []))
            items_with_labels = [(path, label_idx) for path in train_paths]
            num_items = len(items_with_labels)
            val_count = self.validation_split_count

            self.val_items.extend((path, label_idx) for path in explicit_val_paths)

            if explicit_val_paths:
                self.train_items.extend(items_with_labels)
                continue

            if val_count <= 0: # No validation split
                self.train_items.extend(items_with_labels)
            elif num_items <= val_count: # Not enough samples for split
                print(f"Warning: Class '{class_name}' ({num_items} samples) <= validation count ({val_count}). Using all for training.")
                self.train_items.extend(items_with_labels)
            else: # Perform split
                random.shuffle(items_with_labels)
                self.val_items.extend(items_with_labels[:val_count])
                self.train_items.extend(items_with_labels[val_count:])

        if self.train_items: random.shuffle(self.train_items) # Shuffle final training list
        # Validation set is usually not shuffled by default in loader


    def __len__(self):
        return len(self.train_items)

    def __getitem__(self, index):
        img_path, label_idx = self.train_items[index]

        try:
            # Load Image
            img = Image.open(img_path).convert("RGB")

            # Apply Transforms (e.g., AIMv2 Processor)
            pixel_values = None
            if self.transform:
                # Processor typically returns a dict-like object (BatchFeature)
                # Handle potential errors during transform
                try:
                     processed_output = self.transform(images=img, return_tensors="pt")
                     # Extract the pixel_values tensor, remove the batch dimension
                     pixel_values = processed_output['pixel_values'].squeeze(0)
                except Exception as transform_e:
                     print(f"ERROR applying transform to {img_path}: {transform_e}")
                     traceback.print_exc()
                     return None # Skip this item if transform fails

            # Return dictionary expected by training loop
            # Ensure label is a tensor
            label_tensor = torch.tensor(label_idx, dtype=torch.long)

            if pixel_values is not None:
                 return {"pixel_values": pixel_values, "label": label_tensor}
            else: # Should not happen if transform worked, but safety check
                 print(f"Warning: pixel_values is None after transform for {img_path}")
                 return None

        except UnidentifiedImageError:
            print(f"Warning: Skipping file (cannot identify image): {img_path}")
            return None # Skip corrupt images
        except Exception as e:
            print(f"ERROR in ImageFolderDataset __getitem__ for index {index}, path {img_path}: {e}")
            traceback.print_exc()
            return None # Skip item on other errors

    def get_validation_loader(self, batch_size, num_workers=0):
        if not self.val_items:
            print("Validation set is empty.")
            return None
        val_dataset = ValidationSubDataset(self.val_items, self.transform)
        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False, # No shuffle for validation
            drop_last=False,
            pin_memory=False,
            num_workers=num_workers,
            # <<< Use collate_group_by_size for validation too! >>>
            collate_fn=collate_group_by_size
        )

# --- Simple Validation Dataset Wrapper ---
class ValidationSubDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items # List of (path, label_idx) tuples
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # Same logic as main dataset's __getitem__
        img_path, label_idx = self.items[index]
        try:
            img = Image.open(img_path).convert("RGB")
            pixel_values = None
            if self.transform:
                try:
                     processed_output = self.transform(images=img, return_tensors="pt")
                     pixel_values = processed_output['pixel_values'].squeeze(0)
                except Exception as transform_e:
                     print(f"ERROR applying transform to VAL {img_path}: {transform_e}")
                     traceback.print_exc()
                     return None
            label_tensor = torch.tensor(label_idx, dtype=torch.long)
            if pixel_values is not None:
                 return {"pixel_values": pixel_values, "label": label_tensor}
            else: return None
        except UnidentifiedImageError: print(f"Warning: Skipping VAL file (cannot identify): {img_path}"); return None
        except Exception as e: print(f"ERROR in VAL __getitem__ for index {index}, path {img_path}: {e}"); traceback.print_exc(); return None
