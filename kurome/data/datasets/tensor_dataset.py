"""Manifest-backed dataset for spatial tensor artifacts stored in .npz files."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kurome.sample_manifest import load_records, resolve_manifest_path


def collate_tensor_batch(batch):
    """Group tensor samples by spatial shape, mirroring image-mode batching."""
    batch = [item for item in batch if item is not None and "pixel_values" in item and "label" in item]
    if not batch:
        return None

    grouped = defaultdict(list)
    for sample in batch:
        grouped[tuple(sample["pixel_values"].shape)].append(sample)

    output = []
    for samples in grouped.values():
        output.append(torch.utils.data.dataloader.default_collate(samples))
    return output


class TensorArtifactDataset(Dataset):
    """Load `[C, H, W]` tensor artifacts described in a sample manifest."""

    def __init__(
        self,
        manifest_path: str,
        artifact_key: str,
        validation_split_count: int = 0,
        seed: int = 42,
        class_names: list[str] | None = None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.artifact_key = artifact_key
        self.validation_split_count = validation_split_count
        self.seed = seed
        self.class_names = [name.strip() for name in (class_names or []) if isinstance(name, str) and name.strip()]

        self.train_items: list[tuple[Path, int]] = []
        self.val_items: list[tuple[Path, int]] = []
        self.label_map: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}
        self.num_labels = 0
        self.num_channels = 0

        train_items_by_class, val_items_by_class = self._parse_manifest()
        self._map_labels(set(train_items_by_class) | set(val_items_by_class))
        self._split_items(train_items_by_class, val_items_by_class)
        self._infer_num_channels()

    def _ordered_class_names(self, found_classes: set[str]) -> list[str]:
        if self.class_names:
            ordered = [name for name in self.class_names if name in found_classes]
            missing = [name for name in self.class_names if name not in found_classes]
            if missing:
                print(f"Warning: Declared tensor classes with no samples found: {missing}")
            extras = [name for name in found_classes if name not in ordered]
            if extras:
                print(f"Warning: Found undeclared tensor classes not in class_names: {sorted(extras)}. They will be skipped.")
            return ordered
        return sorted(found_classes)

    def _resolve_artifact(self, record: dict) -> tuple[Path, str] | None:
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, dict):
            return None
        payload = artifacts.get(self.artifact_key)
        if payload is None:
            return None
        if isinstance(payload, str):
            return resolve_manifest_path(self.manifest_path, payload), "tensor"
        if isinstance(payload, dict):
            path_value = payload.get("path")
            npz_key = str(payload.get("npz_key") or "tensor")
            resolved = resolve_manifest_path(self.manifest_path, path_value)
            if resolved is None:
                return None
            return resolved, npz_key
        return None

    def _parse_manifest(self):
        train_items_by_class = defaultdict(list)
        val_items_by_class = defaultdict(list)
        found_classes: set[str] = set()

        for idx, record in enumerate(load_records(self.manifest_path), start=1):
            if not isinstance(record, dict):
                continue
            label = record.get("label") or record.get("class_name") or record.get("class")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"Tensor manifest record #{idx} is missing a valid label.")
            label = label.strip()
            if self.class_names and label not in self.class_names:
                continue

            resolved = self._resolve_artifact(record)
            if resolved is None:
                continue
            artifact_path, npz_key = resolved
            if not artifact_path.is_file():
                print(f"Warning: Missing tensor artifact for '{label}': {artifact_path}")
                continue

            found_classes.add(label)
            split = str(record.get("split", "")).strip().lower()
            target = train_items_by_class
            if split in {"val", "valid", "validation"}:
                target = val_items_by_class
            elif split == "test":
                continue
            target[label].append((artifact_path, npz_key))

        self.class_names = self._ordered_class_names(found_classes)
        return train_items_by_class, val_items_by_class

    def _map_labels(self, found_classes: set[str]) -> None:
        self.label_map = {}
        self.idx_to_label = {}
        self.num_labels = 0
        for class_name in self._ordered_class_names(found_classes):
            self.label_map[class_name] = self.num_labels
            self.idx_to_label[self.num_labels] = class_name
            self.num_labels += 1

    def _split_items(self, train_items_by_class, val_items_by_class) -> None:
        random.seed(self.seed)
        self.train_items = []
        self.val_items = []

        for class_name in self.class_names:
            if class_name not in self.label_map:
                continue
            label_idx = self.label_map[class_name]
            explicit_val = list(val_items_by_class.get(class_name, []))
            train_items = list(train_items_by_class.get(class_name, []))
            self.val_items.extend((path, npz_key, label_idx) for path, npz_key in explicit_val)

            if explicit_val:
                self.train_items.extend((path, npz_key, label_idx) for path, npz_key in train_items)
                continue

            if self.validation_split_count <= 0 or len(train_items) <= self.validation_split_count:
                self.train_items.extend((path, npz_key, label_idx) for path, npz_key in train_items)
                continue

            random.shuffle(train_items)
            self.val_items.extend((path, npz_key, label_idx) for path, npz_key in train_items[: self.validation_split_count])
            self.train_items.extend((path, npz_key, label_idx) for path, npz_key in train_items[self.validation_split_count :])

        if self.train_items:
            random.shuffle(self.train_items)

    def _infer_num_channels(self) -> None:
        candidate_items = self.train_items or self.val_items
        if not candidate_items:
            self.num_channels = 0
            return
        path, npz_key, _ = candidate_items[0]
        with np.load(path) as data:
            tensor = data[npz_key]
        if tensor.ndim != 3:
            raise ValueError(f"Tensor artifact '{path}' must be 3D [C,H,W], got shape {tensor.shape}.")
        self.num_channels = int(tensor.shape[0])

    def __len__(self):
        return len(self.train_items)

    def _load_item(self, item):
        path, npz_key, label_idx = item
        with np.load(path) as data:
            if npz_key not in data:
                raise KeyError(f"Artifact '{path}' missing npz key '{npz_key}'.")
            tensor = data[npz_key]
        tensor = torch.from_numpy(np.asarray(tensor, dtype=np.float32))
        if tensor.ndim != 3:
            raise ValueError(f"Expected tensor [C,H,W] from '{path}', got {tuple(tensor.shape)}.")
        return {
            "pixel_values": tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
        }

    def __getitem__(self, index):
        try:
            return self._load_item(self.train_items[index])
        except Exception as exc:
            print(f"Error loading tensor item index {index}: {exc}")
            return None

    def get_validation_loader(self, batch_size, num_workers=0):
        if not self.val_items:
            return None
        val_dataset = ValidationTensorSubDataset(self.val_items)
        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            num_workers=num_workers,
            collate_fn=collate_tensor_batch,
        )


class ValidationTensorSubDataset(Dataset):
    """Validation wrapper matching TensorArtifactDataset semantics."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, npz_key, label_idx = self.items[index]
        try:
            with np.load(path) as data:
                tensor = data[npz_key]
            tensor = torch.from_numpy(np.asarray(tensor, dtype=np.float32))
            return {
                "pixel_values": tensor,
                "label": torch.tensor(label_idx, dtype=torch.long),
            }
        except Exception as exc:
            print(f"Error loading tensor validation item index {index}: {exc}")
            return None
