"""Manifest-backed dataset for precomputed feature-sequence artifacts."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kurome.data.datasets.sequence_dataset import TARGET_LEN, collate_sequences
from kurome.sample_manifest import load_records, resolve_manifest_path


class ManifestFeatureSequenceDataset(Dataset):
    """Load `[N, D]` feature sequences from artifact paths stored in a sample manifest."""

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
        self.target_len = TARGET_LEN
        self.class_names = [name.strip() for name in (class_names or []) if isinstance(name, str) and name.strip()]

        self.train_items: list[tuple[Path, str, int]] = []
        self.val_items: list[tuple[Path, str, int]] = []
        self.label_map: dict[str, int] = {}
        self.idx_to_label: dict[int, str] = {}
        self.num_labels = 0

        train_items_by_class, val_items_by_class = self._parse_manifest()
        self._map_labels(set(train_items_by_class) | set(val_items_by_class))
        self._split_items(train_items_by_class, val_items_by_class)

        print(f"ManifestFeatureSequenceDataset initialized from: {self.manifest_path}")
        print(f"  Artifact key: {self.artifact_key}")
        print(f"  Expecting padded sequence length: {self.target_len}")
        print(f"  Classes found: {self.num_labels} ({list(self.label_map.keys())})")
        print(f"  Training samples: {len(self.train_items)}")
        print(f"  Validation samples: {len(self.val_items)}")

    def _ordered_class_names(self, found_classes: set[str]) -> list[str]:
        if self.class_names:
            ordered = [name for name in self.class_names if name in found_classes]
            missing = [name for name in self.class_names if name not in found_classes]
            if missing:
                print(f"Warning: Declared feature classes with no samples found: {missing}")
            extras = [name for name in found_classes if name not in ordered]
            if extras:
                print(
                    f"Warning: Found undeclared feature classes not in class_names: {sorted(extras)}. "
                    "They will be skipped."
                )
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
            resolved = resolve_manifest_path(self.manifest_path, payload)
            if resolved is None:
                return None
            return resolved, "sequence"
        if isinstance(payload, dict):
            resolved = resolve_manifest_path(self.manifest_path, payload.get("path"))
            if resolved is None:
                return None
            return resolved, str(payload.get("npz_key") or "sequence")
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
                raise ValueError(f"Feature manifest record #{idx} is missing a valid label.")
            label = label.strip()
            if self.class_names and label not in self.class_names:
                continue

            resolved = self._resolve_artifact(record)
            if resolved is None:
                continue
            artifact_path, npz_key = resolved
            if not artifact_path.is_file():
                print(f"Warning: Missing feature artifact for '{label}': {artifact_path}")
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

    def __len__(self):
        return len(self.train_items)

    def _load_sequence(self, path: Path, npz_key: str) -> torch.Tensor:
        with np.load(path) as data:
            if npz_key not in data:
                raise KeyError(f"Artifact '{path}' missing npz key '{npz_key}'.")
            sequence = np.asarray(data[npz_key], dtype=np.float32)
        tensor = torch.from_numpy(sequence)
        if tensor.ndim != 2:
            raise ValueError(f"Expected sequence [N,D] from '{path}', got {tuple(tensor.shape)}.")
        return tensor

    def _sequence_to_sample(self, sequence_tensor: torch.Tensor, label_idx: int) -> dict[str, torch.Tensor]:
        current_len = int(sequence_tensor.shape[0])
        features_dim = int(sequence_tensor.shape[1])
        final_sequence = torch.zeros((self.target_len, features_dim), dtype=torch.float32)
        attention_mask = torch.zeros(self.target_len, dtype=torch.bool)

        if current_len <= 0:
            pass
        elif current_len > self.target_len:
            final_sequence = sequence_tensor[: self.target_len, :]
            attention_mask[:] = True
        else:
            final_sequence[:current_len, :] = sequence_tensor
            attention_mask[:current_len] = True

        return {
            "sequence": final_sequence,
            "mask": attention_mask,
            "label": torch.tensor(label_idx, dtype=torch.long),
        }

    def _load_item(self, item: tuple[Path, str, int]) -> dict[str, torch.Tensor]:
        path, npz_key, label_idx = item
        sequence_tensor = self._load_sequence(path, npz_key)
        return self._sequence_to_sample(sequence_tensor, label_idx)

    def __getitem__(self, index):
        try:
            return self._load_item(self.train_items[index])
        except Exception as exc:
            print(f"Error loading feature item index {index}: {exc}")
            return None

    def get_validation_loader(self, batch_size, num_workers=0, prefetch_factor=2):
        if not self.val_items:
            return None
        val_dataset = ValidationManifestFeatureSequenceDataset(self.val_items, self.target_len)
        return DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_sequences,
            persistent_workers=(num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )


class ValidationManifestFeatureSequenceDataset(Dataset):
    """Validation wrapper matching ManifestFeatureSequenceDataset semantics."""

    def __init__(self, items: list[tuple[Path, str, int]], target_len: int):
        self.items = items
        self.target_len = target_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, npz_key, label_idx = self.items[index]
        try:
            with np.load(path) as data:
                sequence = np.asarray(data[npz_key], dtype=np.float32)
            sequence_tensor = torch.from_numpy(sequence)
            if sequence_tensor.ndim != 2:
                raise ValueError(f"Expected sequence [N,D] from '{path}', got {tuple(sequence_tensor.shape)}.")

            current_len = int(sequence_tensor.shape[0])
            features_dim = int(sequence_tensor.shape[1])
            final_sequence = torch.zeros((self.target_len, features_dim), dtype=torch.float32)
            attention_mask = torch.zeros(self.target_len, dtype=torch.bool)

            if current_len <= 0:
                pass
            elif current_len > self.target_len:
                final_sequence = sequence_tensor[: self.target_len, :]
                attention_mask[:] = True
            else:
                final_sequence[:current_len, :] = sequence_tensor
                attention_mask[:current_len] = True

            return {
                "sequence": final_sequence,
                "mask": attention_mask,
                "label": torch.tensor(label_idx, dtype=torch.long),
            }
        except Exception as exc:
            print(f"Error loading feature validation item index {index}: {exc}")
            return None
