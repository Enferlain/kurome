"""Shared dataloader builder helpers for data adapters."""

from __future__ import annotations

from torch.utils.data import DataLoader


def build_training_dataloader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    collate_fn=None,
    drop_last: bool = True,
    shuffle: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
):
    """Create a training DataLoader with optional worker tuning flags."""
    if persistent_workers is None and prefetch_factor is None:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
    if persistent_workers is None:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
        )
    if prefetch_factor is None:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=num_workers,
            collate_fn=collate_fn,
            persistent_workers=persistent_workers,
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )


def build_validation_dataloader(dataset, *, batch_size: int, num_workers: int):
    """Delegate validation loader creation to dataset adapters that expose the method."""
    return dataset.get_validation_loader(batch_size=batch_size, num_workers=num_workers)


def log_train_val_loader_summary(
    *,
    mode_name: str,
    train_loader,
    train_samples: int,
    val_loader,
    val_split_count: int,
):
    """Print consistent train/validation loader summary messages."""
    print(f"Created {mode_name} training loader with {len(train_loader)} batches ({train_samples} samples).")
    if val_loader and hasattr(val_loader, "dataset") and len(val_loader.dataset) > 0:
        print(
            f"Created {mode_name} validation loader with {len(val_loader)} batches "
            f"({len(val_loader.dataset)} samples)."
        )
    elif val_split_count > 0:
        print(
            f"Validation split requested, but {mode_name} validation loader is empty "
            "or could not be created."
        )
    else:
        print("No validation split requested or validation data available.")
