"""Dataloader setup for end-to-end image training mode."""

from __future__ import annotations

from kurome.data.dataloaders import (
    build_training_dataloader,
    build_validation_dataloader,
    log_train_val_loader_summary,
)
from kurome.data.datasets.image_dataset import ImageFolderDataset, collate_group_by_size


def build_image_training_dataloaders(args, image_processor):
    """Build dataset + dataloaders for image-mode training."""
    if image_processor is None:
        raise RuntimeError("Image processor is required for ImageFolderDataset.")

    image_data_dir = args.data_root
    print(f"Setting up ImageFolderDataset from: {image_data_dir}")
    print(f"Looking for class folders (0, 1, ...) directly inside: {image_data_dir}")

    dataset = ImageFolderDataset(
        root_dir=image_data_dir,
        transform=image_processor,
        validation_split_count=args.val_split_count,
        seed=args.seed,
    )
    args.num_labels = dataset.num_labels
    print(f"DEBUG: Updated args.num_labels from ImageFolderDataset: {args.num_labels}")
    print("DEBUG: Using collate_group_by_size for image-mode dataloader.")

    if len(dataset) == 0:
        print("Warning: Training dataset is empty! Check image path and configuration.")

    train_loader = build_training_dataloader(
        dataset,
        batch_size=args.batch,
        num_workers=args.num_workers,
        collate_fn=collate_group_by_size,
    )
    val_loader = build_validation_dataloader(
        dataset,
        batch_size=args.batch,
        num_workers=args.num_workers,
    )
    log_train_val_loader_summary(
        mode_name="image",
        train_loader=train_loader,
        train_samples=len(dataset),
        val_loader=val_loader,
        val_split_count=args.val_split_count,
    )

    return dataset, train_loader, val_loader
