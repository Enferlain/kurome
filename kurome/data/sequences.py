"""Dataloader setup for feature-sequence training mode."""

from __future__ import annotations

import os

from kurome.data.dataloaders import (
    build_training_dataloader,
    build_validation_dataloader,
    log_train_val_loader_summary,
)
from kurome.data.datasets.sequence_dataset import FeatureSequenceDataset, collate_sequences


def build_feature_sequence_dataloaders(args):
    """Build dataset + dataloaders for sequence feature training."""
    feature_root_dir = os.path.join(args.data_root, args.feature_dir_name)
    print(f"Setting up FeatureSequenceDataset (pooling in __getitem__) from: {feature_root_dir}")

    dataset = FeatureSequenceDataset(
        feature_root_dir=feature_root_dir,
        validation_split_count=args.val_split_count,
        seed=args.seed,
        preload=getattr(args, "preload_data", False),
        preload_limit_gb=getattr(args, "preload_limit_gb", 30.0),
    )
    args.num_labels = dataset.num_labels
    print(f"DEBUG: Updated args.num_labels from FeatureSequenceDataset: {args.num_labels}")
    if len(dataset.train_indices) == 0:
        raise RuntimeError("Training dataset partition is empty.")

    val_loader = build_validation_dataloader(
        dataset,
        batch_size=args.batch,
        num_workers=args.num_workers,
    )

    print("DEBUG: Creating standard DataLoader for training...")
    train_loader = build_training_dataloader(
        dataset,
        batch_size=args.batch,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=getattr(args, "prefetch_factor", 2) if args.num_workers > 0 else None,
        drop_last=getattr(args, "train_drop_last", True),
    )
    log_train_val_loader_summary(
        mode_name="feature-sequence",
        train_loader=train_loader,
        train_samples=len(dataset.train_indices),
        val_loader=val_loader,
        val_split_count=args.val_split_count,
    )

    num_train_samples = len(dataset.train_indices)
    if num_train_samples == 0:
        raise RuntimeError("No training samples available after split.")
    steps_per_epoch = num_train_samples // args.batch
    if not getattr(args, "train_drop_last", True) and num_train_samples % args.batch != 0:
        steps_per_epoch += 1
    args.steps_per_epoch = steps_per_epoch
    print(f"DEBUG: Updated args.steps_per_epoch based on standard DataLoader: {args.steps_per_epoch}")

    return dataset, train_loader, val_loader
