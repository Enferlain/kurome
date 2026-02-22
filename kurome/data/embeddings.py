"""Dataloader setup for embedding-based training mode."""

from __future__ import annotations

import os

import torch

from kurome.data.dataloaders import (
    build_training_dataloader,
    build_validation_dataloader,
    log_train_val_loader_summary,
)
from kurome.data.datasets.embedding_dataset import EmbeddingDataset


def build_embedding_training_dataloaders(args, image_processor=None):
    """Build dataset + dataloaders for embedding-mode training."""
    if image_processor is not None:
        print("Warning: image_processor is ignored in embedding mode.")
    if not hasattr(args, "embed_ver") or not args.embed_ver:
        raise RuntimeError("'embed_ver' is required for embedding-based training.")

    data_root_path = args.data_root
    dataset_version = args.embed_ver
    embedding_data_dir = os.path.join(data_root_path, dataset_version)
    print(f"Setting up EmbeddingDataset using version folder: {embedding_data_dir}")

    dataset = EmbeddingDataset(
        ver=dataset_version,
        root=data_root_path,
        mode=args.arch,
        preload=getattr(args, "preload_data", True),
        validation_split_count=args.val_split_count,
        seed=args.seed,
    )
    if args.arch == "class":
        args.num_labels = dataset.num_labels
        print(f"DEBUG: Updated args.num_labels from EmbeddingDataset: {args.num_labels}")
    collate_fn = getattr(dataset, "collate_ignore_none", torch.utils.data.dataloader.default_collate)
    print(f"DEBUG: Using {getattr(collate_fn, '__name__', 'default_collate')} for embedding dataloader.")

    if len(dataset) == 0:
        print("Warning: Training dataset is empty! Check data path and configuration.")

    train_loader = build_training_dataloader(
        dataset,
        batch_size=args.batch,
        num_workers=getattr(args, "num_workers", 0),
        collate_fn=collate_fn,
    )
    val_loader = build_validation_dataloader(
        dataset,
        batch_size=args.batch,
        num_workers=getattr(args, "num_workers", 0),
    )
    log_train_val_loader_summary(
        mode_name="embedding",
        train_loader=train_loader,
        train_samples=len(dataset),
        val_loader=val_loader,
        val_split_count=args.val_split_count,
    )

    return dataset, train_loader, val_loader
