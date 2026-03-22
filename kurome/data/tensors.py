"""Dataloader setup for manifest-backed forensic tensor training."""

from __future__ import annotations

from kurome.data.dataloaders import (
    build_training_dataloader,
    build_validation_dataloader,
    log_train_val_loader_summary,
)
from kurome.data.datasets.tensor_dataset import TensorArtifactDataset, collate_tensor_batch


def build_tensor_training_dataloaders(args):
    """Build dataset + dataloaders for tensor-mode training."""
    manifest_path = getattr(args, "manifest_path", None)
    artifact_key = getattr(args, "artifact_key", None)
    if not manifest_path:
        raise RuntimeError("TensorArtifactDataset requires args.manifest_path.")
    if not artifact_key:
        raise RuntimeError("TensorArtifactDataset requires args.artifact_key.")

    dataset = TensorArtifactDataset(
        manifest_path=manifest_path,
        artifact_key=artifact_key,
        validation_split_count=args.val_split_count,
        seed=args.seed,
        class_names=getattr(args, "class_names", None),
    )
    args.num_labels = dataset.num_labels
    args.input_channels = dataset.num_channels
    print(f"DEBUG: Updated args.num_labels from TensorArtifactDataset: {args.num_labels}")
    print(f"DEBUG: Updated args.input_channels from TensorArtifactDataset: {args.input_channels}")

    train_loader = build_training_dataloader(
        dataset,
        batch_size=args.batch,
        num_workers=args.num_workers,
        collate_fn=collate_tensor_batch,
    )
    val_loader = build_validation_dataloader(
        dataset,
        batch_size=args.batch,
        num_workers=args.num_workers,
    )
    log_train_val_loader_summary(
        mode_name="tensor",
        train_loader=train_loader,
        train_samples=len(dataset),
        val_loader=val_loader,
        val_split_count=args.val_split_count,
    )

    return dataset, train_loader, val_loader
