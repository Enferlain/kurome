#!/usr/bin/env python3
"""Root task launcher for training, dataset generation, and inference workflows."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

TASKS: dict[str, list[str]] = {
    # Training
    "train-embeddings": [sys.executable, "-m", "kurome.cli.train_embeddings"],
    "train-tensors": [sys.executable, "-m", "kurome.cli.train_embeddings"],
    "train-features": [sys.executable, "-m", "kurome.cli.train_features"],
    # Dataset / feature generation
    "build-embeddings": [sys.executable, "-m", "kurome.cli.generate_embeddings"],
    "build-features": [sys.executable, "-m", "kurome.cli.generate_feature_sequences"],
    "build-manifest": [sys.executable, "-m", "kurome.cli.build_image_manifest"],
    "build-forensic-tensors": [sys.executable, "-m", "kurome.cli.build_forensic_tensors"],
    "prepare-data": [sys.executable, "-m", "kurome.cli.prepare_data"],
    # Inference (folder pipeline)
    "infer-folder": [sys.executable, "-m", "kurome.cli.infer_folder"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch kurome workflows from a single root entrypoint. "
            "Use '--' to forward task-specific arguments."
        )
    )
    parser.add_argument(
        "task",
        choices=sorted(TASKS),
        help="Task name to run.",
    )
    parser.add_argument(
        "task_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected task.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    forwarded = list(args.task_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    cmd = TASKS[args.task] + forwarded
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[launch] cwd={REPO_ROOT}")
    print(f"[launch] exec: {rendered}")

    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
