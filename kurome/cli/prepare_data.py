"""Single entrypoint for building manifests and attached artifacts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare sample manifest plus optional forensic tensor and feature artifacts."
    )
    parser.add_argument("--src", type=Path, required=True, help="Labeled image dataset root.")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest output path. Defaults to <src>/manifest.jsonl.")
    parser.add_argument("--layout", choices=["auto", "class", "split-class"], default="auto")
    parser.add_argument("--class-names", nargs="*", default=None)
    parser.add_argument("--split-mode", choices=["auto", "layout", "random", "none"], default="auto")
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-per-class", type=int, default=None)
    parser.add_argument("--path-mode", choices=["relative", "absolute"], default="relative")
    parser.add_argument("--embed-sidecar", choices=["none", "selected", "raw"], default="selected")
    parser.add_argument("--txt-sidecars", choices=["none", "parsed", "raw", "both"], default="parsed")
    parser.add_argument("--skip-manifest", action="store_true", help="Reuse an existing manifest instead of rebuilding it.")

    parser.add_argument("--forensic-preset", action="append", default=[], help="Forensic tensor preset(s) to build.")
    parser.add_argument("--forensic-output-root", type=Path, default=Path("data/forensic_tensors"))
    parser.add_argument("--forensic-kernel-size", type=int, default=3)
    parser.add_argument("--forensic-blur-radius", type=int, default=2)

    parser.add_argument("--feature-model", action="append", default=[], help="Vision model(s) to generate feature sequences with.")
    parser.add_argument("--feature-output-root", type=Path, default=Path("data"))
    parser.add_argument("--feature-compute-precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--feature-save-precision", type=str, default="fp16", choices=["fp32", "fp16"])
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print(f"[prepare-data] exec: {rendered}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    src = args.src.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else (src / "manifest.jsonl")

    if not args.skip_manifest:
        cmd = [
            sys.executable,
            "-m",
            "kurome.cli.build_image_manifest",
            "--src",
            str(src),
            "--output",
            str(manifest_path),
            "--layout",
            args.layout,
            "--split-mode",
            args.split_mode,
            "--val-ratio",
            str(args.val_ratio),
            "--test-ratio",
            str(args.test_ratio),
            "--seed",
            str(args.seed),
            "--path-mode",
            args.path_mode,
            "--embed-sidecar",
            args.embed_sidecar,
            "--txt-sidecars",
            args.txt_sidecars,
        ]
        if args.class_names:
            cmd.extend(["--class-names", *args.class_names])
        if args.limit_per_class is not None:
            cmd.extend(["--limit-per-class", str(args.limit_per_class)])
        run_step(cmd)

    for preset in args.forensic_preset:
        run_step(
            [
                sys.executable,
                "-m",
                "kurome.cli.build_forensic_tensors",
                "--manifest",
                str(manifest_path),
                "--preset",
                preset,
                "--output-root",
                str(args.forensic_output_root),
                "--artifact-key",
                preset,
                "--input-root",
                str(src),
                "--kernel-size",
                str(args.forensic_kernel_size),
                "--blur-radius",
                str(args.forensic_blur_radius),
            ]
        )

    for model_name in args.feature_model:
        run_step(
            [
                sys.executable,
                "-m",
                "kurome.cli.generate_feature_sequences",
                "--image_dir",
                str(src),
                "--output_dir_root",
                str(args.feature_output_root),
                "--model_name",
                model_name,
                "--compute_precision",
                args.feature_compute_precision,
                "--save_precision",
                args.feature_save_precision,
                "--manifest",
                str(manifest_path),
            ]
        )

    print(f"[prepare-data] manifest ready at {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
