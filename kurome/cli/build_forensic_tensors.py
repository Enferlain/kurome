from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from kurome.sample_manifest import (
    build_source_index,
    load_records,
    manifest_relative_path,
    record_source_image_path,
    upsert_artifact,
    write_records,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS:
                    files.append(child)
    return sorted(set(files))


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def infer_common_root(files: list[Path]) -> Path:
    if not files:
        return Path(".")
    parents = [str(path.parent.resolve()) for path in files]
    return Path(parents[0] if len(parents) == 1 else os.path.commonpath(parents))


def relative_image_path(path: Path, root: Path | None) -> Path:
    if root is None:
        return Path(path.name)
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return Path(path.name)


def median_blur_rgb(rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    filtered = image.filter(ImageFilter.MedianFilter(size=kernel_size))
    return np.asarray(filtered, dtype=np.float32)


def mean_blur_gray(gray: np.ndarray, radius: int) -> np.ndarray:
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    filtered = image.filter(ImageFilter.BoxBlur(radius=radius))
    return np.asarray(filtered, dtype=np.float32)


def signed_residual_rgb(rgb: np.ndarray, kernel_size: int) -> np.ndarray:
    return rgb - median_blur_rgb(rgb, kernel_size=kernel_size)


def residual_magnitude_map(residual_rgb: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(residual_rgb), axis=2)


def signed_luma_residual(residual_rgb: np.ndarray) -> np.ndarray:
    return residual_rgb.mean(axis=2)


def block_boundary_map(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    h, w = gray.shape
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])

    grid = np.zeros((h, w), dtype=np.float32)
    for x in range(7, w - 1, 8):
        grid[:, x] += dx[:, x]
    for y in range(7, h - 1, 8):
        grid[y, :] += dy[y, :]
    return grid


def local_noise_energy_map(residual_rgb: np.ndarray, blur_radius: int) -> np.ndarray:
    magnitude = residual_magnitude_map(residual_rgb)
    energy = magnitude * magnitude
    smoothed = mean_blur_gray(np.clip(energy, 0, 255), radius=blur_radius)
    return np.sqrt(np.maximum(smoothed, 0.0))


def robust_scale_channel(channel: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    channel = channel.astype(np.float32)
    scale = float(np.percentile(np.abs(channel), percentile))
    if scale < 1e-6:
        scale = 1.0
    normalized = np.clip(channel / scale, -1.0, 1.0)
    return normalized.astype(np.float32)


def nonnegative_scale_channel(channel: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    channel = channel.astype(np.float32)
    scale = float(np.percentile(channel, percentile))
    if scale < 1e-6:
        scale = 1.0
    normalized = np.clip(channel / scale, 0.0, 1.0)
    return normalized.astype(np.float32)


def build_tensor_preset_1(rgb: np.ndarray, kernel_size: int, blur_radius: int) -> tuple[np.ndarray, list[str]]:
    del blur_radius
    residual_rgb = signed_residual_rgb(rgb, kernel_size=kernel_size)
    tensor = np.stack(
        [
            nonnegative_scale_channel(residual_magnitude_map(residual_rgb)),
            robust_scale_channel(signed_luma_residual(residual_rgb)),
            nonnegative_scale_channel(block_boundary_map(rgb)),
        ],
        axis=0,
    )
    channels = ["abs_residual", "signed_residual_luma", "block_boundary"]
    return tensor, channels


def build_tensor_preset_2(rgb: np.ndarray, kernel_size: int, blur_radius: int) -> tuple[np.ndarray, list[str]]:
    residual_rgb = signed_residual_rgb(rgb, kernel_size=kernel_size)
    tensor = np.stack(
        [
            robust_scale_channel(residual_rgb[:, :, 0]),
            robust_scale_channel(residual_rgb[:, :, 1]),
            robust_scale_channel(residual_rgb[:, :, 2]),
            nonnegative_scale_channel(residual_magnitude_map(residual_rgb)),
            nonnegative_scale_channel(block_boundary_map(rgb)),
            nonnegative_scale_channel(local_noise_energy_map(residual_rgb, blur_radius=blur_radius)),
        ],
        axis=0,
    )
    channels = [
        "signed_residual_r",
        "signed_residual_g",
        "signed_residual_b",
        "abs_residual",
        "block_boundary",
        "local_noise_energy",
    ]
    return tensor, channels


PRESETS = {
    "preset1": build_tensor_preset_1,
    "preset2": build_tensor_preset_2,
}


def write_npz(
    out_path: Path,
    image_path: Path,
    preset: str,
    tensor: np.ndarray,
    channels: list[str],
    kernel_size: int,
    blur_radius: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        tensor=tensor.astype(np.float32),
        channels=np.array(channels, dtype="<U32"),
        preset=np.array(preset),
        source_path=np.array(str(image_path)),
        original_shape=np.array(tensor.shape[1:], dtype=np.int32),
        kernel_size=np.array(kernel_size, dtype=np.int32),
        blur_radius=np.array(blur_radius, dtype=np.int32),
    )


def export_tensor(
    image_path: Path,
    output_root: Path,
    preset: str,
    kernel_size: int,
    blur_radius: int,
    input_root: Path | None,
) -> tuple[Path, np.ndarray, list[str]]:
    rgb = read_rgb(image_path)
    tensor, channels = PRESETS[preset](rgb, kernel_size=kernel_size, blur_radius=blur_radius)
    rel = relative_image_path(image_path, input_root)
    out_path = output_root / rel.parent / f"{rel.stem}.{preset}.npz"
    write_npz(
        out_path=out_path,
        image_path=image_path,
        preset=preset,
        tensor=tensor,
        channels=channels,
        kernel_size=kernel_size,
        blur_radius=blur_radius,
    )
    return out_path, tensor, channels


def select_manifest_records(
    records: list[dict[str, Any]],
    manifest_path: Path,
    selected_files: set[str] | None,
) -> list[tuple[dict[str, Any], Path]]:
    selected: list[tuple[dict[str, Any], Path]] = []
    for record in records:
        source_path = record_source_image_path(record, manifest_path)
        if source_path is None or source_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if selected_files is not None and str(source_path) not in selected_files:
            continue
        selected.append((record, source_path))
    return selected


def artifact_payload(
    out_path: Path,
    manifest_path: Path,
    artifact_path_mode: str,
    preset: str,
    channels: list[str],
    tensor: np.ndarray,
    kernel_size: int,
    blur_radius: int,
) -> dict[str, Any]:
    return {
        "path": manifest_relative_path(manifest_path, out_path, artifact_path_mode),
        "artifact_type": "forensic_tensor_npz",
        "format": "npz",
        "preset": preset,
        "channels": list(channels),
        "tensor_shape": list(map(int, tensor.shape)),
        "original_shape": list(map(int, tensor.shape[1:])),
        "kernel_size": kernel_size,
        "blur_radius": blur_radius,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic forensic tensor presets for image-model training."
    )
    parser.add_argument("paths", nargs="*", type=Path, help="Image files or folders to process.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional sample manifest to update with artifact entries. If provided, paths can be omitted to process all manifest samples.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Where to write the updated manifest. Defaults to overwriting --manifest.",
    )
    parser.add_argument(
        "--artifact-key",
        type=str,
        default=None,
        help="Artifact key to write into each manifest row. Defaults to the preset name.",
    )
    parser.add_argument(
        "--artifact-path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="How artifact paths are stored inside the manifest.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        required=True,
        help="Which tensor preset to build.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/forensic_tensors"),
        help="Where to save .npz tensor files.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Optional root used to preserve relative output paths.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Median-filter kernel size for residual extraction.",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=2,
        help="Box-blur radius for local noise energy smoothing.",
    )
    parser.add_argument(
        "--output-records-only",
        action="store_true",
        help="Build artifacts and update the manifest, but do not print a per-file line for each sample.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.kernel_size < 3 or args.kernel_size % 2 == 0:
        parser.error("--kernel-size must be an odd integer >= 3")

    selected_files: set[str] | None = None
    if args.paths:
        selected_files = {str(path.resolve()) for path in list_images(args.paths)}
        if not selected_files and args.manifest is None:
            parser.error("no images found")

    files = sorted(Path(path) for path in selected_files) if selected_files is not None else []

    manifest_records: list[dict[str, Any]] | None = None
    manifest_path: Path | None = None
    manifest_out: Path | None = None
    manifest_index: dict[str, int] = {}
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        manifest_out = args.manifest_out.resolve() if args.manifest_out else manifest_path
        manifest_records = load_records(manifest_path)
        manifest_index = build_source_index(manifest_records, manifest_path)

        manifest_selection = select_manifest_records(
            records=manifest_records,
            manifest_path=manifest_path,
            selected_files=selected_files,
        )
        files = [source_path for _, source_path in manifest_selection]

    if not files:
        parser.error("no images found")

    input_root = args.input_root or infer_common_root(files)
    artifact_key = args.artifact_key or args.preset

    print(f"Building {args.preset} tensors for {len(files)} images into {args.output_root}")
    print(f"Input root: {input_root}")
    if manifest_path is not None:
        print(f"Updating manifest: {manifest_path} -> {manifest_out}")
        print(f"Artifact key: {artifact_key}")

    artifact_updates = 0
    for image_path in files:
        out_path, tensor, channels = export_tensor(
            image_path=image_path,
            output_root=args.output_root,
            preset=args.preset,
            kernel_size=args.kernel_size,
            blur_radius=args.blur_radius,
            input_root=input_root,
        )
        if not args.output_records_only:
            print(f"{image_path} -> {out_path}")

        if manifest_records is not None and manifest_path is not None:
            record_idx = manifest_index.get(str(image_path.resolve()))
            if record_idx is not None:
                upsert_artifact(
                    manifest_records[record_idx],
                    artifact_key=artifact_key,
                    payload=artifact_payload(
                        out_path=out_path,
                        manifest_path=manifest_out or manifest_path,
                        artifact_path_mode=args.artifact_path_mode,
                        preset=args.preset,
                        channels=channels,
                        tensor=tensor,
                        kernel_size=args.kernel_size,
                        blur_radius=args.blur_radius,
                    ),
                )
                artifact_updates += 1

    if manifest_records is not None and manifest_out is not None:
        write_records(manifest_out, manifest_records)
        print(f"Updated {artifact_updates} manifest rows in {manifest_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
