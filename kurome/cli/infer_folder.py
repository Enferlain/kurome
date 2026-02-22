"""Batch folder inference CLI for classifier, scorer, and sequence-head models."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model inference over all images in a folder.")
    parser.add_argument("--src", required=True, help="Folder with source images.")
    parser.add_argument("--dst", default="output_passed", help="Output folder for passed images.")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.safetensors).")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to model .config.json; inferred from model path when omitted.",
    )
    parser.add_argument(
        "--arch",
        choices=["score", "class", "head_sequence"],
        required=True,
        help="Model architecture type.",
    )
    parser.add_argument("--min_score", type=int, default=0, help="Lower score bound (inclusive).")
    parser.add_argument("--max_score", type=int, default=100, help="Upper score bound (inclusive).")
    parser.add_argument(
        "--target_label_name",
        type=str,
        default=None,
        help="Label name used for score display/filtering (required for class arch).",
    )
    parser.add_argument(
        "--copy_passed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Copy images in score range into --dst.",
    )
    parser.add_argument(
        "--keep_structure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Preserve source subfolder tree under --dst.",
    )
    parser.add_argument(
        "--use_tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable tiling for classifier inference.",
    )
    parser.add_argument(
        "--tile_strategy",
        choices=["mean", "median", "max", "min"],
        default="mean",
        help="Tiling aggregation strategy.",
    )

    args = parser.parse_args()

    if args.arch == "class" and not args.target_label_name:
        parser.error("--target_label_name is required when --arch is class.")

    if args.arch == "head_sequence" and not args.target_label_name and (
        args.min_score > 0 or args.max_score < 100
    ):
        parser.error("--target_label_name is required when filtering with --arch head_sequence.")

    return args


def _infer_config_path(model_path: str) -> str:
    model_base_name = os.path.basename(model_path)
    if model_base_name.endswith(".safetensors"):
        model_base_name = model_base_name[: -len(".safetensors")]

    for suffix in ("_best_val", "_efinal"):
        if model_base_name.endswith(suffix):
            model_base_name = model_base_name[: -len(suffix)]
            break

    if "_s" in model_base_name:
        parts = model_base_name.split("_s")
        if len(parts) > 1 and parts[-1] and (parts[-1][0].isdigit() or parts[-1][-1] in {"K", "M"}):
            model_base_name = parts[0]

    return os.path.join(os.path.dirname(model_path), f"{model_base_name}.config.json")


def _build_pipeline(args: argparse.Namespace):
    import torch
    from kurome.inference.pipeline import (
        CityAestheticsPipeline,
        CityClassifierPipeline,
        HeadSequencePipeline,
    )

    pipeline_args: dict[str, object] = {}
    if torch.cuda.is_available():
        pipeline_args["device"] = "cuda"
        pipeline_args["clip_dtype"] = torch.float16

    if args.arch == "score":
        return CityAestheticsPipeline(args.model, config_path=args.config, **pipeline_args)
    if args.arch == "class":
        return CityClassifierPipeline(args.model, config_path=args.config, **pipeline_args)
    if args.arch == "head_sequence":
        return HeadSequencePipeline(args.model, config_path=args.config, **pipeline_args)

    raise ValueError(f"Unknown model architecture '{args.arch}'.")


def _prediction_score(pred_dict: dict[str, float], args: argparse.Namespace, labels: dict[str, str]) -> int | None:
    if args.target_label_name:
        target_prob = pred_dict.get(args.target_label_name)
        if target_prob is None:
            return None
        return int(target_prob * 100)

    if args.arch == "score":
        pos_label = labels.get("1", "1")
        target_prob = pred_dict.get(pos_label)
        if target_prob is None:
            return None
        return int(target_prob * 100)

    return None


def _iter_images(src_dir: str) -> list[str]:
    files: list[str] = []
    for root, _, names in os.walk(src_dir):
        for name in names:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                files.append(os.path.join(root, name))
    files.sort()
    return files


def _process_one_image(
    pipeline,
    labels: dict[str, str],
    args: argparse.Namespace,
    src_path: str,
    dst_path: str | None,
    image_cls,
    tqdm_cls,
) -> tuple[bool, bool]:
    try:
        with image_cls.open(src_path) as image:
            img = image.convert("RGB")
    except Exception as exc:
        tqdm_cls.write(f"ERR open [{os.path.basename(src_path)}] {exc}")
        return False, True

    try:
        if args.arch == "score":
            score_val = float(pipeline(img))
            pos_label = labels.get("1", "1")
            neg_label = labels.get("0", "0")
            pred_dict = {pos_label: score_val, neg_label: 1.0 - score_val}
        elif args.arch == "class":
            pred_dict = pipeline(raw_pil_image=img, tiling=args.use_tiling, tile_strat=args.tile_strategy)
        else:
            pred_dict = pipeline(raw_pil_image=img)
    except Exception as exc:
        tqdm_cls.write(f"ERR infer [{os.path.basename(src_path)}] {exc}")
        return False, True

    score = _prediction_score(pred_dict, args, labels)
    if score is None:
        tqdm_cls.write(f"ERR label [{os.path.basename(src_path)}]")
        return False, True

    display_label = f" ({args.target_label_name})" if args.target_label_name else ""
    tqdm_cls.write(f"{score:>3}% [{os.path.basename(src_path)}]{display_label}")

    if not (args.min_score <= score <= args.max_score):
        return False, False

    if dst_path:
        try:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return True, False
        except Exception as exc:
            tqdm_cls.write(f"ERR copy [{os.path.basename(src_path)}] {exc}")
            return False, True

    return False, False


def main() -> int:
    args = _parse_args()
    try:
        from PIL import Image
        from tqdm import tqdm
    except ImportError as exc:
        print(f"Error: missing runtime dependency for inference CLI: {exc}")
        return 1

    if args.config is None:
        args.config = _infer_config_path(args.model)

    if not os.path.isfile(args.config):
        print(f"Error: config file not found: {args.config}")
        return 1

    if args.copy_passed:
        os.makedirs(args.dst, exist_ok=True)

    print(f"Using model: {os.path.basename(args.model)}")
    print(f"Model architecture: {args.arch}")

    try:
        pipeline = _build_pipeline(args)
    except Exception as exc:
        print(f"Error initializing pipeline: {exc}")
        return 1

    labels = getattr(pipeline, "labels", {})
    files = _iter_images(args.src)
    if not files:
        print("No image files found in source directory.")
        return 0

    processed = 0
    copied = 0
    failed = 0

    with tqdm(total=len(files), desc="Processing folder", unit="image") as pbar:
        for src_path in files:
            dst_path = None
            if args.copy_passed:
                dst_dir = args.dst
                if args.keep_structure:
                    src_rel = os.path.relpath(os.path.dirname(src_path), args.src)
                    if src_rel != ".":
                        dst_dir = os.path.join(args.dst, src_rel)
                dst_path = os.path.join(dst_dir, os.path.basename(src_path))

            copied_this, failed_this = _process_one_image(
                pipeline,
                labels,
                args,
                src_path,
                dst_path,
                image_cls=Image,
                tqdm_cls=tqdm,
            )
            processed += 1
            copied += int(copied_this)
            failed += int(failed_this)
            pbar.update(1)

    print("\nProcessing complete.")
    print(f"  Processed: {processed} images")
    if args.copy_passed:
        print(f"  Copied:    {copied} images")
    if failed > 0:
        print(f"  Errors:    {failed} images")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
