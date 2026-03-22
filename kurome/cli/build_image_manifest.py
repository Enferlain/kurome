"""Build a rich JSONL image manifest from a labeled folder tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from kurome.sample_manifest import ensure_sample_structure

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "te": "test",
}


@dataclass
class Sample:
    """A discovered image sample before final manifest serialization."""

    path: Path
    label: str
    split: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a rich JSONL manifest from image folders.")
    parser.add_argument("--src", type=Path, required=True, help="Dataset root to scan.")
    parser.add_argument("--output", type=Path, default=None, help="Output manifest path. Defaults to <src>/manifest.jsonl.")
    parser.add_argument(
        "--layout",
        choices=["auto", "class", "split-class"],
        default="auto",
        help="Folder layout: class/* or split/class/*. 'auto' infers from the top-level directories.",
    )
    parser.add_argument(
        "--class-names",
        nargs="*",
        default=None,
        help="Optional explicit class order. Extra discovered classes will be skipped.",
    )
    parser.add_argument(
        "--split-mode",
        choices=["auto", "layout", "random", "none"],
        default="auto",
        help="How to populate the split field. 'layout' uses split dirs, 'random' creates splits, 'none' leaves split blank.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Validation ratio for random splitting.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Test ratio for random splitting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and random splitting.")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional cap on samples kept per class.")
    parser.add_argument(
        "--hash",
        dest="hash_name",
        choices=["none", "sha256"],
        default="sha256",
        help="Hash algorithm to compute for each image.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Whether manifest 'path' values are relative to --src or absolute.",
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        action="append",
        default=None,
        help="Additional metadata.jsonl files to merge. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-auto-metadata",
        action="store_true",
        help="Do not auto-discover metadata.jsonl files underneath --src.",
    )
    parser.add_argument(
        "--embed-sidecar",
        choices=["none", "selected", "raw"],
        default="selected",
        help="How much matched sidecar metadata to embed in each manifest row.",
    )
    parser.add_argument(
        "--txt-sidecars",
        choices=["none", "parsed", "raw", "both"],
        default="parsed",
        help="How to embed same-stem .txt sidecars found next to images.",
    )
    return parser.parse_args()


def canonical_split(name: str | None) -> str | None:
    """Normalize split names into train/val/test."""
    if not name:
        return None
    return SPLIT_ALIASES.get(name.strip().lower())


def infer_layout(src: Path) -> str:
    """Infer whether src is class/* or split/class/*."""
    dirs = [child for child in src.iterdir() if child.is_dir() and not child.name.startswith(".")]
    if dirs and all(canonical_split(child.name) for child in dirs):
        has_class_children = any(
            grandchild.is_dir()
            for child in dirs
            for grandchild in child.iterdir()
        )
        if has_class_children:
            return "split-class"
    return "class"


def iter_image_files(folder: Path) -> list[Path]:
    """Return sorted image files directly inside a folder."""
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def discover_samples(src: Path, layout: str, class_names: list[str] | None) -> list[Sample]:
    """Discover samples from the dataset root."""
    allowed = set(class_names or [])
    samples: list[Sample] = []

    if layout == "split-class":
        for split_dir in sorted(child for child in src.iterdir() if child.is_dir() and not child.name.startswith(".")):
            split = canonical_split(split_dir.name)
            if split is None:
                continue
            for class_dir in sorted(child for child in split_dir.iterdir() if child.is_dir() and not child.name.startswith(".")):
                label = class_dir.name
                if allowed and label not in allowed:
                    continue
                for image_path in iter_image_files(class_dir):
                    samples.append(Sample(path=image_path, label=label, split=split))
        return samples

    for class_dir in sorted(child for child in src.iterdir() if child.is_dir() and not child.name.startswith(".")):
        label = class_dir.name
        if allowed and label not in allowed:
            continue
        for image_path in iter_image_files(class_dir):
            samples.append(Sample(path=image_path, label=label))
    return samples


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_number}.")
            records.append(payload)
    return records


def infer_metadata_keys(record: dict[str, Any]) -> set[str]:
    """Derive lookup keys for a sidecar metadata record."""
    keys: set[str] = set()
    post_id = record.get("id")
    if post_id is not None:
        keys.add(str(post_id))

    for url_key in ("file_url", "large_file_url", "source"):
        value = record.get(url_key)
        if isinstance(value, str) and value:
            parsed = Path(value.split("?", 1)[0])
            if parsed.name:
                keys.add(parsed.name)
                keys.add(parsed.stem)

    for path_key in ("path", "file", "image_path"):
        value = record.get(path_key)
        if isinstance(value, str) and value:
            parsed = Path(value)
            keys.add(parsed.name)
            keys.add(parsed.stem)

    return {key for key in keys if key}


def discover_metadata_paths(src: Path, explicit_paths: list[Path] | None, auto_discover: bool) -> list[Path]:
    """Collect metadata.jsonl files to merge."""
    paths: list[Path] = []
    seen: set[Path] = set()

    for path in explicit_paths or []:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            paths.append(resolved)

    if auto_discover:
        for candidate in src.rglob("metadata.jsonl"):
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)

    return sorted(paths)


def build_metadata_index(metadata_paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Index sidecar metadata by post id and basename/stem."""
    index: dict[str, dict[str, Any]] = {}
    for metadata_path in metadata_paths:
        for record in load_jsonl_records(metadata_path):
            keys = infer_metadata_keys(record)
            for key in keys:
                index.setdefault(key, record)
    return index


def match_sidecar(record_index: dict[str, dict[str, Any]], image_path: Path) -> dict[str, Any] | None:
    """Find a sidecar metadata record for an image path."""
    keys = [image_path.name, image_path.stem]
    if image_path.stem.isdigit():
        keys.insert(0, image_path.stem)
    for key in keys:
        record = record_index.get(key)
        if record is not None:
            return record
    return None


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_txt_sidecar(image_path: Path) -> Path | None:
    """Return a same-stem .txt sidecar if present."""
    txt_path = image_path.with_suffix(".txt")
    if txt_path.is_file():
        return txt_path
    return None


def read_text_sidecar(path: Path) -> str:
    """Read a text sidecar with forgiving UTF-8 decoding."""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def parse_txt_tags(raw_text: str) -> dict[str, Any]:
    """Parse booru-style comma-separated tags into namespaces and derived fields."""
    tags = [tag.strip() for tag in raw_text.replace("\n", ",").split(",")]
    tags = [tag for tag in tags if tag]

    by_namespace: dict[str, list[str]] = defaultdict(list)
    unscoped_tags: list[str] = []
    for tag in tags:
        if ":" in tag:
            namespace, value = tag.split(":", 1)
            namespace = namespace.strip().lower()
            value = value.strip()
            if namespace and value:
                by_namespace[namespace].append(value)
                continue
        unscoped_tags.append(tag)

    meta_tags = by_namespace.get("meta", [])
    lower_meta_tags = {tag.lower() for tag in meta_tags}
    derived = {
        "has_adversarial_noise_meta": "adversarial noise" in lower_meta_tags,
        "artist_count": len(by_namespace.get("artist", [])),
        "character_count": len(by_namespace.get("character", [])),
        "copyright_count": len(by_namespace.get("copyright", [])),
        "meta_count": len(meta_tags),
        "general_tag_count": len(unscoped_tags),
    }

    return {
        "raw": raw_text,
        "tags": tags,
        "tag_count": len(tags),
        "tags_by_namespace": dict(sorted(by_namespace.items())),
        "unscoped_tags": unscoped_tags,
        "derived": derived,
    }


def summarize_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Keep the most useful metadata fields for downstream grouping and analysis."""
    summary: dict[str, Any] = {}
    field_map = {
        "id": "post_id",
        "created_at": "created_at",
        "source": "source_url",
        "rating": "rating",
        "score": "score",
        "fav_count": "fav_count",
        "tag_string": "tag_string",
        "tag_string_artist": "tag_string_artist",
        "tag_string_character": "tag_string_character",
        "tag_string_copyright": "tag_string_copyright",
        "tag_string_meta": "tag_string_meta",
        "uploader_id": "uploader_id",
        "pixiv_id": "pixiv_id",
        "image_width": "source_image_width",
        "image_height": "source_image_height",
        "file_ext": "source_file_ext",
        "file_size": "source_file_size",
        "md5": "source_md5",
    }
    for source_key, target_key in field_map.items():
        value = sidecar.get(source_key)
        if value not in (None, ""):
            summary[target_key] = value
    summary["metadata_source"] = "danbooru"
    return summary


def assign_random_splits(samples: list[Sample], val_ratio: float, test_ratio: float, seed: int) -> None:
    """Assign splits within each class."""
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    for label, label_samples in grouped.items():
        rng.shuffle(label_samples)
        total = len(label_samples)
        test_count = int(total * test_ratio)
        val_count = int(total * val_ratio)
        if test_ratio > 0 and test_count == 0 and total > 0:
            test_count = 1
        if val_ratio > 0 and val_count == 0 and total - test_count > 0:
            val_count = 1
        if test_count + val_count > total:
            overflow = test_count + val_count - total
            if val_count >= overflow:
                val_count -= overflow
            else:
                test_count = max(0, test_count - (overflow - val_count))
                val_count = 0
        for index, sample in enumerate(label_samples):
            if index < test_count:
                sample.split = "test"
            elif index < test_count + val_count:
                sample.split = "val"
            else:
                sample.split = "train"


def sample_per_class(samples: list[Sample], limit_per_class: int | None, seed: int) -> list[Sample]:
    """Cap each class to at most limit_per_class samples."""
    if limit_per_class is None:
        return samples

    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)
    limited: list[Sample] = []
    for label, label_samples in grouped.items():
        if len(label_samples) <= limit_per_class:
            limited.extend(label_samples)
            continue
        chosen = list(label_samples)
        rng.shuffle(chosen)
        limited.extend(chosen[:limit_per_class])
        print(f"[manifest] limited class '{label}' from {len(label_samples)} to {limit_per_class} samples")
    return limited


def build_record(
    sample: Sample,
    src: Path,
    label_to_index: dict[str, int],
    path_mode: str,
    hash_name: str,
    sidecar_mode: str,
    sidecar: dict[str, Any] | None,
    txt_sidecar_mode: str,
) -> dict[str, Any]:
    """Convert a discovered sample into a manifest row."""
    stat = sample.path.stat()
    with Image.open(sample.path) as image:
        width, height = image.size
        mode = image.mode
        bands = image.getbands()
        image_format = (image.format or sample.path.suffix.lstrip(".")).lower()

    rel_path = sample.path.relative_to(src).as_posix()
    source_image_path = rel_path if path_mode == "relative" else str(sample.path.resolve())
    metadata: dict[str, Any] = {
        "file_name": sample.path.name,
        "stem": sample.path.stem,
        "suffix": sample.path.suffix.lower(),
        "format": image_format,
        "width": width,
        "height": height,
        "aspect_ratio": (width / height) if height else None,
        "mode": mode,
        "channels": len(bands),
        "bands": list(bands),
        "has_alpha": "A" in bands,
        "file_size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "parent_dir": sample.path.parent.name,
        "relative_parent": sample.path.parent.relative_to(src).as_posix(),
    }
    row: dict[str, Any] = {
        "path": rel_path if path_mode == "relative" else str(sample.path.resolve()),
        "source_image_path": source_image_path,
        "label": sample.label,
        "label_index": label_to_index.get(sample.label),
        "split": sample.split,
        **metadata,
        "metadata": dict(metadata),
        "artifacts": {},
    }
    if hash_name == "sha256":
        row["sha256"] = compute_sha256(sample.path)
        row["metadata"]["sha256"] = row["sha256"]

    row["sample_id"] = row.get("sha256") or rel_path
    ensure_sample_structure(row)

    if sidecar is not None and sidecar_mode != "none":
        sidecar_payload = sidecar if sidecar_mode == "raw" else summarize_sidecar(sidecar)
        row["source_metadata"] = sidecar_payload
        row["metadata"]["source_metadata"] = sidecar_payload

    txt_sidecar_path = find_txt_sidecar(sample.path)
    if txt_sidecar_path is not None and txt_sidecar_mode != "none":
        txt_raw = read_text_sidecar(txt_sidecar_path)
        txt_parsed = parse_txt_tags(txt_raw)
        row["txt_sidecar_path"] = (
            txt_sidecar_path.relative_to(src).as_posix()
            if path_mode == "relative"
            else str(txt_sidecar_path.resolve())
        )
        row["metadata"]["txt_sidecar_path"] = row["txt_sidecar_path"]
        if txt_sidecar_mode in {"raw", "both"}:
            row["txt_sidecar_raw"] = txt_raw
            row["metadata"]["txt_sidecar_raw"] = txt_raw
        if txt_sidecar_mode in {"parsed", "both"}:
            txt_payload = {
                "tag_count": txt_parsed["tag_count"],
                "tags": txt_parsed["tags"],
                "tags_by_namespace": txt_parsed["tags_by_namespace"],
                "unscoped_tags": txt_parsed["unscoped_tags"],
                "derived": txt_parsed["derived"],
            }
            row["txt_sidecar"] = txt_payload
            row["metadata"]["txt_sidecar"] = txt_payload

    return row


def validate_ratios(val_ratio: float, test_ratio: float) -> None:
    """Ensure random split ratios make sense."""
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("--val-ratio must be in [0, 1).")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("--test-ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("--val-ratio + --test-ratio must be < 1.")


def main() -> int:
    args = parse_args()
    src = args.src.resolve()
    if not src.is_dir():
        raise SystemExit(f"--src does not exist or is not a directory: {src}")

    validate_ratios(args.val_ratio, args.test_ratio)
    output = args.output.resolve() if args.output else (src / "manifest.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)

    layout = infer_layout(src) if args.layout == "auto" else args.layout
    class_names = list(args.class_names) if args.class_names else None
    samples = discover_samples(src=src, layout=layout, class_names=class_names)
    if not samples:
        raise SystemExit(f"No image files found under {src} using layout '{layout}'.")

    samples = sample_per_class(samples=samples, limit_per_class=args.limit_per_class, seed=args.seed)
    discovered_labels = sorted({sample.label for sample in samples})
    ordered_labels = [label for label in (class_names or discovered_labels) if label in discovered_labels]
    label_to_index = {label: index for index, label in enumerate(ordered_labels)}

    split_mode = args.split_mode
    if split_mode == "auto":
        split_mode = "layout" if any(sample.split for sample in samples) else ("random" if (args.val_ratio or args.test_ratio) else "none")

    if split_mode == "random":
        assign_random_splits(samples=samples, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    elif split_mode == "none":
        for sample in samples:
            sample.split = None
    elif split_mode == "layout":
        for sample in samples:
            sample.split = canonical_split(sample.split)

    metadata_paths = discover_metadata_paths(
        src=src,
        explicit_paths=args.metadata_jsonl,
        auto_discover=not args.no_auto_metadata,
    )
    metadata_index = build_metadata_index(metadata_paths) if metadata_paths else {}

    records: list[dict[str, Any]] = []
    sidecar_matches = 0
    txt_sidecar_matches = 0
    for sample in tqdm(sorted(samples, key=lambda item: (item.label, item.path.as_posix())), desc="Building manifest"):
        try:
            sidecar = match_sidecar(metadata_index, sample.path) if metadata_index else None
            if sidecar is not None:
                sidecar_matches += 1
            if args.txt_sidecars != "none" and find_txt_sidecar(sample.path) is not None:
                txt_sidecar_matches += 1
            records.append(
                build_record(
                    sample=sample,
                    src=src,
                    label_to_index=label_to_index,
                    path_mode=args.path_mode,
                    hash_name=args.hash_name,
                    sidecar_mode=args.embed_sidecar,
                    sidecar=sidecar,
                    txt_sidecar_mode=args.txt_sidecars,
                )
            )
        except (OSError, UnidentifiedImageError) as exc:
            print(f"[manifest] skipping unreadable image: {sample.path} ({exc})")

    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record["label"]] += 1
        split_counts[str(record.get("split") or "unspecified")] += 1

    print(f"[manifest] wrote {len(records)} rows to {output}")
    print(f"[manifest] layout={layout} classes={ordered_labels}")
    print(f"[manifest] counts_by_class={dict(sorted(counts.items()))}")
    print(f"[manifest] counts_by_split={dict(sorted(split_counts.items()))}")
    if metadata_paths:
        print(f"[manifest] merged metadata from {len(metadata_paths)} metadata.jsonl file(s); matched {sidecar_matches} samples")
    if args.txt_sidecars != "none":
        print(f"[manifest] matched same-stem .txt sidecars for {txt_sidecar_matches} samples")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
