"""Shared helpers for sample-centric JSONL manifests and attached artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL or JSON manifest into a list of records."""
    manifest_path = path.resolve()
    suffix = manifest_path.suffix.lower()

    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected JSON object in {manifest_path} line {line_number}.")
                records.append(payload)
        return records

    if suffix == ".json":
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"JSON manifest '{manifest_path}' must contain a list of objects.")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"JSON manifest '{manifest_path}' must contain only objects.")
        return payload

    raise ValueError(f"Unsupported manifest format for '{manifest_path}'. Use .jsonl, .ndjson, or .json.")


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records back to JSONL."""
    output_path = path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def resolve_manifest_path(manifest_path: Path, value: str | Path | None) -> Path | None:
    """Resolve a manifest-stored path value against the manifest location."""
    if value is None:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_path.resolve().parent / candidate).resolve()


def manifest_relative_path(manifest_path: Path, target_path: Path, path_mode: str) -> str:
    """Render a path for storage in a manifest."""
    resolved_target = target_path.resolve()
    if path_mode == "absolute":
        return str(resolved_target)
    manifest_parent = manifest_path.resolve().parent
    try:
        return resolved_target.relative_to(manifest_parent).as_posix()
    except ValueError:
        return os.path.relpath(str(resolved_target), start=str(manifest_parent))


def record_source_image_path(record: dict[str, Any], manifest_path: Path) -> Path | None:
    """Resolve the source image path for a manifest record."""
    for key in ("source_image_path", "path", "image_path", "file"):
        value = record.get(key)
        resolved = resolve_manifest_path(manifest_path, value)
        if resolved is not None:
            return resolved
    return None


def ensure_sample_structure(record: dict[str, Any]) -> dict[str, Any]:
    """Ensure record has sample-centric nested fields."""
    record.setdefault("artifacts", {})
    if not isinstance(record["artifacts"], dict):
        record["artifacts"] = {}
    record.setdefault("metadata", {})
    if not isinstance(record["metadata"], dict):
        record["metadata"] = {}
    if "source_image_path" not in record and "path" in record:
        record["source_image_path"] = record["path"]
    if "sample_id" not in record:
        record["sample_id"] = record.get("sha256") or record.get("path") or record.get("source_image_path")
    return record


def build_source_index(records: list[dict[str, Any]], manifest_path: Path) -> dict[str, int]:
    """Index manifest rows by resolved source image path."""
    index: dict[str, int] = {}
    for idx, record in enumerate(records):
        source_path = record_source_image_path(record, manifest_path)
        if source_path is not None:
            index[str(source_path)] = idx
    return index


def upsert_artifact(
    record: dict[str, Any],
    artifact_key: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Attach or replace an artifact payload on a sample record."""
    ensure_sample_structure(record)
    record["artifacts"][artifact_key] = payload
    return record


def attach_artifacts_by_source(
    manifest_path: Path,
    manifest_out: Path | None,
    artifact_key: str,
    payloads_by_source_path: dict[str, dict[str, Any]],
) -> int:
    """Attach artifact payloads to manifest rows matched by source image path."""
    resolved_manifest = manifest_path.resolve()
    target_manifest = manifest_out.resolve() if manifest_out else resolved_manifest
    records = load_records(resolved_manifest)
    source_index = build_source_index(records, resolved_manifest)

    updates = 0
    for source_path, payload in payloads_by_source_path.items():
        record_idx = source_index.get(str(Path(source_path).resolve()))
        if record_idx is None:
            continue
        upsert_artifact(records[record_idx], artifact_key=artifact_key, payload=payload)
        updates += 1

    write_records(target_manifest, records)
    return updates
