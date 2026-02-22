#!/usr/bin/env python3
"""Ensure compatibility wrapper references stay confined to approved files."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PATTERNS = [
    re.compile(r"\bpython\s+train\.py\b"),
    re.compile(r"\bpython\s+train_features\.py\b"),
    re.compile(r"\bpython\s+inference\.py\b"),
]


def _git_tracked_files(repo_root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _load_allowlist(path: Path) -> set[str]:
    allowlist: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        allowlist.add(line)
    return allowlist


def _is_text_like(path: Path) -> bool:
    return path.suffix.lower() in {
        "",
        ".md",
        ".py",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".sh",
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    allowlist_path = repo_root / "scripts" / "quality" / "wrapper_reference_allowlist.txt"
    allowlist = _load_allowlist(allowlist_path)

    hits: list[tuple[str, int, str]] = []
    for rel in _git_tracked_files(repo_root):
        if rel in allowlist:
            continue
        path = repo_root / rel
        if not path.is_file() or not _is_text_like(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in PATTERNS:
                if pattern.search(line):
                    hits.append((rel, lineno, line.strip()))

    if not hits:
        print("[wrapper-refs] ok (no non-allowlisted wrapper command references)")
        return 0

    print("[wrapper-refs] disallowed wrapper command references found:")
    for rel, lineno, line in hits:
        print(f"  - {rel}:{lineno}: {line}")
    print(
        "move usage to package entrypoints (python -m kurome.cli.*), "
        "or update wrapper_reference_allowlist.txt if intentionally historical."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
