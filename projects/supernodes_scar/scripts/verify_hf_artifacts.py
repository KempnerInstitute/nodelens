#!/usr/bin/env python3
"""Verify a staged supernodes artifact bundle before upload."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

FORBIDDEN_SUFFIXES = (
    ".aux",
    ".blg",
    ".ckpt",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".synctex.gz",
)

PRIVATE_PATTERNS = (
    b"/n/",
    b"holylabs",
    b"holylfs",
    b"kempner_dev",
    b"Users/hsafaai",
    b"/home13/hsafaai",
    b"HF_TOKEN",
    b"WANDB_API_KEY",
)

TEXT_SUFFIXES = (
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> list[str]:
    errors: list[str] = []
    manifest = root / "MANIFEST.sha256"
    if not manifest.exists():
        return ["Missing MANIFEST.sha256"]
    for line_no, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            expected, rel = raw.split(None, 1)
        except ValueError:
            errors.append(f"Bad checksum line {line_no}: {raw}")
            continue
        path = root / rel.strip()
        if not path.exists():
            errors.append(f"Missing file from manifest: {rel}")
            continue
        got = sha256_file(path)
        if got != expected:
            errors.append(f"Checksum mismatch: {rel}")
    return errors


def verify_forbidden_files(root: Path) -> list[str]:
    errors: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or any(path.name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            errors.append(f"Forbidden file: {rel}")
    return errors


def scan_bytes(path: Path, lines) -> str | None:
    for line_no, line in lines:
        if any(pattern in line for pattern in PRIVATE_PATTERNS):
            return f"{path}: private pattern on line {line_no}"
    return None


def verify_private_paths(root: Path) -> list[str]:
    errors: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if path.name.endswith(".gz"):
            with gzip.open(path, "rb") as handle:
                hit = scan_bytes(rel, enumerate(handle, start=1))
        elif not path.name.endswith(TEXT_SUFFIXES):
            hit = None
        else:
            try:
                with path.open("rb") as handle:
                    hit = scan_bytes(rel, enumerate(handle, start=1))
            except UnicodeDecodeError:
                hit = None
        if hit:
            errors.append(hit)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()

    root = args.bundle_dir.resolve()
    errors: list[str] = []
    errors.extend(verify_manifest(root))
    errors.extend(verify_forbidden_files(root))
    errors.extend(verify_private_paths(root))

    if errors:
        for err in errors:
            print(f"ERROR: {err}")
        return 1

    print(f"Artifact bundle verified: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
