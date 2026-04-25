#!/usr/bin/env python3
"""Prepare a sanitized Hugging Face artifact bundle for the supernodes paper."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

EXCLUDED_SUFFIXES = {
    ".aux",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
    ".pyc",
    ".synctex.gz",
}

TEXT_SUFFIXES = {
    ".json",
    ".md",
    ".py",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

PRIVATE_PATH_PATTERNS = (
    (re.compile("/" + r"n/home[0-9]*/[^\s\"',)]+"), "/path/to/user_home"),
    (re.compile("/" + r"n/[^\s\"',)]+"), "/path/to/internal_storage"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def git_is_dirty(repo_root: Path) -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except Exception:
        return None


def should_skip(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts:
        return True
    name = path.name
    return any(name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES)


def sanitize_text(text: str) -> str:
    for pattern, replacement in PRIVATE_PATH_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def iter_sanitized_lines(src: Path):
    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield sanitize_text(line)


def is_text_like(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in TEXT_SUFFIXES)


def copy_one(
    src: Path,
    dst: Path,
    group: str,
    entries: list[dict[str, Any]],
    *,
    sanitize: bool = True,
) -> None:
    if not src.exists() or not src.is_file() or should_skip(src):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sanitize and is_text_like(src):
        with dst.open("w", encoding="utf-8") as handle:
            handle.writelines(iter_sanitized_lines(src))
    else:
        shutil.copy2(src, dst)
    entries.append(
        {
            "path": dst.as_posix(),
            "group": group,
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        }
    )


def copy_text_gzip(src: Path, dst: Path, group: str, entries: list[dict[str, Any]]) -> None:
    if not src.exists() or not src.is_file() or should_skip(src):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dst, "wt", encoding="utf-8", compresslevel=1) as handle:
        handle.writelines(iter_sanitized_lines(src))
    entries.append(
        {
            "path": dst.as_posix(),
            "group": group,
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        }
    )


def copy_tree_files(src_root: Path, dst_root: Path, group: str, entries: list[dict[str, Any]]) -> None:
    if not src_root.exists():
        return
    for src in sorted(p for p in src_root.rglob("*") if p.is_file()):
        if should_skip(src):
            continue
        copy_one(src, dst_root / src.relative_to(src_root), group, entries)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def copy_named_json_sources(
    mapping: dict[str, Any],
    dst_root: Path,
    group: str,
    entries: list[dict[str, Any]],
    source_entries: list[dict[str, Any]],
    *,
    compress: bool = True,
) -> None:
    for key, raw_path in sorted(mapping.items()):
        if not isinstance(raw_path, str):
            continue
        src = Path(raw_path)
        if not src.exists() or should_skip(src):
            continue
        suffix = "".join(src.suffixes) or ".json"
        safe_name = f"{key}{suffix}.gz" if compress else f"{key}{suffix}"
        public_path = (dst_root / safe_name).relative_to(dst_root.parents[1]).as_posix()
        print(f"Staging result: {public_path}", flush=True)
        if compress:
            copy_text_gzip(src, dst_root / safe_name, group, entries)
        else:
            copy_one(src, dst_root / safe_name, group, entries)
        source_entries.append(
            {
                "name": key,
                "artifact_path": public_path,
                "source_kind": "locked_result_json",
                "compressed": compress,
            }
        )


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def relativize_entries(entries: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    clean_entries: list[dict[str, Any]] = []
    for item in entries:
        clean_item = dict(item)
        path = Path(str(clean_item["path"]))
        if path.is_absolute():
            clean_item["path"] = path.relative_to(root).as_posix()
        clean_entries.append(clean_item)
    return clean_entries


def main() -> int:
    script_path = Path(__file__).resolve()
    default_repo_root = script_path.parents[3]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_repo_root)
    parser.add_argument("--paper-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clean", action="store_true", help="Remove output directory before staging.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    paper_dir = (args.paper_dir or repo_root / "drafts" / "LLM_prune").resolve()
    output_dir = (args.output_dir or repo_root / "outputs" / "supernodes_scar_hf").resolve()
    project_dir = repo_root / "projects" / "supernodes_scar"

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []

    copy_one(project_dir / "hf_dataset_card.md", output_dir / "README.md", "dataset_card", entries)
    copy_one(project_dir / "README.md", output_dir / "docs" / "PROJECT_README.md", "docs", entries)
    copy_one(project_dir / "ARTIFACTS.md", output_dir / "docs" / "ARTIFACTS.md", "docs", entries)
    copy_one(project_dir / "REPRODUCIBILITY.md", output_dir / "docs" / "REPRODUCIBILITY.md", "docs", entries)
    copy_one(project_dir / "release_manifest.yaml", output_dir / "docs" / "release_manifest.yaml", "docs", entries)

    copy_one(repo_root / "README.md", output_dir / "code_metadata" / "README.md", "code_metadata", entries)
    copy_one(repo_root / "LICENSE", output_dir / "code_metadata" / "LICENSE", "code_metadata", entries)
    copy_one(repo_root / "pyproject.toml", output_dir / "code_metadata" / "pyproject.toml", "code_metadata", entries)

    copy_tree_files(repo_root / "configs" / "prune_llm", output_dir / "configs" / "prune_llm", "configs", entries)
    copy_tree_files(paper_dir / "paper" / "configs", output_dir / "configs" / "paper_side", "configs", entries)

    copy_tree_files(paper_dir / "figures", output_dir / "paper_artifacts" / "figures", "paper_figures", entries)
    copy_tree_files(paper_dir / "paper_artifacts" / "tables", output_dir / "paper_artifacts" / "tables", "paper_tables", entries)
    copy_one(paper_dir / "paper_artifacts" / "numbers.tex", output_dir / "paper_artifacts" / "numbers.tex", "paper_tables", entries)
    for summary_name in ("olmo_trajectory.json", "olmo_pruning_summary.json"):
        copy_one(
            paper_dir / "paper_artifacts" / summary_name,
            output_dir / "paper_artifacts" / "experiments" / summary_name,
            "derived_experiment_summaries",
            entries,
        )
    copy_tree_files(
        paper_dir / "paper_artifacts" / "experiments",
        output_dir / "paper_artifacts" / "experiments",
        "derived_experiment_summaries",
        entries,
    )

    copy_tree_files(paper_dir / "paper" / "scripts", output_dir / "paper_scripts", "paper_scripts", entries)

    expanded_manifest = load_json(paper_dir / "paper_artifacts" / "repro_manifest_expanded.json")
    collector = expanded_manifest.get("collector_manifest", {})
    if isinstance(collector, dict):
        results_files = collector.get("results_files", {})
        if isinstance(results_files, dict):
            copy_named_json_sources(
                results_files,
                output_dir / "raw_results" / "main_runs",
                "raw_results",
                entries,
                source_entries,
            )
    for source_group in ("extra_external_runs", "repo_side_aggregated_inputs"):
        mapping = expanded_manifest.get(source_group, {})
        if isinstance(mapping, dict):
            copy_named_json_sources(
                mapping,
                output_dir / "raw_results" / source_group,
                "raw_results",
                entries,
                source_entries,
            )

    entries = sorted(relativize_entries(entries, output_dir), key=lambda item: item["path"])
    manifest_path = output_dir / "MANIFEST.json"
    sha_path = output_dir / "MANIFEST.sha256"
    metadata_path = output_dir / "metadata" / "release_metadata.json"
    result_sources_path = output_dir / "metadata" / "result_sources.json"

    metadata = {
        "project": "supernodes_scar",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_commit": git_commit(repo_root),
        "repo_dirty": git_is_dirty(repo_root),
        "artifact_count": len(entries),
        "artifact_bytes": sum(int(item["bytes"]) for item in entries),
        "notes": [
            "Manifest paths are relative to the dataset repository root.",
            "Internal source paths are intentionally not included.",
            "No model weights, raw datasets, logs, or checkpoints are included.",
        ],
    }

    # Include manifest files themselves in the checksum set.
    write_json(metadata_path, metadata)
    write_json(result_sources_path, {"results": sorted(source_entries, key=lambda item: item["artifact_path"])})
    entries.append(
        {
            "path": metadata_path.relative_to(output_dir).as_posix(),
            "group": "metadata",
            "bytes": metadata_path.stat().st_size,
            "sha256": sha256_file(metadata_path),
        }
    )
    entries.append(
        {
            "path": result_sources_path.relative_to(output_dir).as_posix(),
            "group": "metadata",
            "bytes": result_sources_path.stat().st_size,
            "sha256": sha256_file(result_sources_path),
        }
    )
    entries = sorted(entries, key=lambda item: item["path"])
    write_json(manifest_path, entries)

    with sha_path.open("w", encoding="utf-8") as handle:
        for item in entries:
            handle.write(f"{item['sha256']}  {item['path']}\n")

    print(f"Wrote artifact bundle: {output_dir}")
    print(f"Files: {len(entries)}")
    print(f"Bytes: {sum(int(item['bytes']) for item in entries):,}")
    print(f"Manifest: {manifest_path}")
    print(f"Checksums: {sha_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
