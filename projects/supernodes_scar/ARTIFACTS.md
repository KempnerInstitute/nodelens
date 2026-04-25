# Artifact Plan

This document defines what should be shared alongside the paper and why.

## Recommended Public Artifacts

`paper_artifacts/figures/`
: PNG figures used in the arXiv paper. These are useful for quick inspection
and for checking that regenerated figures match the submitted version.

`paper_artifacts/tables/`
: LaTeX table fragments used by the paper.

`paper_artifacts/experiments/`
: Compact JSON summaries used for figure/table generation. These are derived
statistics, not raw datasets.

`raw_results/`
: Locked result JSON files copied from the runs used by the paper, sanitized
and compressed as `.json.gz`. The public paths are stable; internal cluster
paths are not included.

`configs/`
: Paper experiment configs needed to rerun metric estimation, pruning, and
evaluation.

`paper_scripts/`
: Active figure/table aggregation scripts used by the current draft.

`metadata/`
: Release metadata, checksums, git commit, and manifest files.

## Large Or Restricted Items

The public artifact repository should not contain model weights. Users should
download models through their original providers and accept the relevant model
licenses. The artifact repository should also not duplicate raw public
benchmarks; instead, document dataset names and versions in the dataset card.

## Hugging Face vs Zenodo

Hugging Face Datasets is a good fit for browsable, versioned ML artifacts that
users may download programmatically. Zenodo is better for a citable archival
snapshot with a DOI. The strongest release pattern is:

1. GitHub release tag for code.
2. Hugging Face dataset repo for result artifacts.
3. Zenodo archive of the GitHub release, plus optionally the artifact bundle,
   for DOI-based citation.

## Minimal Artifact Schema

Each generated bundle should include:

```text
README.md
MANIFEST.json
MANIFEST.sha256
metadata/release_metadata.json
configs/
paper_artifacts/
paper_scripts/
raw_results/
```

`MANIFEST.json` should record relative path, size, SHA256, and artifact group
for every file. It should not record private absolute paths.
