# Artifact Contents

The Supernodes and SCAR artifact dataset contains derived outputs that help
readers inspect the reported results without rerunning every large-model job.
It does not contain model weights, raw benchmark datasets, checkpoints, or
cluster logs.

## Directory Layout

```text
README.md
MANIFEST.json
MANIFEST.sha256
metadata/
configs/
paper_artifacts/
paper_scripts/
raw_results/
```

`MANIFEST.json`
: Machine-readable inventory. Each entry records the relative path, size,
SHA256 checksum, and artifact group.

`MANIFEST.sha256`
: Checksum file that can be verified with `sha256sum -c MANIFEST.sha256`.

`metadata/`
: Dataset-level metadata, source-result mapping, and bundle-generation
information.

`configs/`
: Experiment configs for metric estimation, pruning, ablation, and 70B
validation runs.

`paper_artifacts/figures/`
: PNG figures used for quick visual inspection.

`paper_artifacts/tables/`
: LaTeX table fragments generated from the locked results.

`paper_artifacts/experiments/`
: Compact JSON summaries used by figure and table scripts.

`paper_scripts/`
: Figure and table aggregation scripts that operate on the included summaries
or on compatible local result folders.

`raw_results/`
: Selected locked result JSON files, sanitized and compressed as `.json.gz`.
These are derived statistics from completed runs, not raw calibration data.

## How To Use The Bundle

After downloading the dataset, verify the checksums:

```bash
sha256sum -c MANIFEST.sha256
```

Inspect the source mapping:

```bash
python -m json.tool metadata/result_sources.json | less
```

Use `raw_results/` for exact numeric values, `paper_artifacts/experiments/` for
compact figure inputs, and `configs/` to rerun matching experiments with the
current NodeLens code.

## Excluded Files

The artifact dataset intentionally excludes:

- model weights and tokenizer files
- raw public benchmark datasets
- checkpoints and optimizer states
- Python caches and compiled bytecode
- LaTeX build products
- local absolute paths, scheduler logs, and access tokens

Models and datasets should be downloaded from their original providers.
