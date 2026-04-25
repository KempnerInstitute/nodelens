# Supernodes and Halos Release

This folder is the public-release entry point for the paper:

> Supernodes and Halos: Loss-Critical Hubs in LLM Feed-Forward Layers

The reusable implementation lives in the main `alignment` package. This project
folder records the paper-specific configs, artifact layout, and release process.

## What To Release

The public release should have two parts:

1. A GitHub release/tag for code, configs, and reproduction scripts.
2. A Hugging Face dataset repository for derived artifacts: result JSON files,
   paper figures, LaTeX tables, checksums, and a dataset card.

This split is intentional. Code belongs in GitHub; generated experiment outputs
and larger derived artifacts are easier to consume and version through the
Hugging Face Hub. A Zenodo DOI can additionally archive the GitHub release for
citation stability.

## Reproduce The Main Runs

Install the package:

```bash
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

Run a paper config:

```bash
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_unified.yaml \
  --base-output-dir /path/to/results
```

Important paper configs include:

```text
configs/prune_llm/llama3_8b_unified.yaml
configs/prune_llm/mistral_7b_unified.yaml
configs/prune_llm/llama2_7b_unified.yaml
configs/prune_llm/qwen2_7b_unified.yaml
configs/prune_llm/llama3_70b_scale_pruning_curves.yaml
configs/prune_llm/llama3_70b_scale_mechanism.yaml
configs/prune_llm/llama3_70b_scale_benchmarks_50_papersafe.yaml
```

The 7B/8B runs are feasible on one A100/H100-class GPU. The 70B validation is a
targeted large-model check and needs substantially more memory or model
parallelism depending on the environment.

## Build The Artifact Bundle

The artifact bundle is prepared locally under `outputs/`, which is ignored by
git:

```bash
python projects/supernodes_scar/scripts/prepare_hf_artifacts.py \
  --output-dir outputs/supernodes_scar_hf \
  --clean

python projects/supernodes_scar/scripts/verify_hf_artifacts.py \
  outputs/supernodes_scar_hf
```

The script copies only releaseable material into a clean directory:

- paper figures and LaTeX tables
- generated numeric summaries and JSON diagnostics
- selected locked result JSON files, sanitized and compressed as `.json.gz`
- experiment configs used by the paper
- active paper-side figure/table scripts
- checksums and a machine-readable manifest
- a Hugging Face dataset-card README

It intentionally excludes model weights, raw calibration datasets, logs,
checkpoints, Python caches, LaTeX build files, and internal absolute paths.

## Upload To Hugging Face

After inspecting `outputs/supernodes_scar_hf`, upload it as a dataset repo:

```bash
huggingface-cli login
huggingface-cli repo create supernodes-scar-artifacts --type dataset
huggingface-cli upload hsafaai/supernodes-scar-artifacts \
  outputs/supernodes_scar_hf \
  --repo-type dataset
```

For very large bundles, use the `huggingface_hub` large-folder upload workflow
instead of the simple CLI upload.

## What Not To Upload

Do not upload:

- Llama, Mistral, Qwen, or OLMo model weights.
- Raw WikiText-2, C4, MMLU, or LM Evaluation Harness datasets.
- Cluster logs, SLURM stdout/stderr, checkpoints, caches, or private paths.
- Any file containing access tokens, usernames beyond public author metadata,
  or absolute Harvard cluster paths.

## Release Checklist

- `python -m pip install -e . --no-deps --dry-run` succeeds.
- `PYTHONPATH=src python -c "import alignment; print(alignment.__version__)"`
  succeeds.
- The artifact bundle has no `.pyc`, `__pycache__`, `.aux`, `.log`, `.out`,
  model checkpoint, or raw dataset files.
- A private-path scan over both plain text files and compressed `.json.gz`
  files returns no internal cluster paths.
- `MANIFEST.sha256` verifies all staged artifacts.
- GitHub release tag, Hugging Face dataset revision, and arXiv version are
  recorded together in the dataset card.

See `REPRODUCIBILITY.md` for the local rerun and figure-regeneration workflow.
