# Supernodes and SCAR Project Workflow

This folder documents the NodeLens workflow used for:

> Supernodes and Halos: Loss-Critical Hubs in LLM Feed-Forward Layers

The reusable implementation lives in `src/nodelens`. This project folder points
to the configs, helper scripts, and derived artifacts used to reproduce the
paper's LLM channel-analysis and structured-pruning results.

## What This Workflow Does

The workflow studies feed-forward network channels in causal language models.
It uses NodeLens to:

- capture FFN activations and gradients on a calibration set
- compute channel metrics such as activation power, Taylor scores, curvature,
  and the SCAR loss proxy
- identify small loss-sensitive channel cores and compare them with
  activation-defined outliers
- run structured FFN pruning and ablation probes
- aggregate numeric summaries into paper figures, tables, and manifest files

The halo analysis is a secondary diagnostic layer on top of the same metric
outputs. It estimates local write-overlap and redundancy structure around the
loss-sensitive core.

## Main Configs

```text
configs/prune_llm/llama3_8b_unified.yaml
configs/prune_llm/mistral_7b_unified.yaml
configs/prune_llm/llama2_7b_unified.yaml
configs/prune_llm/qwen2_7b_unified.yaml
configs/prune_llm/llama3_70b_scale_pruning_curves.yaml
configs/prune_llm/llama3_70b_scale_mechanism.yaml
configs/prune_llm/llama3_70b_scale_benchmarks_50_papersafe.yaml
```

The 7B/8B runs are intended for one A100/H100-class GPU. The 70B configs are
targeted validation runs and need substantially more memory or model
parallelism, depending on the environment.

## Run A Config

Install the package from the repository root:

```bash
conda env create -f environment.yml
conda activate nodelens
pip install -e .
```

Run a project config:

```bash
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_unified.yaml \
  --base-output-dir outputs/supernodes_scar_runs
```

Each run writes a timestamped job directory containing the config copy, logs,
result JSON files, and generated figures.

## Inspect Existing Artifacts

The public artifact dataset contains derived outputs rather than model weights
or raw datasets. It includes compact result JSON files, selected figure/table
inputs, checksums, and metadata describing which public artifact path
corresponds to each paper result.

Download and inspect it with:

```bash
huggingface-cli download hsafaai/supernodes-scar-artifacts \
  --repo-type dataset \
  --local-dir supernodes_scar_artifacts

cd supernodes_scar_artifacts
python -m json.tool MANIFEST.json | head
sha256sum -c MANIFEST.sha256
```

See `ARTIFACTS.md` for the artifact layout and `REPRODUCIBILITY.md` for the
local rerun workflow.

## Build A Local Artifact Bundle

If the expected result folders are present locally, the helper script can build
a clean derived-artifact directory under `outputs/`:

```bash
python projects/supernodes_scar/scripts/prepare_hf_artifacts.py \
  --output-dir outputs/supernodes_scar_hf \
  --clean

python projects/supernodes_scar/scripts/verify_hf_artifacts.py \
  outputs/supernodes_scar_hf
```

The verifier checks checksums and scans the staged bundle for files that should
not be included in a public derived-artifact dataset, such as Python caches,
LaTeX build files, checkpoints, raw datasets, model weights, and local absolute
paths.

## What Is Not Included

This repository and the public artifact dataset do not include:

- Llama, Mistral, Qwen, or OLMo model weights
- raw WikiText-2, C4, MMLU, or LM Evaluation Harness datasets
- cluster logs, SLURM stdout/stderr, checkpoints, or cache directories
- private local paths or access tokens

Users should obtain model weights and benchmark datasets from their original
providers and follow the relevant licenses.
