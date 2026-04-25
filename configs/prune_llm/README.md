# LLM And SCAR Configurations

These configs run Hugging Face causal-language-model experiments for loss
sensitivity, supernode analysis, halo analysis, structured FFN channel pruning,
and SCAR-style pruning baselines.

The configs are useful in two modes:

| Mode | Purpose |
|------|---------|
| Mechanism probes | Compute loss-proxy concentration, supernodes, activation overlap, and halo summaries |
| Pruning probes | Apply structured FFN channel pruning and evaluate perplexity or downstream tasks |

## Main Configs

| Config | Model | Purpose | Typical runtime |
|--------|-------|---------|-----------------|
| `llama3_8b_unified.yaml` | Llama-3.1-8B | Main 8B SCAR suite | 6-8h |
| `llama3_8b_mechanism_probes.yaml` | Llama-3.1-8B | Mechanism-only checks | 1-2h |
| `llama2_7b_unified.yaml` | Llama-2-7B | Cross-model 7B validation | 4-6h |
| `mistral_7b_unified.yaml` | Mistral-7B | Cross-model 7B validation | 4-6h |
| `qwen2_7b_unified.yaml` | Qwen2-7B | Cross-model 7B validation | 4-6h |
| `llama3_70b_scale_mechanism.yaml` | Llama-3.1-70B | Large-model concentration check | Hardware dependent |
| `llama3_70b_scale_pruning_curves.yaml` | Llama-3.1-70B | Large-model structured pruning curves | Hardware dependent |
| `llama3_70b_scale_sparsegpt_curves.yaml` | Llama-3.1-70B | Structured SparseGPT comparison | Hardware dependent |
| `olmo2_7b_ckpt_template.yaml` | OLMo-2-7B checkpoints | Training-trajectory mechanism probe | Per checkpoint |
| `olmo2_7b_pruning_curves.yaml` | OLMo-2-7B | Final-checkpoint pruning replication | 4-6h |

## Quick Start

Run single model:
```bash
python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_unified.yaml
```

Override base output directory:
```bash
python scripts/run_experiment.py \
    --config configs/prune_llm/llama3_8b_unified.yaml \
    --base-output-dir /path/to/your/output/dir
```

## Output Directory Structure

Each job creates a unique directory based on timestamp and SLURM job ID:

```
/path/to/results/Prune_LLM/
├── llama3_8b_paper_results_20241209_143052_12345678/
|   ├── results/              # JSON results files
|   |   ├── results_20241209_143052.json
|   |   └── pruning_results.json
|   ├── logs/                 # Experiment logs
|   |   └── experiment.log
|   ├── figures/              # All visualizations
|   |   ├── fig1_supernode_distribution.pdf
|   |   ├── fig2_halo_redundancy.pdf
|   |   └── fig3_pruning_curves.pdf
|   ├── checkpoints/          # Model checkpoints (if enabled)
|   ├── analysis/             # Post-analysis outputs
|   └── experiment_config.yaml
├── llama2_7b_paper_results_20241209_143100_12345679/
|   └── ...
```

**Directory naming convention:**
- `{experiment_name}_{timestamp}_{job_id}`
- For SLURM jobs: `job_id` = `$SLURM_JOB_ID`
- For local runs: `job_id` = unique 8-character ID

## Pruning Methods

| Category | Methods |
|----------|---------|
| Alignment-based | `rayleigh_quotient`, `gaussian_mi_analytic`, `average_redundancy` |
| SCAR (gradient-based) | `scar_loss_proxy`, `scar_taylor`, `scar_activation_power`, `scar_curvature` |
| Supernode-aware | `supernode_protection_score`, `supernode_connectivity_score` |
| Generalized | `generalized_importance` (no outlier assumption) |
| Cross-layer | `cross_layer_importance` (downstream dependency) |
| Magnitude baseline | `activation_l2_norm` |
| SOTA baselines | `wanda`, `sparsegpt` |

All LLM pruning configs use structured FFN channel pruning unless explicitly
noted. A channel is removed consistently across the corresponding FFN
projection group, which is different from unstructured element-wise pruning.

## Analyses

1. **Supernode Distribution**: Loss proxy histograms, concentration across depth
2. **Supernode Robustness**: Bootstrap stability, Jaccard similarity, cross-metric consistency
3. **Supernode Summary**: Halo vs non-halo metrics by layer
4. **Halo Redundancy**: Within-halo, within-non-halo, cross-group redundancy
5. **Cross-Layer Importance**: Downstream importance, layer transition efficiency
6. **Generalized Importance**: Neighborhood-based scoring without outlier assumption

## Evaluation Benchmarks

**Perplexity**: WikiText-2, C4

**Zero-shot**: HellaSwag, PIQA, BoolQ, WinoGrande, ARC-Easy, ARC-Challenge, OpenBookQA

**Few-shot**: HellaSwag (5-shot), PIQA (5-shot), ARC-Challenge (5-shot), MMLU (5-shot)

## Configuration Options

### Base Output Directory

The `output.base_dir` setting controls where job directories are created:

```yaml
output:
  # Creates: {base_dir}/{experiment_name}_{timestamp}_{job_id}/
  base_dir: "/path/to/results/Prune_LLM"

  # Fallback if base_dir is not set (legacy)
  dir: "./results/paper/llama3_8b"
```

Can be overridden via CLI:
```bash
python scripts/run_experiment.py --config ... --base-output-dir /new/path
```

## Features

Compared to example configs, paper configs include:
- PDF figure output
- All SOTA baselines (Wanda, SparseGPT)
- Supernode robustness analysis
- Generalized importance (outlier-free)
- Cross-layer importance analysis
- Comprehensive evaluation metrics

## Resource Requirements

Approximate requirements for the 7B/8B configs:

| Resource | Typical value |
|----------|---------------|
| GPU | 1x A100 80GB or H100 preferred |
| GPU memory | About 60GB for full 8B pruning/evaluation configs |
| Storage | About 50GB per model run if all figures/results are saved |
| Time | 4-8h for one full 7B/8B model config |

The 70B configs need substantially more memory and may require tensor
parallelism, CPU offload, or a multi-GPU node depending on the local setup.

## Notes On Reproducibility

- The public configs do not include model weights or datasets.
- Gated models require accepting the provider license before running.
- Use `--base-output-dir` to keep large run outputs outside the repository.
- Paper artifact packaging is handled by `projects/supernodes_scar/`.
