# SCAR Paper Experiment Configurations

Configurations for generating results in the SCAR LLM pruning paper.

## Configurations

| Config | Model | Layers | FFN Width | Runtime |
|--------|-------|--------|-----------|---------|
| `llama3_8b_unified.yaml` | LLaMA-3.1-8B | 32 | 14336 | 6-8h |
| `mistral_7b_unified.yaml` | Mistral-7B | 32 | 14336 | 4-6h |
| `llama2_7b_unified.yaml` | LLaMA-2-7B | 32 | 11008 | 4-6h |
| `qwen2_7b_unified.yaml` | Qwen2-7B | 28 | 18944 | 4-6h |

## Quick Start

Run all experiments:
```bash
bash drafts/LLM_prune/paper/slurm/run_all_paper.sh
```

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
/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM/
├── llama3_8b_paper_results_20241209_143052_12345678/
│   ├── results/              # JSON results files
│   │   ├── results_20241209_143052.json
│   │   └── pruning_results.json
│   ├── logs/                 # Experiment logs
│   │   └── experiment.log
│   ├── figures/              # All visualizations
│   │   ├── fig1_supernode_distribution.pdf
│   │   ├── fig2_halo_redundancy.pdf
│   │   └── fig3_pruning_curves.pdf
│   ├── checkpoints/          # Model checkpoints (if enabled)
│   ├── analysis/             # Post-analysis outputs
│   └── experiment_config.yaml
├── llama2_7b_paper_results_20241209_143100_12345679/
│   └── ...
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
  base_dir: "/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM"
  
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

- **GPU**: 1x A100 80GB or H100
- **Memory**: ~60GB GPU memory for 8B models
- **Storage**: ~50GB per model
- **Time**: ~20-30 hours total for all models
