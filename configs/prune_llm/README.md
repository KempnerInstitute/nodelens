# SCAR Paper Experiment Configurations

Configurations for generating results in the SCAR LLM pruning paper.

## Configurations

| Config | Model | Layers | FFN Width | Runtime |
|--------|-------|--------|-----------|---------|
| `llama3_8b_full.yaml` | LLaMA-3.1-8B | 32 | 14336 | 6-8h |
| `mistral_7b_full.yaml` | Mistral-7B | 32 | 14336 | 4-6h |
| `llama2_7b_full.yaml` | LLaMA-2-7B | 32 | 11008 | 4-6h |
| `qwen2_7b_full.yaml` | Qwen2-7B | 28 | 18944 | 4-6h |

## Quick Start

Run all experiments:
```bash
sbatch slurm_jobs/paper/run_all_paper.sh
```

Run single model:
```bash
python scripts/run_experiment.py --config configs/paper/llama3_8b_full.yaml
```

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

## Output Structure

```
results/paper/<model>/
├── metrics/
│   ├── layer_metrics.json
│   ├── supernode_analysis.json
│   └── halo_redundancy.json
├── evaluation/
│   ├── perplexity_results.json
│   └── benchmark_results.json
├── pruning/
│   └── sparsity_curves.json
└── figures/
    ├── fig1_supernode_distribution.pdf
    ├── fig2_halo_redundancy.pdf
    └── fig3_pruning_curves.pdf
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
