# SCAR Paper Experiment Configurations

Comprehensive configurations for generating all results in the SCAR paper.

## Configurations

| Config | Model | Layers | FFN Width | Runtime |
|--------|-------|--------|-----------|---------|
| `llama3_8b_full.yaml` | LLaMA-3.1-8B | 32 | 14336 | ~6-8h |
| `mistral_7b_full.yaml` | Mistral-7B | 32 | 14336 | ~4-6h |
| `llama2_7b_full.yaml` | LLaMA-2-7B | 32 | 11008 | ~4-6h |
| `qwen2_7b_full.yaml` | Qwen2-7B | 28 | 18944 | ~4-6h |

## Quick Start

### Run all experiments:
```bash
sbatch ../slurm_jobs/run_paper_experiments.sh
```

### Run single model:
```bash
python -m alignment.experiments.llm_alignment \
    --config configs/paper/llama3_8b_full.yaml
```

## What's Included

### Pruning Methods (All Configs)

| Category | Methods |
|----------|---------|
| **Alignment-based** | `rayleigh_quotient`, `gaussian_mi_analytic`, `average_redundancy` |
| **SCAR (gradient-based)** | `scar_loss_proxy`, `scar_taylor`, `scar_activation_power`, `scar_curvature` |
| **Supernode-aware** | `supernode_protection_score`, `supernode_connectivity_score` |
| **Generalized** | `generalized_importance` (no outlier assumption) |
| **Cross-layer** | `cross_layer_importance` (SCAR-aligned downstream dependency) |
| **Magnitude baseline** | `activation_l2_norm` |
| **SOTA baselines** | `wanda`, `sparsegpt` |

### Analyses

1. **Supernode Distribution**
   - Loss proxy histograms by layer
   - Concentration across depth
   - Top 1%, 5%, 10% highlighting

2. **Supernode Robustness**
   - Bootstrap stability analysis (10 resamples)
   - Jaccard similarity between metrics
   - Spearman correlation heatmaps
   - Cross-metric consistency

3. **Supernode Summary**
   - Halo vs non-halo metrics by layer
   - Outlier z-score analysis

4. **Halo Redundancy Analysis**
   - Within-halo redundancy
   - Within-non-halo redundancy
   - Cross-group redundancy
   - Depth comparison plots
   - Comprehensive 4-panel figures

5. **Cross-Layer Importance**
   - Downstream importance (next layer dependency)
   - Layer transition efficiency
   - Importance vs redundancy scatter

6. **Generalized Importance**
   - Works without clear supernode structure
   - Neighborhood-based redundancy
   - Downstream propagation

### Evaluation Benchmarks

**Perplexity:**
- WikiText-2
- C4 (validation subset)

**Zero-shot:**
- HellaSwag, PIQA, BoolQ, WinoGrande
- ARC-Easy, ARC-Challenge, OpenBookQA

**Few-shot:**
- HellaSwag (5-shot)
- PIQA (5-shot)
- ARC-Challenge (5-shot)
- MMLU (5-shot, full)

## Output Structure

```
results/paper/<model>/
├── metrics/
│   ├── layer_metrics.json
│   ├── supernode_analysis.json
│   ├── supernode_robustness.json
│   ├── halo_redundancy.json
│   └── cross_layer_analysis.json
├── evaluation/
│   ├── perplexity_results.json
│   └── benchmark_results.json
├── pruning/
│   ├── sparsity_curves.json
│   └── per_method_results.json
└── figures/
    ├── fig1_supernode_distribution.pdf
    ├── fig2_halo_redundancy.pdf
    ├── fig3_cross_layer_importance.pdf
    ├── fig4_pruning_curves.pdf
    ├── supernode_robustness/
    │   ├── jaccard_heatmap.pdf
    │   ├── spearman_heatmap.pdf
    │   └── bootstrap_stability.pdf
    └── supplementary/
```

## Key Differences from `examples/llama3_comprehensive_pruning.yaml`

The paper configs include everything from the comprehensive pruning config plus:

1. ✅ Structured for paper figure generation (PDF output)
2. ✅ All SOTA baselines (Wanda, SparseGPT)
3. ✅ Supernode robustness analysis
4. ✅ Supernode summary/outlier analysis
5. ✅ Generalized importance (no outlier assumption)
6. ✅ Cross-layer importance (SCAR-aligned)
7. ✅ Selection modes (low/high)
8. ✅ Additional evaluation metrics (bits_per_byte)
9. ✅ Comprehensive scatter pair analysis

## Resource Requirements

- **GPU**: 1x A100 80GB (recommended) or H100
- **Memory**: ~60GB GPU memory for 8B models
- **Storage**: ~50GB per model for full results
- **Time**: ~20-30 hours total for all 4 models
