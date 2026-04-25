# Vision Clustering And Pruning Configurations

This directory contains reusable configs for vision channel analysis, metric
clustering, halo-style cross-layer analysis, cascade tests, and structured
channel pruning.

## Overview

The cluster-based analysis pipeline identifies functional types of neurons/channels by clustering them in metric space:

1. **Metric Computation**: RQ (alignment), Redundancy (Gaussian MI), Synergy (with continuous target)
2. **Clustering**: K-means in metric space -> 4 functional types
3. **Cross-Layer Halo Analysis**: Track downstream dependencies
4. **Cascade Testing**: Validate cluster damage predictions
5. **Pruning Experiments**: Compare cluster-aware vs baselines

## Supported Architectures

- ResNet, VGG, AlexNet, MobileNetV2, and related ConvNet models.
- Any model with Conv2d or Linear layers can usually be adapted by adding a config.
- LLM FFN analysis uses the separate configs in `configs/prune_llm/`.

## Configuration Files

| Config | Model | Dataset | Purpose |
|--------|-------|---------|---------|
| `resnet18_cifar10_full.yaml` | ResNet-18 | CIFAR-10 | Full analysis |
| `resnet18_cifar10_unified.yaml` | ResNet-18 | CIFAR-10 | Unified-format full analysis |
| `resnet18_cifar100_unified.yaml` | ResNet-18 | CIFAR-100 | Cross-dataset check |
| `vgg16_cifar10_full.yaml` | VGG-16-BN | CIFAR-10 | Full analysis |
| `mobilenetv2_cifar10_full.yaml` | MobileNetV2 | CIFAR-10 | Full analysis |
| `resnet50_imagenet100.yaml` | ResNet-50 | ImageNet-100 | Large-scale analysis |
| `alexnet_imagenet1k_unified_fastprune.yaml` | AlexNet | ImageNet-1k | Fast pruning-oriented run |

Large private sweep grids are intentionally not kept here. This directory is
for reusable public examples and representative benchmark configs.

## Running Experiments

Use the unified `run_experiment.py` script (same as all other experiments):

```bash
# Run full analysis (experiment_type is read from config)
python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_full.yaml

# Override device
python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_full.yaml --device cuda:1

# Override seed for reproducibility study
python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_full.yaml --seed 123

# Specify output directory
python scripts/run_experiment.py --config configs/vision_prune/vgg16_cifar10_full.yaml \
    --output-dir results/cluster_analysis/vgg16_run1
```

## Key Configuration Options

### Metrics
```yaml
metrics:
  n_calibration_samples: 5000   # Samples for metric computation
  synergy_target: logit_margin  # Continuous target for synergy
  synergy_num_pairs: 10         # Partners per channel for synergy
```

### Clustering
```yaml
clustering:
  n_clusters: 4                 # 4 functional types
  compute_stability: true       # Bootstrap stability analysis
  n_bootstrap: 50               # Number of bootstrap samples
```

### Halo Analysis
```yaml
halo_analysis:
  percentile: 90.0              # Halo membership threshold
  use_activation_weight: true   # Weight influence by activation std
```

### Pruning
```yaml
pruning:
  ratios: [0.1, 0.3, 0.5, 0.7]  # Sparsity levels to test
  methods:
    - magnitude                 # L2 norm baseline
    - taylor                    # Taylor importance
    - network_slimming          # BN gamma
    - composite                 # Per-channel RQ+Red+Syn
    - cluster_aware             # Full cluster + halo aware
```

### Type-aware fine-tuning (optional)
```yaml
pruning:
  methods:
    - cluster_aware
    - cluster_aware_typeft      # Same pruning, enables type-aware FT alias
  fine_tune:
    enabled: true
    epochs: 10
    learning_rate: 0.0001
    track_epoch_accuracy: true
    type_aware:
      enabled: true
      methods: ["cluster_aware", "cluster_aware_typeft"]
      lr_multipliers: {critical: 0.5, synergistic: 1.0, redundant: 1.5, background: 1.5}
      wd_multipliers: {critical: 0.5, synergistic: 1.0, redundant: 1.25, background: 1.5}
      scale_batchnorm: true
```

## Output Structure

```
results/cluster_analysis/resnet18_cifar10/
├── results.json               # Full results
├── figures/
|   ├── cluster_scatter_*.png  # Metric space plots
|   ├── cluster_evolution.png  # Composition by depth
|   ├── influence_matrix_*.png # Cross-layer influence
|   ├── cascade_*.png          # Damage by cluster type
|   └── halo_properties_*.png  # Halo redundancy/synergy
└── metrics/
    └── layer_metrics.npz      # Raw per-channel metrics
```

## Functional Types

The 4-cluster structure identifies:

| Type | Characteristics | Pruning Implication |
|------|-----------------|---------------------|
| **Critical** | High RQ, Low Red, High Syn | Protect (max 30% removal) |
| **Redundant** | Mod RQ, High Red, Low Syn | Target for pruning |
| **Synergistic** | Mod RQ, Low Red, High Syn | Preserve pairs |
| **Background** | Low on all metrics | Safe to remove |

## Choosing A Vision Config

| Need | Recommended config |
|------|--------------------|
| Fast local check | `configs/examples/resnet_pruning.yaml` |
| Full CIFAR-10 clustering and pruning | `resnet18_cifar10_unified.yaml` |
| Compare architecture effects | `vgg16_cifar10_unified.yaml`, `mobilenetv2_cifar10_unified.yaml` |
| Test larger input/model scale | `resnet50_imagenet100_unified.yaml` |
| Run fast pruning without full clustering detail | `alexnet_imagenet1k_unified_fastprune.yaml` |

Use the `_unified.yaml` files for new work. The `_full.yaml` files are kept for
compatibility with older experiments.
