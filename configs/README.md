# Configuration Files

## Structure

```
configs/
├── template.yaml              # Complete template with all options
├── unified_template.yaml      # Unified format template
├── vision_prune/              # Vision model pruning configs
│   ├── resnet18_cifar10_full.yaml
│   ├── resnet18_cifar10_unified.yaml  # Unified format version
│   ├── resnet50_imagenet100.yaml
│   ├── vgg16_cifar10_full.yaml
│   └── mobilenetv2_cifar10_full.yaml
├── prune_llm/                 # LLM pruning configs
│   ├── llama3_8b_full.yaml
│   ├── llama3_8b_unified.yaml  # Unified format version
│   ├── llama2_7b_full.yaml
│   ├── mistral_7b_full.yaml
│   └── qwen2_7b_full.yaml
└── examples/                  # Example configs
    ├── mnist_basic.yaml
    ├── resnet_pruning.yaml
    └── llm_alignment.yaml
```

## Usage

```bash
python scripts/run_experiment.py --config configs/cluster_analysis/resnet18_cifar10_full.yaml
python scripts/run_experiment.py --config configs/paper/llama3_8b_full.yaml
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
```

## Experiment Types

| Type | Description |
|------|-------------|
| `alignment_analysis` | General alignment metrics |
| `llm_alignment` | LLM supernode/SCAR analysis |
| `cluster_analysis` | Metric-space clustering with halos |

## Configuration Blocks

| Block | Purpose |
|-------|---------|
| `experiment` | Name, type, seed, device |
| `model` | Architecture, pretrained, tracked_layers |
| `dataset` | Dataset name, batch_size, data_path |
| `metrics` | Enabled metrics, num_samples, composite_weights |
| `clustering` | n_clusters, compute_stability, n_bootstrap |
| `halo_analysis` | percentile, use_activation_weight |
| `cascade_analysis` | n_remove_per_cluster |
| `supernode` | Detection settings for LLMs |
| `pruning` | Strategy, sparsity_levels, scoring |
| `llm` | LLM-specific: scar_metrics, evaluate_perplexity |

## Metrics

Available metrics for `metrics.enabled`:

- `rayleigh_quotient` - Input-weight alignment
- `activation_l2_norm` - Activation magnitude
- `pairwise_redundancy_gaussian` - Pairwise redundancy
- `synergy_gaussian_mmi` - Synergistic information
- `mutual_information_gaussian` - MI estimate

## Composite Scoring

Define weights in `metrics.composite_weights`:

```yaml
metrics:
  composite_weights:
    activation_l2_norm: 0.2
    rayleigh_quotient: 0.3
    pairwise_redundancy_gaussian: -0.2
```

## Cluster Analysis Configuration

```yaml
experiment_type: cluster_analysis

clustering:
  n_clusters: 4
  compute_stability: true
  n_bootstrap: 50

halo_analysis:
  percentile: 90.0
  use_activation_weight: true

cascade_analysis:
  n_remove_per_cluster: 5
```

## LLM Configuration

```yaml
experiment_type: llm_alignment

model_config:
  model_id: "meta-llama/Llama-3.1-8B"
  torch_dtype: "bfloat16"

do_scar_metrics: true
scar_num_samples: 100

supernode:
  enabled: true
  core_fraction: 0.01
  protect_core: true
```

## Unified Configuration Format

The framework supports a **unified configuration format** that works consistently
across both vision and LLM experiments. Files with `_unified.yaml` suffix use this format.

### Unified Metric Names

| Unified Name | Vision Aliases | LLM Aliases |
|--------------|---------------|-------------|
| `rayleigh_quotient` | `rq`, `compute_rq` | `rayleigh_quotient` |
| `redundancy` | `compute_redundancy` | `gaussian_mi_analytic`, `average_redundancy` |
| `synergy` | `compute_synergy` | `synergy_gaussian_mmi` |
| `magnitude` | `weight_magnitude` | `activation_l2_norm` |
| `scar` | - | `scar_*` (LLM-specific) |

### Unified Structure

```yaml
experiment:
  name: "my_experiment"
  type: "cluster_analysis"  # or "llm_alignment"
  seed: 42
  device: "cuda"
  output_dir: "./results/..."

model:
  name: "resnet18"  # or "hf_causal_lm"
  # Vision: num_classes, pretrained
  # LLM: model_id, dtype, device_map

dataset:
  name: "cifar10"  # or "wikitext"
  batch_size: 128

calibration:
  num_samples: 5000  # Vision: ~5000, LLM: ~128

metrics:
  rayleigh_quotient:
    enabled: true
  redundancy:
    enabled: true
  synergy:
    enabled: true
  magnitude:
    enabled: true
  composite_weights:
    rayleigh_quotient: 0.33
    redundancy: -0.33
    synergy: 0.33

clustering:  # Vision
  enabled: true
  n_clusters: 4

supernode:  # LLM (alternative to clustering)
  enabled: true
  score_metric: "scar_loss_proxy"

pruning:
  enabled: true
  ratios: [0.1, 0.2, 0.3, 0.4, 0.5]
  algorithms: [...]
  selection_modes: ["low", "high"]

visualization:
  enabled: true
  format: "png"

output:
  dir: "./results/..."
  save_metrics: true
```

### Loading Unified Configs

```python
from alignment.configs import load_unified_config

# Works with both old and unified formats!
config = load_unified_config("configs/vision_prune/resnet18_cifar10_unified.yaml")

# Access in a consistent way
print(config.experiment.name)
print(config.model.name)
print(config.pruning.ratios)

# Validate
warnings = config.validate()
```
