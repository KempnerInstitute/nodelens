# Configuration Files

## Structure

```
configs/
├── template.yaml              # Complete template with all options
├── cluster_analysis/          # Cluster-based analysis configs
│   ├── resnet18_cifar10_full.yaml
│   ├── vgg16_cifar10_full.yaml
│   └── mobilenetv2_cifar10_full.yaml
├── paper/                     # LLM paper experiment configs
│   ├── llama3_8b_full.yaml
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
