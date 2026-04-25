# Configuration Files

NodeLens experiments are driven by YAML configs. The same runner is used for
small examples, vision pruning studies, and LLM supernode/SCAR experiments:

```bash
python scripts/run_experiment.py --config path/to/config.yaml
```

The Python package is imported as `nodelens`.

## Directory Map

```text
configs/
|-- template.yaml              # Full legacy-format reference
|-- unified_template.yaml      # Unified-format reference
|-- examples/                  # Small runnable examples and smoke tests
|-- vision_prune/              # Vision clustering, halo, and channel pruning
`-- prune_llm/                 # LLM supernode, SCAR, and paper-scale configs
```

Large private sweep grids are not kept in the public config tree. Public
release material for a paper lives under `projects/<project_name>/`; reusable
experiment configs live here.

## Which Config Should I Use?

| Goal | Start with |
|------|------------|
| Quick install check on MNIST | `configs/examples/mnist_basic.yaml` |
| Small vision pruning example | `configs/examples/resnet_pruning.yaml` |
| Vision metric clustering and halo analysis | `configs/vision_prune/resnet18_cifar10_unified.yaml` |
| Larger vision pruning benchmark | `configs/vision_prune/resnet50_imagenet100_unified.yaml` |
| Minimal LLM supernode example | `configs/examples/llm_alignment.yaml` |
| Main 8B LLM SCAR suite | `configs/prune_llm/llama3_8b_unified.yaml` |
| Cross-model 7B/8B LLM checks | `configs/prune_llm/{llama2,mistral,qwen2}_7b_unified.yaml` |
| 70B mechanism check | `configs/prune_llm/llama3_70b_scale_mechanism.yaml` |
| 70B structured pruning curves | `configs/prune_llm/llama3_70b_scale_pruning_curves.yaml` |
| OLMo checkpoint trajectory | `configs/prune_llm/olmo2_7b_ckpt_template.yaml` |

## Experiment Types

| Type | Used for | Typical configs |
|------|----------|-----------------|
| `alignment_analysis` | General activation, alignment, and pruning analysis for small models | `configs/examples/*.yaml` |
| `cluster_analysis` | Vision channel metrics, metric-space clustering, halo analysis, cascade tests, and structured pruning | `configs/vision_prune/*.yaml` |
| `llm_alignment` | Hugging Face causal LMs, SCAR loss-proxy metrics, supernodes, halos, perplexity, and LLM structured pruning | `configs/prune_llm/*.yaml` |
| `vision_synergy` | Older focused vision synergy experiments | `configs/examples/vision_synergy.yaml` |

## Common Commands

```bash
# Quick smoke test
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# Vision clustering and pruning
python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_unified.yaml

# LLM supernode and SCAR analysis
python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_unified.yaml

# Override output location without editing the YAML
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_unified.yaml \
  --base-output-dir /path/to/results
```

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

Not every block is used by every experiment type. Vision configs usually use
`clustering`, `halo_analysis`, and `cascade_analysis`; LLM configs usually use
`supernode`, `halo_analysis`, `llm`, and `pruning`.

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

## Minimal LLM Configuration

```yaml
experiment:
  type: "llm_alignment"

model:
  name: "hf_causal_lm"
  model_id: "meta-llama/Llama-3.1-8B"
  dtype: "bfloat16"

metrics:
  scar:
    enabled: true
    num_samples: 64
    max_length: 512

supernode:
  enabled: true
  core_fraction: 0.01
  protect_core: true
```

LLM configs require access to the model provider, enough GPU memory, and the
right license acceptance for gated models.

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
from nodelens.configs import load_unified_config

# Works with both old and unified formats!
config = load_unified_config("configs/vision_prune/resnet18_cifar10_unified.yaml")

# Access in a consistent way
print(config.experiment.name)
print(config.model.name)
print(config.pruning.ratios)

# Validate
warnings = config.validate()
```
