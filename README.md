# Alignment Framework

Neural network alignment analysis and pruning framework.

## Overview

Tools for analyzing and pruning neural networks using alignment metrics, information theory, and structured pruning strategies.

**Supported architectures**: MLPs, CNNs (ResNet, VGG), Transformers, LLMs (LLaMA, Mistral)

## Installation

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

## Quick Start

### Run Experiments

```bash
# Vision model analysis
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# CNN pruning
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml

# LLM importance scoring
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml
```

### Programmatic Usage

```python
from alignment import ModelWrapper, get_metric

wrapper = ModelWrapper(model)
rq = get_metric('rayleigh_quotient')

outputs, activations = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights()
scores = rq.compute(activations['layer_input'], weights['layer'])
```

## Configuration

Experiments use YAML configuration files:

```yaml
model:
  name: "resnet18"
  pretrained: true

dataset:
  name: "cifar10"
  batch_size: 128

alignment_methods:
  - "rayleigh_quotient"
  - "pairwise_redundancy_gaussian"

pruning:
  enabled: true
  algorithms: ["alignment"]
  sparsity_levels: [0.3, 0.5, 0.7]
  structured: true
```

See `configs/template.yaml` for all parameters.

## Metrics

| Category | Metrics |
|----------|---------|
| Activation | `activation_l2_norm`, `activation_variance`, `activation_outlier_index` |
| Alignment | `rayleigh_quotient`, `delta_alignment` |
| Information | `mutual_information_gaussian`, `pairwise_redundancy_gaussian`, `gaussian_pid_synergy_mmi` |
| SCAR (LLM) | `scar_activation_power`, `scar_taylor`, `scar_curvature`, `scar_loss_proxy` |

## Pruning Strategies

| Strategy | Description |
|----------|-------------|
| `magnitude` | Prune by weight magnitude |
| `alignment` | Prune by alignment score |
| `hybrid` | Combine magnitude and alignment |
| `random` | Random baseline |
| `global` | Cross-layer pruning |

## Project Structure

```
alignment/
├── configs/           # YAML configuration files
│   ├── examples/      # Example experiments
│   └── template.yaml  # Parameter reference
├── scripts/           # Entry points
│   ├── run_experiment.py
│   └── run_analysis.py
├── src/alignment/     # Main package
│   ├── analysis/      # Visualization
│   ├── experiments/   # Experiment classes
│   ├── metrics/       # Alignment metrics
│   ├── models/        # Model wrappers
│   └── pruning/       # Pruning strategies
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Documentation

- [Usage Guide](docs/usage.md) - Running experiments and configuration
- [API Reference](docs/api_reference.md) - Core classes and functions
- [LLM Guide](docs/llm_guide.md) - LLM-specific analysis and pruning

## Testing

```bash
pytest tests/
pytest tests/unit/ -v
```

## License

See LICENSE file.
