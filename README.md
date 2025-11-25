# Alignment Framework

Neural network alignment analysis and pruning framework for vision models and LLMs.

## Overview

This framework provides tools for:

- Computing alignment metrics between neural network weights and activations
- Information-theoretic analysis (redundancy, synergy, mutual information)
- Structured and unstructured pruning with multiple scoring methods
- Dependency-aware pruning for architectures with skip connections
- LLM-specific analysis including SCAR metrics and attention head pruning

Supported architectures: MLPs, CNNs (ResNet, VGG), Transformers, LLMs (LLaMA, Mistral).

## Installation

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

## Quick Start

### Running Experiments

```bash
# Vision model alignment analysis
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# ResNet pruning on CIFAR-10
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml

# LLM importance scoring
python scripts/run_experiment.py --config configs/examples/llama3_scoring.yaml
```

### Standalone Analysis

```bash
# Generate visualizations from experiment results
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots --quick

# Run specific analyses
python scripts/run_analysis.py --config configs/analysis_template.yaml \
    --analyses histograms pruning_curves
```

### Programmatic Usage

```python
from alignment import ModelWrapper, get_metric

# Wrap model and compute alignment scores
wrapper = ModelWrapper(model)
rq_metric = get_metric('rayleigh_quotient')

outputs, activations = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights()
scores = rq_metric.compute(activations['layer_input'], weights['layer'])
```

## Configuration

Experiments are configured via YAML files. Key sections:

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

See `configs/template.yaml` for all options.

## Available Metrics

| Category | Metrics |
|----------|---------|
| Activation | `activation_l2_norm`, `activation_variance`, `activation_outlier_index` |
| Alignment | `rayleigh_quotient`, `delta_alignment` |
| Information | `mutual_information_gaussian`, `pairwise_redundancy_gaussian`, `gaussian_pid_synergy_mmi` |
| SCAR (LLM) | `scar_activation_power`, `scar_taylor`, `scar_curvature`, `scar_loss_proxy` |

## Pruning Strategies

- **Magnitude**: Prune by weight magnitude
- **Alignment**: Prune by alignment score (low or high)
- **Hybrid**: Combine magnitude and alignment
- **Random**: Random baseline
- **Global**: Cross-layer pruning
- **Dependency-aware**: Maintain shape compatibility in skip connections

## Project Structure

```
alignment/
├── configs/           # Configuration files
│   ├── examples/      # Example experiments
│   ├── projects/      # Project-specific configs
│   └── template.yaml  # Full parameter reference
├── scripts/           # Entry points
│   ├── run_experiment.py
│   └── run_analysis.py
├── src/alignment/     # Main package
│   ├── analysis/      # Visualization and reporting
│   ├── experiments/   # Experiment classes
│   ├── metrics/       # Alignment metrics
│   ├── models/        # Model wrappers
│   └── pruning/       # Pruning strategies
├── tests/             # Unit tests
└── docs/              # Documentation
```

## Documentation

- [Usage Guide](docs/usage.md) - Running experiments
- [LLM Experiments Guide](docs/LLM_EXPERIMENTS_GUIDE.md) - LLM-specific analysis
- [Pruning Guide](docs/PRUNING_CONFIGURATION_GUIDE.md) - Pruning configuration
- [API Reference](docs/api_reference.md) - API documentation

## Testing

```bash
pytest tests/
pytest tests/unit/ -v  # Unit tests only
```

## License

See LICENSE file.
