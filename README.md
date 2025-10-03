# Alignment Analysis Framework

A framework for analyzing neural network alignment, pruning strategies, and information-theoretic properties.

## Overview

This framework provides tools for:
- Computing alignment metrics between neural representations and task structure
- Implementing and testing pruning strategies
- Training and analyzing multiple networks
- Evaluating model performance across various metrics

## Installation

### Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup

```bash
git clone <repository-url>
cd alignment

conda env create -f environment.yml
conda activate alignment

pip install -e .
```

## Quick Start

Run an experiment using a configuration file:

```bash
python scripts/run_experiment.py --config configs/examples/resnet18_analysis.yaml
```

Or use the Python API:

```python
from alignment.configs.config_loader import load_config
from alignment.experiments import GeneralAlignmentExperiment

config = load_config('configs/examples/resnet18_analysis.yaml')
experiment = GeneralAlignmentExperiment(config)
results = experiment.run()
```

## Supported Models

**Vision Models:**
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19)
- EfficientNet (B0-B7)
- Vision Transformers (ViT, DeiT)
- AlexNet, MobileNet, DenseNet

**Custom Models:**
- Multi-layer Perceptrons
- Convolutional Neural Networks
- Custom architectures via registry

**Language Models:**
- HuggingFace Causal LM (GPT, LLaMA, Mistral)

## Datasets

Supported datasets include:
- MNIST, Fashion-MNIST
- CIFAR-10, CIFAR-100
- ImageNet
- WikiText, C4 (for language models)
- Custom datasets via registry

## Configuration

Experiments are configured via YAML files. Example:

```yaml
experiment:
  name: "my_experiment"
  seed: 42

model:
  name: "resnet18"
  pretrained: true

dataset:
  name: "cifar10"
  batch_size: 128

alignment:
  metrics: ["rayleigh_quotient", "mutual_information_gaussian"]

pruning:
  enabled: true
  algorithms: ["alignment"]
  sparsity_levels: [0.2, 0.5]
```

See `configs/examples/` for complete examples.

## Project Structure

```
alignment/
├── src/alignment/
│   ├── core/            # Registry and base classes
│   ├── models/          # Model loaders and wrappers
│   ├── metrics/         # Alignment metrics
│   ├── pruning/         # Pruning strategies
│   ├── experiments/     # Experiment framework
│   ├── data/            # Dataset handling
│   └── configs/         # Configuration management
├── configs/             # Configuration files
├── examples/            # Example scripts
├── scripts/             # Experiment runners
├── tests/               # Tests
└── docs/                # Documentation
```

## Available Metrics

The framework implements over 30 alignment metrics, including:
- Rayleigh Quotient
- Mutual Information (various estimators)
- Spectral Alignment
- Cosine Similarity
- Partial Information Decomposition (PID)
- Task-specific metrics

## Pruning Strategies

Supported pruning approaches:
- Magnitude-based pruning
- Gradient-based pruning
- Alignment-based pruning
- Random pruning (baseline)
- Structured and unstructured pruning

## Documentation

Build the documentation:

```bash
cd docs
make html
```

View at `docs/build/html/index.html`.

## Testing

Run tests:

```bash
pytest tests/
```

## License

See LICENSE file for details.
