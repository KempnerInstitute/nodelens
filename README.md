# Alignment Framework

Neural network alignment analysis and intelligent pruning framework.

## Overview

This framework provides tools for analyzing neural networks through information-theoretic metrics and performing redundancy-aware pruning and quantization.

Key capabilities:
- Alignment metrics for analyzing neural network weight-input relationships
- Information-theoretic analysis tools
- Pruning strategies with multiple scoring methods
- Quantization with alignment-guided precision selection
- Architecture support for MLPs, CNNs, Transformers, and LLMs
- Data loading for vision and text datasets
- Evaluation and visualization tools

## Installation

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

See [docs/installation.md](docs/installation.md) for details.

## Usage

Run experiments using configuration files:

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

The framework supports:
- Training networks from scratch or loading pre-trained models
- Computing alignment and information-theoretic scores
- Applying pruning with different strategies and distributions
- Quantization with various precision settings

Example configurations are available in `configs/examples/`:
- `mnist_basic.yaml` - Basic alignment analysis on MNIST
- `resnet_pruning.yaml` - Pruning ResNet on CIFAR-10
- `llama3_scoring.yaml` - Computing scores for LLaMA models
- `llama3_pruning.yaml` - Pruning transformer networks

See `configs/template.yaml` for all available configuration options.

## Documentation

- [Usage Guide](docs/usage.md) - Running experiments with configs
- [User Guide](docs/user_guide.md) - Detailed framework documentation
- [API Reference](docs/api_reference.md) - API documentation
- [Quick Reference](docs/quick_reference.md) - Code examples

## Examples

Python scripts demonstrating framework capabilities:

```bash
python examples/07_mnist_intelligent_pruning.py
python examples/08_llama_ffn_pruning.py
python examples/09_attention_neuron_vs_head_pruning.py
```



## Testing

```bash
pytest tests/
```

## License

See LICENSE file.
