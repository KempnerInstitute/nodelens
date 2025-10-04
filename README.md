# Alignment Framework

Neural Network Alignment Analysis & Intelligent Pruning

A research framework for analyzing neural networks through information-theoretic metrics and performing redundancy-aware pruning.

---

## Features

- Alignment metrics (Rayleigh Quotient, class-conditioned RQ, mutual information)
- Information-theoretic analysis (redundancy, synergy, PID)
- 16 pruning strategies (magnitude, gradient-based, redundancy-aware)
- Architecture support (MLPs, CNNs, Transformers, LLMs)
- Data loading (vision datasets, text datasets for LLMs)
- Evaluation (classification accuracy, language model perplexity)
- Visualization (publication-quality plots and reports)

---

## Installation

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

See [docs/installation.md](docs/installation.md) for details.

---

## Quick Start

### Run from Config File

```bash
# MNIST analysis
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# ResNet pruning on CIFAR-10
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml

# LLaMA-3 per-neuron scoring
python scripts/run_experiment.py --config configs/examples/llama3_scoring.yaml
```

### Python API

```python
from alignment import ModelWrapper, get_metric

wrapper = ModelWrapper(model)
rq = get_metric('rayleigh_quotient')

outputs, acts = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights()

scores = rq.compute(acts['layer_input'], weights['layer'])
```

---

## Documentation

- [Installation](docs/installation.md) - Setup guide
- [Usage](docs/usage.md) - Running experiments with YAML configs
- [User Guide](docs/user_guide.md) - Complete guide
- [API Reference](docs/api_reference.md) - API documentation
- [Quick Reference](docs/quick_reference.md) - Code examples
- [Changelog](docs/changelog.md) - Version history

Full documentation: [docs/README.md](docs/README.md)

---

## Configuration

All experiments configured via YAML. See `configs/template.yaml` for complete parameter reference.

Example configs in `configs/examples/`:
- `mnist_basic.yaml` - Simple analysis
- `resnet_pruning.yaml` - Vision model pruning
- `llama3_scoring.yaml` - LLM neuron importance
- `llama3_pruning.yaml` - LLM pruning

---

## Examples

Python code examples in `examples/`:

```bash
python examples/07_mnist_intelligent_pruning.py
python examples/08_llama_ffn_pruning.py
python examples/09_attention_neuron_vs_head_pruning.py
```

---

## Testing

```bash
pytest tests/
```

---

## License

See LICENSE file.
