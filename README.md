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

### Basic Analysis

```python
from alignment import ModelWrapper, get_metric

wrapper = ModelWrapper(model)
rq = get_metric('rayleigh_quotient')

outputs, acts = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights()

scores = rq.compute(acts['layer_input'], weights['layer'])
```

### Run from Config

```bash
# MNIST analysis
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# ResNet pruning
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml

# LLaMA-3 scoring
python scripts/run_experiment.py --config configs/examples/llama3_scoring.yaml
```

---

## Documentation

- [Installation](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [API Reference](docs/api_reference.md)
- [Quick Reference](docs/quick_reference.md)
- [Changelog](docs/changelog.md)

Build full documentation: `cd docs && make html`

---

## Configuration

All experiments configured via YAML files. See `configs/template.yaml` for complete parameter reference.

Example:

```yaml
experiment:
  name: "my_experiment"

model:
  name: "resnet18"
  pretrained: true

dataset:
  name: "cifar10"
  batch_size: 128

metrics:
  enabled: ['rayleigh_quotient']

pruning:
  enabled: true
  strategy: 'ultimate'
  target_sparsity: 0.7
```

---

## Examples

Python examples in `examples/` directory:

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
