# Alignment Framework

Neural network alignment analysis and intelligent pruning framework.

---

## Overview

This framework provides tools for analyzing neural networks through information-theoretic metrics and performing redundancy-aware pruning and quantization.

**Key capabilities:**
- Alignment metrics (Rayleigh Quotient, class-conditioned RQ, mutual information)
- Information-theoretic analysis (redundancy, synergy, PID)
- Pruning strategies (16 methods including magnitude, gradient-based, redundancy-aware)
- Quantization (INT8, INT4, mixed-precision with alignment-guided bit allocation)
- Architecture support (MLPs, CNNs, Transformers, LLMs)
- Data loading (vision and text datasets)
- Evaluation (classification accuracy, perplexity for language models)
- Visualization (plots and reports)

---

## Installation

```bash
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

Details: [docs/installation.md](docs/installation.md)

---

## Usage

### Command Line

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
python scripts/run_experiment.py --config configs/examples/llama3_scoring.yaml
python scripts/run_experiment.py --config configs/examples/llama3_quantization.yaml
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

### Quantization

```python
from alignment.quantization import quantize_model, find_optimal_bit_allocation

# INT8 quantization
results = quantize_model(model, precision='int8')

# Mixed precision using alignment scores
layer_importance = {layer: rq_scores[layer].mean() for layer in layers}
bit_allocation = find_optimal_bit_allocation(model, layer_importance, target_avg_bits=6.0)
```

---

## Documentation

- [Installation](docs/installation.md) - Setup and dependencies
- [Usage Guide](docs/usage.md) - Running experiments
- [User Guide](docs/user_guide.md) - Complete documentation
- [API Reference](docs/api_reference.md) - API details
- [Quick Reference](docs/quick_reference.md) - Code examples
- [Changelog](docs/changelog.md) - Version history

Full documentation: [docs/README.md](docs/README.md)

---

## Configuration

Experiments are configured via YAML files. See `configs/template.yaml` for all parameters.

Available examples in `configs/examples/`:
- mnist_basic.yaml
- resnet_pruning.yaml
- llama3_scoring.yaml
- llama3_pruning.yaml
- llama3_quantization.yaml

---

## Code Examples

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
