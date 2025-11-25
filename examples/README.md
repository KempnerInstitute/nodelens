# Examples

Example scripts demonstrating framework capabilities.

## Available Examples

- `01_quick_start.py` - Basic usage
- `02_complete_experiment.py` - Full workflow
- `03_pruning_strategies.py` - Pruning comparison
- `06_redundancy_aware_pruning.py` - Information-theoretic pruning
- `07_mnist_intelligent_pruning.py` - MNIST pruning
- `08_llama_ffn_pruning.py` - LLM feed-forward pruning
- `09_attention_neuron_vs_head_pruning.py` - Attention analysis

## Running

```bash
python examples/01_quick_start.py
python examples/07_mnist_intelligent_pruning.py
```

## Configuration-Based

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```
