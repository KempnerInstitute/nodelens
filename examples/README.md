# Examples

Example scripts demonstrating framework capabilities.

## Available Examples

**Basic Usage**
- `01_quick_start.py` - Model wrapping and basic metrics
- `02_complete_experiment.py` - Full workflow with training and pruning
- `03_pruning_strategies.py` - Pruning strategy comparison
- `04_visualization_gallery.py` - Visualization examples

**Advanced Usage**
- `06_redundancy_aware_pruning.py` - Information-theoretic pruning
- `07_mnist_intelligent_pruning.py` - Complete MNIST pruning workflow
- `08_llama_ffn_pruning.py` - LLaMA feed-forward pruning
- `09_attention_neuron_vs_head_pruning.py` - Attention analysis

## Running Examples

Each example is self-contained:

```bash
python examples/01_quick_start.py
python examples/07_mnist_intelligent_pruning.py
python examples/08_llama_ffn_pruning.py
```

## Configuration-Based Experiments

For YAML-based experiments:
```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
```

See `configs/examples/` for available configurations.