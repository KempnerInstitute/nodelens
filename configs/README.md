# Configuration Guide

All experiments are configured via YAML files.

---

## Template

`template.yaml` - Complete template with all available parameters documented inline.

---

## Examples

Compact, ready-to-use configurations:

### LLaMA-3

- `examples/llama3_scoring.yaml` - Compute per-neuron importance scores
- `examples/llama3_pruning.yaml` - Redundancy-aware pruning

### Vision Models

- `examples/resnet_pruning.yaml` - ResNet-18 pruning with dependency handling
- `examples/mnist_basic.yaml` - Simple MNIST analysis

---

## Usage

Run any experiment:

```bash
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
```

Override parameters:

```bash
python scripts/run_experiment.py \
  --config configs/examples/resnet_pruning.yaml \
  --device cuda:1 \
  --batch-size 64 \
  --target-sparsity 0.5
```

---

## Creating Custom Configs

1. Copy template: `cp configs/template.yaml configs/my_config.yaml`
2. Modify parameters (all options documented inline)
3. Run: `python scripts/run_experiment.py --config configs/my_config.yaml`

---

## Parameter Categories

- `experiment`: Name, seed, device, output directory
- `model`: Architecture, pretrained, layers to track
- `dataset`: Data source, batch size, preprocessing
- `metrics`: Which metrics to compute and their parameters
- `training`: Training parameters (if training from scratch)
- `pruning`: Pruning strategy, distribution, scoring method
- `layer_config`: Architecture-specific settings (CNN, transformer)
- `analysis`: Analysis options (class-conditioned, save options)
- `visualization`: Plot generation settings
- `advanced`: Backend, parallelization, optimization options

See `template.yaml` for complete parameter documentation.
