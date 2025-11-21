# Configuration Guide

Experiments are configured using YAML files.

## Template

`template.yaml` - Complete template with all available parameters.

## Example Configurations

Ready-to-use configurations in `examples/`:

**Vision Models**
- `mnist_basic.yaml` - MLP analysis on MNIST
- `resnet_pruning.yaml` - ResNet-18 pruning on CIFAR-10

**LLaMA Models**
- `llama3_scoring.yaml` - Compute importance scores
- `llama3_pruning.yaml` - Apply pruning

Project-oriented pipelines live under `configs/projects/`:

- `vision_synergy.yaml` – ResNet-18 alignment/synergy analysis with composite pruning
- `llm_supernode.yaml` – LLaMA-3 supernode-aware pruning workflow

## Usage

Run experiment:
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

## Creating Configurations

1. Copy template: `cp configs/template.yaml configs/my_config.yaml`
2. Edit parameters as needed
3. Run: `python scripts/run_experiment.py --config configs/my_config.yaml`

## Configuration Sections

- `experiment` - Name, seed, device, output directory
- `model` - Architecture, pretrained weights, layers to track
- `dataset` - Data source, batch size, preprocessing
- `metrics` - Metrics to compute and their parameters
- `training` - Training parameters
- `pruning` - Pruning strategy, distribution, scoring
- `layer_config` - Architecture-specific settings
- `analysis` - Analysis options
- `visualization` - Plot settings
- `advanced` - Backend and optimization options

See `template.yaml` for detailed parameter documentation.
