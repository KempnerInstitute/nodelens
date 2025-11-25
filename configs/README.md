# Configuration

YAML configuration files for experiments.

## Structure

```
configs/
├── template.yaml      # Complete parameter reference
├── examples/          # Example configurations
│   ├── mnist_basic.yaml
│   ├── resnet_pruning.yaml
│   └── llm_alignment.yaml
└── projects/          # Project-specific configs
```

## Usage

```bash
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
```

## Creating Configurations

1. Copy template: `cp configs/template.yaml configs/my_config.yaml`
2. Edit parameters
3. Run: `python scripts/run_experiment.py --config configs/my_config.yaml`

## Key Sections

- `experiment` - Name, seed, device
- `model` - Architecture, pretrained weights
- `dataset` - Data source, batch size
- `alignment_methods` - Metrics to compute
- `pruning` - Strategy, sparsity levels
- `visualization` - Plot settings

See `template.yaml` for all parameters.
