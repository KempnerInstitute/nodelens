# Scripts

Scripts for running experiments.

## Main Script

`run_experiment.py` - Experiment runner with YAML configuration support.

Usage:
```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
```

The script handles:
- Model loading (pretrained or training from scratch)
- Metric computation
- Pruning with various strategies
- Evaluation and visualization

## Configuration

Experiments are configured via YAML files. See `configs/template.yaml` for all parameters and `configs/examples/` for working examples.

## Examples vs Scripts

- `examples/` - Standalone demonstration scripts
- `scripts/` - Configuration-based experiment runner 