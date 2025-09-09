## Examples

This directory contains example scripts demonstrating the alignment framework functionality.

### Available Examples

1. **01_quick_start.py** - Minimal example showing basic model wrapping and metrics
2. **02_complete_experiment.py** - Full workflow with training, pruning, and visualization
3. **03_pruning_strategies.py** - Comparison of different pruning methods
4. **04_visualization_gallery.py** - Comprehensive visualization capabilities

### Running Examples

Each example can be run independently:

```bash
python examples/01_quick_start.py
python examples/02_complete_experiment.py
python examples/03_pruning_strategies.py
python examples/04_visualization_gallery.py
```

### Configuration-Based Experiments

For experiments using YAML configurations, see:
```bash
python scripts/run_experiment.py --config configs/examples/resnet18_analysis.yaml
```

See the `configs/examples/` directory for configuration templates.