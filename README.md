# Neural Network Alignment Analysis Framework

A comprehensive framework for analyzing neural network alignment through various metrics and pruning strategies.

## Features

- **36+ Alignment Metrics**: Rayleigh quotient, mutual information, spectral analysis, and more
- **Advanced Pruning**: Multiple strategies with low/high/random modes and parallel execution
- **Comprehensive Experiments**: Fully configurable system supporting all models and datasets
- **Automatic Analysis**: Built-in visualization and reporting tools
- **GPU Optimized**: Efficient implementations with automatic memory management

## Installation

```bash
# Clone the repository
git clone https://github.com/KempnerInstitute/alignment.git
cd alignment

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e .[all]
```

## Quick Start

### 1. Learning the Framework

Start with the examples to understand how things work:

```bash
# Basic demonstration
python examples/quick_demo.py

# Complete workflow example  
python examples/standard_alignment_experiment.py

# Advanced pruning features
python examples/pruning_strategies_demo.py
```

### 2. Running Experiments

For actual research, use the unified experiment runner:

```bash
# Run with the default configuration
python scripts/run_experiment.py --config configs/unified_config.yaml

# Run with a minimal test configuration
python scripts/run_experiment.py --config configs/examples/quick_test.yaml
```

The examples are self-contained demos for learning, while `scripts/run_experiment.py` is the production tool for research.

## Available Examples

1. **Quick Demo** (`examples/quick_demo.py`)
   - Basic alignment metric computation
   - Simple pruning demonstration
   - ~5 minute runtime

2. **Standard Experiment** (`examples/standard_alignment_experiment.py`)
   - Complete workflow from training to analysis
   - Multiple pruning strategies comparison
   - Comprehensive visualizations
   - ~10 minute runtime

3. **Pruning Strategies Demo** (`examples/pruning_strategies_demo.py`)
   - All available pruning algorithms
   - Performance comparisons
   - Advanced features demonstration

4. **Visualization Demo** (`examples/pruning_visualization_demo.py`)
   - Advanced plotting capabilities
   - Interactive visualizations
   - Custom analysis tools

5. **Production Experiments** (`scripts/run_experiment.py`)
   - Fully configurable via YAML
   - Supports all models, datasets, and metrics
   - Experiment types: standard, cascading, layer-isolated
   - Example: `python scripts/run_experiment.py --config configs/unified_config.yaml`

## Comprehensive Experiments

The framework provides a unified experiment system through YAML configuration:

```yaml
# Example configuration
name: "my_experiment"
model_name: "resnet18"
dataset_name: "cifar10"

training_config:
  epochs: 20
  batch_size: 128
  learning_rate: 0.001

alignment_metrics:
  - "rayleigh_quotient"
  - "mutual_information_gaussian"
  - "spectral_gap"

pruning_strategy: "magnitude"
pruning_config:
  amount: 0.5
```

Run with:
```bash
python examples/comprehensive_alignment_experiment.py --config my_config.yaml
```

## Available Metrics

The framework includes 36+ metrics organized by type:

### Rayleigh Quotient Based
- `rayleigh_quotient` (aliases: `rq`, `RQ`)
- `delta_alignment`
- `normalized_delta_alignment`

### Information Theoretic
- `mutual_information_gaussian` (aliases: `mi_gaussian`, `mi_0`)
- `mutual_information_binning` (aliases: `mi_binning`, `mi_1`)
- `average_redundancy`, `node_redundancy`, `layer_redundancy`
- `total_correlation`, `interaction_information`

### Similarity Metrics
- `weight_cosine_similarity`
- `activation_cosine_similarity`
- `weight_activation_alignment`

### Spectral Metrics
- `spectral_gap`, `eigenvalue_alignment`
- `spectral_clustering`, `eigenvalue_entropy`

See the [documentation](https://kempnerinstitute.github.io/alignment/) for the complete list.

## Project Structure

```
alignment/
├── src/alignment/
│   ├── core/           # Base classes and protocols
│   ├── models/         # Model wrappers and architectures
│   ├── metrics/        # Alignment metrics
│   ├── pruning/        # Pruning strategies
│   ├── experiments/    # Experiment framework
│   ├── data/           # Dataset handling
│   ├── training/       # Training utilities
│   ├── analysis/       # Analysis and visualization
│   └── infrastructure/ # Runtime support (distributed, storage, config)
├── examples/           # Example scripts
├── configs/            # Configuration files
├── tests/              # Unit and integration tests
└── docs/               # Documentation
```

## Documentation

Full documentation is available at: https://kempnerinstitute.github.io/alignment/

Key documentation:
- [Getting Started Guide](docs/source/user_guide/getting_started.md)
- [Metrics Reference](docs/source/METRICS_REFERENCE.md)
- [Pruning Strategies](docs/source/user_guide/pruning.md)
- [Pruning Concepts](PRUNING_CONCEPTS.md) - Structured vs unstructured pruning explained
- [Experiment Types Guide](EXPERIMENT_TYPES_GUIDE.md) - Different pruning experiment patterns
- [API Reference](https://kempnerinstitute.github.io/alignment/api/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{alignment2024,
  title={Neural Network Alignment Analysis Framework},
  author={Kempner Institute},
  year={2024},
  url={https://github.com/KempnerInstitute/alignment}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This framework was developed at the Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University.




