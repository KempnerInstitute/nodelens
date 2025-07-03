# Alignment Analysis Framework

A comprehensive framework for analyzing neural network alignment, pruning, and information-theoretic properties.

## Features

- **Alignment Analysis**: Measure how neural representations align with data and task structure
- **Pruning Experiments**: Test various pruning strategies and their effects on model performance  
- **Multi-Network Analysis**: Train and analyze multiple networks in parallel
- **30+ Metrics**: Rayleigh quotient, mutual information, spectral metrics, and more
- **Extensible Design**: Easy to add custom metrics and strategies

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd alignment

# Create conda environment
conda env create -f environment.yml
conda activate alignment

# Install in development mode
pip install -e .
```

## Quick Start

```python
from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig

# Configure and run experiment
config = GeneralAlignmentConfig(
    experiment_name="mnist_alignment",
    dataset_name="mnist",
    model_name="mlp",
    hidden_sizes=[128, 64],
    num_epochs=10,
    compute_alignment=True,
    alignment_metrics=["rayleigh_quotient", "mutual_information_gaussian"]
)

experiment = GeneralAlignmentExperiment(config)
results = experiment.run()
```

## Documentation

Full documentation is available at `docs/build/html/index.html` after building:

```bash
cd docs
make html
```

Key documentation sections:
- [User Guide](docs/source/user_guide/) - Installation, configuration, experiments
- [API Reference](docs/source/api/) - Detailed API documentation
- [Examples](examples/) - Example scripts and notebooks

## Project Structure

```
alignment/
├── src/alignment/     # Main package
│   ├── core/         # Core functionality
│   ├── models/       # Model architectures
│   ├── metrics/      # Alignment metrics
│   ├── pruning/      # Pruning strategies
│   ├── experiments/  # Experiment classes
│   └── analysis/     # Analysis tools
├── configs/          # Configuration files
├── examples/         # Example scripts
├── tests/           # Unit tests
└── docs/            # Documentation
```

## Examples

See the [examples](examples/) directory for:
- Basic alignment analysis
- Pruning experiments
- Multi-network training
- Custom metrics implementation

## Contributing

Contributions are welcome! Please see our contributing guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This framework was developed at the Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University.




