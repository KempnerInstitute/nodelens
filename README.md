# Alignment Analysis Framework

A comprehensive framework for analyzing neural network alignment, pruning, and information-theoretic properties.

## Overview

This framework provides tools for:
- **Alignment Analysis**: Measure how neural representations align with data and task structure
- **Pruning Experiments**: Test various pruning strategies and their effects on model performance
- **Multi-Network Analysis**: Train and analyze multiple networks in parallel
- **Information Theory Metrics**: Compute mutual information, Rayleigh quotients, and other metrics
- **Visualization**: Generate comprehensive plots and reports

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

### Basic Alignment Analysis

```python
from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig

# Create configuration
config = GeneralAlignmentConfig(
    experiment_name="mnist_alignment",
    dataset_name="mnist",
    model_name="mlp",
    hidden_sizes=[128, 64],
    num_epochs=10,
    compute_alignment=True,
    alignment_metrics=["rayleigh_quotient", "mutual_information_gaussian"]
)

# Run experiment
experiment = GeneralAlignmentExperiment(config)
results = experiment.run()
```

### Multi-Network Analysis

```python
# Train multiple networks in parallel
config = GeneralAlignmentConfig(
    experiment_name="multi_network_analysis",
    num_networks=5,  # Train 5 networks
    dataset_name="mnist",
    model_name="cnn",
    num_epochs=20
)

experiment = GeneralAlignmentExperiment(config)
results = experiment.run()
```

### Pruning Experiments

```python
from alignment.pruning.experiments import LayerIsolatedPruningExperiment

# Layer-wise pruning
config = LayerIsolatedConfig(
    experiment_name="layer_pruning",
    dataset_name="mnist",
    model_name="mlp",
    pruning_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
    pruning_strategy="magnitude"
)

experiment = LayerIsolatedPruningExperiment(config)
results = experiment.run()
```

## Project Structure

```
alignment/
├── src/alignment/
│   ├── core/              # Core functionality
│   ├── models/            # Model architectures
│   ├── data/              # Data loading and processing
│   ├── metrics/           # Alignment and information metrics
│   ├── pruning/           # Pruning strategies and experiments
│   │   ├── strategies/    # Core pruning algorithms
│   │   └── experiments/   # High-level pruning experiments
│   ├── experiments/       # Main experiment classes
│   ├── analysis/          # Analysis and visualization tools
│   └── training/          # Training utilities
├── configs/               # Configuration files
├── examples/              # Example scripts
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## Key Components

### Experiments
- `GeneralAlignmentExperiment`: Main experiment class with multi-network support
- `LayerIsolatedPruningExperiment`: Layer-wise pruning analysis
- `GlobalDropoutExperiment`: Global pruning across all layers
- `CascadingLayerPruningExperiment`: Cascading pruning strategy
- `EigenvectorDropoutExperiment`: Eigenvector-based pruning

### Metrics
- **Rayleigh Quotient**: Measures alignment between representations and data
- **Mutual Information**: Information-theoretic similarity measures
- **Spectral Metrics**: Eigenvalue and singular value analysis
- **Task-Specific Metrics**: Accuracy, loss, and other performance metrics

### Pruning Strategies
- **Magnitude-based**: Prune weights with smallest magnitudes
- **Gradient-based**: Use gradient information for importance
- **Random**: Baseline random pruning
- **Alignment-based**: Use alignment metrics to guide pruning

## Configuration

Experiments can be configured via YAML files or programmatically:

```yaml
# configs/example_config.yaml
experiment_name: "alignment_analysis"
experiment_type: "general_alignment"
seed: 42

# Model configuration
model_name: "mlp"
hidden_sizes: [128, 64]

# Training configuration
num_epochs: 20
batch_size: 128
learning_rate: 0.001

# Alignment configuration
compute_alignment: true
alignment_metrics:
  - "rayleigh_quotient"
  - "mutual_information_gaussian"
```

## Running Experiments

### Command Line

```bash
# Run with config file
python scripts/run_experiment.py --config configs/example_config.yaml

# Run pruning experiment
python run_pruning_experiment.py --config configs/pruning_config.yaml

# Visualize results
python visualize_experiment_results.py --results_dir results/experiment_name
```

### Python API

```python
from alignment.experiments import create_experiment_from_config

# Load and run experiment
config = load_config("configs/example_config.yaml")
experiment = create_experiment_from_config(config)
results = experiment.run()

# Access results
print(f"Final accuracy: {results['final_metrics']['accuracy']}")
print(f"Alignment scores: {results['alignment_metrics']}")
```

## Documentation

See the [docs](docs/) directory for detailed documentation on:
- [Experiment Types](docs/EXPERIMENT_TYPES_GUIDE.md)
- [Pruning Concepts](docs/PRUNING_CONCEPTS.md)
- [Analysis Tools](docs/PRUNING_ANALYSIS_SUMMARY.md)

## Contributing

Contributions are welcome! Please see our contributing guidelines and ensure all tests pass before submitting PRs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This framework was developed at the Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University.




