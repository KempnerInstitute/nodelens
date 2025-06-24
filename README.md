# Network Alignment Analysis

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://kempnerinstitute.github.io/alignment/)

This repository provides a comprehensive framework for analyzing neural network representations using alignment metrics. It includes 36 different metrics spanning information theory, spectral analysis, similarity measures, and task-specific alignment. The framework supports distributed training, batch processing, and advanced visualization capabilities.

## Key Features

- **36 Alignment Metrics**: Comprehensive suite including Rayleigh quotient, mutual information, spectral metrics, and more
- **Advanced Pruning**: Multiple pruning strategies with parallel execution and high/low/random modes
- **Flexible Architecture**: Easy to extend with custom metrics and experiments
- **Performance Optimized**: GPU acceleration, parallel processing, and batch computation
- **Experiment Framework**: Structured approach to running and tracking experiments
- **Visualization Tools**: Built-in plotting and analysis capabilities

## Setup

The code requires a basic ML python environment. Setup can be done with a standard python environment manager like conda (or mamba). To get started, clone the repository from GitHub, then navigate to the cloned folder.

```bash
mamba env create -f environment.yml
mamba activate networkAlignmentAnalysis
```

## Installation

After creating and activating the environment, you can install the package:

```bash
pip install -e .[all]
```

To verify the installation:

```bash
python src/alignment/examples/run_experiment_from_config.py configs/config_alignment_experiment.yaml
```

## Documentation

📚 **[View Full Documentation](https://kempnerinstitute.github.io/alignment/)**

The codebase is fully documented with comprehensive guides and API references:

### Core Documentation
- **[User Guide](docs/source/ALIGNMENT_MODULE_GUIDE.md)**: Complete guide to using the alignment module
- **[Metrics Reference](docs/source/METRICS_REFERENCE.md)**: Detailed mathematical descriptions of all 36 metrics
- **[All Metrics List](docs/source/ALL_METRICS_LIST.md)**: Quick reference of available metrics
- **[Pruning Strategies](docs/source/user_guide/pruning_strategies.md)**: Comprehensive guide to all pruning strategies
- **[Architecture Guide](docs/source/developer_guide/architecture.md)**: Framework architecture and design principles

### Module Guides
- **[Pruning Module](src/alignment/pruning/README.md)**: Comprehensive pruning strategies and experiments
- **[Training Module](src/alignment/training/README.md)**: Training utilities and multi-network training
- **[Experiments Module](src/alignment/experiments/README.md)**: Experiment framework and runners
- **[Analysis Module](src/alignment/analysis/README.md)**: Result aggregation and reporting
- **[Data Module](src/alignment/data/README.md)**: Dataset handling and preprocessing
- **[Infrastructure Module](src/alignment/infrastructure/README.md)**: Computing and storage utilities

### Additional Resources
- **[Gaussian MI Documentation](archive/refactoring_docs/GAUSSIAN_MI_SUMMARY.md)**: Details on the Gaussian mutual information metric with Edgeworth expansions
- **[Task-Specific Metrics](archive/refactoring_docs/TASK_SPECIFIC_REORG_SUMMARY.md)**: Documentation for domain-specific alignment metrics
- **[API Reference](docs/source/api/)**: Comprehensive API documentation

### Quick Start Example

```python
import torch
from alignment.core import ModelWrapper
from alignment.metrics import get_metric

# Wrap your model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)
wrapped_model = ModelWrapper(model)

# Compute alignment metrics
metric = get_metric("rayleigh_quotient")()
scores = metric.compute(inputs=inputs, weights=model[0].weight)
```

## Available Metrics

The framework provides 36 metrics across 6 categories:

1. **Rayleigh Quotient** (3 metrics): Variance capture analysis
2. **Information-Theoretic** (14 metrics): Mutual information, redundancy, PID
3. **Similarity** (7 metrics): Cosine similarity, correlation, alignment
4. **Spectral** (8 metrics): Eigenvalue analysis, spectral gaps
5. **Task-Specific** (8 metrics): Classification, language modeling, vision, RL
6. **Higher-Order** (4 metrics): Multi-way information interactions

See [METRICS_REFERENCE.md](docs/source/METRICS_REFERENCE.md) for detailed descriptions.

## Codebase Structure

```
alignment/
├── src/alignment/       # Core source code
│   ├── core/           # Base classes and registry
│   ├── metrics/        # All metric implementations
│   ├── models/         # Model wrappers and architectures
│   ├── experiments/    # Experiment framework
│   ├── utils/          # Utilities (batch processing, visualization)
│   └── examples/       # Example scripts
├── tests/              # Unit and integration tests
├── configs/            # Configuration files
├── docs/               # Documentation
└── results/            # Output directory
```

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{alignment_framework,
  title = {Neural Network Alignment Analysis Framework},
  author = {Kempner Institute},
  year = {2024},
  url = {https://github.com/KempnerInstitute/alignment}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.




