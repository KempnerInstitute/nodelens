# Codebase Organization

This document describes the organization and structure of the alignment framework codebase.

## Directory Structure

```
alignment/
├── src/alignment/          # Core source code
│   ├── core/              # Base classes, protocols, and registry
│   ├── metrics/           # Alignment metrics implementations
│   ├── models/            # Model architectures and wrappers
│   ├── data/              # Dataset handling and loaders
│   ├── training/          # Training utilities
│   ├── pruning/           # Pruning strategies and experiments
│   ├── experiments/       # Experiment framework
│   ├── analysis/          # Result analysis and reporting
│   ├── infrastructure/    # Computing and storage utilities
│   └── examples/          # Example scripts
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── docs/                  # Documentation
├── results/               # Output directory (gitignored)
└── checkpoints/          # Model checkpoints (gitignored)
```

## Module Descriptions

### Core (`core/`)
Foundation of the framework with base classes, protocols, and the metric registry system.
- `base.py`: Abstract base classes for all components
- `protocols.py`: Type protocols for consistency
- `registry.py`: Dynamic registration system
- `wrappers.py`: Model wrapper implementation

### Metrics (`metrics/`)
Comprehensive collection of 36+ alignment metrics organized by category:
- `rayleigh/`: Rayleigh quotient variants
- `information/`: Information-theoretic metrics (MI, PID, etc.)
- `similarity/`: Cosine similarity, correlation metrics
- `spectral/`: Eigenvalue and spectral analysis
- `task_specific/`: Domain-specific metrics

### Models (`models/`)
Model architectures and utilities:
- `architectures/`: Standard models (MLP, CNN, ResNet, etc.)
- `base.py`: Extended model wrapper functionality

### Data (`data/`)
Dataset handling and preprocessing:
- `datasets/`: Dataset wrappers (MNIST, CIFAR, etc.)
- `processing/`: Data preprocessing utilities
- `loaders.py`: DataLoader utilities
- `base.py`: Base dataset classes

### Training (`training/`)
Training utilities and strategies:
- `base.py`: BaseTrainer with comprehensive features
- `multi_network.py`: Train multiple networks simultaneously

### Pruning (`pruning/`)
Complete pruning framework:
- `strategies/`: Various pruning algorithms
- `structured/`: Structured pruning methods
- `experiments/`: Pruning-specific experiments
- `base.py`: Base pruning classes

### Experiments (`experiments/`)
Experiment framework for structured analysis:
- `base.py`: Base experiment class
- `runner.py`: Experiment execution utilities
- `general_alignment.py`: General-purpose alignment experiment

### Analysis (`analysis/`)
Result analysis and visualization:
- `aggregation/`: Result and metric aggregation
- `reporting/`: Report generation (HTML, Markdown, JSON)
- `visualization/`: Plotting and visualization utilities

### Infrastructure (`infrastructure/`)
System-level utilities:
- `computing/`: Distributed computing, GPU utilities
- `storage/`: Checkpoint and result management
- `configuration/`: Configuration handling

## Design Principles

1. **Modularity**: Each module has a single, clear purpose
2. **Extensibility**: Easy to add new metrics, models, or experiments
3. **Consistency**: Uniform APIs across all components
4. **Documentation**: Comprehensive documentation at all levels
5. **Testing**: Unit tests for critical functionality

## Key Features

### Registry System
Dynamic registration allows easy extension:
```python
@register_metric("my_metric")
class MyMetric(BaseMetric):
    # Implementation
```

### Unified Configuration
YAML-based configuration for experiments:
```yaml
name: "experiment_name"
model_name: "resnet18"
metrics: ["rayleigh_quotient", "spectral_gap"]
```

### Comprehensive Metrics
36+ metrics spanning multiple categories:
- Information theory
- Spectral analysis
- Similarity measures
- Task-specific alignment

### Flexible Experiments
From simple metric computation to complex training pipelines:
```python
experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()
```

## Usage Patterns

### Basic Metric Computation
```python
from alignment import ModelWrapper, get_metric

wrapped_model = ModelWrapper(model)
metric = get_metric("rayleigh_quotient")()
score = metric.compute(inputs=data, weights=model.weight)
```

### Running Experiments
```python
from alignment.experiments import GeneralAlignmentExperiment

experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()
```

### Analyzing Results
```python
from alignment.analysis import ResultAggregator, HTMLReporter

aggregator = ResultAggregator()
aggregator.load_from_directory("./results/")
reporter = HTMLReporter("Analysis")
reporter.generate("report.html")
```

## Best Practices

1. **Use the Registry**: Register custom components for easy reuse
2. **Configuration Files**: Keep experiments reproducible
3. **Structured Experiments**: Use the experiment framework
4. **Automated Analysis**: Generate reports automatically
5. **Version Control**: Track configurations and results

## Contributing

When adding new functionality:
1. Follow the existing module structure
2. Add comprehensive documentation
3. Include unit tests
4. Update relevant README files
5. Use type hints consistently 