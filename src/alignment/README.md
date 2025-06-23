# Neural Network Alignment Framework

A comprehensive framework for studying neural network alignment properties through information-theoretic metrics and pruning strategies.

## Features

- **Alignment Metrics**: Rayleigh Quotient (RQ), Mutual Information (MI), Partial Information Decomposition (PID), and 30+ other metrics
- **Metric Categories**: Information-theoretic, Similarity, Spectral, Task-specific, and Rayleigh-based metrics
- **Model Support**: Works with any PyTorch model through ModelWrapper interface
- **Batch Processing**: Efficient metric computation with GPU support
- **Visualization**: Comprehensive plotting and analysis tools
- **Experiment Tracking**: Integration with Weights & Biases and TensorBoard
- **Pruning Utilities**: Advanced neuron importance analysis and pruning strategies

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd alignment

# Install the package
pip install -e .

# Or install with specific extras
pip install -e ".[dev,docs,viz]"
```

## Quick Start

```python
import torch
from alignment.core import ModelWrapper
from alignment.metrics import get_metric

# Create and wrap your model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)
wrapped_model = ModelWrapper(model)

# Prepare data
inputs = torch.randn(100, 784)

# Extract activations
activations = wrapped_model.extract_activations(inputs)

# Compute metrics
rq_metric = get_metric("rayleigh_quotient")()
scores = rq_metric.compute(
    inputs=activations['0'],  # First layer activations
    weights=model[0].weight
)
```

## Project Structure

```
alignment/
├── core/                   # Core functionality
│   ├── base.py            # Base metric class
│   ├── model_wrapper.py   # Model wrapping utilities
│   └── registry.py        # Metric registry system
├── metrics/               # All metric implementations
│   ├── information/       # MI, PID, and information metrics
│   ├── rayleigh/         # Rayleigh quotient variants
│   ├── similarity/       # Similarity and correlation metrics
│   ├── spectral/         # Spectral analysis metrics
│   └── task_specific/    # Task-specific alignment metrics
├── utils/                 # Utility functions
│   ├── batch_processing.py    # Batch and parallel processing
│   ├── experiment_tracking.py # Experiment tracking
│   ├── pruning.py            # Pruning utilities
│   └── optimized/            # GPU-accelerated functions
├── visualization/         # Visualization tools
├── data/                  # Dataset utilities
├── models/               # Model architectures
└── examples/             # Example scripts
```

## Available Metrics

The framework provides 36+ metrics across 6 categories:

### Information-Theoretic Metrics
- Mutual Information (Gaussian, Binning, Analytic)
- Partial Information Decomposition (PID)
- Conditional Mutual Information
- Total Correlation
- Interaction Information

### Rayleigh Quotient Metrics
- Standard Rayleigh Quotient
- Alternative formulations
- Patchwise analysis

### Similarity Metrics
- Activation/Weight Cosine Similarity
- Node Correlation and Redundancy
- Weight-Activation Alignment

### Spectral Metrics
- Spectral Gap and Norm Ratio
- Eigenvalue Entropy and Alignment
- Power Iteration Analysis

### Task-Specific Metrics
- Classification Alignment
- Language Model Alignment
- Vision Task Alignment
- Reinforcement Learning Alignment

## Advanced Usage

### Batch Processing

```python
from alignment.utils.batch_processing import BatchMetricProcessor

processor = BatchMetricProcessor(
    metrics=['rayleigh_quotient', 'mutual_information_gaussian'],
    batch_size=1000,
    use_gpu=True
)

results = processor.process(model, dataloader)
```

### Visualization

```python
from alignment.visualization import AlignmentVisualizer

visualizer = AlignmentVisualizer()
visualizer.plot_metric_distributions(results)
visualizer.create_report("alignment_report.html")
```

### Experiment Tracking

```python
from alignment.utils.experiment_tracking import create_tracker

tracker = create_tracker("wandb", project="alignment-analysis")
tracker.log_metrics(results)
```

## Examples

See the `examples/` directory for comprehensive demonstrations:
- `quick_demo.py` - Basic usage example
- `advanced_analysis.py` - Advanced features showcase
- `comprehensive_demo.py` - Full pipeline example
- `pruning_demo.py` - Pruning analysis example

## Contributing

Contributions are welcome! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 