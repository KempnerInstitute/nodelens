# Neural Network Alignment Framework

A comprehensive framework for studying neural network alignment properties through information-theoretic metrics and pruning strategies.

## Features

- **Alignment Metrics**: Rayleigh Quotient (RQ), Mutual Information (MI), Partial Information Decomposition (PID), CKA, CCA
- **Pruning Strategies**: Progressive dropout, eigenvector-based, layer-isolated, and cascading pruning
- **Model Support**: Pre-defined architectures (MLP, CNN) and support for custom models
- **Experiment Framework**: Reproducible experiment management with comprehensive configuration
- **Tensorized Dropout**: Efficient structured pruning implementation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd alignment/src/alignment_refactor

# Install the package
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from alignment_refactor.models.architectures.standard_models import create_model
from alignment_refactor.experiments.progressive_dropout import ProgressiveDropoutExperiment
from alignment_refactor.experiments.base import ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    name="mnist_pruning",
    model_name="mlp",
    dataset_name="mnist",
    metrics=["rayleigh_quotient"],
    device="cuda"
)

# Run experiment
experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements-docs.txt

# Build HTML documentation
cd docs
make html

# View documentation
open build/html/index.html
```

### Live Documentation Server

```bash
cd docs
make livehtml
# Navigate to http://localhost:8000
```

## Project Structure

```
alignment_refactor/
├── core/                   # Core functionality (tensorized dropout, registry)
├── models/                 # Model architectures and wrappers
│   ├── architectures/      # Standard models (MLP, CNN)
│   └── wrapper.py          # ModelWrapper for activation tracking
├── metrics/                # Alignment and information metrics
│   ├── rayleigh/          # Rayleigh quotient metrics
│   ├── information/       # MI, PID metrics
│   └── similarity/        # CKA, CCA metrics
├── experiments/           # Experiment implementations
│   ├── base.py           # Base experiment class
│   ├── progressive_dropout.py
│   ├── eigenvector.py
│   ├── layer_isolated.py
│   └── cascading.py
├── data/                  # Dataset handling
├── utils/                 # Utility functions
├── examples/              # Example scripts
└── docs/                  # Documentation source
```

## Examples

### Basic MLP Pruning

```python
from alignment_refactor.models.architectures.standard_models import MLP
from alignment_refactor.models import ModelWrapper
from alignment_refactor.metrics import RayleighQuotient

# Create model
model = MLP(input_dim=784, hidden_dims=[300, 200], output_dim=10)

# Wrap for tracking
tracked_layers = ['network.0', 'network.3']
wrapped_model = ModelWrapper(model, tracked_layers=tracked_layers)

# Compute metrics
metric = RayleighQuotient()
# ... forward pass and metric computation
```

### Running Multiple Experiments

```python
from alignment_refactor.experiments.runner import ExperimentRunner

runner = ExperimentRunner(base_config=config)

# Add experiments
for pruning_rate in [0.1, 0.3, 0.5]:
    runner.add_experiment(
        "progressive_dropout",
        config_overrides={"pruning_rate": pruning_rate}
    )

results = runner.run_all()
```

## Configuration

Experiments are configured using the `ExperimentConfig` dataclass:

```python
config = ExperimentConfig(
    # Experiment identification
    name="experiment_name",
    description="Detailed description",
    
    # Model configuration
    model_name="mlp",
    model_config={"hidden_dims": [300, 200]},
    
    # Training configuration
    batch_size=128,
    learning_rate=0.001,
    training_epochs=10,
    
    # Metrics configuration
    metrics=["rayleigh_quotient", "mutual_information"],
    metric_configs={
        "rayleigh_quotient": {"scale_by_norm": False}
    }
)
```

## Available Metrics

- **Rayleigh Quotient (RQ)**: Measures neuron alignment with input variance
- **Mutual Information (MI)**: Quantifies information shared between layers
- **Partial Information Decomposition (PID)**: Decomposes information into unique, redundant, and synergistic components
- **Centered Kernel Alignment (CKA)**: Measures similarity between representations
- **Canonical Correlation Analysis (CCA)**: Finds maximally correlated projections

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 