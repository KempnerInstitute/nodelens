# Configuration System

YAML-based configuration system for reproducible experiments.

## Quick Start

```yaml
# Basic configuration
name: "my_experiment"
model_name: "mlp"
dataset_name: "mnist"
metrics: ["rayleigh_quotient"]
dropout_fractions: [0.0, 0.2, 0.4, 0.6, 0.8]
```

## Templates

Use provided templates as starting points:
- `templates/mnist_mlp.yaml` - Basic MLP on MNIST
- `templates/cifar10_cnn.yaml` - CNN on CIFAR-10
- `simplified_config.yaml` - Minimal configuration
- `clean_config.yaml` - Well-organized example

## Components

Configuration supports composable components from `config_components.py`:
- `ModelConfig` - Model architecture settings
- `DataConfig` - Dataset and preprocessing
- `TrainingConfig` - Training parameters
- `MetricConfig` - Alignment metrics configuration

## Usage

```python
from alignment.experiments import GeneralAlignmentExperiment

# From YAML file
experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()

# Command line overrides
python run_experiment.py config.yaml --device cuda:1 --batch-size 256
```

## Environment Variables

Use environment variables with defaults:
```yaml
data_path: ${DATA_PATH:./data}
device: ${DEVICE:cuda}
``` 