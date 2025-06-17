# Quick Start Guide

This guide provides a quick introduction to using the Neural Network Alignment framework.

## Basic Concepts

The framework is organized around several key concepts:

1. **Models**: Neural network architectures (MLP, CNN, etc.)
2. **Metrics**: Measures of alignment and information (RQ, MI, PID, etc.)
3. **Experiments**: Structured ways to run pruning and analysis
4. **ModelWrapper**: Automatic activation and weight tracking

## Quick Start with Config Files

The easiest way to run experiments is using configuration files:

### 1. Create a Config File

Create a file `my_experiment.yaml`:

```yaml
name: "mnist_dropout_study"
description: "Progressive dropout on MNIST MLP"

# Model configuration
model_name: "mlp"
model_config:
  input_dim: 784
  hidden_dims: [300, 200, 100]
  output_dim: 10
  dropout_rate: 0.5

# Dataset
dataset_name: "mnist"
data_path: "./data"
batch_size: 128

# Metrics to track
metrics: ["rayleigh_quotient", "mutual_information"]

# Experiment parameters
dropout_fractions: [0.0, 0.2, 0.4, 0.6, 0.8]
training_epochs: 10
device: "cuda"
```

### 2. Run the Experiment

```bash
python src/alignment_refactor/examples/run_experiment_from_config.py my_experiment.yaml
```

### 3. Command-Line Overrides

Override any parameter from the command line:

```bash
# Change device and batch size
python run_experiment_from_config.py my_experiment.yaml --device cpu --batch-size 256

# Skip training phase
python run_experiment_from_config.py my_experiment.yaml --no-train
```

### 4. Use Templates

Start with pre-configured templates:

```bash
# Copy a template
cp src/alignment_refactor/configs/templates/mnist_mlp.yaml my_config.yaml

# Edit and run
python run_experiment_from_config.py my_config.yaml
```

## Your First Experiment (Programmatic Approach)

### Step 1: Create a Model

```python
from alignment_refactor.models.architectures.standard_models import create_model

# Create an MLP for MNIST
model = create_model('mlp', 'mnist', hidden_dims=[300, 200, 100])

# Or create a CNN for CIFAR-10
model = create_model('cnn2p2', 'cifar10')
```

### Step 2: Wrap the Model for Tracking

```python
from alignment_refactor.models import ModelWrapper

# Identify layers to track (Linear layers in MLP)
tracked_layers = ['network.0', 'network.3', 'network.6']  
wrapped_model = ModelWrapper(model, tracked_layers=tracked_layers)
```

### Step 3: Compute Metrics

```python
from alignment_refactor.metrics import RayleighQuotient
import torch

# Create dummy data
inputs = torch.randn(32, 784)  # Batch of MNIST-like data

# Forward pass with activation tracking
outputs, activations = wrapped_model.forward_with_activations(inputs)

# Compute Rayleigh Quotient
rq_metric = RayleighQuotient()
weights = wrapped_model.get_layer_weights()

for layer_name in tracked_layers:
    layer_input = activations[f"{layer_name}_input"].flatten(start_dim=1)
    layer_weight = weights[layer_name]
    
    rq_scores = rq_metric.compute(inputs=layer_input, weights=layer_weight)
    print(f"{layer_name}: Mean RQ = {rq_scores.mean().item():.4f}")
```

### Step 4: Run a Pruning Experiment

```python
from alignment_refactor.experiments.progressive_dropout import ProgressiveDropoutExperiment
from alignment_refactor.experiments.base import ExperimentConfig

# Configure the experiment
config = ExperimentConfig(
    name="mnist_progressive_pruning",
    model_name="mlp",
    dataset_name="mnist",
    model_config={
        "hidden_dims": [300, 200, 100],
        "dropout_rate": 0.5
    },
    metrics=["rayleigh_quotient"],
    batch_size=128,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Run the experiment
experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()

# Results contain accuracy vs pruning fraction data
print(f"Final accuracies: {results['final_accuracies']}")
```

## Common Use Cases

### 1. Comparing Pruning Strategies

```python
# Run multiple experiments with different strategies
strategies = ['magnitude', 'random', 'gradient']
results = {}

for strategy in strategies:
    config = ExperimentConfig(
        name=f"comparison_{strategy}",
        pruning_strategy=strategy,
        # ... other config options
    )
    exp = ProgressiveDropoutExperiment(config)
    results[strategy] = exp.run()
```

### 2. Analyzing Layer Importance

```python
from alignment_refactor.experiments.layer_isolated import LayerIsolatedPruningExperiment

config = ExperimentConfig(
    name="layer_importance_analysis",
    model_name="cnn2p2",
    dataset_name="cifar10",
    metrics=["rayleigh_quotient", "mutual_information"]
)

experiment = LayerIsolatedPruningExperiment(config)
results = experiment.run()

# Results show impact of pruning each layer individually
for layer, accuracy in results['layer_accuracies'].items():
    print(f"Pruning {layer}: {accuracy}% accuracy retained")
```

### 3. Using Multiple Metrics

```python
from alignment_refactor.metrics import (
    RayleighQuotient, 
    MutualInformationGaussian,
    PartialInformationDecomposition
)

# Create multiple metrics
metrics = {
    'rq': RayleighQuotient(),
    'mi': MutualInformationGaussian(),
    'pid': PartialInformationDecomposition()
}

# Compute all metrics
results = {}
for name, metric in metrics.items():
    scores = metric.compute(inputs=layer_input, weights=layer_weight)
    results[name] = scores.mean().item()
```

## Working with Datasets

```python
from alignment_refactor.data import get_dataset

# Load MNIST
mnist = get_dataset('mnist')
train_loader = mnist.train_loader
test_loader = mnist.test_loader

# Load CIFAR-10 with custom batch size
cifar = get_dataset('cifar10', batch_size=64)
```

## Saving and Loading Results

```python
# Experiments automatically save results
experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()

# Results are saved to checkpoint_dir/experiment_name/
# You can also save manually
import pickle
with open('my_results.pkl', 'wb') as f:
    pickle.dump(results, f)
```

## Tips for Getting Started

1. **Start Small**: Begin with MNIST and small models to understand the framework
2. **Use GPU**: Set `device="cuda"` in configs for faster experiments
3. **Monitor Progress**: Experiments show progress bars during execution
4. **Check Logs**: Detailed logs are saved in the checkpoint directory
5. **Visualize Results**: Use matplotlib to plot accuracy vs pruning curves

## Next Steps

- Explore [different experiment types](experiments.md)
- Learn about [configuration options](configuration.md)
- Understand [available metrics](metrics.md)
- Create [custom models](models.md)

## Example Scripts

Check the `examples/` directory for complete working examples:
- `mnist_mlp_pruning.py`: Basic MLP pruning on MNIST
- `simple_pruning_demo.py`: Quick demonstration
- `interactive_pruning_tutorial.py`: Comprehensive tutorial
- `using_standard_models.py`: Model usage examples 