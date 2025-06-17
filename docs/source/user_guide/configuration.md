# Configuration Guide

This guide provides comprehensive information about configuring experiments in the Neural Network Alignment framework.

## Configuration System Overview

## ExperimentConfig Structure

The main configuration class is `ExperimentConfig`:

```python
from alignment_refactor.experiments.base import ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    description="Detailed experiment description",
    tags=["pruning", "mnist", "baseline"],
    
    # Model configuration
    model_name="mlp",
    model_config={
        "hidden_dims": [300, 200, 100],
        "dropout_rate": 0.5,
        "activation": "relu"
    },
    
    # Dataset configuration
    dataset_name="mnist",
    dataset_config={
        "data_path": "./data",
        "augmentation": False
    },
    
    # Training configuration
    batch_size=128,
    num_workers=4,
    device="cuda",
    seed=42,
    
    # Metrics configuration
    metrics=["rayleigh_quotient", "mutual_information"],
    metric_configs={
        "rayleigh_quotient": {"scale_by_norm": False},
        "mutual_information": {"estimation_method": "gaussian"}
    }
)
```

## Configuration Parameters

### Experiment Identification

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Unique experiment identifier |
| `description` | str | "" | Detailed experiment description |
| `tags` | List[str] | [] | Tags for categorizing experiments |

### Model Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "resnet18" | Model architecture name |
| `model_config` | Dict | {} | Model-specific parameters |
| `pretrained` | bool | False | Use pretrained weights |
| `tracked_layers` | List[str] | None | Layers to track activations |

#### Model-Specific Parameters

**MLP Configuration:**
```python
model_config = {
    "input_dim": 784,
    "hidden_dims": [300, 200, 100],
    "output_dim": 10,
    "dropout_rate": 0.5,
    "activation": "relu"  # Options: "relu", "tanh", "sigmoid"
}
```

**CNN Configuration:**
```python
model_config = {
    "in_channels": 3,
    "conv_channels": [32, 64],
    "kernel_sizes": [5, 5],
    "hidden_fc_dim": 128,
    "dropout_rate": 0.5
}
```

### Dataset Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_name` | str | "cifar10" | Dataset to use |
| `dataset_config` | Dict | {} | Dataset-specific parameters |
| `data_path` | str | None | Path to dataset files |

### Training Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 128 | Batch size for training |
| `num_workers` | int | 4 | Data loader workers |
| `device` | str | "cuda" | Computing device |
| `seed` | int | 42 | Random seed |
| `train_before_dropout` | bool | True | Train before pruning |
| `training_epochs` | int | 10 | Number of training epochs |
| `learning_rate` | float | 0.001 | Learning rate |
| `optimizer` | str | "adam" | Optimizer type |

### Metrics Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | List[str] | ["rayleigh_quotient"] | Metrics to compute |
| `metric_configs` | Dict | {} | Metric-specific configurations |
| `scale_by_norm` | bool | False | Scale alignment scores by weight norm |
| `force_cpu_for_large_metric_ops` | bool | True | Use CPU for memory-intensive operations |

### Pruning Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dropout_fractions` | List[float] | [0.0, 0.2, 0.4, 0.6, 0.8] | Pruning percentages |
| `pruning_strategy` | str | "magnitude" | Pruning strategy |
| `structured_pruning` | bool | False | Use structured pruning |

### Checkpointing Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint_dir` | str | "./checkpoints" | Checkpoint directory |
| `checkpoint_interval` | int | 1000 | Steps between checkpoints |
| `save_best` | bool | True | Save best performing model |

### Logging Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | str | "./logs" | Log directory |
| `log_interval` | int | 100 | Steps between log entries |
| `wandb_project` | str | None | Weights & Biases project |
| `wandb_entity` | str | None | Weights & Biases entity |

## Loading and Saving Configurations

### From/To Dictionary

```python
# Convert to dictionary
config_dict = config.to_dict()

# Create from dictionary
config = ExperimentConfig.from_dict(config_dict)
```

### From/To JSON

```python
# Save to JSON
config.save("config.json")

# Load from JSON
config = ExperimentConfig.load("config.json")
```

### From/To YAML

```python
import yaml

# Save to YAML
with open("config.yaml", "w") as f:
    yaml.dump(config.to_dict(), f)

# Load from YAML
with open("config.yaml", "r") as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = ExperimentConfig.from_dict(config_dict)
```

## Using YAML Configuration Files

### Overview

The framework provides a complete YAML-based configuration system for defining and running experiments. This is the recommended approach for reproducible experiments.

### Basic YAML Configuration

Create a file `experiment.yaml`:

```yaml
# Basic experiment configuration
name: "my_first_experiment"
description: "Testing progressive dropout on MNIST"

# Model setup
model_name: "mlp"
model_config:
  input_dim: 784
  hidden_dims: [300, 200, 100]
  output_dim: 10
  dropout_rate: 0.5
  activation: "relu"

# Dataset
dataset_name: "mnist"
data_path: "./data"
batch_size: 128

# Training
device: "cuda"
training_epochs: 10
learning_rate: 0.001

# Metrics
metrics: ["rayleigh_quotient"]

# Experiment specific
dropout_fractions: [0.0, 0.2, 0.4, 0.6, 0.8]
```

### Running Experiments from YAML

Use the provided script:

```bash
python src/alignment_refactor/examples/run_experiment_from_config.py experiment.yaml
```

With command-line overrides:

```bash
# Override specific parameters
python run_experiment_from_config.py experiment.yaml \
    --device cpu \
    --batch-size 256 \
    --epochs 20 \
    --no-train
```

### Configuration Templates

The framework provides templates in `configs/templates/`:

1. **mnist_mlp.yaml** - MLP experiments on MNIST
2. **cifar10_cnn.yaml** - CNN experiments on CIFAR-10
3. **comprehensive_example.yaml** - All available options

Copy and modify a template:

```bash
cp configs/templates/mnist_mlp.yaml my_experiment.yaml
# Edit my_experiment.yaml
python run_experiment_from_config.py my_experiment.yaml
```

### Environment Variables in YAML

Use environment variables with defaults:

```yaml
# Use environment variables
data_path: ${DATA_PATH:./data}
device: ${DEVICE:cuda}
checkpoint_dir: ${CHECKPOINT_DIR:./checkpoints}
log_dir: ${LOG_DIR:./logs}

# With complex paths
wandb_project: ${WANDB_PROJECT:neural-alignment}
results_dir: ${RESULTS_DIR:${HOME}/experiments/results}
```

### Advanced YAML Examples

#### Multi-Metric Configuration

```yaml
name: "multi_metric_analysis"

metrics:
  - "rayleigh_quotient"
  - "mutual_information"
  - "cka"
  - "cca"

metric_configs:
  rayleigh_quotient:
    scale_by_norm: false
    aggregation_op: "mean"
    force_cpu: true
    
  mutual_information:
    estimation_method: "gaussian"
    num_samples: 5000
    
  cka:
    kernel: "rbf"
    threshold: 0.01
    
  cca:
    n_components: 50
    reg: 0.1
```

#### Experiment-Specific Configurations

**Progressive Dropout:**
```yaml
name: "progressive_dropout_study"
dropout_fractions: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_mode: "scaled"  # or "unscaled"
pruning_strategy: "magnitude"  # or "random", "structured"
```

**Eigenvector Alignment:**
```yaml
name: "eigenvector_analysis"
num_components: 20
compute_full_eigenspectrum: true
```

**Layer-Isolated Pruning:**
```yaml
name: "layer_importance"
pruning_percentages: [0.1, 0.3, 0.5, 0.7, 0.9]
layer_pruning_order: "sequential"  # or "reverse", "random"
```

#### Complete Production Example

```yaml
name: "cifar10_resnet_alignment_study"
description: "Comprehensive alignment study on ResNet-18 with CIFAR-10"
tags: ["production", "cifar10", "resnet", "alignment"]

# Model configuration
model_name: "resnet18"
pretrained: true
model_config:
  num_classes: 10

# Dataset configuration
dataset_name: "cifar10"
data_path: ${CIFAR10_PATH:/data/cifar10}
dataset_config:
  normalize: true
  augmentation: true
  
# Training configuration
batch_size: 256
num_workers: 8
device: ${DEVICE:cuda:0}
seed: 42

train_before_dropout: true
training_epochs: 50
learning_rate: 0.1
optimizer: "sgd"

optimizer_config:
  momentum: 0.9
  weight_decay: 0.0001
  nesterov: true

lr_scheduler: "cosine"
lr_scheduler_config:
  T_max: 50
  eta_min: 0.0001

# Metrics configuration
metrics:
  - "rayleigh_quotient"
  - "mutual_information"
  - "cka"

tracked_layers:  # Specific ResNet layers
  - "layer1.0.conv1"
  - "layer2.0.conv1"
  - "layer3.0.conv1"
  - "layer4.0.conv1"

# Experiment configuration
dropout_fractions: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
dropout_mode: "scaled"
exclude_classification_layer: true

# Output configuration
checkpoint_dir: ${CHECKPOINT_DIR:./checkpoints/cifar10_resnet}
checkpoint_interval: 1000
save_best: true

log_dir: ${LOG_DIR:./logs/cifar10_resnet}
log_interval: 50

# Weights & Biases
wandb_project: ${WANDB_PROJECT:alignment-studies}
wandb_entity: ${WANDB_ENTITY:my-team}
wandb_config:
  log_model: true
  log_gradients: true
  gradient_log_freq: 100

# Advanced options
gradient_accumulation_steps: 1
mixed_precision: true
deterministic: true

# Plotting
plotting:
  save_plots: true
  plot_format: "pdf"
  dpi: 300
  plot_types:
    - "metric_vs_dropout"
    - "layer_comparison"
    - "eigenvalue_spectrum"
```

### YAML Best Practices

1. **Use Comments**: Document your configuration choices
2. **Environment Variables**: Use for machine-specific paths
3. **Consistent Naming**: Follow the naming conventions
4. **Version Control**: Track your YAML files in git
5. **Validation**: The config loader validates your YAML automatically

### Config File Organization

Recommended directory structure:

```
experiments/
├── configs/
│   ├── baselines/
│   │   ├── mnist_baseline.yaml
│   │   └── cifar10_baseline.yaml
│   ├── ablations/
│   │   ├── dropout_modes.yaml
│   │   └── pruning_strategies.yaml
│   └── production/
│       └── final_experiments.yaml
├── results/
└── scripts/
```

### Troubleshooting YAML Configs

Common issues and solutions:

1. **Invalid YAML Syntax**
   ```yaml
   # Wrong - no space after colon
   name:"experiment"
   
   # Correct
   name: "experiment"
   ```

2. **Type Mismatches**
   ```yaml
   # Wrong - string instead of list
   dropout_fractions: "0.1, 0.2, 0.3"
   
   # Correct
   dropout_fractions: [0.1, 0.2, 0.3]
   ```

3. **Missing Required Fields**
   - The config validator will tell you which fields are missing
   - Check the comprehensive example for all required fields

## Configuration Templates

### Minimal Configuration

```python
config = ExperimentConfig(
    name="minimal_experiment",
    model_name="mlp",
    dataset_name="mnist"
)
```

### Full Configuration Example

```python
config = ExperimentConfig(
    # Identification
    name="comprehensive_pruning_study",
    description="Study of pruning strategies on CIFAR-10",
    tags=["pruning", "cifar10", "cnn", "baseline"],
    
    # Model
    model_name="cnn2p2",
    model_config={
        "in_channels": 3,
        "conv_channels": [64, 128],
        "kernel_sizes": [3, 3],
        "hidden_fc_dim": 256,
        "dropout_rate": 0.3
    },
    pretrained=False,
    
    # Dataset
    dataset_name="cifar10",
    dataset_config={
        "data_path": "./data",
        "augmentation": True,
        "normalize": True
    },
    
    # Training
    batch_size=64,
    num_workers=8,
    device="cuda:0",
    seed=42,
    train_before_dropout=True,
    training_epochs=20,
    learning_rate=0.001,
    optimizer="adamw",
    
    # Metrics
    metrics=["rayleigh_quotient", "mutual_information", "cka"],
    metric_configs={
        "rayleigh_quotient": {
            "scale_by_norm": False,
            "aggregation_op": "mean"
        },
        "mutual_information": {
            "estimation_method": "gaussian",
            "num_samples": 1000
        },
        "cka": {
            "kernel": "linear"
        }
    },
    
    # Pruning
    dropout_fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    pruning_strategy="metric_based",
    structured_pruning=True,
    
    # Checkpointing
    checkpoint_dir="./checkpoints/cifar10_cnn",
    checkpoint_interval=500,
    save_best=True,
    
    # Logging
    log_dir="./logs/cifar10_cnn",
    log_interval=50,
    wandb_project="alignment-studies",
    wandb_entity=None
)
```

## Environment Variables

The framework supports environment variable overrides:

```bash
# Override device
export ALIGNMENT_DEVICE="cpu"

# Override batch size
export ALIGNMENT_BATCH_SIZE="256"

# Override checkpoint directory
export ALIGNMENT_CHECKPOINT_DIR="/path/to/checkpoints"
```

## Validation

The configuration system performs automatic validation:

```python
# This will raise an error
config = ExperimentConfig(
    name="invalid",
    batch_size=-1  # Invalid: must be positive
)

# This will raise an error
config = ExperimentConfig(
    name="invalid",
    device="invalid_device"  # Invalid: must be cuda or cpu
)
```

## Best Practices

1. **Use Descriptive Names**: Include key parameters in experiment names
2. **Set Seeds**: Always set seeds for reproducibility
3. **Document Configurations**: Use the description field
4. **Version Control**: Save configurations with experiment results
5. **Use Templates**: Create reusable configuration templates

## Advanced Configuration

### Dynamic Configuration

```python
def create_config(model_size="small", dataset="mnist"):
    """Create configuration based on parameters"""
    
    hidden_dims = {
        "small": [100, 50],
        "medium": [300, 200, 100],
        "large": [500, 400, 300, 200]
    }
    
    return ExperimentConfig(
        name=f"{model_size}_{dataset}_experiment",
        model_name="mlp",
        model_config={
            "hidden_dims": hidden_dims[model_size]
        },
        dataset_name=dataset
    )
```

### Configuration Inheritance

```python
# Base configuration
base_config = ExperimentConfig(
    model_name="mlp",
    dataset_name="mnist",
    batch_size=128,
    device="cuda"
)

# Create variations
configs = []
for lr in [0.001, 0.01, 0.1]:
    config = ExperimentConfig(**base_config.to_dict())
    config.learning_rate = lr
    config.name = f"lr_study_{lr}"
    configs.append(config)
```

### Conditional Configuration

```python
import torch

config = ExperimentConfig(
    name="adaptive_experiment",
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=256 if torch.cuda.is_available() else 64,
    num_workers=8 if torch.cuda.is_available() else 2
)
``` 