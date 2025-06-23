# Experiments Module

This module provides a comprehensive framework for running structured experiments with neural network alignment analysis. It includes base classes, experiment runners, and specialized experiments for various analysis scenarios.

## Overview

The experiments module offers:
- **Base Framework**: Extensible base classes for creating custom experiments
- **Experiment Runner**: Utilities for managing and executing multiple experiments
- **General Alignment Experiment**: Complete pipeline for training, metric computation, and pruning
- **Specialized Analysis**: Focused experiments for specific research questions

## Module Structure

```
experiments/
├── base.py                  # Base experiment class and configuration
├── runner.py               # Experiment runner for batch execution
├── general_alignment.py    # General-purpose alignment analysis
└── README.md              # This file
```

Note: Pruning-specific experiments have been moved to `pruning/experiments/` for better organization.

## Available Experiments

### 1. General Alignment Experiment (`general_alignment.py`)

A comprehensive experiment that provides a complete pipeline for alignment analysis.

**Features:**
- Model training on any dataset
- Computation of multiple alignment metrics
- Pruning with various strategies
- Fine-tuning after pruning
- Automatic performance tracking
- Detailed analysis and reporting

**Configuration Example:**
```yaml
# general_alignment_config.yaml
name: "mnist_alignment_analysis"
dataset_name: "mnist"
model_name: "mlp"

training_config:
  epochs: 20
  learning_rate: 0.001
  batch_size: 64

alignment_metrics:
  - "rayleigh_quotient"
  - "mutual_information_gaussian"
  - "weight_cosine_similarity"

pruning_strategy: "magnitude"
pruning_config:
  amount: 0.5

train_model: true
compute_initial_metrics: true
apply_pruning: true
fine_tune_after_pruning: true
```

**Usage:**
```python
from alignment.experiments import GeneralAlignmentExperiment

# From configuration file
experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()

# Or programmatically
from alignment.experiments import GeneralAlignmentConfig

config = GeneralAlignmentConfig(
    dataset_name="cifar10",
    model_name="resnet18",
    alignment_metrics=["rayleigh_quotient", "spectral_gap"],
    pruning_strategy="gradient"
)
experiment = GeneralAlignmentExperiment(config)
results = experiment.run()
```

### 2. Base Experiment Class

All experiments inherit from `BaseExperiment` which provides:

- **Configuration Management**: Structured configuration handling
- **Model Wrapping**: Automatic model preparation for metric computation
- **Metric Computation**: Standardized interface for all metrics
- **Result Tracking**: Automatic logging and checkpointing
- **Device Management**: GPU/CPU handling

**Creating Custom Experiments:**
```python
from alignment.experiments import BaseExperiment, ExperimentConfig
from dataclasses import dataclass

@dataclass
class MyExperimentConfig(ExperimentConfig):
    custom_param: float = 0.5
    analysis_type: str = "correlation"

class MyCustomExperiment(BaseExperiment):
    def __init__(self, config: MyExperimentConfig):
        super().__init__(config)
        self.config = config
    
    def run(self) -> Dict[str, Any]:
        # Implement your experiment logic
        results = {}
        
        # Use wrapped model for metrics
        metrics = self.compute_metrics(data)
        
        # Custom analysis
        analysis = self.perform_custom_analysis()
        
        # Save results
        self.save_results(results)
        return results
```

### 3. Experiment Runner

Manages and executes multiple experiments efficiently.

**Features:**
- Sequential or parallel execution
- Grid search over parameters
- Result aggregation
- Progress tracking
- Error handling

**Usage:**
```python
from alignment.experiments import ExperimentRunner

runner = ExperimentRunner(
    results_dir="./results",
    parallel=True,
    max_workers=4
)

# Add individual experiments
runner.add_experiment(
    "general_alignment",
    config_overrides={"pruning_strategy": "magnitude"}
)

# Or grid search
runner.add_grid_search(
    "general_alignment",
    param_grid={
        "pruning_strategy": ["magnitude", "gradient", "random"],
        "pruning_config.amount": [0.3, 0.5, 0.7]
    }
)

# Run all experiments
results = runner.run_all()

# Find best configuration
best = runner.get_best_experiment("accuracy", "final", minimize=False)
```

## Configuration System

### YAML Configuration
```yaml
# Base configuration
name: "experiment_name"
description: "Detailed description"
seed: 42
device: "cuda"

# Model configuration
model_name: "resnet18"
model_config:
  num_classes: 10
  pretrained: false

# Dataset configuration
dataset_name: "cifar10"
dataset_config:
  data_path: "./data"
  augmentation: true

# Metrics to compute
metrics:
  - "rayleigh_quotient"
  - "mutual_information_gaussian"

# Logging
log_dir: "./logs"
checkpoint_dir: "./checkpoints"
```

### Programmatic Configuration
```python
from alignment.experiments import ExperimentConfig

config = ExperimentConfig(
    name="my_experiment",
    model_name="vgg16",
    metrics=["spectral_gap", "eigenvalue_entropy"],
    device="cuda:0"
)
```

## Running Experiments

### Command Line Interface
```bash
# Run single experiment
python -m alignment.experiments.run config.yaml

# With overrides
python -m alignment.experiments.run config.yaml \
    --device cuda:1 \
    --batch-size 128 \
    --epochs 50
```

### Python API
```python
# Simple execution
experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()

# With custom data
experiment = GeneralAlignmentExperiment(config)
results = experiment.run(
    train_loader=my_train_loader,
    val_loader=my_val_loader
)
```

### Batch Execution
```python
# Run multiple configurations
configs = [
    "configs/exp1.yaml",
    "configs/exp2.yaml",
    "configs/exp3.yaml"
]

for config_path in configs:
    experiment = GeneralAlignmentExperiment.from_yaml(config_path)
    results = experiment.run()
```

## Results and Analysis

### Result Structure
```python
results = {
    "initial_metrics": {
        "rayleigh_quotient": {"layer1": 0.85, "layer2": 0.92},
        "spectral_gap": {"layer1": 0.23, "layer2": 0.31}
    },
    "pruning_results": {
        "masks": {...},
        "sparsity": {"layer1": 0.5, "layer2": 0.5, "overall": 0.5}
    },
    "final_metrics": {...},
    "performance_history": {
        "train_loss": [...],
        "val_accuracy": [...]
    },
    "analysis": {
        "metric_changes": {...},
        "sparsity_impact": {...}
    }
}
```

### Automatic Analysis
The framework automatically computes:
- Metric changes (before/after pruning)
- Performance retention
- Layer-wise statistics
- Correlation analysis

### Integration with Analysis Module
```python
from alignment.analysis import ResultAggregator, HTMLReporter

# Aggregate results from multiple experiments
aggregator = ResultAggregator()
aggregator.load_from_directory("./results/")

# Generate comprehensive report
reporter = HTMLReporter("Experiment Analysis")
reporter.add_dataframe("Results", aggregator.to_dataframe())
reporter.generate("analysis.html")
```

## Best Practices

1. **Use Configuration Files**: Keep experiments reproducible with YAML configs
2. **Set Random Seeds**: Ensure reproducibility with fixed seeds
3. **Log Everything**: Use the built-in logging for debugging
4. **Save Checkpoints**: Enable checkpointing for long experiments
5. **Validate Early**: Test with small datasets first
6. **Use Parallel Execution**: Run multiple experiments simultaneously
7. **Automate Analysis**: Generate reports automatically after experiments

## Examples

### Complete Workflow
```python
from alignment.experiments import (
    GeneralAlignmentExperiment,
    ExperimentRunner
)
from alignment.analysis import ResultAggregator, HTMLReporter

# 1. Define experiment configurations
base_config = {
    "dataset_name": "mnist",
    "model_name": "mlp",
    "alignment_metrics": ["rayleigh_quotient", "spectral_gap"]
}

# 2. Run experiments with different pruning strategies
runner = ExperimentRunner()
for strategy in ["magnitude", "gradient", "random"]:
    runner.add_experiment(
        GeneralAlignmentExperiment,
        config_overrides={
            **base_config,
            "pruning_strategy": strategy,
            "name": f"mnist_{strategy}_pruning"
        }
    )

results = runner.run_all()

# 3. Analyze results
aggregator = ResultAggregator()
for name, result in results.items():
    aggregator.add_results(name, result)

# 4. Generate report
df = aggregator.to_dataframe()
best_strategy = df.loc[df['final_accuracy'].idxmax(), 'experiment']
print(f"Best strategy: {best_strategy}")
```

## Extending the Framework

To add new experiment types:

1. Create a new class inheriting from `BaseExperiment`
2. Implement the `run()` method
3. Add custom configuration if needed
4. Register with the experiment registry (optional)
5. Document the experiment purpose and usage

The framework is designed to be extensible while maintaining consistency across different experiment types. 