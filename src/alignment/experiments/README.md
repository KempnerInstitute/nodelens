# Experiments Module

This module contains specialized experiment runners for analyzing neural network alignment under various conditions.

## Overview

Each experiment class extends `BaseExperiment` and implements a specific analysis methodology for understanding how neural networks process information and how their representations align with different metrics.

## Available Experiments

### 1. Cascading Experiment (`cascading.py`)
Analyzes the cascading effects of pruning on network alignment.

**Key Features:**
- Progressive pruning with alignment tracking
- Multiple pruning strategies (magnitude, random, structured)
- Tracks how alignment metrics change as network sparsity increases
- Supports iterative pruning with fine-tuning

**Usage:**
```python
from alignment.experiments import CascadingExperiment

experiment = CascadingExperiment(
    model=model,
    metrics=['rayleigh_quotient', 'mutual_information_gaussian'],
    pruning_ratios=[0.1, 0.3, 0.5, 0.7, 0.9]
)
results = experiment.run(dataloader)
```

### 2. Progressive Dropout Experiment (`progressive_dropout.py`)
Studies alignment under varying dropout rates.

**Key Features:**
- Applies dropout progressively during analysis
- Measures robustness of alignment metrics to dropout
- Useful for understanding representation stability

**Usage:**
```python
from alignment.experiments import ProgressiveDropoutExperiment

experiment = ProgressiveDropoutExperiment(
    model=model,
    metrics=['spectral_gap', 'eigenvalue_entropy'],
    dropout_rates=[0.0, 0.1, 0.3, 0.5, 0.7]
)
results = experiment.run(dataloader)
```

### 3. Layer Isolated Experiment (`layer_isolated.py`)
Analyzes each layer in isolation to understand layer-specific contributions.

**Key Features:**
- Isolates individual layers for analysis
- Computes layer-specific alignment metrics
- Helps identify critical layers for alignment

**Usage:**
```python
from alignment.experiments import LayerIsolatedExperiment

experiment = LayerIsolatedExperiment(
    model=model,
    metrics=['activation_cosine_similarity', 'node_correlation']
)
results = experiment.run(dataloader)
```

### 4. Eigenvector Experiment (`eigenvector.py`)
Focuses on eigenvector-based analysis of network representations.

**Key Features:**
- Computes and tracks eigenvector properties
- Analyzes eigenspace alignment across layers
- Studies spectral properties of weight matrices

**Usage:**
```python
from alignment.experiments import EigenvectorExperiment

experiment = EigenvectorExperiment(
    model=model,
    num_eigenvectors=10,
    track_evolution=True
)
results = experiment.run(dataloader)
```

## Base Experiment Class

All experiments inherit from `BaseExperiment` which provides:

- **Configuration management**: Handle experiment parameters
- **Metric computation**: Standardized metric calculation
- **Result tracking**: Automatic logging and saving
- **Visualization**: Built-in plotting capabilities
- **Distributed support**: Multi-GPU experiment execution

## Creating Custom Experiments

To create a new experiment:

```python
from alignment.experiments.base import BaseExperiment

class CustomExperiment(BaseExperiment):
    def __init__(self, model, metrics, **kwargs):
        super().__init__(model, metrics, **kwargs)
        # Custom initialization
    
    def setup(self):
        """Prepare experiment (called once)."""
        pass
    
    def run_iteration(self, batch_data):
        """Run single iteration (called per batch)."""
        # Implement experiment logic
        return results
    
    def aggregate_results(self, all_results):
        """Combine results from all iterations."""
        # Aggregate and return final results
        return aggregated_results
```

## Configuration Files

Experiments can be configured via YAML files:

```yaml
experiment:
  type: cascading
  metrics:
    - rayleigh_quotient
    - mutual_information_gaussian
  pruning:
    strategy: magnitude
    ratios: [0.1, 0.3, 0.5, 0.7, 0.9]
    structured: false
  tracking:
    log_interval: 10
    save_checkpoints: true
```

## Running Experiments

### Command Line
```bash
python -m alignment.experiments.runner --config experiments/configs/cascading.yaml
```

### Python API
```python
from alignment.experiments.runner import ExperimentRunner

runner = ExperimentRunner.from_config("experiments/configs/cascading.yaml")
results = runner.run()
```

## Results and Analysis

Experiment results include:
- **Metrics**: All computed alignment metrics
- **Metadata**: Experiment configuration and runtime info
- **Visualizations**: Automatically generated plots
- **Checkpoints**: Model states at different stages

Results are saved in structured format for easy analysis and comparison. 