# Network Alignment Analysis Documentation

## Introduction

This repository contains tools and methodologies for understanding the structure of neural networks through alignment analysis. It provides a framework for conducting alignment-related experiments, measuring various network properties, and analyzing the results.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Codebase Structure](#codebase-structure)
4. [Core Capabilities](#core-capabilities)
5. [Metrics System](#metrics-system)
6. [Experiment Framework](#experiment-framework)
7. [Performance Optimizations](#performance-optimizations)
8. [API Reference](#api-reference)
9. [Examples](#examples)

## Overview

The Network Alignment Analysis toolkit is designed to help researchers and practitioners analyze neural networks by measuring alignment between weight matrices and input activations. This approach provides insights into network structure, training dynamics, and pruning strategies.

Key capabilities include:
- Measuring various alignment metrics between neural networks
- Conducting pruning experiments with different strategies
- Training multiple networks in parallel with optimized implementations
- Analyzing network properties during training and evaluation

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd alignment

# Create and activate the environment
mamba env create -f environment.yml
mamba activate networkAlignmentAnalysis

# Install the package
pip install -e .[all]
```

### Basic Usage

To run a simple experiment with default settings:

```bash
python experiment.py configs/config_alignment_stats.yaml
```

For more detailed usage instructions, see the [Usage Guide](usage).

## Codebase Structure

The codebase is organized into the following directories:

- `src/alignment/`: Core source code implementing alignment metrics and algorithms
  - `metrics.py`: Implementation of all alignment metrics
  - `utils/`: Utility functions for data handling, visualization, etc.
  - `experiment/`: Experiment framework for running alignment experiments
  - `networks/`: Network architectures and training utilities
  - `pruning/`: Implementation of various pruning strategies

- `tests/`: Unit and integration tests
- `scripts/`: Utility scripts for running experiments and analysis
- `benchmarks/`: Performance evaluation scripts
- `configs/`: Configuration files for experiments
- `results/`: Output directory for experiment results
- `doc/`: Documentation

## Core Capabilities

### Neural Network Training

The framework supports training multiple neural networks in parallel, with several optimized implementations:

- Sequential training: Trains each network independently
- Tensorized training: Trains networks in parallel using batched operations
- Fully-tensorized training: Most efficient method that combines networks into a single model ensemble

### Progressive Dropout

The system implements progressive dropout techniques to analyze network alignment:

- Multiple pruning strategies (high RQ, low RQ, random)
- Optimized multi-strategy implementation for performance
- Analysis tools for evaluating pruning impact

### Metrics Calculation

A comprehensive metrics system for measuring various network properties:

- Rayleigh Quotient (RQ) metrics
- Mutual Information (MI) metrics
- Redundancy metrics
- Partial Information Decomposition (PID) metrics
- Weight similarity metrics

## Metrics System

The alignment metrics system measures various properties of neural networks:

### Available Metrics

- **Rayleigh Quotient (RQ) Metrics**
  - `rayleigh_quotient`/`rq`: Standard Rayleigh Quotient calculation
  - `rq_alt_denom`: Alternative RQ calculation with different denominator

- **Mutual Information (MI) Metrics**
  - `mi_gaussian`/`mi_g`: MI using Gaussian approximation
  - `mi_direct`/`mi_bin`: MI using direct binning approach
  - `mi_proj_vs_mean_input`: MI between neuron's projected input and mean input

- **Redundancy Metrics**
  - `redundancy_gaussian`/`red_g`: Redundancy between neurons using Gaussian approximation
  - `node_redundancy`: Redundancy between input features based on correlation

- **Partial Information Decomposition (PID) Metrics**
  - `pid_shared_info`/`pid_si`: Shared information component
  - `pid_unique_info_neuron`/`pid_uiy`/`pid_ui1`: Unique information in neuron 1
  - `pid_unique_info_other`/`pid_uiz`/`pid_ui2`: Unique information in neuron 2
  - `pid_synergy_info`/`pid_ci`: Synergistic information component

- **Weight Similarity Metrics**
  - `weight_cosine_similarity`: Cosine similarity between weight vectors
  - `weight_dot_similarity`: Dot product similarity between weight vectors
  - `weight_euclidean_distance`: Euclidean distance between weight vectors

For more details, see the [Metrics Documentation](metrics/README).

## Experiment Framework

The experiment framework provides tools for setting up, running, and analyzing alignment experiments:

- Configuration-based experiment setup
- Automated logging and result tracking
- Integration with Weights & Biases for experiment monitoring
- Distributed training support for HPC environments

See the [Experiment Documentation](experiment/README) for details.

## Performance Optimizations

The codebase includes several optimizations for efficient experimentation:

### Tensorized Training

- **Fully Tensorized Training**: Combines networks into a single model ensemble for up to 3x faster training

### Multi-Strategy Dropout

- **Parallelized Strategy Evaluation**: Process all pruning strategies simultaneously for 2.5-3x speedup

See the [Performance Documentation](performance/README) for benchmarks and usage details.

## API Reference

### Core Modules

- `alignment.metrics`: Metrics calculation and management
- `alignment.experiment`: Experiment framework
- `alignment.networks`: Network architecture definitions
- `alignment.pruning`: Pruning strategy implementations
- `alignment.utils`: Utility functions

### Key Functions

#### Metrics API

```python
from alignment.metrics import get_metric, compute_all_node_scores

# Get a specific metric
metric = get_metric("rq", scale_by_norm=True)
scores = metric.compute_per_node_scores(layer_inputs, layer_weights)

# Compute multiple metrics for a whole network
scores = compute_all_node_scores(
    model=model,
    metric_configs=[{"name": "rq", "scale_by_norm": True}, {"name": "mi_gaussian"}],
    device="cuda",
    data_loader=train_loader,
    num_batches=5
)
```

#### Experiment API

```python
from alignment.experiment import ExperimentRunner

# Create and run an experiment
runner = ExperimentRunner(config_path="configs/my_experiment.yaml")
results = runner.run()
```

#### Pruning API

```python
from alignment.pruning import ProgressiveDropout

# Create a progressive dropout controller
dropout = ProgressiveDropout(
    model=model,
    strategy="high_rq",
    use_multi_strategy=True
)

# Apply progressive dropout
results = dropout.run_dropout_experiment(data_loader, steps=10)
```

## Examples

Example scripts demonstrating key functionality:

- [Basic Alignment Analysis](examples/basic_alignment)
- [Progressive Dropout Experiment](examples/progressive_dropout)
- [Multi-Network Training](examples/multi_network_training)

For more examples, see the `scripts/` directory. 