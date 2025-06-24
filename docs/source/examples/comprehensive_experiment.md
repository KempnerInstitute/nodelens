# Comprehensive Experiment Guide

The comprehensive experiment system allows you to run any type of alignment analysis through a unified, configurable interface.

## Overview

The `comprehensive_alignment_experiment.py` script provides:
- Support for all models and datasets
- Access to 36+ alignment metrics
- Multiple pruning strategies
- Advanced training options
- Automatic analysis and visualization
- Full YAML configuration

## Basic Usage

### Quick Test

```bash
python examples/comprehensive_alignment_experiment.py --config configs/quick_test_config.yaml
```

This runs a minimal experiment on MNIST with ResNet18 in about 2-3 minutes.

### Full Experiment

```bash
python examples/comprehensive_alignment_experiment.py --config configs/comprehensive_alignment_config.yaml
```

This uses the comprehensive configuration with all options documented.

### Custom Configuration

```bash
# Override specific parameters
python examples/comprehensive_alignment_experiment.py \
    --config configs/comprehensive_alignment_config.yaml \
    --model_name resnet50 \
    --dataset_name cifar100 \
    --training_config.epochs 50
```

## Configuration Structure

The YAML configuration is organized into sections:

### 1. Experiment Metadata

```yaml
name: "my_experiment"
description: "Detailed description of the experiment"
tags: ["alignment", "pruning", "resnet"]
```

### 2. Model Configuration

```yaml
model_name: "resnet18"  # Any torchvision model
model_config:
  num_classes: 10
  pretrained: false
```

Supported models:
- MLP
- ResNet (18, 34, 50, 101, 152)
- VGG (16, 19)
- AlexNet
- DenseNet (121, 169, 201)
- EfficientNet (B0-B7)
- MobileNet (v2, v3)

### 3. Dataset Configuration

```yaml
dataset_name: "cifar10"
dataset_config:
  data_path: "./data"
  augmentation: true
  normalize: true
```

Supported datasets:
- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- ImageNet
- SVHN

### 4. Training Configuration

```yaml
training_config:
  epochs: 20
  batch_size: 128
  learning_rate: 0.001
  optimizer: "adam"  # sgd, adam, adamw, rmsprop
  scheduler: "cosine"  # none, step, cosine, exponential, plateau
  scheduler_params:
    T_max: 20
    eta_min: 0.0
```

### 5. Alignment Metrics

```yaml
alignment_metrics:
  # Rayleigh Quotient based
  - "rayleigh_quotient"
  - "delta_alignment"
  
  # Information-theoretic
  - "mutual_information_gaussian"
  - "average_redundancy"
  - "node_redundancy"
  
  # Similarity metrics
  - "weight_cosine_similarity"
  - "activation_cosine_similarity"
  
  # Spectral metrics
  - "spectral_gap"
  - "eigenvalue_alignment"

metric_configs:
  rayleigh_quotient:
    scale_by_norm: true
    aggregation_op: "mean"
```

See [Metrics Reference](../METRICS_REFERENCE.md) for all 36+ available metrics.

### 6. Pruning Configuration

```yaml
pruning_strategy: "magnitude"  # magnitude, gradient, random
pruning_config:
  amount: 0.5
  structured: false
  mode: "global"  # global, layer, block

# Optional: use metric to guide pruning
pruning_based_on_metric: "rayleigh_quotient"
```

### 7. Experiment Workflow

```yaml
# Control which steps to run
train_model: true
compute_initial_metrics: true
apply_pruning: true
fine_tune_after_pruning: true
fine_tune_epochs: 10
```

## Output Structure

Results are saved to `logs/<experiment_name>/`:

```
logs/
└── my_experiment/
    ├── my_experiment_20240315_143022.log    # Detailed execution log
    ├── results.json                         # Complete results
    ├── report.html                          # Interactive HTML report
    └── visualizations/
        ├── summary.png                      # Quick summary
        ├── rayleigh_quotient_comparison.png # Metric comparisons
        ├── sparsity_by_layer.png           # Pruning visualization
        └── alignment_heatmap.png           # Comprehensive heatmap
```

## Advanced Features

### Distributed Training

```yaml
distributed: true
world_size: 4
backend: "nccl"
```

### Memory Optimization

```yaml
force_cpu_for_large_metric_ops: true
mixed_precision_metrics: true
```

### Experimental Features

```yaml
experimental:
  pruning_warmup: 5
  gradual_pruning: true
  pruning_schedule: "cosine"
  compute_hessian: true
  track_gradient_flow: true
```

## Example Configurations

### Minimal Configuration

```yaml
name: "quick_test"
model_name: "resnet18"
dataset_name: "mnist"
training_config:
  epochs: 2
alignment_metrics:
  - "rayleigh_quotient"
```

### Research Configuration

```yaml
name: "alignment_research"
model_name: "resnet50"
dataset_name: "cifar100"

training_config:
  epochs: 100
  batch_size: 256
  learning_rate: 0.1
  optimizer: "sgd"
  scheduler: "cosine"

alignment_metrics:
  - "rayleigh_quotient"
  - "mutual_information_gaussian"
  - "spectral_gap"
  - "weight_cosine_similarity"

pruning_strategy: "magnitude"
pruning_config:
  amount: 0.9
  iterative: true
  iterations: 10
```

## Command-Line Interface

### Override Any Parameter

```bash
python comprehensive_alignment_experiment.py --config base.yaml \
    --name "custom_experiment" \
    --model_name "vgg16" \
    --training_config.epochs 50 \
    --training_config.learning_rate 0.01 \
    --pruning_config.amount 0.7
```

### Common Overrides

```bash
# Change device
--device cpu

# Skip training
--train_model false

# Change batch size
--training_config.batch_size 64

# Add metrics
--alignment_metrics "['rayleigh_quotient', 'spectral_gap']"
```

## Tips and Best Practices

1. **Start Small**: Use `quick_test_config.yaml` to verify setup
2. **Incremental Testing**: Test with few epochs before full runs
3. **Monitor Resources**: Watch GPU memory with large models
4. **Use Checkpointing**: Enable for long experiments
5. **Parallel Experiments**: Run multiple configs with different seeds

## Troubleshooting

### Out of Memory
- Reduce batch size
- Enable `force_cpu_for_large_metric_ops`
- Process metrics sequentially

### Slow Metric Computation
- Reduce number of metrics
- Use smaller validation set
- Enable mixed precision

### Configuration Errors
- Check YAML syntax
- Verify parameter names match documentation
- Use provided configs as templates

## Integration with Analysis

After experiments complete:

```python
from alignment.analysis import ResultAggregator, HTMLReporter

# Load results
aggregator = ResultAggregator()
aggregator.load_from_directory("logs/")

# Generate comparison report
reporter = HTMLReporter("Experiment Comparison")
reporter.add_dataframe("Results", aggregator.to_dataframe())
reporter.generate("comparison.html")
```

## Next Steps

1. Review the [comprehensive config](../../configs/comprehensive_alignment_config.yaml)
2. Explore available [metrics](../METRICS_REFERENCE.md)
3. Learn about [pruning strategies](../user_guide/pruning.md)
4. Check the [API reference](../api/experiments.rst) 