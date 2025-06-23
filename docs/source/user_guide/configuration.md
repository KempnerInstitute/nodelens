# Configuration Guide

This guide explains how to configure experiments and metrics in the alignment framework.

## Metric Configuration

### Basic Metric Usage

```python
from alignment.metrics import get_metric

# Create metric with default parameters
mi_metric = get_metric("mutual_information_gaussian")()

# Create metric with custom parameters
rq_metric = get_metric("rayleigh_quotient")(relative=True, normalize_covariance=True)
```

### Available Metric Parameters

Each metric has specific configuration options:

```python
# Rayleigh Quotient metrics
rq_config = {
    "relative": True,  # Use relative RQ formulation
    "normalize_covariance": True,  # Normalize by covariance
    "epsilon": 1e-5  # Regularization term
}

# Mutual Information metrics
mi_config = {
    "normalize": True,  # Normalize MI values
    "num_bins": 30,  # For binning-based MI
    "kernel_bandwidth": None  # For kernel density estimation
}

# PID metrics
pid_config = {
    "normalize": True,  # Normalize PID components
    "num_samples": 1000  # Number of samples for estimation
}
```

## Batch Processing Configuration

### BatchMetricProcessor Options

```python
from alignment.utils.batch_processing import BatchMetricProcessor

processor = BatchMetricProcessor(
    metrics=['rayleigh_quotient', 'mutual_information_gaussian'],
    metric_configs={
        'rayleigh_quotient': {'relative': True},
        'mutual_information_gaussian': {'normalize': True}
    },
    batch_size=256,
    use_gpu=True,
    num_workers=4,
    memory_limit_gb=8.0,
    accumulation_strategy='mean',  # 'mean', 'concat', or 'list'
    show_progress=True
)
```

### Streaming Configuration

For very large datasets:

```python
from alignment.utils.batch_processing import StreamingMetricComputer

streaming_computer = StreamingMetricComputer(
    metrics=['rayleigh_quotient'],
    buffer_size=10000,
    use_gpu=True
)

# Process data in streams
for batch in dataloader:
    streaming_computer.update(batch)

results = streaming_computer.compute()
```

## Pruning Configuration

### PruningConfig Options

```python
from alignment.utils.pruning import PruningConfig, PruningUtilities

config = PruningConfig(
    method='magnitude',  # 'magnitude', 'random', 'gradient', 'importance'
    amount=0.5,  # Fraction to prune
    structured=False,  # Structured vs unstructured pruning
    iterative=True,  # Iterative pruning
    iterations=10,  # Number of iterations
    recovery_epochs=5  # Epochs between pruning steps
)

pruner = PruningUtilities(
    importance_metric='rayleigh_quotient',
    config=config
)
```

### Pruning Schedules

```python
from alignment.utils.pruning import create_pruning_schedule

# Polynomial schedule
poly_schedule = create_pruning_schedule(
    initial_sparsity=0.0,
    final_sparsity=0.9,
    start_epoch=0,
    end_epoch=100,
    frequency=10,
    schedule_type='polynomial',
    power=3
)

# Linear schedule
linear_schedule = create_pruning_schedule(
    initial_sparsity=0.0,
    final_sparsity=0.9,
    start_epoch=0,
    end_epoch=100,
    frequency=5,
    schedule_type='linear'
)

# Custom schedule
def custom_schedule(epoch):
    return min(0.9, epoch * 0.01)
```

## Visualization Configuration

### AlignmentVisualizer Options

```python
from alignment.visualization import AlignmentVisualizer

visualizer = AlignmentVisualizer(
    style='seaborn',  # matplotlib style
    figsize=(10, 6),  # Default figure size
    dpi=100,  # Resolution
    save_format='png'  # 'png', 'pdf', 'svg'
)

# Configure specific plots
visualizer.plot_metric_distributions(
    results,
    bins=50,
    show_kde=True,
    show_stats=True,
    save_path='distributions.png'
)

visualizer.plot_layer_comparison(
    results,
    metric='rayleigh_quotient',
    show_variance=True,
    annotate=True,
    save_path='layer_comparison.png'
)
```

## Experiment Tracking Configuration

### Weights & Biases

```python
from alignment.utils.experiment_tracking import create_tracker

tracker = create_tracker(
    'wandb',
    project='alignment-analysis',
    config={
        'model': 'resnet18',
        'dataset': 'cifar10',
        'metrics': ['rq', 'mi'],
        'pruning_rate': 0.5
    },
    tags=['pruning', 'analysis'],
    name='experiment_001'
)
```

### TensorBoard

```python
tracker = create_tracker(
    'tensorboard',
    log_dir='./runs/experiment_001',
    comment='RQ analysis on ResNet18'
)
```

## Complete Configuration Example

Here's a complete example combining all configuration options:

```python
import torch
from alignment.core import ModelWrapper
from alignment.metrics import METRIC_REGISTRY
from alignment.utils.batch_processing import BatchMetricProcessor
from alignment.utils.pruning import PruningUtilities, PruningConfig
from alignment.visualization import AlignmentVisualizer
from alignment.utils.experiment_tracking import create_tracker

# Model configuration
model = create_your_model()
wrapped_model = ModelWrapper(model)

# Metric configuration
metric_configs = {
    'rayleigh_quotient': {
        'relative': True,
        'normalize_covariance': True
    },
    'mutual_information_gaussian': {
        'normalize': True
    },
    'spectral_gap': {
        'normalize': True
    }
}

# Batch processing configuration
processor = BatchMetricProcessor(
    metrics=list(metric_configs.keys()),
    metric_configs=metric_configs,
    batch_size=256,
    use_gpu=torch.cuda.is_available(),
    num_workers=4,
    show_progress=True
)

# Pruning configuration
pruning_config = PruningConfig(
    method='magnitude',
    amount=0.5,
    structured=False,
    iterative=True,
    iterations=10
)

pruner = PruningUtilities(
    importance_metric='rayleigh_quotient',
    config=pruning_config
)

# Visualization configuration
visualizer = AlignmentVisualizer(
    style='seaborn',
    figsize=(12, 8),
    dpi=150
)

# Experiment tracking
tracker = create_tracker(
    'wandb',
    project='alignment-study',
    config={
        'model_type': 'custom',
        'metrics': list(metric_configs.keys()),
        'pruning': pruning_config.__dict__
    }
)

# Run analysis
results = processor.process(model, dataloader)
tracker.log_metrics(results)

# Visualize
visualizer.create_report(results, 'analysis_report.html')

# Prune model
importance_scores = pruner.compute_importance(model, dataloader)
pruned_model = pruner.prune(model, importance_scores)
```

## Configuration Best Practices

1. **Start with defaults**: Most metrics have sensible defaults
2. **Use GPU when available**: Set `use_gpu=torch.cuda.is_available()`
3. **Monitor memory usage**: Set appropriate `memory_limit_gb` for batch processing
4. **Save configurations**: Store configs in JSON/YAML for reproducibility
5. **Use consistent naming**: Name experiments systematically for easy tracking

## Environment Variables

The framework respects several environment variables:

```bash
# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1"  # Use specific GPUs
export ALIGNMENT_GPU_MEMORY_FRACTION="0.8"  # Limit GPU memory usage

# Parallel processing
export ALIGNMENT_NUM_WORKERS="4"  # Default number of workers
export ALIGNMENT_BATCH_SIZE="256"  # Default batch size

# Paths
export ALIGNMENT_CACHE_DIR="~/.cache/alignment"  # Cache directory
export ALIGNMENT_LOG_LEVEL="INFO"  # Logging level
``` 