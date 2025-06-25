# Unified Alignment Experiment System

This unified system provides a single entry point for all neural network alignment experiments. It replaces the previous fragmented approach with a comprehensive, configuration-driven system.

## Overview

The unified experiment system can handle:
- **Any Dataset**: MNIST, CIFAR-10/100, ImageNet, SVHN, etc.
- **Any Model**: MLP, CNN, ResNet, VGG, AlexNet, custom architectures
- **Any Metric**: Rayleigh Quotient, Mutual Information, CKA, custom metrics
- **Any Pruning Strategy**: Magnitude, Gradient, Fisher, Random, Taylor expansion
- **Any Experiment Type**: Standard pruning, Progressive dropout, Layer-wise analysis, etc.

## Quick Start

```bash
# Run standard pruning on MNIST with MLP
python run_unified_experiment.py --config configs/unified_config.yaml

# Run with specific experiment type
python run_unified_experiment.py --config configs/examples/mnist_mlp_standard.yaml

# Override parameters from command line
python run_unified_experiment.py --config configs/unified_config.yaml \
    --experiment_type progressive_dropout \
    --device cuda:1 \
    --seed 123
```

## Configuration Structure

All experiments are controlled through YAML configuration files. The main sections are:

### 1. Experiment Metadata
```yaml
experiment_name: "my_experiment"
experiment_type: "standard_pruning"  # or "progressive_dropout", "layer_isolated_pruning", etc.
device: "cuda"
seed: 42
```

### 2. Dataset Configuration
```yaml
dataset:
  name: "mnist"
  data_path: "./data"
  batch_size: 128
  num_workers: 4
```

### 3. Model Configuration
```yaml
model:
  name: "mlp"  # or "cnn", "resnet18", etc.
  output_dim: 10
  mlp_config:  # Model-specific parameters
    input_dim: 784
    hidden_dims: [512, 256, 128]
    activation: "relu"
```

### 4. Training Configuration
```yaml
training:
  epochs: 10
  optimizer: "adam"
  learning_rate: 0.001
  scheduler: "cosine"
```

### 5. Pruning Configuration
```yaml
pruning:
  strategy: "magnitude"
  scope: "global"
  amount: 0.5
  fine_tune: true
  fine_tune_epochs: 5
```

## Experiment Types

### 1. Standard Pruning (`standard_pruning`)
Traditional weight pruning with various strategies:
- Train model → Apply pruning → Fine-tune → Evaluate

### 2. Progressive Dropout (`progressive_dropout`)
Analyze model behavior under increasing dropout rates:
- Tests multiple dropout strategies (high/low/random importance)
- Generates dropout vs. accuracy curves

### 3. Alignment Analysis (`alignment_analysis`)
Comprehensive analysis combining multiple experiments:
- Progressive dropout analysis
- Layer importance scoring
- Cross-layer alignment measurements

### 4. Layer-Isolated Pruning (`layer_isolated_pruning`)
Analyze each layer independently:
- Prune one layer at a time
- Measure impact on performance
- Restore weights between layers

### 5. Cascading Layer Pruning (`cascading_layer_pruning`)
Sequential layer pruning:
- Prune layers in order (forward/backward)
- Each layer's pruning affects subsequent layers

## Example Configurations

### Basic Examples
- `configs/unified_config.yaml` - Master configuration with all options
- `configs/examples/mnist_mlp_standard.yaml` - Standard pruning on MNIST
- `configs/examples/cifar10_resnet_progressive.yaml` - Progressive dropout on CIFAR-10
- `configs/examples/layer_isolated_analysis.yaml` - Layer-wise analysis

### Creating Custom Configurations
1. Copy an example configuration
2. Modify the relevant sections
3. Run with: `python run_unified_experiment.py --config your_config.yaml`

## Output Structure

```
results/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── config.yaml          # Configuration used
    ├── experiment.log       # Detailed logs
    ├── results.json         # Experiment results
    ├── checkpoints/         # Model checkpoints
    ├── plots/              # Visualization plots
    │   ├── dropout_accuracy.png
    │   ├── layer_importance.png
    │   └── ...
    └── report.html         # Comprehensive HTML report
```

## Visualization

The system automatically generates relevant visualizations based on experiment type:
- **Dropout vs. Accuracy curves**
- **Layer importance heatmaps**
- **Weight distribution histograms**
- **Training curves**
- **Alignment matrices**

## Advanced Usage

### Command Line Overrides
Override any configuration parameter:
```bash
python run_unified_experiment.py --config base.yaml \
    --dataset.name cifar10 \
    --model.name resnet18 \
    --training.epochs 50 \
    --pruning.amount 0.8
```

### Multi-GPU Support
Configure distributed training:
```yaml
resources:
  distributed: true
  backend: "nccl"
```

### Custom Metrics
Add custom alignment metrics:
```yaml
alignment:
  metrics:
    - "rayleigh_quotient"
    - "mutual_information"
    - "custom_metric"  # Implement in alignment/metrics/
```

## Migration from Old System

### Old Config Files
The following configs are deprecated and replaced by the unified system:
- Individual experiment configs (mnist_mlp_*.yaml, etc.)
- Separate pruning configs
- Multiple experiment-specific configs

### Old Scripts
Replace usage of:
- `examples/unified_experiment.py` → `run_unified_experiment.py`
- Multiple experiment scripts → Single unified runner

### Key Differences
1. **Single entry point** instead of multiple scripts
2. **Unified configuration** structure
3. **Consistent output** format across all experiments
4. **Automatic visualization** generation
5. **Comprehensive reporting** with HTML output

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use CPU for metric computation

2. **Dataset not found**
   - Check data_path in config
   - Enable download: true

3. **Model not supported**
   - Check model name spelling
   - Implement custom model in alignment/models/

### Debug Mode
Enable detailed logging:
```yaml
logging:
  level: "DEBUG"
  console: true
```

## Contributing

To add new features:

1. **New Dataset**: Add to `alignment/data/datasets/`
2. **New Model**: Add to `alignment/models/architectures/`
3. **New Metric**: Add to `alignment/metrics/`
4. **New Experiment Type**: Extend `run_unified_experiment.py`

## Citation

If you use this unified experiment system, please cite:
```bibtex
@software{unified_alignment_2024,
  title = {Unified Neural Network Alignment Experiment System},
  year = {2024},
  url = {https://github.com/yourusername/alignment}
}
``` 