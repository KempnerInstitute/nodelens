# Unified Alignment Experiment System

This unified system provides a single entry point for all neural network alignment experiments. It replaces the previous fragmented approach with a comprehensive, configuration-driven system.

**Note**: The main experiment runner `run_experiment.py` is kept in the root directory as the primary entry point to the alignment framework. All configuration files are in `configs/`, examples in `examples/`, and the core library code in `src/alignment/`.

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
python run_experiment.py --config configs/unified_config.yaml

# Run with specific experiment type
python run_experiment.py --config configs/examples/mnist_mlp_standard.yaml

# Override parameters from command line
python run_experiment.py --config configs/unified_config.yaml \
    --experiment_type progressive_dropout \
    --device cuda:1 \
    --seed 123
```

## Alignment-Based Pruning

The unified system now fully supports alignment-based pruning, which uses neuron-input alignment metrics to guide pruning decisions.

### Key Features

1. **Multiple Alignment Metrics**:
   - `rayleigh_quotient`: Classic neuron-input alignment based on variance
   - `mutual_information_gaussian`: Information shared between neurons and inputs
   - `weight_cosine_similarity`: Cosine similarity between weight vectors
   - `gradient_similarity`: Alignment based on gradient information
   - `cka`: Centered Kernel Alignment

2. **Structured Pruning by Default**: 
   - Alignment metrics compute scores per neuron, making them naturally suited for structured pruning
   - Removes entire neurons/channels instead of individual weights
   - More hardware-efficient than unstructured pruning

3. **Hybrid Pruning**:
   - Combines magnitude and alignment scores
   - Configurable weighting with `hybrid_alpha` parameter

**Note on Pruning Scope**: The `scope` parameter that appeared in some configs has been removed as it was never implemented. Currently, all pruning is done per-layer. Use `structured: true` to prune entire neurons instead of individual weights.

### Configuration Example

```yaml
pruning:
  # Compare different approaches
  algorithms: ["magnitude", "alignment", "hybrid"]
  
  # Specify alignment metric
  alignment_metric: "rayleigh_quotient"
  
  # For hybrid: 70% alignment, 30% magnitude
  hybrid_alpha: 0.7
  
  # Structured pruning (default for alignment)
  structured: true
  
  # Test different selection modes
  selection_mode: ["low", "high"]
  
  # Multiple sparsity levels
  sparsity_levels: [0.3, 0.5, 0.7]
```

### Running Alignment Pruning

```bash
# Basic alignment pruning
python run_experiment.py --config configs/unified_config.yaml \
    --pruning.algorithms alignment \
    --pruning.alignment_metric rayleigh_quotient

# Compare multiple metrics
python run_experiment.py --config configs/examples/mnist_alignment_pruning.yaml

# Hybrid approach with custom weighting
python run_experiment.py --config configs/unified_config.yaml \
    --pruning.algorithms hybrid \
    --pruning.hybrid_alpha 0.8
```

See `PRUNING_CONCEPTS.md` for detailed explanations of structured vs unstructured pruning.

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

### 5. Pruning Strategies

The system supports multiple pruning algorithms via the `algorithms` parameter:

- **magnitude**: Traditional weight magnitude pruning
- **gradient**: Gradient-based importance
- **fisher**: Fisher information approximation
- **alignment**: Neuron-input alignment using specified metric
- **hybrid**: Combine magnitude and alignment scores

### 6. Pruning Scope

The `scope` parameter controls how pruning is applied across layers:

- **layer** (default): Each layer pruned independently to target sparsity
- **global**: Pool scores from all layers and prune globally
- **cascading**: Prune layers sequentially, recomputing scores after each (alignment only)

Example with cascading scope:
```yaml
pruning:
  algorithms: ["alignment"]
  scope: "cascading"
  cascading_direction: "forward"  # or "backward"
  alignment_metric: "rayleigh_quotient"
  sparsity_levels: [0.3, 0.5, 0.7]
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
3. Run with: `python run_experiment.py --config your_config.yaml`

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
python run_experiment.py --config base.yaml \
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
- `examples/unified_experiment.py` → `run_experiment.py`
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
4. **New Experiment Type**: Extend `run_experiment.py`

## Citation

If you use this unified experiment system, please cite:
```bibtex
@software{unified_alignment_2024,
  title = {Unified Neural Network Alignment Experiment System},
  year = {2024},
  url = {https://github.com/yourusername/alignment}
}
``` 