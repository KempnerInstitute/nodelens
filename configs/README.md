# Configuration Files

This directory contains YAML configuration files for running alignment experiments. The configurations are designed to work with the unified experiment runner.

## Usage

```bash
# From repository root
python scripts/run_experiment.py --config configs/example_quick_test.yaml
```

## Available Configurations

### 1. `unified_config.yaml`
Comprehensive configuration with all possible options documented. Use this as a reference for creating your own configs.

```bash
python scripts/run_experiment.py --config configs/unified_config.yaml
```

### 2. Example Configurations

Located in `configs/examples/`:

- **`quick_test.yaml`**: Minimal config for quick testing
- **`mnist_mlp.yaml`**: Simple MLP on MNIST
- **`cifar10_resnet.yaml`**: ResNet on CIFAR-10
- **`alignment_comparison.yaml`**: Compare multiple alignment metrics
- **`pruning_comparison.yaml`**: Compare pruning strategies
- **`cascading_alignment_pruning.yaml`**: Cascading scope example

## Creating Custom Configurations

Start with `unified_config.yaml` as a template. You can override any setting:

```yaml
experiment_name: "my_custom_experiment"
experiment_type: "standard_pruning"

model:
  name: "resnet50"
  pretrained: true

dataset:
  name: "cifar10"
  
pruning:
  algorithms: ["magnitude", "alignment"]
  sparsity_levels: [0.3, 0.5, 0.7]
```

## Configuration Priority

Settings are applied in this order (later overrides earlier):
1. Default values in code
2. Configuration file
3. Command-line arguments

Example:
```bash
python scripts/run_experiment.py --config configs/unified_config.yaml --device cpu
```

## Key Configuration Sections

- **experiment_type**: Type of experiment to run
- **model**: Model architecture and settings
- **dataset**: Dataset and data loading settings
- **training**: Training hyperparameters
- **pruning**: Pruning strategies and settings
- **alignment**: Alignment metrics to compute
- **visualization**: Plotting and reporting options

## Master Configuration

- **`master_config.yaml`**: The complete configuration file with ALL available parameters documented. Use this as a reference for all possible options.

## Example Configurations

### Quick Testing
- **`test_comprehensive.yaml`**: Basic test configuration for quick experiments
- **`example_quick_test.yaml`**: Minimal configuration for fast testing and debugging

### Standard Experiments
- **`example_resnet_cifar.yaml`**: Standard ResNet experiments on CIFAR datasets
- **`example_pruning_comparison.yaml`**: Configuration for comparing different pruning strategies

## Usage

### Basic Usage
```bash
# Run with a configuration file
python scripts/run_experiment.py --config configs/example_quick_test.yaml
```

### Override Parameters
You can override any parameter from the command line:

```bash
# Change model
python scripts/run_experiment.py --config configs/master_config.yaml --model_name resnet50

# Change dataset
python scripts/run_experiment.py --config configs/master_config.yaml --dataset_name cifar100

# Change training parameters
python scripts/run_experiment.py --config configs/master_config.yaml \
    --training_config.epochs 300 \
    --training_config.batch_size 256 \
    --training_config.learning_rate 0.01

# Change pruning strategy
python scripts/run_experiment.py --config configs/master_config.yaml \
    --pruning_strategy gradient \
    --pruning_config.amount 0.7
```

### Running Multiple Experiments
To compare different configurations:

```bash
# Compare pruning strategies
for strategy in magnitude gradient random; do
    python scripts/run_experiment.py \
        --config configs/example_pruning_comparison.yaml \
        --pruning_strategy $strategy \
        --name pruning_comparison_$strategy
done

# Compare pruning amounts
for amount in 0.3 0.5 0.7 0.9; do
    python scripts/run_experiment.py \
        --config configs/example_resnet_cifar.yaml \
        --pruning_config.amount $amount \
        --name resnet_pruning_$amount
done
```

## Configuration Structure

### Essential Sections

1. **Metadata**
   - `name`: Experiment name
   - `description`: Experiment description
   - `seed`: Random seed for reproducibility

2. **Model Configuration**
   - `model_name`: Architecture (resnet18, vgg16, mlp, etc.)
   - `model_config`: Model-specific parameters

3. **Dataset Configuration**
   - `dataset_name`: Dataset (mnist, cifar10, cifar100, etc.)
   - `dataset_config`: Dataset-specific parameters including augmentation

4. **Training Configuration**
   - `training_config`: All training hyperparameters
   - Includes optimizer, scheduler, loss function settings

5. **Metrics Configuration**
   - `alignment_metrics`: List of metrics to compute
   - `metric_configs`: Metric-specific parameters

6. **Pruning Configuration**
   - `pruning_strategy`: Which pruning method to use
   - `pruning_config`: Pruning-specific parameters

7. **Analysis Configuration**
   - `analysis_config`: Visualization and reporting options

8. **Workflow Control**
   - Boolean flags to control which parts of the experiment to run

## Best Practices

1. **Version Control**: Keep configurations in version control
2. **Naming**: Use descriptive names for experiments
3. **Documentation**: Add comments explaining non-obvious choices
4. **Validation**: Test configurations with small epochs first
5. **Reproducibility**: Always set the seed for reproducible results

## Available Models

- **CNNs**: resnet18, resnet34, resnet50, resnet101, resnet152
- **VGGs**: vgg11, vgg13, vgg16, vgg19
- **DenseNets**: densenet121, densenet161, densenet169, densenet201
- **MobileNets**: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- **EfficientNets**: efficientnet_b0 through efficientnet_b7
- **Vision Transformers**: vit_b_16, vit_b_32, vit_l_16, vit_l_32
- **Swin Transformers**: swin_t, swin_s, swin_b, swin_l
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base, convnext_large
- **Custom**: mlp, cnn, transformer

## Available Datasets

- **Vision**: mnist, fashion_mnist, cifar10, cifar100, imagenet, tiny_imagenet, svhn, stl10
- **Custom**: Implement custom dataset loaders

## Available Pruning Strategies

- **Magnitude-based**: magnitude, iterative_magnitude, global_magnitude
- **Gradient-based**: gradient, fisher, momentum
- **Random**: random, bernoulli
- **Parallel**: parallel_mode, tensorized, async_parallel

## Available Metrics

- rayleigh_quotient
- mutual_information
- canonical_correlation
- spectral_gap
- weight_similarity
- activation_similarity
- gradient_similarity
- representation_similarity 