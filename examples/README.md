# Alignment Framework Examples

This directory contains example scripts demonstrating various features of the alignment framework.

## Examples Overview

### 1. `quick_demo.py` (127 lines)
A minimal example showing the basic workflow:
- Load a pre-trained model
- Compute alignment metrics
- Apply pruning
- Visualize results

**Usage:**
```bash
python quick_demo.py
```

### 2. `standard_alignment_experiment.py` (347 lines)
A complete workflow demonstrating:
- Training a model from scratch
- Computing multiple alignment metrics
- Applying pruning at different sparsity levels
- Generating comprehensive visualizations

**Usage:**
```bash
python standard_alignment_experiment.py
```

### 3. `pruning_strategies_demo.py` (416 lines)
Comprehensive demonstration of all pruning features:
- Multiple pruning strategies (magnitude, gradient, random)
- Different pruning modes (low, high, random)
- Parallel pruning execution
- Performance comparison

**Usage:**
```bash
python pruning_strategies_demo.py
```

### 4. `pruning_visualization_demo.py` (247 lines)
Advanced visualization capabilities:
- Weight distribution plots
- Sparsity patterns
- Layer-wise analysis
- Interactive visualizations

**Usage:**
```bash
python pruning_visualization_demo.py
```

### 5. `unified_experiment.py` (471 lines)
The main experiment script that can run any configuration specified in the master config file:
- Fully configurable via YAML
- Supports all model architectures
- All alignment metrics
- Multiple pruning strategies
- Advanced training options
- Comprehensive analysis and reporting

**Usage:**
```bash
# Run with master config (all options documented)
python unified_experiment.py --config ../configs/master_config.yaml

# Run quick test
python unified_experiment.py --config ../configs/simple_test.yaml

# Override parameters from command line
python unified_experiment.py --config ../configs/simple_test.yaml \
    --model_name resnet50 --dataset_name cifar10 --training_config.epochs 10

# Run without training (just compute metrics)
python unified_experiment.py --config ../configs/simple_test.yaml \
    --train_model false --apply_pruning false
```

### 6. `parallel_experiment_demo.py` (180 lines)
Demonstrates parallel experiment capabilities:
- Training multiple networks with different seeds
- Statistical analysis across multiple runs
- Parallel metric computation
- Variance analysis

**Usage:**
```bash
python parallel_experiment_demo.py
```

## Configuration Files

The `configs/` directory contains example configuration files:

- `master_config.yaml`: Complete configuration with ALL possible options documented
- `simple_test.yaml`: Simplified config for quick testing
- `quick_test_config.yaml`: Minimal configuration for demos
- Other configs: Various experiment configurations

## Key Features Demonstrated

1. **Model Support**: MLP, ResNet, VGG, AlexNet, DenseNet, EfficientNet, MobileNet
2. **Datasets**: MNIST, Fashion-MNIST, CIFAR-10/100, ImageNet, SVHN
3. **Metrics**: 36+ metrics including:
   - Rayleigh quotient variants
   - Information-theoretic metrics (MI, redundancy, PID)
   - Similarity metrics
   - Spectral metrics
   - Task-specific metrics
4. **Pruning**: Multiple strategies with various modes and configurations
5. **Training**: Optimizers, schedulers, mixed precision, distributed training
6. **Analysis**: Automatic visualizations, HTML reports, interactive dashboards

## Output Structure

Running examples creates the following structure:
```
logs/
├── experiment_name/
│   ├── experiment.log
│   ├── results.json
│   ├── report.html
│   └── visualizations/
│       ├── metric_comparison.png
│       ├── pruning_impact.png
│       ├── layer_metrics/
│       └── dashboard.html
checkpoints/
├── experiment_name_step_N.pt
results/
├── experiment_name/
│   └── various_outputs.json
```

## Tips

1. Start with `quick_demo.py` to understand the basic workflow
2. Use `standard_alignment_experiment.py` as a template for custom experiments
3. Explore `pruning_strategies_demo.py` for advanced pruning techniques
4. Use `unified_experiment.py` with custom configs for research
5. Run `parallel_experiment_demo.py` for statistical analysis across seeds

## Requirements

All examples require:
- PyTorch >= 2.0
- CUDA-capable GPU (recommended)
- Dependencies from `pyproject.toml`

Install with:
```bash
pip install -e .[all]
```

## Getting Started

1. **Ensure the alignment package is installed:**
   ```bash
   pip install -e .  # From repository root
   ```

2. **Activate your environment:**
   ```bash
   conda activate networkAlignmentAnalysis
   ```

3. **Run the quick demo:**
   ```bash
   python quick_demo.py
   ```

4. **Try the standard experiment:**
   ```bash
   python standard_alignment_experiment.py
   ```

## Example Outputs

### Alignment Metrics (from quick_demo.py)
```
Layer 0:
  RQ scores: mean=0.0013, std=0.0003
  Weight similarity: mean=0.0006

Layer 2:
  RQ scores: mean=0.0038, std=0.0010
  Weight similarity: mean=-0.0001
```

### Pruning Results (from standard_alignment_experiment.py)
```
Magnitude pruning:
  50% sparsity: 97.26% accuracy (drop: 0.39%)
  90% sparsity: 48.26% accuracy (drop: 49.39%)

Random pruning:
  50% sparsity: 79.88% accuracy (drop: 17.77%)
  90% sparsity: 14.85% accuracy (drop: 82.80%)
```

### Parallel Experiment Results (from parallel_experiment_demo.py)
```
Training 3 networks with different seeds...
Magnitude low mode (50% sparsity): 96.2% ± 0.8%
Random low mode (50% sparsity): 78.4% ± 2.1%
Statistical significance: p < 0.001
```

## Key Concepts Demonstrated

1. **Model Wrapping**: Track activations and weights automatically
2. **Metric Computation**: Rayleigh quotient, mutual information, weight similarity
3. **Pruning Strategies**: Magnitude, gradient, random with different modes
4. **Parallel Execution**: Compute multiple pruning variations efficiently
5. **Visualization**: Publication-ready plots and analysis grids
6. **Statistical Analysis**: Multi-seed experiments with confidence intervals
7. **Complete Workflow**: From training to analysis in one script

## Customization

Each example can be customized by modifying:
- Model architecture
- Dataset (MNIST, CIFAR, etc.)
- Metrics to compute
- Pruning strategies and sparsity levels
- Visualization styles

## Example Structure

- **Quick demos** (`quick_demo.py`): Simple, focused demonstrations
- **Complete experiments** (`standard_alignment_experiment.py`, `unified_experiment.py`): Full workflows
- **Feature demos** (`pruning_strategies_demo.py`, `pruning_visualization_demo.py`): Deep dives into specific features
- **Statistical analysis** (`parallel_experiment_demo.py`): Multi-seed experiments

## Troubleshooting

If you encounter import errors:
1. Ensure you're in the correct conda environment
2. Verify the package is installed: `pip show alignment`
3. Check Python path includes the src directory

For GPU/CUDA errors:
- Examples automatically fall back to CPU if CUDA is unavailable
- Set device explicitly: `device = torch.device('cpu')`

## Next Steps

After running these examples:
1. Modify `standard_alignment_experiment.py` for your own experiments
2. Create custom YAML configs for `unified_experiment.py`
3. Explore the API documentation for advanced features
4. Create custom metrics and pruning strategies
5. Use the experiment framework for systematic studies

For more information, see the main documentation at `docs/`. 