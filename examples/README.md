# Alignment Framework Examples

This directory contains example scripts demonstrating various features of the alignment framework.

## Examples Overview

### 1. `quick_demo.py` (107 lines)
A minimal example showing the basic workflow:
- Load a pre-trained model
- Compute alignment metrics
- Apply pruning
- Visualize results

**Usage:**
```bash
python quick_demo.py
```

### 2. `standard_alignment_experiment.py` (321 lines)
A complete workflow demonstrating:
- Training a model from scratch
- Computing multiple alignment metrics
- Applying pruning at different sparsity levels
- Generating comprehensive visualizations

**Usage:**
```bash
python standard_alignment_experiment.py
```

### 3. `pruning_strategies_demo.py` (388 lines)
Comprehensive demonstration of all pruning features:
- Multiple pruning strategies (magnitude, gradient, random)
- Different pruning modes (low, high, random)
- Parallel pruning execution
- Performance comparison

**Usage:**
```bash
python pruning_strategies_demo.py
```

### 4. `pruning_visualization_demo.py` (218 lines)
Advanced visualization capabilities:
- Weight distribution plots
- Sparsity patterns
- Layer-wise analysis
- Interactive visualizations

**Usage:**
```bash
python pruning_visualization_demo.py
```

### 5. `comprehensive_alignment_experiment.py` (NEW - 450+ lines)
The most complete example demonstrating ALL framework features:
- Fully configurable via YAML
- All 36+ alignment metrics
- Multiple model architectures
- Various datasets
- Advanced training options
- Comprehensive analysis and reporting

**Usage:**
```bash
# Run with comprehensive config (all options documented)
python comprehensive_alignment_experiment.py --config ../configs/comprehensive_alignment_config.yaml

# Run quick test
python comprehensive_alignment_experiment.py --config ../configs/quick_test_config.yaml

# Override parameters from command line
python comprehensive_alignment_experiment.py --config ../configs/quick_test_config.yaml \
    --model_name resnet50 --dataset_name cifar10 --training_config.epochs 10

# Run without training (just compute metrics)
python comprehensive_alignment_experiment.py --config ../configs/quick_test_config.yaml \
    --train_model false --compute_initial_metrics true
```

## Configuration Files

The `configs/` directory contains example configuration files:

- `comprehensive_alignment_config.yaml`: Complete configuration with ALL possible options documented
- `quick_test_config.yaml`: Simplified config for quick testing
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
│   ├── experiment_name_TIMESTAMP.log
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
2. Use `standard_alignment_experiment.py` as a template for your experiments
3. Explore `pruning_strategies_demo.py` for advanced pruning techniques
4. Use `comprehensive_alignment_experiment.py` with custom configs for research

## Requirements

All examples require:
- PyTorch >= 2.0
- CUDA-capable GPU (recommended)
- Dependencies from `pyproject.toml`

Install with:
```bash
pip install -e .[all]
```

## Available Examples

### 1. `quick_demo.py` - Quick Start Demo
A simple introduction to the framework showing:
- Model wrapping and activation tracking
- Computing alignment metrics (Rayleigh Quotient, Weight Similarity)
- Basic pruning with magnitude strategy
- ~100 lines, runs in seconds

**Run:** `python quick_demo.py`

### 2. `standard_alignment_experiment.py` - Complete Workflow
A comprehensive experiment template that:
- Trains a neural network on MNIST
- Computes alignment metrics before pruning
- Applies different pruning strategies (magnitude, random)
- Evaluates performance at various sparsity levels
- Generates publication-ready visualizations
- Saves all results as JSON files

**Run:** `python standard_alignment_experiment.py`

**Output files in `results/standard_experiment/`:**
- `training_history.json` - Training metrics
- `alignment_metrics.json` - Layer-wise alignment scores
- `pruning_results.json` - Performance at different sparsities
- `pruning_performance.png` - Accuracy/loss curves
- `comparison_grid.png` - Comprehensive 6-panel analysis

### 3. `pruning_strategies_demo.py` - Pruning Strategies
Demonstrates all pruning capabilities:
- Different pruning modes (low, high, random)
- Parallel pruning execution
- GPU-optimized tensorized operations
- Gradient-based pruning
- Performance comparisons

**Run:** `python pruning_strategies_demo.py`

### 4. `pruning_visualization_demo.py` - Visualization Demo
Shows visualization capabilities with both simulated and real pruning:
- Performance comparison plots
- Confidence intervals
- Multi-seed statistical analysis
- Comprehensive comparison grids
- Real pruning demonstration with statistics

**Run:** `python pruning_visualization_demo.py`

**Output:** Plots saved to `results/pruning_visualization/`

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

### Real Pruning Analysis (from pruning_visualization_demo.py)
```
Low mode: Kept weights 3.0x larger than pruned (0.0569 vs 0.0191)
High mode: Kept weights 3.0x smaller than pruned (0.0191 vs 0.0569)
Random mode: Balanced distribution (0.0380 for both)
```

## Key Concepts Demonstrated

1. **Model Wrapping**: Track activations and weights automatically
2. **Metric Computation**: Rayleigh quotient, mutual information, weight similarity
3. **Pruning Strategies**: Magnitude, gradient, random with different modes
4. **Parallel Execution**: Compute multiple pruning variations efficiently
5. **Visualization**: Publication-ready plots and analysis grids
6. **Complete Workflow**: From training to analysis in one script

## Customization

Each example can be customized by modifying:
- Model architecture
- Dataset (MNIST, CIFAR, etc.)
- Metrics to compute
- Pruning strategies and sparsity levels
- Visualization styles

## Example Structure

- **Quick demos** (`quick_demo.py`): Simple, focused demonstrations
- **Complete experiments** (`standard_alignment_experiment.py`): Full workflows with all components
- **Feature demos** (`pruning_strategies_demo.py`, `pruning_visualization_demo.py`): Deep dives into specific features

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
2. Explore the API documentation for advanced features
3. Create custom metrics and pruning strategies
4. Use the experiment framework for systematic studies

For more information, see the main documentation at `docs/`. 