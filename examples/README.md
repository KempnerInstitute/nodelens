# Alignment Framework Examples

This folder contains working examples demonstrating various features of the alignment framework.

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

### 4. `direct_pruning_viz_demo.py` - Visualization Demo
Shows visualization capabilities without running full experiments:
- Performance comparison plots
- Confidence intervals
- Multi-seed statistical analysis
- Comprehensive comparison grids

**Run:** `python direct_pruning_viz_demo.py`

**Output:** Plots saved to `results/direct_viz_demo/`

### 5. `pruning_parallel_demo.py` - Parallel Pruning
Demonstrates parallel pruning features:
- Apply multiple pruning modes simultaneously
- Compare mask overlaps
- Efficiency analysis

**Run:** `python pruning_parallel_demo.py`

### 6. `simple_pruning_visualization_demo.py` - Simple Visualization
A simplified version showing:
- Basic pruning visualization
- Strategy comparison
- Different pruning modes

**Run:** `python simple_pruning_visualization_demo.py`

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

### Alignment Metrics
```
Layer 0:
  RQ scores: mean=0.0013, std=0.0003
  Weight similarity: mean=0.0006

Layer 2:
  RQ scores: mean=0.0038, std=0.0010
  Weight similarity: mean=-0.0001
```

### Pruning Results
```
Magnitude pruning:
  50% sparsity: 97.26% accuracy (drop: 0.39%)
  90% sparsity: 48.26% accuracy (drop: 49.39%)

Random pruning:
  50% sparsity: 79.88% accuracy (drop: 17.77%)
  90% sparsity: 14.85% accuracy (drop: 82.80%)
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