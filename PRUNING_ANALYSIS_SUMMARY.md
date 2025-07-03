# Pruning Analysis Summary

## Issues Identified

### 1. Model Size
Your current model is extremely small:
- **Architecture**: MLP with hidden sizes [16, 8]
- **Total parameters**: 12,786
- **Layer breakdown**:
  - Layer 1: 784 → 16 (12,544 params - 98% of total!)
  - Layer 2: 16 → 8 (128 params - 1%)
  - Layer 3: 8 → 10 (80 params - 0.6%)

This causes several problems:
- The first layer dominates the parameter count
- Even small pruning amounts have large effects
- The network is too small to be representative of real neural networks

### 2. Sparsity Level Mismatch
The actual sparsity levels don't match your configured levels because:
- With such a small network, discrete pruning decisions cause larger jumps
- The alignment-based pruning may not find enough neurons with similar scores
- Rounding effects are more pronounced with fewer parameters

### 3. Performance Degradation
The network quickly degrades to random performance (11.24% on MNIST = random) because:
- The network has very limited capacity to begin with
- Pruning even a few neurons removes critical pathways
- The first layer (784→16) is already a severe bottleneck

## What's Being Pruned

With alignment-based pruning:
1. **Rayleigh Quotient** is computed for each neuron
2. This measures how well-aligned the neuron's weights are with the input covariance
3. Selection modes:
   - `low`: Prunes neurons with low alignment (poorly aligned with inputs)
   - `high`: Prunes neurons with high alignment (well aligned with inputs)
   - `random`: Random selection for comparison

## Recommendations

### 1. Use a Realistic Model
```yaml
model:
  name: "mlp"
  hidden_sizes: [512, 256, 128]  # or [1024, 512, 256]
```

### 2. Test Fewer Sparsity Levels Initially
```yaml
sparsity_levels: [0.0, 0.3, 0.5, 0.7, 0.9]
```

### 3. Consider Structured Pruning
Since alignment metrics compute one score per neuron, structured pruning makes more sense:
```yaml
structured: true  # Prunes entire neurons instead of individual weights
```

### 4. Compare Different Strategies
```yaml
algorithms: ["alignment", "magnitude", "random"]
```

### 5. Run the Realistic Test
```bash
python scripts/run_experiment.py --config configs/test_realistic_pruning.yaml
```

## Expected Results with Proper Setup

With a reasonably-sized network, you should see:
- Gradual performance degradation as sparsity increases
- Different behaviors for low vs high alignment pruning
- Magnitude pruning likely performing well
- Alignment-based pruning potentially finding different important neurons

## Understanding Alignment Pruning

The key insight is that alignment-based pruning identifies neurons based on their relationship with inputs, not just weight magnitude. This can potentially:
- Find neurons that are important for the task despite small weights
- Remove neurons that have large weights but poor alignment
- Provide a different perspective on network importance 