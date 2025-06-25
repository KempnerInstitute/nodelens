# Pruning Code Organization

This document explains the organization of pruning code and the distinction between strategies and experiments.

## Directory Structure

```
pruning/
├── strategies/           # Pruning algorithms
│   ├── alignment_based.py    # Basic alignment pruning + global variant
│   ├── cascading.py         # Cascading alignment pruning
│   ├── magnitude.py         # Magnitude-based pruning
│   ├── gradient.py          # Gradient-based pruning
│   └── ...
└── experiments/          # Full experimental workflows
    ├── cascading_layer.py   # Cascading layer analysis experiment
    ├── layer_wise.py        # Layer-isolated analysis experiment
    └── ...
```

## Key Distinction: Strategies vs Experiments

### Pruning Strategies (`pruning/strategies/`)
- **What**: Core pruning algorithms
- **Purpose**: Implement HOW to prune (the algorithm)
- **Scope**: Just the pruning operation
- **Example**: `AlignmentPruning`, `CascadingAlignmentPruning`

### Pruning Experiments (`pruning/experiments/`)
- **What**: Complete experimental workflows
- **Purpose**: Organize pruning studies and analysis
- **Scope**: Training, pruning, evaluation, visualization
- **Example**: `CascadingLayerPruningExperiment`, `LayerIsolatedPruningExperiment`

## Alignment-Based Strategies

### 1. Standard Alignment Pruning (`AlignmentPruning`)
- Computes alignment scores once
- Prunes each layer to target sparsity
- Layer-wise by default

### 2. Global Alignment Pruning (`GlobalAlignmentPruning`)
- Collects scores from all layers
- Sorts globally
- Prunes worst neurons across entire network

### 3. Cascading Alignment Pruning (`CascadingAlignmentPruning`)
- Prunes layer 1
- Recomputes scores for layer 2 (with layer 1 pruned)
- Continues sequentially
- Accounts for pruning effects on subsequent layers

## When to Use What

### Use Strategies When:
- You want to apply a specific pruning algorithm
- You're integrating pruning into your own workflow
- You need fine control over the pruning process

```python
from alignment.pruning.strategies import CascadingAlignmentPruning

strategy = CascadingAlignmentPruning(
    metric='rayleigh_quotient',
    direction='forward'
)
masks = strategy.prune_model(model, get_inputs_fn)
```

### Use Experiments When:
- You want a complete analysis workflow
- You need training, evaluation, and visualization
- You're studying pruning effects systematically

```python
from alignment.pruning.experiments import CascadingLayerPruningExperiment

experiment = CascadingLayerPruningExperiment(config)
results = experiment.run()  # Full workflow
```

## Configuration Mapping

| Strategy | Scope | Config Parameters |
|----------|-------|-------------------|
| `AlignmentPruning` | `layer` | `algorithms: ["alignment"]`, `scope: "layer"` |
| `GlobalAlignmentPruning` | `global` | `algorithms: ["alignment"]`, `scope: "global"` |
| `CascadingAlignmentPruning` | `cascading` | `algorithms: ["alignment"]`, `scope: "cascading"` |

Legacy support:
- `algorithms: ["cascading_alignment"]` → Redirects to `algorithms: ["alignment"]` with `scope: "cascading"`

## Should We Reorganize Further?

### Current `alignment_based.py` Contains:
1. `AlignmentPruning` - Basic strategy
2. `HybridPruning` - Magnitude + alignment
3. `GlobalAlignmentPruning` - Global variant

### Options:
1. **Keep as is** - Related strategies in one file
2. **Split by variant**:
   - `alignment.py` - Basic AlignmentPruning
   - `global_alignment.py` - GlobalAlignmentPruning
   - `hybrid.py` - HybridPruning
   - `cascading.py` - CascadingAlignmentPruning (already done)

### Recommendation:
Keep current organization. The file is manageable and groups related concepts. The cascading strategy is different enough to warrant its own file.

## Future Enhancements

1. **Cascading for other algorithms**: Extend cascading scope to magnitude, gradient pruning
2. **Adaptive Cascading**: Adjust pruning amount based on layer importance
3. **Multi-metric Cascading**: Use different metrics for different layers 