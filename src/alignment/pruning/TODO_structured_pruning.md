# TODO: Structured Pruning Implementation

## Overview
Structured pruning removes entire channels, filters, or neurons rather than individual weights. This maintains regular tensor structure for hardware efficiency.

## Planned Implementations

### 1. Channel Pruning
Remove entire output channels from convolutional layers.
```python
class ChannelPruning(BasePruningStrategy):
    # Prune entire output channels based on importance
```

### 2. Filter Pruning  
Remove entire convolutional filters.
```python
class FilterPruning(BasePruningStrategy):
    # Prune 3D filters from Conv layers
```

### 3. Block Pruning
Remove structured blocks of weights.
```python
class BlockPruning(BasePruningStrategy):
    # Prune n×m blocks for hardware efficiency
```

### 4. Pattern Pruning
Remove weights in specific patterns.
```python
class PatternPruning(BasePruningStrategy):
    # Prune in patterns (e.g., 2:4 sparsity)
```

### 5. N:M Pruning
Remove N weights out of every M weights.
```python
class NMPruning(BasePruningStrategy):
    # Implement N:M structured sparsity
```

## Implementation Notes

1. Create `structured/` subdirectory
2. Implement base `StructuredPruningStrategy` class
3. Add importance scoring methods specific to structured pruning
4. Ensure compatibility with existing pruning experiments
5. Add tests for each structured pruning method

## Benefits of Structured Pruning
- Hardware acceleration friendly
- Maintains regular memory access patterns
- Can achieve actual speedups (unlike unstructured)
- Better for deployment scenarios 