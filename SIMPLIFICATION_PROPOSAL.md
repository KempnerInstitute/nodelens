# Codebase Simplification Proposal

## Overview

After analyzing the codebase, I've identified several areas where we can simplify and consolidate duplicate functionality. Here are the main issues and proposed solutions:

## 1. Multi-Network Support Integration

**Issue**: Separate `ParallelPruningExperiment` class duplicates functionality from `GeneralAlignmentExperiment`

**Solution**: 
- Integrate multi-network support directly into `general_alignment.py` 
- Add `num_networks` parameter (default=1) to `GeneralAlignmentConfig`
- Use conditional logic to handle single vs. multi-network cases
- Delete `parallel_pruning_experiment.py` after integration

**Benefits**:
- Single unified experiment type
- No code duplication
- Backward compatible
- Simpler user experience

## 2. Training Method Duplication

**Issue**: Multiple `_train_model()` implementations across different experiment types:
- `general_alignment.py`
- `layer_wise.py`
- `eigenvector_based.py`
- `cascading_layer.py`

**Solution**:
- Move common training logic to `BaseExperiment` class
- Allow experiments to override specific parts via hooks:
  ```python
  def train_model(self):
      # Common setup
      self._pre_training_hook()
      
      # Common training loop
      for epoch in range(epochs):
          self._train_epoch(epoch)
          self._post_epoch_hook(epoch)
      
      # Common cleanup
      self._post_training_hook()
  ```

## 3. Metric Computation Duplication

**Issue**: Similar metric computation logic repeated across experiments

**Solution**:
- Standardize metric computation in `BaseExperiment.compute_metrics()`
- Remove experiment-specific implementations unless they need special handling
- Use configuration to control metric behavior instead of code duplication

## 4. Configuration Class Proliferation

**Issue**: Many similar configuration classes with minor differences:
- `LayerIsolatedConfig`
- `CascadingConfig`
- `EigenvectorConfig`
- etc.

**Solution**:
- Use composition instead of inheritance
- Create specific configuration sections:
  ```python
  @dataclass
  class PruningConfig:
      strategy: str
      amount: float
      # ... pruning-specific fields
  
  @dataclass
  class GeneralConfig(ExperimentConfig):
      pruning: Optional[PruningConfig] = None
      multi_network: Optional[MultiNetworkConfig] = None
      # ... other optional sections
  ```

## 5. Parallel Strategy Duplication

**Issue**: Multiple parallel implementations:
- `ParallelModePruning`
- `AsyncParallelPruning`
- `train_networks_parallel()`
- `_compute_metrics_parallel()`

**Solution**:
- Create a single `ParallelExecutor` utility class
- Use it consistently across all parallel operations
- Example:
  ```python
  executor = ParallelExecutor(num_workers=4)
  results = executor.map(compute_metrics, networks)
  ```

## 6. File Organization

**Issue**: Similar functionality spread across multiple files

**Solution**:
- Consolidate related functionality:
  - Merge all pruning experiments into a single configurable class
  - Move common utilities to shared modules
  - Group related strategies together

## 7. Remove "Enhanced" Naming

**Issue**: Files with "_enhanced", "_improved", etc. create confusion

**Solution**:
- Integrate improvements into main files
- Use version control for history
- Delete redundant versions

## Implementation Plan

### Phase 1: Multi-Network Integration (Immediate)
1. Update `general_alignment.py` to support `num_networks` parameter
2. Migrate functionality from `parallel_pruning_experiment.py`
3. Delete parallel experiment files
4. Update documentation

### Phase 2: Training Consolidation (Week 1)
1. Extract common training logic to `BaseExperiment`
2. Update each experiment to use base implementation
3. Remove duplicate code

### Phase 3: Configuration Simplification (Week 2)
1. Design new configuration structure
2. Migrate existing configs
3. Update experiment initialization

### Phase 4: Parallel Execution (Week 3)
1. Create `ParallelExecutor` utility
2. Replace all parallel implementations
3. Standardize parallel patterns

### Phase 5: File Consolidation (Week 4)
1. Merge related files
2. Update imports
3. Clean up directory structure

## Expected Benefits

1. **Reduced Complexity**: ~40% less code to maintain
2. **Better User Experience**: Single experiment type handles all cases
3. **Easier Testing**: Less duplicate test cases needed
4. **Faster Development**: Clear patterns for adding new features
5. **Better Performance**: Shared optimizations benefit all experiments

## Backward Compatibility

All changes will maintain backward compatibility:
- Existing configs will continue to work
- Old experiment types will be aliased to new unified version
- Deprecation warnings for old patterns
- Migration guide for users

## Metrics for Success

- Lines of code reduced by 30-40%
- Number of duplicate functions reduced to near zero
- Time to add new features reduced by 50%
- User confusion issues eliminated 