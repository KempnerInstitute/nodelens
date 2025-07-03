# Proposal: Unified Multi-Network Support in General Alignment

## Summary

You're absolutely right - there's no good reason to have separate experiment classes for single vs. multiple network training. We should integrate multi-network support directly into the `general_alignment` experiment to provide a seamless experience.

## Benefits of Integration

1. **Single Configuration**: Users only need to learn one config format
2. **Backward Compatible**: Existing configs continue to work (num_networks=1 by default)
3. **Code Reuse**: No duplication of pruning logic, metric computation, etc.
4. **Seamless Scaling**: Easy to go from 1 to N networks by changing one parameter
5. **Consistent Interface**: Same experiment type handles all cases

## Proposed Changes

### 1. Enhanced Configuration

Add these fields to `GeneralAlignmentConfig`:

```python
@dataclass
class GeneralAlignmentConfig(ExperimentConfig):
    # ... existing fields ...
    
    # Multi-network configuration
    num_networks: int = 1  # Number of networks to train (1 = single network, >1 = parallel)
    seeds: Optional[List[int]] = None  # Random seeds for each network
    use_tensorized_training: bool = True  # Use efficient tensorized training when possible
    
    # Statistical analysis for multi-network experiments
    compute_statistics: bool = True  # Compute mean/std across networks
    confidence_level: float = 0.95  # Confidence interval level
    show_confidence_intervals: bool = True  # Show error bars in plots
    save_all_network_results: bool = True  # Save individual results for each network
```

### 2. Conditional Logic in Methods

Each major method checks `self.config.num_networks`:

```python
def _train_model(self):
    if self.config.num_networks == 1:
        # Original single-network training
        return self._train_single_network()
    else:
        # Multi-network training with aggregation
        return self._train_multiple_networks()

def _train_multiple_networks(self):
    # Use tensorized training if possible (much faster)
    if self._can_use_tensorized_training():
        return train_networks_fully_tensorized(...)
    else:
        # Fall back to parallel processing
        return self._train_networks_parallel(...)
```

### 3. Results Structure

Results adapt based on number of networks:

```python
# Single network (backward compatible)
results = {
    "train_losses": [...],
    "train_accs": [...],
    "pruning_results": {...}
}

# Multiple networks (enhanced)
results = {
    "individual_results": [  # Results for each network
        {"train_losses": [...], "train_accs": [...], ...},
        {"train_losses": [...], "train_accs": [...], ...},
    ],
    "aggregated": {  # Statistical aggregation
        "train_losses": {
            "mean": [...],
            "std": [...],
            "confidence_interval": [...]
        },
        ...
    }
}
```

### 4. Visualization Enhancements

Plots automatically show error bars when `num_networks > 1`:

```python
def _generate_visualizations(self):
    if self.config.num_networks == 1:
        # Original single-network plots
        self._plot_single_network_results()
    else:
        # Enhanced plots with confidence intervals
        self._plot_multi_network_results()
```

## Implementation Strategy

### Phase 1: Minimal Changes
1. Add `num_networks` parameter to config
2. Modify `_train_model()` to handle multiple networks
3. Keep all other methods working on single network at a time

### Phase 2: Full Integration
1. Update pruning experiments to aggregate across networks
2. Enhance visualization with statistical plots
3. Add parallel metric computation

### Phase 3: Optimization
1. Integrate tensorized training for efficiency
2. Add caching for repeated computations
3. Optimize memory usage for large experiments

## Example Usage

### Single Network (Existing Behavior)
```yaml
experiment_name: "mnist_alignment"
num_networks: 1  # Or omit entirely, defaults to 1
model:
  name: "mlp"
  hidden_sizes: [512, 256, 128]
```

### Multiple Networks (New Feature)
```yaml
experiment_name: "mnist_alignment_statistical"
num_networks: 5  # Train 5 networks
seeds: [42, 43, 44, 45, 46]  # Optional, auto-generated if not provided
model:
  name: "mlp"
  hidden_sizes: [512, 256, 128]
```

## Code Example

Here's how the integrated `_train_model` might look:

```python
def _train_model(self) -> Dict[str, Any]:
    """Train model(s) and collect alignment metrics."""
    if not self.config.do_train:
        logger.info("Skipping training (do_train=False)")
        return {}
    
    # Single network path (backward compatible)
    if self.config.num_networks == 1:
        logger.info(f"Training model for {self.config.training_epochs} epochs")
        return self._train_single_network_impl()
    
    # Multi-network path
    logger.info(f"Training {self.config.num_networks} networks for {self.config.training_epochs} epochs")
    
    # Create networks with different seeds
    networks = []
    for i, seed in enumerate(self.config.seeds[:self.config.num_networks]):
        torch.manual_seed(seed)
        network = self._create_model()
        networks.append(network)
    
    # Train efficiently
    if self.config.use_tensorized_training and self._can_use_tensorized():
        # Fast path: tensorized training
        results = self._train_networks_tensorized(networks)
    else:
        # Fallback: parallel training
        results = self._train_networks_parallel(networks)
    
    # Aggregate results
    return self._aggregate_training_results(results)
```

## Advantages Over Current Approach

1. **User Experience**: One experiment type to learn, not multiple
2. **Maintenance**: Single codebase to maintain
3. **Features**: All features available regardless of network count
4. **Migration**: Easy to convert existing experiments
5. **Flexibility**: Can start with 1 network and scale up without changing experiment type

## Migration Path

1. Current `parallel_pruning_experiment.py` becomes deprecated
2. Its functionality gets absorbed into `general_alignment.py`
3. Add compatibility layer to convert old configs
4. Eventually remove the separate parallel experiment

## Conclusion

This integration makes the framework more intuitive and powerful. Users get statistical analysis "for free" just by setting `num_networks > 1`, while maintaining full backward compatibility for existing single-network experiments. 