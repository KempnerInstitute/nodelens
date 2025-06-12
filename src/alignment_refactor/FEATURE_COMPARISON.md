# Feature Comparison: Original vs Refactored Codebase

This document compares features between the original alignment codebase and the refactored version.

## ✅ Implemented Features

### Core Infrastructure
| Feature | Original | Refactored | Status |
|---------|----------|------------|--------|
| Protocol-based interfaces | ❌ | ✅ | Improved |
| Registry system | Partial | ✅ | Enhanced |
| Type annotations | Partial | ✅ | Complete |
| Memory management | Basic | ✅ Auto CPU offloading | Enhanced |
| Distributed support | ✅ DDP | ✅ Built-in | Maintained |

### Metrics
| Metric | Original | Refactored | Status |
|--------|----------|------------|--------|
| Rayleigh Quotient (RQ) | ✅ | ✅ `RayleighQuotient` | ✅ |
| Patch-wise RQ | ✅ | ✅ `PatchWiseRayleighQuotient` | ✅ |
| Delta Alignment | ✅ | ✅ `DeltaAlignment` | ✅ |
| Normalized Delta | ✅ | ✅ `NormalizedDeltaAlignment` | ✅ |
| Mutual Information (Gaussian) | ✅ MI_0 | ✅ `MutualInformationGaussian` | ✅ |
| Mutual Information (Binning) | ✅ MI_1 | ✅ `MutualInformationBinning` | ✅ |
| Conditional MI | ✅ | ✅ `ConditionalMutualInformation` | ✅ |
| Average Redundancy | ✅ | ✅ `AverageRedundancy` | ✅ |
| Node Redundancy | ✅ | ✅ `NodeRedundancy` | ✅ |
| Layer Redundancy | ✅ | ✅ `LayerRedundancy` | ✅ |
| Weight Cosine Similarity | ✅ | ✅ `WeightCosineSimilarity` | ✅ |
| Activation Cosine Similarity | ✅ | ✅ `ActivationCosineSimilarity` | ✅ |
| Weight-Activation Alignment | ✅ | ✅ `WeightActivationAlignment` | ✅ |
| **PID Metrics** | ✅ BROJA | ✅ `SharedInformation`, `UniqueInformationX/Y`, `SynergisticInformation` | ✅ |

### Models
| Feature | Original | Refactored | Status |
|---------|----------|------------|--------|
| MLP | ✅ | ✅ Via torchvision | ✅ |
| CNN2P2 | ✅ | ✅ Via torchvision | ✅ |
| External models (torchvision) | ✅ | ✅ | ✅ |
| AlignmentNetwork wrapper | ✅ | ✅ `ModelWrapper` | Improved |
| Activation tracking | ✅ | ✅ `ActivationTracker` | Enhanced |
| Auto layer discovery | ❌ | ✅ | New |

### Datasets
| Dataset | Original | Refactored | Status |
|---------|----------|------------|--------|
| MNIST | ✅ | ✅ | ✅ |
| CIFAR-10 | ✅ | ✅ | ✅ |
| CIFAR-100 | ✅ | ✅ | ✅ |
| ImageNet | ✅ | ✅ | ✅ |
| Custom datasets | ✅ | ✅ Via `DatasetWrapper` | ✅ |

### Experiments
| Experiment Type | Original | Refactored | Status |
|-----------------|----------|------------|--------|
| Progressive Dropout | ✅ | ✅ `ProgressiveDropoutExperiment` | ✅ |
| **Layer-Isolated Pruning** | ✅ | ❌ | **TODO** |
| **Cascading Layer Pruning** | ✅ | ❌ | **TODO** |
| **Eigenvector Dropout** | ✅ | ❌ | **TODO** |

### Training Features
| Feature | Original | Refactored | Status |
|---------|----------|------------|--------|
| Standard training | ✅ | ✅ | ✅ |
| **Fully tensorized training** | ✅ | ❌ | **TODO** |
| Sequential training | ✅ | ✅ | ✅ |
| train_before_dropout option | ✅ | ✅ In config | ✅ |
| Multiple optimizers | ✅ | ✅ | ✅ |
| Loss functions | ✅ | ✅ | ✅ |

## ❌ Missing Features (TODO)

### 2. Additional Experiment Types

#### Layer-Isolated Pruning
- **Original**: `dropout_pruning_mode = "layer_isolated"`
- **Required**: `LayerIsolatedPruningExperiment` class
- Prunes each layer independently based on alignment scores

#### Cascading Layer Pruning  
- **Original**: `dropout_pruning_mode = "cascading_layer"`
- **Required**: `CascadingLayerPruningExperiment` class
- Progressive pruning that cascades through layers

#### Eigenvector Dropout
- **Original**: `run_eigenvector = True`
- **Required**: `EigenvectorDropoutExperiment` class
- Uses PCA to identify principal components
- Drops neurons based on eigenvalue ranking

### 3. Advanced CNN Processing Modes
- **filter_patch_summary**: Aggregates patches per filter (partially supported via patchwise mode)
- **filter_specific_covariance_rq**: Filter-specific covariance computation (needs implementation)
- **batch_patch_combined**: Combined batch and patch processing (partially supported)
- These affect metric computation, not just preprocessing

### 4. Training Methods
- **Fully Tensorized Training**: Train multiple networks simultaneously
  - Requires `train_networks_fully_tensorized` implementation
  - Efficient for training network ensembles

### 5. Multi-Strategy Dropout
- **Original**: `use_multi_strategy_dropout = True`
- Different dropout strategies (magnitude-based, gradient-based, etc.)

### 6. Additional Configuration Options
Most configuration options are present, but some specific fields from original:
- ✅ DDP configuration (basic support present)
- ✅ WandB integration (basic support present)
- ✅ Checkpoint management
- ❌ `train_before_dropout` option (needs implementation in experiments)
- ❌ `scale_by_norm` global setting
- ❌ `force_cpu_for_large_metric_ops` global setting
- ❌ `exclude_classification_layer` option
- ❌ `cnn_rq_aggregation_op` setting (mean, max, var, sum)

## 📋 Implementation Priority

1. **High Priority**:
   - ~~PID metrics (research-critical)~~ ✅ DONE
   - Layer-isolated and cascading pruning experiments
   - Eigenvector dropout experiment

2. **Medium Priority**:
   - Fully tensorized training
   - Advanced CNN processing modes
   - Multi-strategy dropout

3. **Low Priority**:
   - Additional logging options
   - Minor configuration tweaks

## 🔄 Migration Notes

### For PID Metrics:
```python
# Create in metrics/information/pid.py
@register_metric("pid_shared")
class PartialInformationDecomposition(BaseInformationMetric):
    def compute(self, inputs, outputs, **kwargs):
        # Implement BROJA PID computation
        pass
```

### For Additional Experiments:
```python
# Create in experiments/
@register_experiment("layer_isolated_pruning")
class LayerIsolatedPruningExperiment(BaseExperiment):
    def run(self):
        # Implement layer-isolated pruning logic
        pass
```

### For CNN Modes:
- Extend the metric computation to handle `filter_patch_summary` and `filter_specific_covariance_rq`
- These affect how patches are aggregated in CNN layers

## Summary

The refactored codebase has successfully implemented:
- ✅ All core metrics including PID (100% coverage)
- ✅ Clean, modular architecture with protocols and registries
- ✅ Memory-efficient operations with CPU offloading
- ✅ Full model and dataset support
- ✅ Basic experiment infrastructure

Still missing (but can be added incrementally):
1. Three experiment types (layer-isolated, cascading, eigenvector)
2. Fully tensorized training method
3. Some advanced CNN processing modes (filter_specific_covariance_rq)
4. Some configuration options (train_before_dropout, scale_by_norm, etc.)

The refactored architecture makes it easy to add these features while maintaining code quality. 