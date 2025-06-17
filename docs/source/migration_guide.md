# Migration Guide: From Old Alignment to Refactored Codebase

This guide helps users transition from the old `src/alignment` codebase to the new `src/alignment_refactor` structure.

## Key Changes Overview

### 1. **Experiment System**

**Old Codebase (`src/alignment/experiments/`):**
- `experiment.py` - Base `Experiment` class
- `alignment_experiments.py` - Single large file (1300+ lines) containing ALL experiment classes:
  - `AlignmentExperiment` (base class for alignment experiments)
  - `ProgressiveDropoutExperiment`
  - `EigenvectorDropoutExperiment`
  - `LayerIsolatedPruningExperiment`
  - `AlignmentAnalysisExperiment`
  - `CascadingLayerPruningExperiment`

**New Codebase (`src/alignment_refactor/experiments/`):**
- `base.py` - Base `BaseExperiment` class
- Separate file for each experiment type:
  - `progressive_dropout.py` - Progressive dropout experiments
  - `eigenvector.py` - Eigenvector-based pruning experiments
  - `cascading.py` - Cascading layer pruning experiments
  - `layer_isolated.py` - Layer-isolated pruning experiments
- `runner.py` - `ExperimentRunner` for managing multiple experiments

**Key Difference**: The new structure is more modular - each experiment type has its own file instead of all being in one large file.

**Migration Example:**
```python
# Old way
from alignment.experiments.alignment_experiments import ProgressiveDropoutExperiment
from alignment.config import ExperimentConfig

config = ExperimentConfig.from_yaml('config.yaml')
experiment = ProgressiveDropoutExperiment(config)
results, networks = experiment.run()

# New way
from alignment_refactor.experiments import ExperimentRunner
from alignment_refactor.experiments.progressive_dropout import ProgressiveDropoutExperiment
from alignment_refactor.experiments.base import ExperimentConfig

config = ExperimentConfig.from_dict(yaml_config_dict)
# Option 1: Direct experiment usage
experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()

# Option 2: Using runner for multiple experiments
runner = ExperimentRunner(base_config=config)
runner.add_experiment(ProgressiveDropoutExperiment)
all_results = runner.run_all()
```

### 2. **Network Models**

**Old Codebase:**
- `src/alignment/models/models.py` with predefined networks:
  - `MLP` - Multi-layer perceptron with configurable hidden layers
  - `CNN2P2` - CNN with 2 conv layers and 2 pooling layers
  - Model creation functions: `create_mlp()`, `create_cnn2p2()`, `create_alexnet()`
  - Dataset-specific parameters in `DATASET_PARAMETERS`
- `AlignmentNetwork` wrapper for layer tracking

**New Codebase:**
- `src/alignment_refactor/models/architectures/` - Currently empty, **missing pre-defined models**
- `ModelWrapper` for unified interface and activation tracking
- Need to implement standard models or use torchvision directly

**Migration Example:**
```python
# Old way
from alignment.models.models import MLP, CNN2P2
from alignment.models.registry import create_model

# Direct instantiation
model = MLP(input_dim=784, hidden_dims=[100, 100, 50], output_dim=10, dropout_rate=0.5)

# Or via registry
model = create_model(model_config)  # Returns AlignmentNetwork wrapper

# New way - REQUIRES IMPLEMENTATION
# Option 1: Direct torchvision models
import torchvision.models as models
model = models.resnet18(num_classes=10)
wrapped_model = ModelWrapper(model, tracked_layers=['layer1.0.conv1', 'layer2.0.conv1'])

# Option 2: Custom implementation needed
# You need to implement MLP, CNN2P2 in src/alignment_refactor/models/architectures/
# or copy from the example standard_models.py I created
```

### 3. **Metrics and Analysis**

**Old Codebase:**
- Metrics computed inline in experiments
- Limited metric options
- Manual activation tracking

**New Codebase:**
- `src/alignment_refactor/metrics/` - Comprehensive metric library
- Modular metric computation
- Automatic activation tracking with ModelWrapper

**Available Metrics:**
- Information Theory: `MutualInformation`, `SharedInformation`, `PartialInformationDecomposition`
- Rayleigh: `RayleighQuotient`, `GeneralizedRayleighQuotient`
- Similarity: `CKA`, `CCA`, `Procrustes`

### 4. **Data Handling**

**Old Codebase:**
- `src/alignment/datasets.py` - Complete dataset implementations
  - `DataSet` base class with standard interface
  - Pre-defined datasets: `MNIST`, `CIFAR10`, `CIFAR100`, `ImageNet2012`
  - `DATASET_REGISTRY` for dataset lookup
  - `get_dataset()` and `load_dataset()` helper functions
  - Transform parameters by model-dataset combination

**New Codebase:**
- `src/alignment_refactor/data/` - More modular structure
  - `base.py` - Base dataset class
  - `datasets/` - Individual dataset implementations
  - `loaders.py` - DataLoader utilities
- Similar functionality but different API

**Migration Example:**
```python
# Old way
from alignment.datasets import get_dataset, load_dataset
from alignment.config import DatasetConfig

# Using registry
dataset_class = get_dataset('MNIST')
dataset = dataset_class(device='cuda', transform_parameters={'flatten': True})

# Or with config
dataset = load_dataset(dataset_config, batch_size=128)

# New way
from alignment_refactor.data import get_dataset

dataset = get_dataset('mnist')  # Note: lowercase naming convention
```

### 5. **Training Framework**

**Old Codebase:**
- `src/alignment/training.py` - Comprehensive training utilities
  - `train_networks()` - Main training function for multiple networks
  - `train_model()` - Single model training
  - Support for callbacks, different optimizers, weight decay
  - Built-in DDP support
- Training logic integrated into experiment classes

**New Codebase:**
- Training is built into experiment classes
- Less modular than old codebase - no separate training utilities
- Each experiment type handles its own training

**Migration Example:**
```python
# Old way
from alignment.training import train_networks

history = train_networks(
    networks=[model1, model2],
    dataset=dataset,
    num_epochs=10,
    learning_rate=0.001,
    device='cuda',
    callbacks=[metric_tracker]
)

# New way - training is part of experiment
experiment = ProgressiveDropoutExperiment(config)
results = experiment.run()  # Training happens internally
```

## Feature Mapping

| Old Feature | New Location | Notes |
|-------------|--------------|-------|
| `alignment_experiments.py` classes | `experiments/*.py` files | Split into separate modules |
| ProgressiveDropoutExperiment | `experiments/progressive_dropout.py` | Standalone module |
| EigenvectorDropoutExperiment | `experiments/eigenvector.py` | Standalone module |  
| LayerIsolatedPruningExperiment | `experiments/layer_isolated.py` | Standalone module |
| CascadingLayerPruningExperiment | `experiments/cascading.py` | Standalone module |
| AlignmentNetwork wrapper | `ModelWrapper` | Enhanced with more features |
| `models/models.py` (MLP, CNN2P2) | **MISSING** - needs implementation | Pre-defined models not ported |
| `create_model()` registry | **MISSING** - no factory pattern | Model creation simplified |
| YAML config loading | `ExperimentConfig` | Different structure |
| Metrics (RQ, MI_Gaussian, etc.) | `metrics/` module | Enhanced implementations |
| Training functions | Built into experiment classes | Less modular than old |
| Plotting utilities | **MISSING** in new codebase | Need to port from old |

## Migration Steps

### 1. Update Imports
```python
# Replace old imports
# from alignment.models import MLP
# from alignment.utils import compute_metric

# With new imports
from alignment_refactor.models import ModelFactory
from alignment_refactor.metrics import RayleighQuotient
```

### 2. Convert Configuration Files
Old YAML format:
```yaml
model_type: MLP
hidden_sizes: [300, 200]
dataset: MNIST
```

New YAML format:
```yaml
model:
  type: mlp
  params:
    hidden_sizes: [300, 200]
    activation: relu
    
data:
  dataset: mnist
  batch_size: 128
  
training:
  optimizer: adam
  learning_rate: 0.001
```

### 3. Update Experiment Scripts
```python
# Old experiment script
from alignment.experiments import run_alignment_experiment

results = run_alignment_experiment(
    model_type='MLP',
    dataset='MNIST',
    metrics=['rayleigh']
)

# New experiment script
from alignment_refactor.experiments import ExperimentRunner
from alignment_refactor.configs import ExperimentConfig

config = ExperimentConfig(
    model={'type': 'mlp', 'params': {'hidden_sizes': [300, 200]}},
    data={'dataset': 'mnist', 'batch_size': 128},
    metrics=['rayleigh_quotient']
)

runner = ExperimentRunner(config)
results = runner.run()
```

## New Features Not in Old Codebase

1. **Tensorized Dropout** - Advanced dropout implementation for structured pruning
2. **ModelWrapper** - Enhanced model wrapping with automatic activation tracking
3. **Enhanced Metrics** - PID, CKA, CCA (in addition to RQ, MI)
4. **Modular Experiment Structure** - Each experiment type in separate file

## Major Missing Features in New Codebase

### 1. **Pre-defined Neural Networks**
The old codebase had ready-to-use implementations:
- `MLP` with configurable hidden layers
- `CNN2P2` with 2 conv + 2 pool layers  
- Model creation functions with dataset-specific parameters
- `DATASET_PARAMETERS` dictionary for automatic configuration

**Impact:** Users must implement their own models or use torchvision directly

### 2. **Plotting and Visualization**
The old `src/alignment/utils/plotting.py` provided extensive plotting:
- `plot_dropout_results()` - Accuracy/loss vs dropout fraction
- `plot_experiment_summary()` - 2x2 grid summary plots
- `plot_mean_score_of_pruned_nodes()` - Layer-wise pruning analysis
- `plot_per_layer_pruning_percentage()` - Pruning distribution
- `plot_metric_evolution()` - Metric tracking over time
- `log_plots_to_wandb()` - W&B integration

**Impact:** No built-in visualization - users must create their own plots

### 3. **Model Registry and Factory**
The old codebase had:
- `create_model()` function with automatic configuration
- Model registry with dataset-specific parameters
- Transform parameters by model-dataset combination

**Impact:** More manual model creation and configuration

### 4. **Training Utilities**
The old `training.py` provided:
- Standalone `train_networks()` function
- Support for training multiple network replicates
- Built-in callback system
- Flexible optimizer configuration

**Impact:** Training logic is now embedded in experiments, less reusable

### 5. **Configuration System**
The old codebase had comprehensive configs:
- `ExperimentConfig`, `DatasetConfig`, `ModelConfig`, etc.
- YAML loading with validation
- Nested configuration with defaults

**Impact:** Simpler but less structured configuration in new codebase

### 6. **Utilities and Helpers**
Missing utilities from `src/alignment/utils/`:
- `activation_utils.py` - Activation extraction helpers
- `model_utils.py` - Model manipulation utilities
- `metrics_utils.py` - Metric computation helpers
- `evaluation.py` - Model evaluation utilities

**Impact:** Users need to implement these utilities themselves

## Common Issues and Solutions

### Issue 1: Missing Model Types
**Problem:** Old model types like CNN2P2 not found
**Solution:** Define custom models using the new architecture system:
```python
from alignment_refactor.models.architectures.base import BaseArchitecture

class CNN2P2(BaseArchitecture):
    def __init__(self):
        # Define your architecture
        pass
```

### Issue 2: Config File Incompatibility
**Problem:** Old YAML configs don't work
**Solution:** Use the config converter utility:
```python
from alignment_refactor.utils import convert_old_config
new_config = convert_old_config('old_config.yaml')
```

### Issue 3: Metric Computation Changes
**Problem:** Metrics computed differently
**Solution:** Use the new metric interface:
```python
# Old way: manual computation
# New way: standardized interface
metric = RayleighQuotient()
scores = metric.compute(inputs, weights)
```

## Best Practices for Migration

1. **Start Small**: Migrate one experiment at a time
2. **Implement Missing Components First**:
   - Create standard models (MLP, CNN2P2) if needed
   - Port essential plotting functions
   - Set up proper configuration structure
3. **Test Thoroughly**: 
   - Verify metrics compute the same values
   - Check that pruning behaves identically
   - Compare final accuracies between old and new
4. **Leverage New Features When Appropriate**:
   - Use ModelWrapper for cleaner activation tracking
   - Take advantage of new metrics (PID, CKA)
5. **Create Wrapper Functions**: 
   - Bridge old API to new where possible
   - Maintain backward compatibility for configs

## Quick Reference: Common Migrations

| Task | Old Code | New Code |
|------|----------|----------|
| Run progressive dropout | `ProgressiveDropoutExperiment(config).run()` | Same, but import from different module |
| Create MLP | `MLP(input_dim=784, hidden_dims=[300,200])` | Need to implement or use torchvision |
| Load MNIST | `get_dataset('MNIST', build=True)` | `get_dataset('mnist')` |
| Train model | `train_networks([model], dataset, epochs=10)` | Embedded in experiment.run() |
| Compute RQ | `metric = get_metric('RQ')`<br>`metric.compute_metric(acts, weights)` | `metric = RayleighQuotient()`<br>`metric.compute(inputs, weights)` |
| Plot results | `plot_dropout_results(results, save_dir)` | Need to implement yourself |

## Recommended Migration Path

1. **Phase 1: Core Components**
   - Port model definitions to `models/architectures/`
   - Implement essential plotting functions
   - Create config converters

2. **Phase 2: Experiments**
   - Migrate experiments one by one
   - Verify metric computations match
   - Test on small datasets first

3. **Phase 3: Enhancement**
   - Add new metrics where beneficial
   - Leverage ModelWrapper features
   - Optimize for performance

## Resources

### Available in New Codebase
- Example scripts: `src/alignment_refactor/examples/`
  - `mnist_mlp_pruning.py` - Basic MLP pruning example
  - `simple_pruning_demo.py` - Quick pruning demonstration
  - `interactive_pruning_tutorial.py` - Comprehensive tutorial
- Standard models: `src/alignment_refactor/models/architectures/standard_models.py`
  - MLP and CNN2P2 implementations matching old codebase
- Migration guide: `src/alignment_refactor/migration_guide.md` (this file)

### Files to Port from Old Codebase
If you need specific functionality, consider porting these files:
- `src/alignment/utils/plotting.py` - Visualization functions
- `src/alignment/models/registry.py` - Model registration system
- `src/alignment/config.py` - Comprehensive configuration classes
- `src/alignment/utils/evaluation.py` - Evaluation utilities

## Summary

The new `alignment_refactor` codebase provides a cleaner, more modular structure but is missing several convenience features from the old codebase:

**Key Missing Components:**
1. Pre-defined standard models (MLP, CNN2P2) - now provided in `standard_models.py`
2. Comprehensive plotting utilities - need to port from old codebase
3. Model registry and factory pattern - simplified but less flexible
4. Standalone training utilities - now embedded in experiments
5. Rich configuration system - simplified in new version

**Migration Strategy:**
- Start by implementing missing components you need
- Use the provided standard models as a starting point
- Port plotting functions as needed
- Test thoroughly to ensure compatibility

The refactored codebase offers better modularity and new features like tensorized dropout and enhanced metrics, but requires more setup for basic tasks. Plan your migration based on which features are essential for your use case. 