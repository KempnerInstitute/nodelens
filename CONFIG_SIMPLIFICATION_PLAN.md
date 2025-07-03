# Configuration Simplification Plan

## Current State Analysis

### Common Configuration Patterns

After analyzing the experiment configurations, I've identified these common patterns:

1. **Training Configuration** (appears in all pruning experiments)
   - `train_before_dropout: bool`
   - `training_epochs: int`
   - `learning_rate: float`
   - `optimizer: str`

2. **Dropout/Pruning Configuration** (all pruning experiments)
   - `dropout_rates: List[float]`
   - `dropout_mode: str`
   - `num_random_trials: int`

3. **Evaluation Configuration** (all experiments)
   - `eval_batches: Optional[int]`
   - `exclude_classification_layer: bool`

4. **Pruning-specific Configuration** (pruning experiments)
   - `pruning_metric: str`
   - `pruning_strategy: str`

5. **CNN Configuration** (some experiments)
   - `cnn_mode: str`

## Proposed Solution: Composition over Inheritance

### Step 1: Create Composable Configuration Classes

```python
@dataclass
class TrainingConfig:
    """Training-related configuration."""
    train_before_dropout: bool = True
    training_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    scheduler_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PruningConfig:
    """Pruning/dropout configuration."""
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    dropout_mode: str = "scaled"
    num_random_trials: int = 3
    pruning_metric: str = "rayleigh_quotient"
    pruning_strategy: str = "low"

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    eval_batches: Optional[int] = None
    exclude_classification_layer: bool = True

@dataclass
class CNNConfig:
    """CNN-specific configuration."""
    cnn_mode: str = "unfold"
```

### Step 2: Update Experiment Configs to Use Composition

```python
@dataclass
class LayerIsolatedConfig(ExperimentConfig):
    """Configuration for layer-isolated pruning experiment."""
    # Compose configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    
    # Experiment-specific fields only
    compute_layer_scores: bool = True
```

### Step 3: Create Factory Functions for Common Patterns

```python
def create_pruning_experiment_config(
    name: str,
    description: str,
    model_name: str,
    dataset_name: str,
    # Override defaults
    training_epochs: int = 10,
    dropout_rates: List[float] = None,
    **kwargs
) -> ExperimentConfig:
    """Factory for creating pruning experiment configs."""
    training = TrainingConfig(training_epochs=training_epochs)
    pruning = PruningConfig(
        dropout_rates=dropout_rates or [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    )
    # ... etc
```

## Benefits

1. **Reduced Duplication**: Common fields defined once
2. **Better Organization**: Related fields grouped together
3. **Easier Defaults**: Change defaults in one place
4. **Type Safety**: Each component has clear types
5. **Flexibility**: Mix and match components as needed

## Migration Strategy

### Phase 1: Create Composable Classes
1. Create `src/alignment/experiments/config_components.py`
2. Define all component classes
3. Add factory functions

### Phase 2: Update One Experiment
1. Start with `LayerIsolatedConfig` as pilot
2. Update to use composition
3. Ensure backward compatibility
4. Test thoroughly

### Phase 3: Migrate Remaining Experiments
1. Update each experiment config
2. Maintain backward compatibility
3. Update documentation

### Phase 4: Deprecate Old Pattern
1. Add deprecation warnings
2. Update all examples
3. Remove old code in future version

## Backward Compatibility

To maintain compatibility, we can:

1. **Property Forwarding**: Add properties that forward to composed objects
```python
@property
def training_epochs(self) -> int:
    return self.training.training_epochs

@training_epochs.setter
def training_epochs(self, value: int):
    self.training.training_epochs = value
```

2. **Custom __init__**: Handle both old and new style initialization
```python
def __init__(self, **kwargs):
    # Handle old-style flat kwargs
    training_kwargs = {}
    for key in ['training_epochs', 'learning_rate', ...]:
        if key in kwargs:
            training_kwargs[key] = kwargs.pop(key)
    
    # Create components
    self.training = TrainingConfig(**training_kwargs)
    # ... etc
```

## Expected Outcomes

- **Code Reduction**: ~30-40% reduction in config code
- **Maintenance**: Easier to add new config options
- **Clarity**: Clear separation of concerns
- **Reusability**: Components can be reused across projects 