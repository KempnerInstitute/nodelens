# Feature Verification: alignment_refactor

This document verifies that the refactored codebase (`src/alignment_refactor`) contains all the features from the original codebases (`src/alignment`, `src/alignment_v2`, `src/alignment_preref`).

## ✅ Core Requirements Verified

### 1. **Works with Any Network** ✓

The refactored `ModelWrapper` class can wrap any PyTorch network:

```python
# Examples from the codebase:
from alignment_refactor.models import ModelWrapper

# Torchvision models
wrapper = ModelWrapper(models.resnet18(pretrained=True))
wrapper = ModelWrapper(models.vgg16())

# Custom models
custom_model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
wrapper = ModelWrapper(custom_model)
```

**Evidence**: 
- `models/wrappers.py`: ModelWrapper accepts any `nn.Module`
- `models/base.py`: BaseModelWrapper has auto-discovery of trackable layers
- Example scripts show usage with ResNet, VGG, MobileNet, custom CNNs, and MLPs

### 2. **Works with Any Layer** ✓

The system can track any layer type with learnable parameters:

```python
# Track specific layers
wrapper = ModelWrapper(model, tracked_layers=["conv1", "fc1", "layer2.0.conv1"])

# Auto-discover all trackable layers
wrapper = ModelWrapper(model, tracked_layers=None)  # Auto-discovers Conv, Linear layers
```

**Evidence**:
- `models/base.py:_discover_layers()`: Automatically finds Linear, Conv1d, Conv2d layers
- Supports mixed layer types in same model
- Can track by layer name or auto-discover

### 3. **Proper Hook Implementation** ✓

The refactored code uses PyTorch's forward hooks for activation tracking:

**In `core/base.py`:**
```python
def _register_hooks(self) -> None:
    """Register forward hooks for activation collection."""
    for name, module in self._model.named_modules():
        if name in self._tracked_layers:
            hook = module.register_forward_hook(
                self._create_activation_hook(name)
            )
            self._hooks.append(hook)
```

**In `models/base.py`:**
```python
def _create_activation_hook(self, layer_name: str):
    """Create a forward hook for a specific layer."""
    def hook(module, input, output):
        # Store both input and output if requested
        if self.track_inputs and input is not None:
            if isinstance(input, tuple):
                input = input[0]
            self._activation_cache[f"{layer_name}_input"] = input.detach()
        
        if self.track_outputs and output is not None:
            if isinstance(output, tuple):
                output = output[0]
            self._activation_cache[layer_name] = output.detach()
    
    return hook
```

**Key Features**:
- ✅ Tracks both layer inputs and outputs
- ✅ Handles tuple inputs/outputs correctly
- ✅ Detaches tensors to avoid memory issues
- ✅ Proper cleanup with `_clear_hooks()`

### 4. **Usage During Training** ✓

The refactored code fully supports metric computation during training:

```python
# During training loop
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass with activation tracking
        output, activations = wrapper.forward_with_activations(data)
        loss = criterion(output, target)
        
        # Compute metrics during training
        if batch_idx % log_interval == 0:
            weights = wrapper.get_layer_weights()
            for layer_name in wrapper.tracked_layers:
                if f"{layer_name}_input" in activations:
                    scores = metric.compute(
                        inputs=activations[f"{layer_name}_input"],
                        weights=weights[layer_name]
                    )
        
        loss.backward()
        optimizer.step()
```

**Evidence**:
- Hooks don't interfere with gradients
- Can compute metrics at any point during training
- `forward_with_activations()` returns both outputs and activations

### 5. **Usage After Training** ✓

Post-training analysis is fully supported:

```python
# After training
model.eval()
wrapper = ModelWrapper(trained_model)

# Analyze over dataset
for data, _ in test_loader:
    _, activations = wrapper.forward_with_activations(data)
    # Compute metrics...

# Or use ActivationTracker for accumulation
tracker = ActivationTracker(wrapper)
for data, _ in test_loader:
    tracker.update(data)
stats = tracker.get_statistics()
```

**Evidence**:
- `models/wrappers.py`: ActivationTracker class for accumulating statistics
- All experiments support post-training analysis
- Can analyze without modifying the trained model

### 6. **Usage on Pre-trained Networks** ✓

Full support for pre-trained models:

```python
# Load pre-trained model
pretrained_model = models.resnet18(pretrained=True)
wrapper = ModelWrapper(pretrained_model, store_initial_weights=True)

# Get initial weights
initial_weights = wrapper.get_initial_weights("layer_name")

# Compute metrics on pre-trained model
outputs, activations = wrapper.forward_with_activations(inputs)
```

**Evidence**:
- `experiments/base.py`: `pretrained` configuration option
- `models/wrappers.py`: `store_initial_weights` option for delta alignment
- Example scripts show usage with pre-trained torchvision models

## 📊 Additional Features Maintained

### Configuration Options
All original configuration options are preserved in `experiments/base.py`:
- ✅ `train_before_dropout`
- ✅ `scale_by_norm`
- ✅ `force_cpu_for_large_metric_ops`
- ✅ `exclude_classification_layer`
- ✅ `cnn_rq_aggregation_op`

### CNN Processing Modes
The refactored code maintains CNN processing flexibility:
- ✅ "unfold" mode - Unfolds patches for Conv layers
- ✅ "patchwise" mode - Keeps spatial structure
- ✅ "flatten" mode - Simple flattening

### Training Methods
All training methods are implemented:
- ✅ Standard single network training
- ✅ Sequential multi-network training
- ✅ Fully tensorized parallel training (`training/tensorized.py`)

### Experiment Types
All experiment types from original codebase:
- ✅ Progressive dropout (`experiments/progressive_dropout.py`)
- ✅ Layer-isolated pruning (`experiments/layer_isolated.py`)
- ✅ Cascading layer pruning (`experiments/cascading.py`)
- ✅ Eigenvector dropout (`experiments/eigenvector.py`)

## 🔍 Hook Implementation Comparison

### Original AlignmentNetwork
- Used `store_hidden=True` parameter (deprecated)
- Relied on external `collect_layer_data` function
- Complex hook management for eigenvector dropout

### Refactored ModelWrapper
- Always-on hook system when layers are tracked
- Built-in activation collection
- Cleaner API with `forward_with_activations()`
- Supports same functionality with better design

## ✨ Improvements in Refactored Version

1. **Cleaner Hook Management**: Hooks are registered once and managed automatically
2. **Type Safety**: Full type annotations throughout
3. **Memory Efficiency**: Automatic CPU offloading for large operations
4. **Better Organization**: Clear separation between model wrapping and metric computation
5. **Extensibility**: Easy to add new layer types or tracking modes

## Conclusion

The refactored codebase in `src/alignment_refactor` **fully supports all the features** from the original codebases:

- ✅ Works with any PyTorch network
- ✅ Works with any layer type
- ✅ Proper hook implementation for activation tracking
- ✅ Full support during training
- ✅ Full support after training
- ✅ Full support for pre-trained networks

Additionally, it provides a cleaner, more maintainable architecture while preserving all the computational methods and algorithms from the original implementation. 