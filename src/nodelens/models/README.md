# Models Module

Model wrappers and loaders for the alignment metrics framework.

## Quick Start

```python
# Load a pretrained vision model
from nodelens.models import ModelWrapper
import torchvision.models as tvm

model = tvm.resnet18(pretrained=True)
wrapper = ModelWrapper(model)

# Forward pass with activation capture
outputs, activations = wrapper.forward_with_activations(input_batch)
# activations: {'layer1_input': ..., 'layer1_output': ..., ...}

# Get weights for alignment computation
weights = wrapper.get_layer_weights()
# weights: {'layer1': tensor[out_features, in_features], ...}
```

## Components

### Model Wrappers

| Class | Purpose | When to Use |
|-------|---------|-------------|
| `ModelWrapper` | General-purpose wrapper | Default for most experiments |
| `AlignmentNetwork` | Backward-compatible wrapper | Legacy code compatibility |
| `TransformerWrapperEnhanced` | Transformer with Q/K/V tracking | Per-head attention analysis |
| `LLaMAWrapper` | LLaMA-specific wrapper | FFN/attention analysis for LLaMA models |

### Model Loaders (hub.py)

Use these in YAML configs via the model registry:

```yaml
# Vision model from torchvision
model:
  name: "resnet18"  # Shorthand - auto-loads via torchvision

# LLM from Hugging Face
model:
  name: "hf_causal_lm"
  model_id: "meta-llama/Llama-3.1-8B"
  dtype: "bfloat16"
  device_map: "auto"
```

| Loader | Description |
|--------|-------------|
| `TorchvisionModel` | ResNet, VGG, MobileNet, etc. from torchvision |
| `TIMMModel` | Any model from timm library |
| `HFVisionModel` | Vision transformers from Hugging Face |
| `HFCausalLM` | Causal LMs (LLaMA, Mistral, GPT) from Hugging Face |

### Custom Architectures (architectures/)

Simple models for experiments:

```python
from nodelens.models import MLP, CNN2P2, create_model

# Create MLP for MNIST
model = create_model('mlp', 'mnist', hidden_dims=[300, 200])

# Create CNN for CIFAR-10
model = create_model('cnn2p2', 'cifar10')
```

### Hook Management (hooks.py)

Automatic hook lifecycle management:

```python
from nodelens.models.hooks import HookManager

hook_mgr = HookManager()

# Temporary hooks with automatic cleanup
with hook_mgr.temporary_hooks(model, ['layer1', 'layer2']) as cache:
    output = model(input_batch)
    layer1_output = cache['layer1_output']
    layer2_input = cache['layer2_input']
# Hooks automatically removed after context

# Or use PersistentHookManager for long-running tracking
from nodelens.models.hooks import PersistentHookManager

persistent_mgr = PersistentHookManager()
persistent_mgr.register_persistent_hooks(model, ['layer1'])
# ... multiple forward passes ...
persistent_mgr.cleanup()  # Manual cleanup required
```

## Integration with Experiments

The experiment runner automatically wraps models:

```python
# In experiments/base.py
self.wrapped_model = ModelWrapper(self.model, **wrapper_kwargs)

# Activation capture via service layer
from nodelens.services.activation_capture import ActivationCaptureService
service = ActivationCaptureService(wrapped_model)
data = service.capture(input_batch)
```

## Layer Auto-Discovery

Wrappers automatically discover trackable layers:

```python
wrapper = ModelWrapper(model)
print(wrapper.tracked_layers)
# ['layer1.0.conv1', 'layer1.0.conv2', ..., 'fc']

# Or specify explicitly
wrapper = ModelWrapper(model, tracked_layers=['layer4', 'fc'])
```

## CNN Preprocessing Modes

For convolutional layers, choose preprocessing:

| Mode | Description | Memory | Best For |
|------|-------------|--------|----------|
| `unfold` | Unfold with kernel params | High | Exact RQ computation |
| `patchwise` | Keep patch structure | Medium | Patch-level analysis |
| `flatten` | Simple flatten | Low | Quick experiments |

```python
wrapper = ModelWrapper(model, preprocessing_mode='unfold')
```
