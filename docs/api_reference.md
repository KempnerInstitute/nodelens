# API Reference

Quick reference for framework APIs.

## Core Classes

### ModelWrapper

```python
from alignment import ModelWrapper

wrapper = ModelWrapper(
    model,                    # PyTorch model
    tracked_layers=None,      # List of layer names or None (auto-detect)
    track_inputs=True,        # Capture layer inputs
    track_outputs=True        # Capture layer outputs
)

# Methods:
wrapper.forward_with_activations(inputs)  # Returns (outputs, activations_dict)
wrapper.get_layer_weights(layers=None)     # Returns dict of weights
wrapper.tracked_layers                     # List of tracked layer names
```

### BaseMetric

All metrics inherit from `BaseMetric` and implement:

```python
metric.requires_inputs   # bool: needs layer inputs
metric.requires_weights  # bool: needs layer weights
metric.requires_outputs  # bool: needs layer outputs
metric.compute(inputs, weights, outputs, **kwargs)  # Returns scores
```

## Metrics

### Rayleigh Quotient

```python
from alignment.metrics import get_metric

rq = get_metric('rayleigh_quotient',
    relative=True,           # Normalize by trace
    regularization=1e-6,     # Numerical stability
    min_samples=2           # Minimum batch size
)

scores = rq.compute(inputs, weights)  # [num_neurons]

# Class-conditioned:
results = rq.compute_class_conditioned(
    inputs, weights, targets,
    return_delta_rq=True
)
# Returns: {'rq_uncond', 'rq_cond', 'delta_rq'}
```

### Redundancy

```python
redundancy = get_metric('pairwise_redundancy_gaussian',
    mode='output_based',     # 'output_based' or 'covariance_based'
    num_pairs=10,            # Number of partners to sample
    sampling_strategy='random', # 'random', 'nearest', or 'all'
    aggregation='mean'       # 'mean', 'median', 'max', 'sum'
)

scores = redundancy.compute(outputs=layer_outputs)  # [N]
matrix = redundancy.compute(outputs, return_matrix=True)  # [N,N]
```

### Synergy

```python
synergy = get_metric('synergy_gaussian_mmi',
    num_pairs=10
)

scores = synergy.compute(inputs, weights, targets=labels)  # [N]
```

### Gradient-Based

```python
from alignment.metrics.gradient_based import GradientAlignment

grad_align = GradientAlignment(
    local_signal='hebbian',  # 'hebbian', 'anti_hebbian', 'oja', etc.
    normalize=True
)

# After backward pass:
alignment = grad_align.compute(
    inputs, outputs,
    gradients=layer.weight.grad
)
# Returns correlation between local signal and backprop
```

## Services

### ActivationCaptureService

```python
from alignment.services import ActivationCaptureService

capture = ActivationCaptureService(
    model_wrapper,
    default_mode='flatten',  # Preprocessing mode
    conv_mode='patchwise'    # For conv layers
)

data = capture.capture(
    input_batch,
    layers=['conv1'],
    include_weights=True,
    preprocess=True
)

# Returns ActivationData with:
# - data.inputs: Dict[layer_name, tensor]
# - data.outputs: Dict[layer_name, tensor]
# - data.weights: Dict[layer_name, tensor]
```

### NodeScoringService

```python
from alignment.services import NodeScoringService

scorer = NodeScoringService(
    metrics={...},
    alpha_mi=0.3,
    beta_synergy=0.2,
    gamma_redundancy=0.3,
    delta_rq=0.2
)

scores = scorer.compute_composite_scores(inputs, weights, targets)

# Returns CompositeScores with:
# - scores.rq
# - scores.redundancy
# - scores.synergy
# - scores.composite
```

### MaskOperations

```python
from alignment.services import MaskOperations

# Structured mask (per-neuron/channel)
mask = MaskOperations.create_structured_mask(
    scores,
    amount=0.5,        # Prune 50%
    mode='low',        # 'low', 'high', or 'random'
    min_keep=1         # Minimum neurons to keep
)

# Statistics
stats = MaskOperations.get_mask_statistics(mask)
# Returns: {total_elements, kept_elements, pruned_elements, sparsity, density}

# Global threshold across layers
masks = MaskOperations.global_threshold_mask(
    layer_scores_dict,
    global_amount=0.5
)
```

## Pruning

### Quick Pruning

```python
from alignment.pruning.orchestrator import prune_with_all_options

result = prune_with_all_options(
    model,
    target_sparsity=0.7,
    distribution='adaptive_sensitivity',
    scoring='composite',
    direction='low',
    use_dynamic=False,
    val_loader=val_loader,
    eval_fn=evaluate,
    fine_tune_epochs=20
)
```

### Dependency-Aware Pruning

```python
from alignment.pruning.dependency_aware import DependencyAwarePruning

pruner = DependencyAwarePruning(model)

result = pruner.prune(
    layer_scores={'conv1': scores1, 'conv2': scores2},
    amount=0.5,
    mode='low',
    dry_run=False  # Set True to preview without applying
)
```

### Parallel Comparison

```python
from alignment.pruning.parallel_optimizer import ParallelPruningOptimizer

optimizer = ParallelPruningOptimizer(num_workers=4)

results = optimizer.compare_strategies_parallel(
    model,
    strategies=['magnitude', 'alignment', 'composite'],
    amounts=[0.3, 0.5, 0.7],
    data_loader=val_loader,
    eval_fn=evaluate
)
```

## Training

### Training Callback

```python
from alignment.training.callbacks import AlignmentMetricsCallback

callback = AlignmentMetricsCallback(
    metrics={'rq': get_metric('rayleigh_quotient')},
    layers=['conv1'],
    frequency=100,          # Compute every N steps
    sample_size=512,        # Subsample batch for efficiency
    tracker=None            # Optional: WandB/TensorBoard tracker
)

# In training loop:
callback.on_batch_end(wrapper, inputs, targets, step)

# Get history:
history = callback.get_history()
```

## Model Wrappers

### Generic Wrapper

```python
from alignment import ModelWrapper

wrapper = ModelWrapper(model)  # Auto-detects all trackable layers
```

### Enhanced Transformer

```python
from alignment.models.transformer_enhanced import TransformerWrapperEnhanced

wrapper = TransformerWrapperEnhanced(
    transformer_model,
    track_qkv=True,          # Track Q/K/V projections
    track_per_head=True,     # Extract per-head representations
    aggregation='sequence_mean',  # 'sequence_mean' or 'token_level'
    num_heads=32,            # Auto-detect if None
    head_dim=128             # Auto-detect if None
)

# Extract per-head:
head_repr = wrapper.extract_attention_heads(attn_output)
```

### LLaMA Wrapper

```python
from alignment.models.transformer_enhanced import LLaMAWrapper

wrapper = LLaMAWrapper(
    llama_model,
    track_ffn=True,         # Track FFN layers
    track_attention=True    # Track attention
)

# Access FFN layers:
wrapper.ffn_layers  # {'expansion': [...], 'contraction': [...]}

# Access attention:
wrapper.attention_layers  # {'q': [...], 'k': [...], 'v': [...]}
```

## Utility Functions

### Layer Detection

```python
from alignment.core.layer_detector import detect_trackable_layers

layers = detect_trackable_layers(
    model,
    min_neurons=1,
    roles=None  # Filter by roles: 'linear', 'conv', 'ffn_expansion', etc.
)
```

### Covariance Estimation

```python
from alignment.data.processing import estimate_covariance

cov = estimate_covariance(
    X,
    method='ledoit_wolf',  # 'none', 'diagonal', 'ledoit_wolf', 'oas'
    regularization=1e-6
)
```

## Configuration Parameters

### Metric Parameters

**RayleighQuotient:**
- `relative` (bool): Normalize by trace
- `regularization` (float): Diagonal regularization (default: 1e-6)
- `min_samples` (int): Minimum batch size

**PairwiseRedundancyGaussian:**
- `mode` (str): 'output_based' (fast) or 'covariance_based'
- `num_pairs` (int): Partners to sample (default: 10)
- `sampling_strategy` (str): 'random', 'nearest', or 'all'
- `aggregation` (str): 'mean', 'median', 'max', 'sum'

### Pruning Parameters

**Strategy:** 'magnitude', 'alignment', 'composite', 'movement', 'adaptive', 'ultimate'

**Distribution:** 'uniform', 'global_threshold', 'adaptive_sensitivity', 'importance_weighted', 'cascading', 'size_proportional', 'hybrid'

**Scoring:** 'magnitude', 'rayleigh_quotient', 'composite', 'movement'

**Direction:** 'low' (prune unimportant), 'high' (ablation), 'random' (baseline)

## Common Patterns

### Analyze Pretrained Model

```python
from alignment import ModelWrapper, get_metric

wrapper = ModelWrapper(pretrained_model)
rq = get_metric('rayleigh_quotient')

outputs, acts = wrapper.forward_with_activations(validation_batch)
weights = wrapper.get_layer_weights()

for layer in wrapper.tracked_layers:
    scores = rq.compute(acts[f'{layer}_input'], weights[layer])
    print(f"{layer}: mean={scores.mean():.4f}, std={scores.std():.4f}")
```

### Prune with Best Strategy

```python
from alignment.pruning.strategies.ultimate import create_ultimate_pruner

pruner = create_ultimate_pruner(target_sparsity=0.7, mode='full')

result = pruner.prune(
    model,
    train_loader,
    val_loader,
    trainer_fn=train,
    eval_fn=evaluate
)
```

### Compare Multiple Strategies

```python
from alignment.pruning.parallel_optimizer import ParallelPruningOptimizer

optimizer = ParallelPruningOptimizer()

results = optimizer.compare_strategies_parallel(
    model,
    strategies=['magnitude', 'composite', 'ultimate'],
    amounts=[0.5, 0.7],
    data_loader=val_loader,
    eval_fn=evaluate
)
```

## Examples

See `examples/` directory for complete workflows:

- `07_mnist_intelligent_pruning.py` - End-to-end pruning
- `08_llama_ffn_pruning.py` - LLM feed-forward analysis
- `09_attention_neuron_vs_head_pruning.py` - Attention layer analysis

