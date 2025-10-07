# Quick Reference

Common code patterns and examples.

## Basic Usage

### Import Core Components

```python
from alignment import ModelWrapper, get_metric
from alignment.services import (
    ActivationCaptureService,
    NodeScoringService,
    MaskOperations
)
from alignment.pruning.orchestrator import prune_with_all_options
```

### Analyze Pretrained Model

```python
# Wrap model (auto-detects trackable layers)
wrapper = ModelWrapper(model)

# Get metric
rq = get_metric('rayleigh_quotient')

# Capture activations
outputs, activations = wrapper.forward_with_activations(input_batch)
weights = wrapper.get_layer_weights()

# Compute scores
for layer in wrapper.tracked_layers:
    scores = rq.compute(activations[f'{layer}_input'], weights[layer])
    print(f"{layer}: {scores.mean():.4f}")
```

### Prune with Single Metric

```python
from alignment.pruning.orchestrator import prune_with_all_options

result = prune_with_all_options(
    model,
    target_sparsity=0.5,
    scoring='rayleigh_quotient',
    val_loader=val_loader,
    eval_fn=evaluate
)
```

### Prune with Composite Scores

```python
result = prune_with_all_options(
    model,
    target_sparsity=0.7,
    distribution='adaptive_sensitivity',  # Per-layer amounts
    scoring='composite',                  # Multi-metric
    direction='low',                      # Prune unimportant
    val_loader=val_loader,
    eval_fn=evaluate,
    fine_tune_epochs=20
)
```

## Metrics

### Rayleigh Quotient

```python
rq = get_metric('rayleigh_quotient',
    relative=True,
    regularization=1e-6
)
scores = rq.compute(inputs, weights)  # [num_neurons]
```

### Class-Conditioned RQ

```python
results = rq.compute_class_conditioned(
    inputs, weights, targets,
    return_delta_rq=True
)
# Returns: dict with 'rq_uncond', 'rq_cond', 'delta_rq'
```

### Redundancy

```python
redundancy = get_metric('pairwise_redundancy_gaussian',
    mode='output_based',  # Fast for large models
    num_pairs=10
)
scores = redundancy.compute(outputs=layer_outputs)
```

### Synergy

```python
synergy = get_metric('synergy_gaussian_mmi', num_pairs=10)
scores = synergy.compute(inputs, weights, targets=labels)
```

### Composite Scoring

```python
from alignment.services import NodeScoringService

scorer = NodeScoringService(
    metrics={
        'rq': get_metric('rayleigh_quotient'),
        'redundancy': get_metric('pairwise_redundancy_gaussian', mode='output_based')
    },
    gamma_redundancy=0.4,
    delta_rq=0.3
)

scores = scorer.compute_composite_scores(inputs, weights, targets)
```

## Pruning

### Available Strategies

```python
strategies = [
    'magnitude',
    'random',
    'gradient',
    'alignment',
    'composite',
    'movement',
    'adaptive',
    'ultimate'
]
```

### Distribution Methods

```python
distributions = [
    'uniform',
    'global_threshold',
    'adaptive_sensitivity',
    'importance_weighted',
    'size_proportional'
]
distributions = [
    'uniform',                  # Same % per layer
    'global_threshold',         # Global score threshold
    'adaptive_sensitivity',     # Based on layer importance
    'importance_weighted',      # By average scores
    'cascading',               # Sequential
    'size_proportional',       # Based on layer size
    'hybrid'                   # Combination
]
```

### Pruning Direction

```python
direction='low'     # Prune low-scoring neurons
direction='high'    # Prune high-scoring neurons (ablation)
direction='random'  # Random pruning (baseline)
```

## Training Integration

### Track Metrics During Training

```python
from alignment.training.callbacks import AlignmentMetricsCallback

callback = AlignmentMetricsCallback(
    metrics={'rq': get_metric('rayleigh_quotient')},
    layers=['conv1'],
    frequency=100
)

# In training loop:
for batch_idx, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss.backward()
    optimizer.step()
    
    callback.on_batch_end(wrapper, inputs, targets, global_step)
```

### Gradient-Based Metrics

```python
from alignment.metrics.gradient_based import GradientAlignment

grad_align = GradientAlignment(local_signal='hebbian')

# After backward:
alignment = grad_align.compute(
    inputs, outputs,
    gradients=layer.weight.grad
)
# High alignment = Hebbian rule works for this neuron
```

## Architecture-Specific

### CNNs

```python
# Automatic handling of spatial dimensions
wrapper = ModelWrapper(cnn_model)
rq = get_metric('rayleigh_quotient')

# Activations automatically preprocessed
outputs, acts = wrapper.forward_with_activations(images)
scores = rq.compute(acts['conv1_input'], weights['conv1'])
```

### Transformers

```python
from alignment.models.transformer_enhanced import TransformerWrapperEnhanced

wrapper = TransformerWrapperEnhanced(
    transformer_model,
    track_qkv=True,
    track_per_head=True
)

# Analyze per-head
head_repr = wrapper.extract_attention_heads(attn_output)
redundancy = get_metric('pairwise_redundancy_gaussian', mode='output_based')
head_scores = redundancy.compute(outputs=head_repr)
```

### LLMs

```python
from alignment.models.transformer_enhanced import LLaMAWrapper

wrapper = LLaMAWrapper(llama_model, track_ffn=True)

# FFN neurons (e.g., 11,008 in LLaMA-3)
ffn_up = model.model.layers[0].mlp.up_proj
inputs_2d = hidden_states.mean(dim=1)  # Sequence mean
outputs = ffn_up(inputs_2d)

redundancy = get_metric('pairwise_redundancy_gaussian', mode='output_based')
scores = redundancy.compute(outputs=outputs)
# [11008] - one per neuron
```

## Configuration

### Minimal Config

```yaml
experiment:
  name: "my_experiment"
model:
  name: "resnet18"
  pretrained: true
dataset:
  name: "cifar10"
metrics:
  enabled: ['rayleigh_quotient']
```

### Pruning Config

```yaml
pruning:
  enabled: true
  strategy: 'ultimate'
  target_sparsity: 0.7
  distribution: 'adaptive_sensitivity'
  scoring: 'composite'
  fine_tune:
    enabled: true
    epochs: 20
```

Run: `python scripts/run_experiment.py --config my_config.yaml`

## Performance Tips

### For Large Models (LLMs)

```python
# Use output-based mode (30x faster)
redundancy = get_metric('pairwise_redundancy_gaussian',
                       mode='output_based',
                       num_pairs=10)
```

### For Small Batches

```python
# Use covariance shrinkage
from alignment.dataops.processing import estimate_covariance
cov = estimate_covariance(X, method='ledoit_wolf')
```

### For Multiple Networks

```python
from alignment.pruning.parallel_optimizer import ParallelPruningOptimizer

optimizer = ParallelPruningOptimizer()
results = optimizer.prune_ensemble_parallel(networks, ...)
```

## Common Patterns

### Complete Pruning Workflow

```python
from alignment import ModelWrapper
from alignment.services import ActivationCaptureService, NodeScoringService
from alignment.metrics import get_metric
from alignment.pruning.dependency_aware import prune_model_with_dependencies

# 1. Setup
wrapper = ModelWrapper(model)
capture = ActivationCaptureService(wrapper)
scorer = NodeScoringService(metrics={
    'rq': get_metric('rayleigh_quotient'),
    'redundancy': get_metric('pairwise_redundancy_gaussian', mode='output_based')
})

# 2. Capture
data = capture.capture(validation_batch)

# 3. Score
layer_scores = scorer.compute_layerwise_scores(data, targets)
scores_dict = {name: ls.composite for name, ls in layer_scores.items()}

# 4. Prune (with dependency handling)
result = prune_model_with_dependencies(
    model,
    scores_dict,
    amount=0.5
)

# 5. Fine-tune
train(model, train_loader, epochs=20)

# 6. Evaluate
accuracy = evaluate(model, test_loader)
```

## Troubleshooting

### Singular Covariance

```python
# Increase regularization
rq = get_metric('rayleigh_quotient', regularization=1e-4)
```

### Memory Leaks

```python
# HookManager automatically handles cleanup
# No action needed (integrated in v0.2.0)
```

### Shape Mismatches in CNN Pruning

```python
# Use dependency-aware pruning
from alignment.pruning.dependency_aware import prune_model_with_dependencies
result = prune_model_with_dependencies(model, scores, amount)
```

See [user_guide.md](user_guide.md) for detailed documentation.

