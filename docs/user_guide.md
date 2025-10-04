# Alignment Framework User Guide

**Comprehensive guide to using the alignment framework for neural network analysis and pruning**

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Computing Metrics](#computing-metrics)
3. [Pruning Strategies](#pruning-strategies)
4. [Architecture Support](#architecture-support)
5. [Configuration](#configuration)
6. [Advanced Features](#advanced-features)

---

## Core Concepts

### Rayleigh Quotient (RQ)

Measures how well neuron weights align with input principal components:

```
RQ(w) = (w^T Σ w) / (w^T w · tr(Σ))
```

Higher RQ indicates better alignment with dominant input variance.

### Class-Conditioned RQ (ΔRQ)

Measures task-relevant alignment:

```
ΔRQ = RQ(overall) - E[RQ(class-conditioned)]
```

Positive ΔRQ indicates the neuron captures discriminative features.

### Redundancy

Measures overlap between neuron pairs:

```
R(i,j) = I(Y_i; Y_j) = -0.5 · log(1 - ρ²)
```

High redundancy means neurons capture similar information.

### Synergy

Measures complementary information:

```
S(Z; Y_i, Y_j) = I(Z; Y_i, Y_j) - I(Z; Y_i) - I(Z; Y_j) + min(I(Z; Y_i), I(Z; Y_j))
```

High synergy means neurons provide unique joint information.

---

## Computing Metrics

### Basic Usage

```python
from alignment import ModelWrapper, get_metric

# Wrap model
wrapper = ModelWrapper(model, tracked_layers=['conv1', 'fc1'])

# Get metric
rq_metric = get_metric('rayleigh_quotient')

# Capture activations
outputs, acts = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights()

# Compute per-neuron scores
scores = rq_metric.compute(
    inputs=acts['conv1_input'],
    weights=weights['conv1']
)
```

### Available Metrics

**Alignment:**
- `rayleigh_quotient` - Standard RQ
- `delta_alignment` - Change in alignment over training

**Information-Theoretic:**
- `gaussian_mi_analytic` - Mutual information (Gaussian approximation)
- `pairwise_redundancy_gaussian` - Per-neuron redundancy
- `synergy_gaussian_mmi` - Per-neuron synergy
- `conditional_mutual_information` - Conditional MI

**Gradient-Based:**
- `gradient_alignment` - Alignment between local signal and backprop
- `local_learning_rule_search` - Find optimal local rules per neuron

**Task-Specific:**
- `classification_alignment` - Classification-specific importance
- `vision_task_alignment` - Vision-specific patterns

### Composite Scoring

Combine multiple metrics for robust importance estimation:

```python
from alignment.services import NodeScoringService

scorer = NodeScoringService(
    metrics={
        'rq': get_metric('rayleigh_quotient'),
        'redundancy': get_metric('pairwise_redundancy_gaussian', mode='output_based'),
        'synergy': get_metric('synergy_gaussian_mmi')
    },
    alpha_mi=0.0,
    beta_synergy=0.3,
    gamma_redundancy=0.4,
    delta_rq=0.3
)

scores = scorer.compute_composite_scores(inputs, weights, targets)
# score = β·synergy - γ·redundancy + δ·log(RQ)
```

---

## Pruning Strategies

### Available Strategies

**Basic:**
- `magnitude` - L1/L2 norm
- `random` - Random baseline
- `gradient` - Gradient-based
- `fisher` - Fisher information

**Alignment-Based:**
- `alignment` - RQ-based structured pruning
- `global_alignment` - Global threshold across layers
- `hybrid` - Magnitude + Alignment combination

**Advanced:**
- `movement` - Prune weights moving toward zero (training-aware)
- `adaptive` - Adaptive per-layer amounts based on sensitivity
- `ultimate` - Multi-stage combining all best practices

**Novel:**
- `composite` - Redundancy-aware (preserves synergistic neurons)

### Distribution Across Layers

**Uniform:** Same percentage per layer
```python
# 70% removed from each layer
```

**Global Threshold:** Single threshold across all layers
```python
# Naturally varying amounts based on score distribution
```

**Adaptive Sensitivity:** Per-layer amounts based on importance
```python
# Sensitive layers: 50%
# Robust layers: 85%
# Overall: 70% average
```

### Using the Orchestrator

```python
from alignment.pruning.orchestrator import prune_with_all_options

result = prune_with_all_options(
    model,
    target_sparsity=0.7,
    
    # Distribution: how to allocate across layers
    # Options: 'uniform', 'global_threshold', 'adaptive_sensitivity',
    #          'importance_weighted', 'cascading', 'size_proportional', 'hybrid'
    distribution='adaptive_sensitivity',
    
    # Scoring: what importance metric
    # Options: 'magnitude', 'rayleigh_quotient', 'composite', 'movement'
    scoring='composite',
    
    # Direction: which neurons to remove
    # Options: 'low' (remove unimportant), 'high' (ablation), 'random' (baseline)
    direction='low',
    
    # Data
    train_loader=train_loader,  # For fine-tuning
    val_loader=val_loader,       # For evaluation
    trainer_fn=train_function,   # Training function
    eval_fn=eval_function,       # Evaluation function
    
    # Options
    fine_tune_epochs=20
)
```

---

## Architecture Support

### MLPs

```python
layer = model.fc1  # Linear(784, 256)
scores = rq.compute(inputs, layer.weight)
# [256] - one score per neuron
```

### CNNs

```python
conv = model.conv1  # Conv2d(64, 128, 3, 3)
scores = rq.compute(inputs, conv.weight)
# [128] - one score per channel

# Dependency-aware pruning automatically handles channel propagation
```

### Transformers & LLMs

```python
from alignment.models.transformer_enhanced import LLaMAWrapper

wrapper = LLaMAWrapper(llama_model, track_ffn=True)

# FFN neurons: up_proj has 11,008 neurons
ffn_up = model.model.layers[0].mlp.up_proj
scores = rq.compute(inputs, ffn_up.weight)
# [11008] - one score per FFN neuron

# Attention: Can analyze per-head or per-neuron
```

---

## Configuration

### Quick Start Config

```yaml
# configs/quickstart.yaml
experiment:
  name: "quickstart"

model:
  name: "resnet18"
  pretrained: true

dataset:
  name: "cifar10"
  batch_size: 128

metrics:
  enabled: ['rayleigh_quotient']
```

### Complete Options

See `configs/template_master_v2.yaml` for all parameters with documentation.

Key sections:
- `experiment`: Name, seed, device, output
- `model`: Architecture, pretrained, layers to track
- `dataset`: Data source, batch size, augmentation
- `metrics`: Which metrics to compute and their parameters
- `pruning`: Strategy, distribution, scoring, fine-tuning
- `training`: Training parameters, metric tracking
- `advanced`: Backend selection, parallelization

---

## Advanced Features

### Training-Time Metrics

Track alignment evolution during training with zero overhead:

```python
from alignment.training.callbacks import AlignmentMetricsCallback

callback = AlignmentMetricsCallback(
    metrics={'rq': get_metric('rayleigh_quotient')},
    layers=['conv1', 'fc1'],
    frequency=100  # Every 100 steps
)

# In training loop:
for inputs, targets in train_loader:
    outputs = model(inputs)
    loss.backward()
    optimizer.step()
    
    callback.on_batch_end(wrapper, inputs, targets, global_step)

# Analyze evolution
history = callback.get_history()
```

### Gradient-Based Local Learning

Design bio-plausible learning rules:

```python
from alignment.metrics.gradient_based import LocalLearningRuleSearch

searcher = LocalLearningRuleSearch()

# After backward pass:
best_rules = searcher.compute(
    inputs, outputs,
    gradients=layer.weight.grad
)
# Returns best local rule per neuron
```

### Pairwise Metrics

Any pairwise metric can aggregate to single-neuron scores:

```python
redundancy = get_metric('pairwise_redundancy_gaussian',
                       mode='output_based',  # Fast!
                       num_pairs=10,         # Sample 10 partners
                       aggregation='mean')   # How to aggregate

scores = redundancy.compute(outputs=layer_outputs)
# [N] - per-neuron redundancy

# Or get full matrix:
matrix = redundancy.compute(outputs, return_matrix=True)
# [N, N] - all pairwise relationships
```

### Dependency-Aware Pruning

Automatically handles inter-layer dependencies:

```python
from alignment.pruning.dependency_aware import prune_model_with_dependencies

result = prune_model_with_dependencies(
    model,
    layer_scores={'conv1': scores1, 'conv2': scores2},
    amount=0.5,
    verbose=True
)

# Automatically propagates:
# conv1.out_channels → conv2.in_channels
# Maintains shape compatibility
```

---

## Performance

### Computation Time

| Model Scale | Per-Layer Time | Throughput |
|-------------|----------------|------------|
| Small (N=256) | ~28ms | 35 layers/sec |
| Medium (N=1024) | ~105ms | 10 layers/sec |
| Large (N=4096) | ~440ms | 2.3 layers/sec |

### Speedup Techniques

- **Output-based redundancy**: 30x faster for large models
- **Shared computation**: 2-3x when computing multiple metrics
- **Parallel strategies**: M strategies in ~1.3x time

---

## Testing

```bash
# Run all tests
pytest tests/

# Scientific correctness validation
python tests/unit/metrics/test_scientific_correctness.py

# Specific module
pytest tests/unit/services/
```

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit pull request

---

## Support

- **Documentation**: See guides in repository root
- **Examples**: `examples/` directory
- **Issues**: GitHub issues
- **API Reference**: Run `cd docs && make html`

