# Alignment Framework User Guide

Comprehensive guide to neural network alignment analysis and pruning.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Available Metrics](#available-metrics)
3. [Computing Metrics](#computing-metrics)
4. [Pruning Strategies](#pruning-strategies)
5. [Architecture Support](#architecture-support)
6. [Configuration](#configuration)

## Core Concepts

### Alignment

Alignment measures how well neuron weights align with the structure of their input activations. The Rayleigh Quotient quantifies the proportion of input variance captured by each neuron's weight vector.

For weight vector w and input covariance Σ:
```
RQ(w) = (w^T Σ w) / (w^T w · tr(Σ))
```

Higher RQ values indicate the neuron aligns with dominant input variance directions.

### Class-Conditioned Alignment

Class-conditioned analysis compares alignment across different classes versus overall alignment. The difference (ΔRQ) indicates task-relevant alignment:

```
ΔRQ = RQ(overall) - E[RQ(class-conditioned)]
```

Positive ΔRQ means the neuron captures discriminative features between classes.

### Information Theory

The framework uses information-theoretic measures to quantify relationships between neurons:

- **Mutual Information**: Shared information between two variables
- **Redundancy**: Information overlap between neuron pairs
- **Synergy**: Information that emerges only from joint neuron outputs
- **Partial Information Decomposition (PID)**: Decomposes information into unique, redundant, and synergistic components

## Available Metrics

### Alignment Metrics

**Rayleigh Quotient (RQ)**
- Measures alignment with input covariance structure
- Returns per-neuron scores indicating variance captured
- Supports relative normalization and regularization
- Works with linear, convolutional, and transformer layers

**Class-Conditioned RQ**
- Computes RQ separately for each class
- Returns ΔRQ showing task-relevant alignment
- Requires labeled data

**Spectral Alignment**
- Analyzes eigenvalue decomposition of weight-covariance product
- Provides detailed spectral properties

### Information-Theoretic Metrics

**Mutual Information (MI)**
- Gaussian MI using analytic formula: MI = -0.5 log(1 - ρ²)
- Can compute MI between neuron outputs and targets
- Supports both continuous and categorical targets

**Pairwise Redundancy**
- Computes redundancy between neuron pairs
- Output-based mode for efficiency with large models
- Covariance-based mode for precise estimates
- Sampling strategies: random, nearest, or all pairs
- Returns per-neuron average redundancy scores

**Synergy (MMI)**
- Measures complementary information from neuron pairs
- Uses Minimum Mutual Information redundancy definition
- Returns per-neuron synergy scores
- Requires target labels

**Partial Information Decomposition (PID)**
- Decomposes information into unique, redundant, and synergistic parts
- Shared information: overlap between two neurons
- Unique information: contribution from single neurons
- Synergistic information: emergent from joint outputs

**Higher-Order Metrics**
- Total correlation: generalization of MI to multiple variables
- Interaction information: higher-order dependencies
- Connected information: network-level measures

### Gradient-Based Metrics

**Gradient Alignment**
- Compares gradient-based learning with local learning rules
- Analyzes alignment between backpropagation and Hebbian-style updates
- Useful for biologically-plausible learning analysis

**Local Learning Rule Search**
- Finds optimal local learning rules per neuron
- Searches over combinations of pre/post-synaptic activity
- Returns best-fit local rule for each neuron

### Task-Specific Metrics

**Classification Alignment**
- Measures discriminative power for classification tasks
- Analyzes separation of class representations

**Vision Task Alignment**
- Specialized for vision architectures
- Analyzes feature selectivity patterns

### Similarity Metrics

**Centered Kernel Alignment (CKA)**
- Compares representations between layers or models
- Invariant to isotropic scaling

**Canonical Correlation Analysis (CCA)**
- Finds maximally correlated directions between representations

## Computing Metrics

### Basic Usage

```python
from alignment import ModelWrapper, get_metric

# Wrap model to track layers
wrapper = ModelWrapper(model, tracked_layers=['conv1', 'fc1'])

# Get metric instance
rq_metric = get_metric('rayleigh_quotient')

# Forward pass with activation capture
outputs, acts = wrapper.forward_with_activations(inputs)
weights = wrapper.get_layer_weights()

# Compute per-neuron scores
scores = rq_metric.compute(
    inputs=acts['conv1_input'],
    weights=weights['conv1']
)
```

### Composite Scoring

Combine multiple metrics for robust importance estimation:

```python
from alignment.services import NodeScoringService

scorer = NodeScoringService(
    metrics={
        'rq': get_metric('rayleigh_quotient'),
        'redundancy': get_metric('pairwise_redundancy_gaussian'),
        'synergy': get_metric('synergy_gaussian_mmi')
    },
    alpha_mi=0.0,
    beta_synergy=0.3,
    gamma_redundancy=0.4,
    delta_rq=0.3
)

scores = scorer.compute_composite_scores(inputs, weights, targets)
```

The composite score combines: synergy (positive), redundancy (negative), and alignment.

## Pruning Strategies

### Overview

Pruning removes less important network parameters to reduce model size and computational cost. The framework provides multiple strategies based on different importance criteria.

### Magnitude-Based Strategies

**MagnitudePruning**
- Prunes weights with smallest absolute values
- Fast and simple baseline
- Works for unstructured and structured pruning
- Applies L1 or L2 norm for importance scores

**IterativeMagnitudePruning**
- Applies magnitude pruning in multiple rounds
- Allows network to recover between pruning steps
- Typically achieves better accuracy than one-shot pruning

**GlobalMagnitudePruning**
- Applies single threshold across all layers
- Natural adaptation to layer sensitivity
- Some layers may be pruned more than others

### Gradient-Based Strategies

**GradientPruning**
- Uses gradient magnitude as importance signal
- Can use gradient alone or gradient-weight product (Taylor approximation)
- Requires backward pass to compute gradients

**FisherPruning**
- Uses Fisher information for importance
- Accounts for both gradient and Hessian information
- More accurate but computationally expensive

**MomentumPruning**
- Incorporates momentum in importance computation
- Smooths importance over training iterations

### Alignment-Based Strategies

**AlignmentPruning**
- Uses Rayleigh Quotient or other alignment metrics
- Structured pruning (removes entire neurons/channels)
- Considers input-weight relationships

**GlobalAlignmentPruning**
- Global threshold across layers based on alignment
- Automatically handles layer-wise sensitivity

**HybridPruning**
- Combines magnitude and alignment scores
- Balances simple magnitude with structural alignment

**CascadingAlignmentPruning**
- Prunes layers sequentially
- Recomputes alignment after each layer
- Accounts for pruning effects on downstream layers

### Random Strategies

**RandomPruning**
- Prunes weights randomly
- Baseline for comparison
- Useful for ablation studies

**LayerwiseRandomPruning**
- Random pruning with per-layer amounts

**BernoulliPruning**
- Stochastic pruning with Bernoulli sampling

### Parallel Strategies

**ParallelModePruning**
- Applies multiple pruning modes simultaneously (low/high/random)
- Returns separate masks for each mode
- Useful for comparing ablations

**TensorizedPruning**
- GPU-optimized parallel execution
- Efficient for multiple strategies

**AsyncParallelPruning**
- Asynchronous parallel execution
- For distributed pruning experiments

### Advanced Strategies

**MovementPruning**
- Tracks weight movement during training
- Prunes weights moving toward zero
- Training-aware strategy

**AdaptiveMovementPruning**
- Adapts pruning amount per layer based on movement patterns

### Pruning Distribution

Distribution strategies determine how pruning is allocated across layers.

**Uniform Distribution**
- Same sparsity percentage for all layers
- Simple and predictable
- May not respect layer-specific sensitivity

**Global Threshold**
- Single importance threshold across all layers
- Natural adaptation based on score distributions
- Different layers pruned by different amounts

**Adaptive Sensitivity**
- Per-layer pruning amounts based on sensitivity analysis
- Sensitive layers pruned less
- Robust layers pruned more
- Maintains target overall sparsity

**Importance Weighted**
- Allocates pruning inversely to layer importance
- More important layers retain more parameters

**Size Proportional**
- Pruning amount scales with layer size
- Larger layers can tolerate more pruning

**Cascading**
- Sequential layer-by-layer pruning
- Recomputes scores after each layer

### Pruning Modes

**Low Mode** (default)
- Prunes weights/neurons with lowest importance scores
- Standard pruning approach

**High Mode**
- Prunes weights/neurons with highest importance scores
- Used for ablation studies to test importance hypotheses

**Random Mode**
- Random pruning regardless of scores
- Baseline for comparison

### Structured vs Unstructured

**Unstructured Pruning**
- Prunes individual weights
- Arbitrary sparsity patterns
- Requires sparse tensor support for speedup

**Structured Pruning**
- Prunes entire neurons or channels
- Maintains dense tensor operations
- Immediate speedup without specialized hardware
- Handles dependencies automatically

### Using Configuration Files

Specify pruning in YAML configuration:

```yaml
pruning:
  enabled: true
  strategy: 'composite'
  target_sparsity: 0.7
  distribution: 'adaptive_sensitivity'
  scoring: 'rayleigh_quotient'
  direction: 'low'
  structured: true
  dependency_aware: true
  
  fine_tune:
    enabled: true
    epochs: 20
    learning_rate: 0.0001
  
  composite_weights:
    beta_synergy: 0.3
    gamma_redundancy: 0.4
    delta_rq: 0.3
```

Then run:
```bash
python scripts/run_experiment.py --config configs/my_pruning.yaml
```

## Architecture Support

### Linear Layers (MLPs)

For fully-connected layers, metrics compute per-neuron scores:

```python
layer = model.fc1  # Linear(in_features=784, out_features=256)
scores = rq.compute(inputs, layer.weight)  # Returns [256]
```

Each neuron receives an importance score. Structured pruning removes entire neurons.

### Convolutional Layers (CNNs)

For convolutional layers, metrics operate on channels:

```python
conv = model.conv1  # Conv2d(in_channels=64, out_channels=128, kernel_size=3)
scores = rq.compute(inputs, conv.weight)  # Returns [128]
```

Each output channel receives a score. The framework handles:
- Spatial weight dimensions automatically
- Different conv modes: unfold, patchwise, channel variance
- Dependency propagation between layers

**CNN-Specific Modes:**
- `unfold`: Unfolds spatial dimensions into feature vectors
- `patchwise`: Treats each spatial location separately
- `channel_variance`: Computes variance across spatial dimensions
- `batch_patch_combined`: Combines batch and spatial statistics

### Transformer Layers

For transformer architectures, the framework supports:

**Feed-Forward Networks (FFN)**
```python
ffn_layer = model.layers[0].mlp.up_proj
scores = rq.compute(inputs, ffn_layer.weight)
```

Each FFN neuron receives a score. For LLaMA models, typical FFN dimensions are 11,008 neurons.

**Attention Layers**
```python
# Per-head analysis
wrapper = TransformerWrapper(model, track_per_head=True)

# Per-neuron within attention projections
q_proj = model.layers[0].self_attn.q_proj
scores = rq.compute(inputs, q_proj.weight)
```

Can analyze:
- Query, Key, Value projections separately
- Per-head importance
- Per-neuron importance within projections
- Attention output projections

**Aggregation Options:**
- `sequence_mean`: Average over sequence length
- `token_level`: Per-token analysis

### Large Language Models

The framework provides specialized wrappers for LLMs:

```python
from alignment.models.transformer_enhanced import LLaMAWrapper

wrapper = LLaMAWrapper(
    llama_model,
    track_ffn=True,
    track_attention=True
)
```

Supports analysis and pruning of:
- FFN intermediate layers (up_proj, down_proj)
- Attention projections (q_proj, k_proj, v_proj, o_proj)
- Layer-wise or model-wide analysis

## Configuration

Experiments are configured using YAML files. The framework provides a template with all available options.

### Basic Configuration

```yaml
experiment:
  name: "my_experiment"
  seed: 42
  device: "cuda"
  output_dir: "./results"

model:
  name: "resnet18"
  pretrained: true
  tracked_layers: null  # Auto-detect

dataset:
  name: "cifar10"
  data_path: "./data"
  batch_size: 128

metrics:
  enabled: ['rayleigh_quotient']
```

### Configuration Sections

**experiment**
- Experiment name, random seed, device selection
- Output directory for results

**model**
- Model architecture name or path
- Whether to use pretrained weights
- Checkpoint path for custom models
- Which layers to track for analysis

**dataset**
- Dataset name and path
- Batch size and data loading workers
- Data augmentation settings

**metrics**
- List of metrics to compute
- Per-metric configuration parameters
- Regularization and sampling settings

**pruning**
- Pruning strategy and target sparsity
- Distribution method across layers
- Scoring method for importance
- Structured vs unstructured
- Fine-tuning configuration

**training**
- Training parameters if training from scratch
- Learning rate, optimizer, scheduler
- Metric tracking during training

**advanced**
- Covariance computation methods
- Hook management settings
- Layer-specific configurations

See `configs/template.yaml` for complete documentation of all parameters.

## Testing

Run tests to verify installation and functionality:

```bash
# All tests
pytest tests/

# Specific test modules
pytest tests/unit/metrics/
pytest tests/unit/pruning/

# Scientific correctness validation
pytest tests/unit/metrics/test_scientific_correctness.py
```

## Examples

The `examples/` directory contains scripts demonstrating framework capabilities:

- `01_quick_start.py` - Basic usage
- `02_complete_experiment.py` - Full experiment workflow
- `03_pruning_strategies.py` - Pruning strategy comparison
- `06_redundancy_aware_pruning.py` - Information-theoretic pruning
- `07_mnist_intelligent_pruning.py` - MNIST pruning example
- `08_llama_ffn_pruning.py` - LLM feed-forward pruning
- `09_attention_neuron_vs_head_pruning.py` - Attention layer analysis

