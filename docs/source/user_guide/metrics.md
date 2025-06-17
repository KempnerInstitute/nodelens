# Metrics Guide

This guide provides detailed information about the alignment and information-theoretic metrics available in the framework.

## Overview

The framework implements various metrics to analyze neural network behavior:

- **Alignment Metrics**: Rayleigh Quotient (RQ), Generalized Rayleigh Quotient
- **Information Theory**: Mutual Information (MI), Partial Information Decomposition (PID)
- **Similarity Metrics**: Centered Kernel Alignment (CKA), Canonical Correlation Analysis (CCA)
- **Statistical Measures**: Procrustes distance, correlation measures

## Rayleigh Quotient (RQ)

The Rayleigh Quotient measures how well aligned a neuron's weight vector is with the principal components of its input distribution.

### Mathematical Definition

For a weight vector **w** and input covariance matrix **C**:

```
RQ(w) = (w^T C w) / (w^T w)
```

### Usage

```python
from alignment_refactor.metrics import RayleighQuotient

# Create metric instance
rq_metric = RayleighQuotient(
    scale_by_norm=False,  # Whether to scale by weight norm
    aggregation_op="mean"  # How to aggregate for CNN layers
)

# Compute scores
rq_scores = rq_metric.compute(
    inputs=layer_inputs,  # Shape: (batch_size, input_dim)
    weights=layer_weights  # Shape: (output_dim, input_dim)
)
# Returns: tensor of shape (output_dim,) - one score per neuron
```

### Parameters

- `scale_by_norm`: If True, divides RQ by weight norm for scale invariance
- `aggregation_op`: For CNNs - "mean", "max", "sum" over spatial dimensions
- `force_cpu`: Move computation to CPU for large matrices

### Interpretation

- **High RQ**: Neuron aligns with high-variance input directions (important)
- **Low RQ**: Neuron aligns with low-variance directions (potentially redundant)

## Mutual Information (MI)

Mutual Information quantifies the amount of information shared between inputs and outputs.

### Usage

```python
from alignment_refactor.metrics import MutualInformationGaussian

# Create metric instance
mi_metric = MutualInformationGaussian(
    estimation_method="gaussian",  # Estimation method
    num_samples=1000  # Number of samples for estimation
)

# Compute MI
mi_scores = mi_metric.compute(
    inputs=layer_inputs,
    weights=layer_weights,
    outputs=layer_outputs  # Required for MI
)
```

### Estimation Methods

1. **Gaussian**: Assumes Gaussian distributions (fast, approximate)
2. **KNN**: k-nearest neighbors estimator (more accurate, slower)
3. **Binning**: Histogram-based estimation (simple, requires discretization)

## Partial Information Decomposition (PID)

PID decomposes the information that multiple inputs provide about an output into unique, redundant, and synergistic components.

### Usage

```python
from alignment_refactor.metrics import PartialInformationDecomposition

# Create metric instance
pid_metric = PartialInformationDecomposition(
    method="broja",  # PID estimation method
    max_variables=100  # Maximum variables to consider
)

# Compute PID components
pid_results = pid_metric.compute(
    inputs=layer_inputs,
    weights=layer_weights,
    outputs=layer_outputs
)

# Results contain:
# - unique_information: Information unique to each input
# - redundant_information: Information shared by inputs
# - synergistic_information: Information only available from combination
```

### Methods

- `"broja"`: BROJA estimator (recommended)
- `"barrett"`: Barrett's Gaussian PID
- `"williams"`: Williams & Beer framework

## Centered Kernel Alignment (CKA)

CKA measures the similarity between representations in different layers or networks.

### Usage

```python
from alignment_refactor.metrics import CKA

# Create metric instance
cka_metric = CKA(
    kernel="linear",  # Kernel type
    threshold=0.01  # Eigenvalue threshold
)

# Compare two representations
similarity = cka_metric.compute(
    X=representation1,  # Shape: (n_samples, n_features1)
    Y=representation2   # Shape: (n_samples, n_features2)
)
# Returns: scalar similarity score in [0, 1]
```

### Kernel Options

- `"linear"`: Linear kernel (fast, captures linear relationships)
- `"rbf"`: RBF/Gaussian kernel (captures non-linear relationships)

## Canonical Correlation Analysis (CCA)

CCA finds linear transformations that maximize correlation between two sets of variables.

### Usage

```python
from alignment_refactor.metrics import CCA

# Create metric instance
cca_metric = CCA(
    n_components=50,  # Number of canonical components
    reg=1e-3  # Regularization parameter
)

# Compute CCA similarity
similarity = cca_metric.compute(
    X=representation1,
    Y=representation2
)
```

## Generalized Rayleigh Quotient

Extension of RQ for comparing two covariance structures.

### Usage

```python
from alignment_refactor.metrics import GeneralizedRayleighQuotient

grq_metric = GeneralizedRayleighQuotient()

# Compute GRQ
grq_scores = grq_metric.compute(
    inputs=layer_inputs,
    weights=layer_weights,
    reference_cov=reference_covariance  # Optional reference covariance
)
```

## Shared Information Metrics

Measures information shared between different neurons or layers.

### Usage

```python
from alignment_refactor.metrics import SharedInformation

shared_info = SharedInformation(
    method="correlation"  # Method for measuring sharing
)

scores = shared_info.compute(
    activations1=layer1_activations,
    activations2=layer2_activations
)
```

## Using Multiple Metrics

### Batch Computation

```python
from alignment_refactor.metrics import MetricCollection

# Create collection of metrics
metrics = MetricCollection([
    RayleighQuotient(),
    MutualInformationGaussian(),
    CKA()
])

# Compute all metrics
results = metrics.compute_all(
    inputs=inputs,
    weights=weights,
    outputs=outputs
)
# Returns: dict with metric names as keys
```

### In Experiments

```python
config = ExperimentConfig(
    metrics=["rayleigh_quotient", "mutual_information", "cka"],
    metric_configs={
        "rayleigh_quotient": {"scale_by_norm": True},
        "mutual_information": {"estimation_method": "knn"},
        "cka": {"kernel": "rbf"}
    }
)
```

## Performance Considerations

### Memory Usage

Some metrics require large covariance matrices:
- Use `force_cpu=True` for large layers
- Consider batch processing for very large datasets
- Use approximation methods when exact computation is infeasible

### Computation Time

Relative computation costs:
1. **Fast**: RQ, linear CKA
2. **Medium**: Gaussian MI, CCA
3. **Slow**: KNN-based MI, PID, RBF CKA

### GPU Acceleration

Most metrics support GPU computation:
```python
# Ensure inputs are on GPU
inputs = inputs.cuda()
weights = weights.cuda()

# Metric computation will use GPU
scores = metric.compute(inputs, weights)
```

## Custom Metrics

### Creating a Custom Metric

```python
from alignment_refactor.metrics.base import BaseMetric

class MyCustomMetric(BaseMetric):
    def __init__(self, parameter=1.0):
        super().__init__(name="my_custom_metric")
        self.parameter = parameter
    
    def compute(self, inputs, weights, outputs=None):
        # Your metric computation
        scores = custom_computation(inputs, weights, self.parameter)
        return scores
```

### Registering Custom Metrics

```python
from alignment_refactor.core.registry import register_metric

@register_metric("my_metric")
class MyMetric(BaseMetric):
    # Implementation
    pass
```

## Metric Selection Guidelines

### For Pruning

1. **RQ**: Best for identifying neurons aligned with input variance
2. **MI**: Good for information-based pruning
3. **Magnitude**: Simple baseline

### For Layer Analysis

1. **CKA**: Compare representations across layers
2. **CCA**: Linear similarity between layers
3. **Shared Information**: Information flow analysis

### For Network Comparison

1. **CKA**: Compare different networks' representations
2. **Procrustes**: Geometric similarity
3. **PID**: Information decomposition

## Troubleshooting

### Numerical Issues

```python
# Add regularization for stability
rq_metric = RayleighQuotient(
    epsilon=1e-6  # Regularization term
)

# Use double precision for sensitive computations
inputs = inputs.double()
weights = weights.double()
```

### Memory Errors

```python
# Force CPU computation
metric = RayleighQuotient(force_cpu=True)

# Process in smaller batches
batch_size = 100
results = []
for i in range(0, len(inputs), batch_size):
    batch_result = metric.compute(
        inputs[i:i+batch_size],
        weights
    )
    results.append(batch_result)
```

### Interpretation Issues

- Normalize metrics for comparison across layers
- Consider layer-specific baselines
- Use multiple metrics for robust conclusions 