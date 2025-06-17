# Metrics Organization

## Overview

All metrics in this framework measure some form of alignment between neural network components. They are organized by their computational methodology rather than by what they measure, since they all ultimately measure alignment.

## Metric Categories

### 1. Rayleigh Quotient-Based Metrics (`metrics/rayleigh/`)

These metrics use the Rayleigh Quotient formulation to measure how well weight vectors align with the principal components of their inputs.

**Core Formula**: RQ(w) = (w^T C w) / (w^T w)

**Metrics**:
- `RayleighQuotient`: Standard RQ computation
- `PatchWiseRayleighQuotient`: RQ for convolutional layers with patch-wise computation
- `DeltaAlignment`: RQ of weight changes (W_current - W_initial)
- `NormalizedDeltaAlignment`: Scale-invariant version of delta alignment

### 2. Information-Theoretic Metrics (`metrics/information/`)

These metrics use information theory concepts to measure alignment through mutual information, redundancy, and information decomposition.

**Metrics**:
- `MutualInformationGaussian`: MI assuming Gaussian distributions
- `MutualInformationBinning`: MI using histogram binning
- `PartialInformationDecomposition`: PID components (SharedInfo, UniqueInfo, Synergy)
- `AverageRedundancy`: Average redundancy between neurons
- `NodeRedundancy`: Redundancy of individual nodes

### 3. Similarity-Based Metrics (`metrics/similarity/`)

These metrics measure alignment through various similarity measures between weights, activations, or neurons.

**Metrics**:
- `WeightCosineSimilarity`: Cosine similarity between weight vectors
- `ActivationCosineSimilarity`: Cosine similarity between activation patterns
- `NodeCorrelation`: Correlation between node activations
- `WeightDotSimilarity`: Dot product similarity between weights
- `WeightEuclideanDistance`: Euclidean distance between weight vectors

## Design Principles

1. **Method-Based Organization**: Metrics are grouped by how they compute alignment, not what aspect they measure
2. **Consistent Interface**: All metrics implement the same protocol for easy substitution
3. **Composability**: Metrics can be combined and aggregated in various ways
4. **Extensibility**: New metrics can be added to the appropriate category based on their method

## Usage Examples

```python
# Rayleigh Quotient-based
from alignment_refactor.metrics.rayleigh import RayleighQuotient
rq = RayleighQuotient(relative=True)
scores = rq.compute(inputs=X, weights=W)

# Information-theoretic
from alignment_refactor.metrics.information import MutualInformationGaussian
mi = MutualInformationGaussian()
scores = mi.compute(inputs=X, outputs=Y)

# Similarity-based
from alignment_refactor.metrics.similarity import WeightCosineSimilarity
cos_sim = WeightCosineSimilarity()
scores = cos_sim.compute(weights=W)
```

## Adding New Metrics

When adding a new metric, consider:
1. What is the primary computational method? (This determines the folder)
2. Does it extend an existing metric? (Consider inheritance)
3. What data does it require? (inputs, weights, outputs)
4. Can it benefit from distributed computation?

Place the metric in the appropriate folder based on its primary computational approach. 