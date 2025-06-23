# Alignment Metrics Reference

This document provides a comprehensive reference for all metrics available in the alignment framework, including mathematical derivations and usage examples.

> **Note**: For detailed information about how each metric is computed in practice, including probability estimation methods and numerical considerations, see [METRICS_IMPLEMENTATION_DETAILS.md](METRICS_IMPLEMENTATION_DETAILS.md).

## Table of Contents

1. [Rayleigh Quotient Metrics](#rayleigh-quotient-metrics)
2. [Information-Theoretic Metrics](#information-theoretic-metrics)
3. [Similarity Metrics](#similarity-metrics)
4. [Spectral Metrics](#spectral-metrics)
5. [Task-Specific Metrics](#task-specific-metrics)
6. [Higher-Order Information Metrics](#higher-order-information-metrics)

## Overview

The alignment framework provides 36 metrics organized into categories based on their mathematical properties and intended use cases. Each metric implements the `BaseMetric` interface and can be accessed through the metric registry.

## Rayleigh Quotient Metrics

### 1. Rayleigh Quotient (rayleigh_quotient)

**Mathematical Definition:**

For a weight vector **w** and input covariance matrix **C**:

```
RQ(w) = (w^T C w) / (w^T w)
```

**Derivation:**

The Rayleigh quotient measures how much variance in the input space is captured by a neuron's weight vector. It is the ratio of the quadratic form to the norm of the weight vector.

**Usage:**
```python
metric = get_metric("rayleigh_quotient")(relative=True)
scores = metric.compute(inputs=inputs, weights=weights)
```

### 2. Rayleigh Quotient Alternative (rq_alternative)

**Mathematical Definition:**

```
RQ_alt(w) = (w^T C w) / tr(C)
```

**Derivation:**

This alternative formulation normalizes by the trace of the covariance matrix instead of the weight norm, providing a different perspective on variance capture that can be more stable in some cases.

### 3. Patchwise Rayleigh Quotient (rq_patchwise)

**Mathematical Definition:**

For convolutional layers with patch-based processing:

```
RQ_patch(w) = (1/N) Σ_i RQ(w_i)
```

where w_i represents the weights for patch i.

## Information-Theoretic Metrics

### 1. Mutual Information - Gaussian (mutual_information_gaussian)

**Mathematical Definition:**

For Gaussian-distributed variables X and Y:

```
I(X;Y) = (1/2) log(det(Σ_X) * det(Σ_Y) / det(Σ_XY))
```

where Σ_X, Σ_Y are marginal covariances and Σ_XY is the joint covariance.

**Derivation:**

For Gaussian distributions, mutual information has a closed-form solution based on the ratio of determinants of covariance matrices.

### 2. Mutual Information - Binning (mutual_information_binning)

**Mathematical Definition:**

```
I(X;Y) = Σ_i,j p(x_i, y_j) log(p(x_i, y_j) / (p(x_i) * p(y_j)))
```

**Derivation:**

This non-parametric estimator discretizes continuous variables into bins and estimates probabilities from histograms.

### 3. Gaussian MI with Edgeworth Expansion (gaussian_mi_analytic)

**Mathematical Definition:**

```
I(X;Y) = I_Gaussian(X;Y) + Σ_k E_k(X,Y)
```

where E_k represents k-th order Edgeworth corrections.

**First-order correction (skewness):**
```
E_1 = (1/6) * ρ * γ_1^X * γ_1^Y
```

where γ_1 = κ_3/σ^3 is the normalized third cumulant.

**Second-order correction (kurtosis):**
```
E_2 = (1/24) * ρ^2 * (γ_2^X + γ_2^Y) + (1/72) * (γ_1^X)^2 + (γ_1^Y)^2
```

where γ_2 = κ_4/σ^4 is the normalized fourth cumulant.

### 4. Conditional Mutual Information (conditional_mutual_information)

**Mathematical Definition:**

```
I(X;Y|Z) = Σ_x,y,z p(x,y,z) log(p(x,y|z) / (p(x|z) * p(y|z)))
```

**Derivation:**

Measures the mutual information between X and Y given knowledge of Z, capturing conditional dependencies.

### 5. Average Redundancy (average_redundancy)

**Mathematical Definition:**

```
R_avg = (1/N(N-1)) Σ_i≠j I(X_i; X_j)
```

**Derivation:**

Computes the average pairwise mutual information between neurons, measuring redundancy in representations.

### 6. Layer Redundancy (layer_redundancy)

**Mathematical Definition:**

Similar to average redundancy but computed across entire layers rather than individual neurons.

### 7. MI Projection vs Mean Input (mi_projection)

**Mathematical Definition:**

```
I_proj(W, X) = I(W^T X; μ_X)
```

where μ_X is the mean input vector.

## Partial Information Decomposition (PID) Metrics

### 1. Shared Information (pid_shared)

**Mathematical Definition:**

Using the BROJA framework:

```
I_shared(X1, X2 → Y) = min(I(X1→Y), I(X2→Y))
```

**Derivation:**

Represents information about Y that is redundantly provided by both X1 and X2.

### 2. Unique Information X (pid_unique_x)

**Mathematical Definition:**

```
I_unique(X1→Y|X2) = I(X1→Y) - I_shared(X1,X2→Y)
```

**Derivation:**

Information that X1 provides about Y that X2 does not provide.

### 3. Unique Information Y (pid_unique_y)

**Mathematical Definition:**

```
I_unique(X2→Y|X1) = I(X2→Y) - I_shared(X1,X2→Y)
```

### 4. Synergistic Information (pid_synergy)

**Mathematical Definition:**

```
I_syn(X1,X2→Y) = I(X1,X2→Y) - I_unique(X1→Y) - I_unique(X2→Y) - I_shared(X1,X2→Y)
```

**Derivation:**

Information about Y that can only be obtained by considering X1 and X2 together.

## Similarity Metrics

### 1. Activation Cosine Similarity (activation_cosine_similarity)

**Mathematical Definition:**

```
cos(a, b) = (a · b) / (||a|| * ||b||)
```

**Derivation:**

Measures the cosine of the angle between activation vectors.

### 2. Weight Cosine Similarity (weight_cosine_similarity)

**Mathematical Definition:**

Similar to activation cosine similarity but applied to weight vectors:

```
cos(w1, w2) = (w1 · w2) / (||w1|| * ||w2||)
```

### 3. Weight Dot Product Similarity (weight_dot_similarity)

**Mathematical Definition:**

```
sim(w1, w2) = w1 · w2
```

### 4. Weight Euclidean Distance (weight_euclidean_distance)

**Mathematical Definition:**

```
d(w1, w2) = ||w1 - w2||_2
```

### 5. Node Correlation (node_correlation)

**Mathematical Definition:**

```
corr(x, y) = cov(x, y) / (σ_x * σ_y)
```

**Derivation:**

Pearson correlation coefficient between neuron activations.

### 6. Node Redundancy (node_redundancy)

**Mathematical Definition:**

Measures redundancy between input features using correlation:

```
R = (1/N^2) Σ_i,j |corr(x_i, x_j)|
```

### 7. Weight Activation Alignment (weight_activation_alignment)

**Mathematical Definition:**

```
align(w, a) = cos(w, a) = (w · a) / (||w|| * ||a||)
```

**Derivation:**

Measures how aligned weight vectors are with activation patterns.

## Spectral Metrics

### 1. Spectral Gap (spectral_gap)

**Mathematical Definition:**

```
gap = (λ_1 - λ_2) / λ_1
```

where λ_1, λ_2 are the largest and second-largest eigenvalues.

**Derivation:**

Measures the dominance of the principal component in the weight matrix.

### 2. Spectral Norm Ratio (spectral_norm_ratio)

**Mathematical Definition:**

```
ratio = σ_max / σ_F
```

where σ_max is the largest singular value and σ_F is the Frobenius norm.

### 3. Eigenvalue Entropy (eigenvalue_entropy)

**Mathematical Definition:**

```
H(λ) = -Σ_i p_i log(p_i)
```

where p_i = λ_i / Σ_j λ_j are normalized eigenvalues.

**Derivation:**

Measures the distribution of variance across eigenmodes.

### 4. Spectral Clustering Score (spectral_clustering_score)

**Mathematical Definition:**

Based on spectral clustering quality metrics, measuring how well eigenspaces separate data.

### 5. Eigenvalue Alignment (eigenvalue_alignment)

**Mathematical Definition:**

Uses Wasserstein distance between eigenvalue distributions:

```
W_p(λ_1, λ_2) = (Σ_i |λ_1^(i) - λ_2^(i)|^p)^(1/p)
```

### 6. Spectral Clustering Alignment (spectral_clustering)

**Mathematical Definition:**

Measures alignment between weight eigenspaces and data clustering structure.

### 7. Power Iteration Convergence (power_iteration)

**Mathematical Definition:**

```
rate = 1 / iterations_to_convergence
```

**Derivation:**

Faster convergence indicates clearer spectral structure.

### 8. Spectral Alignment (spectral_alignment)

**Mathematical Definition:**

General spectral alignment combining multiple spectral properties.

## Task-Specific Metrics

### General Task Metrics

#### 1. Task Alignment (task_alignment)

**Mathematical Definition:**

For gradient-based alignment:

```
align(w, g) = |w · ∇_x L|
```

where L is the task loss and g are input gradients.

**Derivation:**

Measures how neuron weights align with task-relevant gradients.

#### 2. Class Selectivity (class_selectivity)

**Mathematical Definition:**

Using Fisher's discriminant ratio:

```
S = σ_between^2 / σ_within^2
```

**Derivation:**

Ratio of between-class to within-class variance.

#### 3. Feature Importance (feature_importance)

**Mathematical Definition:**

For permutation importance:

```
I_i = E[L(f(X_perm_i)) - L(f(X))]
```

where X_perm_i has feature i permuted.

#### 4. Representation Quality (representation_quality)

**Mathematical Definition:**

Using linear probe accuracy:

```
Q = R^2 = 1 - SS_res / SS_tot
```

### Domain-Specific Metrics

#### 1. Classification Alignment (classification_alignment)

**Mathematical Definition:**

Measures alignment with decision boundaries using entropy:

```
H = -Σ_i p_i log(p_i)
```

High entropy indicates proximity to decision boundaries.

#### 2. Language Model Alignment (language_model_alignment)

**Mathematical Definition:**

For next-token prediction:

```
align = -E[log P(x_t+1 | h_t)]
```

where h_t are hidden states.

#### 3. Vision Task Alignment (vision_task_alignment)

**Mathematical Definition:**

For spatial coherence:

```
C = corr(f(x), f(shift(x)))
```

Measures correlation with spatially shifted versions.

#### 4. Reinforcement Learning Alignment (reinforcement_learning_alignment)

**Mathematical Definition:**

For value function alignment:

```
align = corr(h, V(s))
```

where h are hidden states and V(s) is the value function.

## Higher-Order Information Metrics

### 1. Total Correlation (total_correlation)

**Mathematical Definition:**

```
TC(X_1, ..., X_n) = Σ_i H(X_i) - H(X_1, ..., X_n)
```

**Derivation:**

KL divergence between joint distribution and product of marginals.

### 2. Interaction Information (interaction_information)

**Mathematical Definition:**

For three variables:

```
I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
```

**Derivation:**

Measures information present only when all variables are considered together.

### 3. Connected Information (connected_information)

**Mathematical Definition:**

Using inclusion-exclusion principle for n-way interactions.

### 4. Synergistic Information (synergistic_information)

**Mathematical Definition:**

For a group of neurons:

```
Syn = H_joint - Σ_i H_i
```

under Gaussian assumption.

## Usage Examples

### Basic Usage

```python
from src.alignment.core.registry import METRIC_REGISTRY

# Get a metric
metric = METRIC_REGISTRY.get("rayleigh_quotient")()

# Compute scores
scores = metric.compute(inputs=inputs, weights=weights)
```

### Advanced Usage

```python
# Configure metric with parameters
metric = METRIC_REGISTRY.get("gaussian_mi_analytic")(
    expansion_order=2,
    noise_std=0.1,
    per_neuron=True
)

# Compute with additional arguments
scores = metric.compute(
    inputs=inputs,
    weights=weights,
    outputs=outputs
)
```

## Metric Properties

Each metric has properties indicating its requirements:
- `requires_inputs`: Whether input activations are needed
- `requires_weights`: Whether weight matrices are needed  
- `requires_outputs`: Whether output activations are needed

## Performance Considerations

1. **Computational Complexity:**
   - Binning-based metrics: O(n log n) for sorting
   - Gaussian metrics: O(d^3) for matrix operations
   - Spectral metrics: O(n^3) for eigendecomposition

2. **Memory Requirements:**
   - Most metrics: O(n*d) for storing activations
   - Higher-order metrics: O(n^k) for k-way interactions

3. **Numerical Stability:**
   - Use regularization for matrix inversions
   - Add small constants to avoid log(0)
   - Check for NaN/Inf values

## References

1. Rayleigh Quotient: Horn, R. A., & Johnson, C. R. (2012). Matrix analysis.
2. Mutual Information: Cover, T. M., & Thomas, J. A. (2006). Elements of information theory.
3. PID: Bertschinger, N., Rauh, J., Olbrich, E., Jost, J., & Ay, N. (2014). Quantifying unique information.
4. Spectral Methods: Golub, G. H., & Van Loan, C. F. (2013). Matrix computations.
5. Edgeworth Expansion: McCullagh, P. (1987). Tensor methods in statistics. 