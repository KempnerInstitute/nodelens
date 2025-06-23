# Metrics Implementation Details

This document provides detailed information about how each metric is computed in practice, including estimation methods and numerical considerations.

## Information-Theoretic Metrics

### Mutual Information - Binning Method

**Implementation Details:**
- **Joint Probability Estimation**: Uses histogram binning with configurable number of bins (default: 10)
- **Binning Strategy**: Equal-width bins across the data range
- **Probability Calculation**: 
  ```python
  # Pseudo-code
  hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
  p_xy = hist_2d / hist_2d.sum()  # Joint probability
  p_x = p_xy.sum(axis=1)  # Marginal x
  p_y = p_xy.sum(axis=0)  # Marginal y
  ```
- **Smoothing**: Adds small epsilon (1e-10) to avoid log(0)
- **Edge Cases**: Returns 0 for constant inputs

### Mutual Information - Gaussian Method

**Implementation Details:**
- **Covariance Estimation**: 
  ```python
  cov_x = torch.cov(x.T)  # Marginal covariance of X
  cov_y = torch.cov(y.T)  # Marginal covariance of Y
  cov_joint = torch.cov(torch.cat([x, y], dim=1).T)  # Joint covariance
  ```
- **Regularization**: Adds small diagonal term (1e-6 * I) for numerical stability
- **Formula Application**:
  ```python
  MI = 0.5 * (log(det(cov_x)) + log(det(cov_y)) - log(det(cov_joint)))
  ```
- **Assumptions**: Data is assumed to be multivariate Gaussian

### Gaussian MI with Edgeworth Expansion

**Implementation Details:**
- **Cumulant Calculation**:
  ```python
  # Third cumulant (skewness)
  mean_centered = x - x.mean(dim=0)
  kappa_3 = (mean_centered ** 3).mean(dim=0)
  gamma_1 = kappa_3 / (x.std(dim=0) ** 3)
  
  # Fourth cumulant (kurtosis)
  kappa_4 = (mean_centered ** 4).mean(dim=0) - 3 * (x.std(dim=0) ** 4)
  gamma_2 = kappa_4 / (x.std(dim=0) ** 4)
  ```
- **Expansion Terms**:
  - Order 0: Gaussian MI only
  - Order 1: Adds skewness correction
  - Order 2: Adds kurtosis and mixed terms
  - Order 3: Adds higher-order cross terms
- **Numerical Safeguards**: Clamps corrections to prevent negative MI

### Conditional Mutual Information

**Implementation Details:**
- **3D Histogram**: Creates joint histogram for (X, Y, Z)
- **Conditional Probability**:
  ```python
  p_xy_given_z = p_xyz / p_z  # For each z bin
  p_x_given_z = p_xz / p_z
  p_y_given_z = p_yz / p_z
  ```
- **Summation**: Weighted sum over all Z bins

## Rayleigh Quotient Metrics

### Standard Rayleigh Quotient

**Implementation Details:**
- **Covariance Computation**:
  ```python
  # Input covariance
  C = torch.cov(inputs.T) if inputs.shape[0] > 1 else torch.eye(d)
  
  # For each weight vector w
  RQ = (w @ C @ w) / (w @ w)
  ```
- **Batched Computation**: Processes multiple neurons simultaneously
- **Relative Mode**: Divides by mean eigenvalue of C for scale invariance

### Patchwise Rayleigh Quotient

**Implementation Details:**
- **Patch Extraction**: Uses unfold operation for convolutional weights
- **Local Covariance**: Computes covariance for each spatial location
- **Aggregation**: Averages RQ values across all patches

## Spectral Metrics

### Spectral Gap

**Implementation Details:**
- **SVD Computation**: Uses `torch.svd()` for stability
- **Gap Calculation**:
  ```python
  U, S, V = torch.svd(weights)
  gap = (S[0] - S[1]) / S[0] if normalized else S[0] - S[1]
  ```
- **Dimension Handling**: 
  - Conv weights: Reshape to 2D (out_channels, in_features)
  - 3D tensors: Average over batch dimension

### Eigenvalue Entropy

**Implementation Details:**
- **Eigenvalue Extraction**: Via SVD for numerical stability
- **Normalization**: 
  ```python
  eigenvalues = S ** 2  # Squared singular values
  p_i = eigenvalues / eigenvalues.sum()
  ```
- **Entropy**: Shannon entropy with base-2 logarithm
- **Regularization**: Adds 1e-10 to eigenvalues before log

### Power Iteration

**Implementation Details:**
- **Initialization**: Random unit vector
- **Iteration**:
  ```python
  for i in range(max_iter):
      v = weights @ v
      v = v / torch.norm(v)
      if torch.norm(v - v_prev) < tol:
          break
  ```
- **Convergence Rate**: 1 / number_of_iterations

## Similarity Metrics

### Cosine Similarity

**Implementation Details:**
- **Normalization**: L2 norm for each vector
- **Computation**:
  ```python
  cos_sim = (a @ b) / (torch.norm(a) * torch.norm(b))
  ```
- **Batch Processing**: Handles multiple vector pairs
- **Numerical Stability**: Clamps denominators at 1e-8

### Node Correlation

**Implementation Details:**
- **Pearson Correlation**:
  ```python
  # Center the data
  x_centered = x - x.mean()
  y_centered = y - y.mean()
  
  # Correlation
  corr = (x_centered @ y_centered) / (x.std() * y.std() * len(x))
  ```
- **Absolute Mode**: Returns |corr| when specified

## Task-Specific Metrics

### Class Selectivity

**Implementation Details:**
- **Between-Class Variance**:
  ```python
  class_means = [outputs[labels == c].mean(dim=0) for c in classes]
  grand_mean = outputs.mean(dim=0)
  var_between = sum(n_c * (mu_c - grand_mean)**2) / n_total
  ```
- **Within-Class Variance**:
  ```python
  var_within = sum((outputs[labels == c] - mu_c)**2).sum() / n_total
  ```
- **Fisher Ratio**: var_between / (var_within + epsilon)

### Feature Importance

**Implementation Details:**
- **Gradient-Based**:
  ```python
  gradients = torch.autograd.grad(loss, inputs)[0]
  importance = (gradients * inputs).abs().mean(dim=0)
  ```
- **Permutation-Based**:
  ```python
  baseline_loss = compute_loss(model, inputs)
  perm_inputs = inputs.clone()
  perm_inputs[:, i] = perm_inputs[torch.randperm(n), i]
  perm_loss = compute_loss(model, perm_inputs)
  importance[i] = perm_loss - baseline_loss
  ```

### Classification Alignment

**Implementation Details:**
- **Entropy Calculation**:
  ```python
  probs = F.softmax(outputs, dim=-1)
  entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
  ```
- **Per-Neuron Analysis**: Computes entropy for each neuron's contribution

## Higher-Order Information Metrics

### Total Correlation

**Implementation Details:**
- **Entropy Estimation**: Uses binning for marginal and joint entropies
- **Computation**:
  ```python
  TC = sum(H(X_i)) - H(X_1, ..., X_n)
  ```
- **Dimensionality**: Limited to 10 variables for computational feasibility

### Interaction Information

**Implementation Details:**
- **Three-Way Interaction**:
  ```python
  I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
  ```
- **Sign**: Can be negative (indicating redundancy)

## Partial Information Decomposition

### BROJA Framework

**Implementation Details:**
- **When BROJA_2PID Available**:
  - Uses optimization-based approach
  - Computes exact PID decomposition
  - Handles continuous variables via binning
  
- **Fallback Implementation**:
  ```python
  # Simplified approximation
  I_shared = min(I(X1; Y), I(X2; Y))
  I_unique_X1 = I(X1; Y) - I_shared
  I_unique_X2 = I(X2; Y) - I_shared
  I_synergy = I(X1, X2; Y) - I(X1; Y) - I(X2; Y) + I_shared
  ```

## Performance Optimizations

### Batch Processing
- Most metrics support batched computation across neurons
- Uses vectorized operations where possible
- GPU acceleration via PyTorch operations

### Memory Management
- Chunked processing for large datasets
- In-place operations where safe
- Careful tensor allocation

### Numerical Stability
- Regularization terms in matrix operations
- Epsilon values for logarithms and divisions
- Gradient clipping for optimization-based methods

## Configuration Parameters

### Common Parameters
- `device`: CPU or CUDA device selection
- `dtype`: Float precision (default: float32)
- `batch_size`: For chunked processing

### Metric-Specific Parameters
- **Binning Metrics**: `n_bins` (default: 10)
- **Spectral Metrics**: `top_k` eigenvalues
- **Iterative Methods**: `max_iterations`, `tolerance`
- **Regularization**: `epsilon`, `regularization_strength` 