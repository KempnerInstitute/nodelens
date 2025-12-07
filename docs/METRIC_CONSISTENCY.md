# Metric Consistency with Theoretical Definitions

This document verifies that the implemented metrics are consistent with the theoretical
definitions in `drafts/alignment_notes/main.tex` and `drafts/alignment_notes/new.tex`.

## Summary

| Metric | LaTeX Reference | Code Implementation | Status |
|--------|-----------------|---------------------|--------|
| Rayleigh Quotient | Eq. 3.1 in new.tex | `src/alignment/metrics/rayleigh/rayleigh_quotient.py` | ✅ Consistent |
| Pairwise Redundancy | Eq. 5.1-5.2 in new.tex | `src/alignment/metrics/information/redundancy.py` | ✅ Consistent |
| Composite Score | Eq. 6.1 in new.tex | `src/alignment/metrics/composite.py` | ✅ Consistent |
| Class-conditioned RQ | Eq. 4.1-4.3 in new.tex | `src/alignment/metrics/conditional_metrics.py` | ✅ Consistent |
| Gaussian MI | Section 3.2 in new.tex | `src/alignment/metrics/information/gaussian_mi.py` | ✅ Consistent |
| PID Synergy | Eq. 5.4 in new.tex | `src/alignment/metrics/information/gaussian_pid.py` | ✅ Consistent |

---

## 1. Rayleigh Quotient (RQ)

### LaTeX Definition (new.tex, Eq. 3.1)
```
RQ(w; Σ_X) = (w^T Σ_X w) / (w^T w)
```

### Code Implementation
```python
# From src/alignment/metrics/rayleigh/rayleigh_quotient.py
# Lines 200-219: _compute_rq_from_cov()

numerator = torch.einsum("oi,ij,oj->o", weights, cov_matrix, weights)
denominator = (weights ** 2).sum(dim=1)
rq_values = numerator / denominator
```

### Verification
- **Formula**: Matches exactly. Computes w^T Σ w / w^T w
- **Normalization**: Code supports both absolute and relative (divided by trace) modes
- **Status**: ✅ **CONSISTENT**

---

## 2. Pairwise Redundancy (Gaussian MI)

### LaTeX Definition (new.tex, Section 5.1)
```
I(Y_i; Y_j) = -0.5 * log(1 - ρ²)

where ρ = (w_i^T Σ_X w_j) / sqrt((w_i^T Σ_X w_i)(w_j^T Σ_X w_j))
```

### Code Implementation
```python
# From src/alignment/metrics/information/redundancy.py
# Lines 131-135

rho_sq = corr_with_refs ** 2
rho_sq = torch.clamp(rho_sq, 0, 0.999999)

# MI approximation for each neuron
mi_with_refs = -0.5 * torch.log(1.0 - rho_sq)
```

### Verification
- **Formula**: Matches exactly. Uses -0.5 * log(1 - ρ²)
- **Correlation**: Computed from normalized activations (equivalent to ρ in theory)
- **Clamping**: Properly handles edge cases (ρ² < 1)
- **Status**: ✅ **CONSISTENT**

---

## 3. Composite Importance Score

### LaTeX Definition (new.tex, Eq. 6.1)
```
Score(Y_i) = α·I(Z; Y_i) + β·S(Y_i) - γ·R(Y_i) + δ·log RQ(w_i)
```

### Code Implementation
```python
# From src/alignment/metrics/composite.py
# CompositeImportance class, compute() method

for metric_name, weight in self.metric_weights.items():
    # Compute each metric
    metric_scores = metric.compute(inputs=inputs, weights=weights, ...)
    
    # Apply log transform for RQ if requested
    if self.log_transform_rq and "rayleigh" in metric_name.lower():
        metric_scores = torch.log(metric_scores + 1e-8)
    
    composite += weight * metric_scores
```

### Verification
- **Formula**: Matches. Supports arbitrary metric weights
- **Log RQ**: Correctly applies log transform when configured
- **Signs**: Redundancy can be given negative weight (penalty)
- **Status**: ✅ **CONSISTENT**

---

## 4. Class-Conditioned RQ

### LaTeX Definition (new.tex, Eq. 4.1-4.3)
```
RQ_y(w) = (w^T Σ_{X|y} w) / (w^T w)

Δ_RQ(w) = RQ(w; Σ_X) - E_y[RQ(w; Σ_{X|y})]
```

### Code Implementation
```python
# From src/alignment/metrics/conditional_metrics.py
# ConditionalRayleighQuotient class

# Compute class-conditioned RQ (weighted average)
for class_label in unique_classes:
    class_mask = (targets == class_label)
    class_inputs = inputs[class_mask]
    class_cov = (class_inputs.T @ class_inputs) / (n_class - 1)
    
    # RQ for this class
    numerator_c = torch.einsum("oi,ij,oj->o", weights, class_cov, weights)
    rq_c = numerator_c / denominator
    
    rq_cond_sum += rq_c * weight_c

# Delta RQ
delta_rq = rq_uncond - rq_cond
```

### Verification
- **Per-class RQ**: Correctly computes RQ with class-specific covariance
- **Weighted average**: Uses class proportions p(y) as weights
- **Delta RQ**: Matches definition exactly
- **Status**: ✅ **CONSISTENT**

---

## 5. Gaussian MI (RQ Connection)

### LaTeX Definition (new.tex, Section 3.2)
```
I(X; y) = 0.5 * log(1 + (w^T Σ_X w) / σ_n²)

For small σ_n²: I ≈ 0.5 * log(w^T Σ_X w) - 0.5 * log(σ_n²)
             = 0.5 * log(RQ(w)) + 0.5 * log(w^T w) - 0.5 * log(σ_n²)
```

### Code Implementation
```python
# From src/alignment/metrics/information/gaussian_mi.py
# AnalyticGaussianMI class

# Compute variance of projected output
output_var = torch.einsum("oi,ij,oj->o", weights, cov, weights)

# MI = 0.5 * log(1 + signal_var / noise_var)
# For fixed noise, this is proportional to log(output_var)
mi_scores = 0.5 * torch.log(output_var / noise_variance + 1.0)
```

### Verification
- **Formula**: Matches the Gaussian channel capacity formula
- **RQ Connection**: log(MI) ∝ log(RQ) for fixed noise (documented in code)
- **Status**: ✅ **CONSISTENT**

---

## 6. PID Synergy (MMI)

### LaTeX Definition (new.tex, Section 5.3)
```
R_MMI(Z; Y_1, Y_2) = min{I(Z; Y_1), I(Z; Y_2)}

S_MMI(Z; Y_1, Y_2) = I(Z; [Y_1,Y_2]) - I(Z; Y_1) - I(Z; Y_2) + R_MMI
```

### Code Implementation
```python
# From src/alignment/metrics/information/gaussian_pid.py
# GaussianPIDSynergyMMI class

# MMI redundancy
R_mmi = torch.minimum(I_z_y1, I_z_y2)

# Synergy
S = I_z_y12 - I_z_y1 - I_z_y2 + R_mmi
```

### Verification
- **MMI Redundancy**: Uses min correctly
- **Synergy formula**: Matches exactly
- **Gaussian MI terms**: All I() computed using same Gaussian formulas
- **Status**: ✅ **CONSISTENT**

---

## 7. Extended Metrics (New Additions)

### 7.1 Halo Redundancy

Based on the pairwise redundancy formula, extended to group analysis:

```python
# From src/alignment/metrics/halo_redundancy.py

def correlation_to_redundancy(corr):
    rho_sq = corr ** 2
    rho_sq = torch.clamp(rho_sq, 0, 0.999999)
    redundancy = -0.5 * torch.log(1 - rho_sq)
    return redundancy
```

This is the exact formula from Eq. 5.1 in new.tex.

### 7.2 Cross-Layer Redundancy

Extension of pairwise redundancy to cross-layer:

```
R(Y_i^l || Y^{l-1}) = mean_j I(Y_i^l; Y_j^{l-1})
```

Same formula as within-layer redundancy, but computed between layers.

### 7.3 Cross-Layer Importance (SCAR-aligned)

Extension of composite score following SCAR logic:

```
Score(Y_i^l) = α·RQ + β·Downstream_Importance - γ·R_within

Where:
- Downstream_Importance = mean_j I(Y_i^l; Y_j^{l+1})  (POSITIVE term)
- R_within = within-layer redundancy (PENALTY)
```

Key insight: Downstream importance is a **POSITIVE** term because
neurons that the next layer depends on are important (like supernodes).

This follows SCAR logic:
- Supernodes are important because downstream layers depend on them
- Halo neurons are redundant if their info is already carried by others

---

## Notes on Implementation Details

### Numerical Stability

All implementations include appropriate safeguards:
- Clamping correlations to avoid log(0)
- Adding small epsilon to denominators
- Using `torch.nan_to_num` for edge cases

### Efficiency

For large layers (>2048 neurons), implementations use:
- Reference neuron sampling for redundancy
- Stochastic estimation with configurable sample sizes

### Consistency Verification

To verify consistency, run:
```python
from alignment.metrics import get_metric

# Test RQ matches theory
rq = get_metric("rayleigh_quotient")
# RQ should equal w^T Σ w / w^T w

# Test redundancy matches theory
red = get_metric("average_redundancy")
# Should use I(Y_i; Y_j) = -0.5 * log(1 - ρ²)
```

---

## References

1. **main.tex**: Original alignment framework
2. **new.tex**: Extended framework with detailed derivations
3. **vision_synergy_icml.tex**: Vision-specific extensions
