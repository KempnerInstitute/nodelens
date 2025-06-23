# Complete List of Alignment Metrics

This document provides a complete list of all 36 metrics available in the alignment framework, organized by category.

## Total: 36 Metrics

### Rayleigh Quotient Metrics (3)

1. **rayleigh_quotient**: Standard Rayleigh quotient measuring variance capture
   - Formula: RQ(w) = (w^T C w) / (w^T w)
   
2. **rq_patchwise**: Patch-based RQ for convolutional layers
   - Formula: RQ_patch = mean(RQ(w_i)) for patches i

### Information-Theoretic Metrics (10)

3. **mutual_information_gaussian**: MI assuming Gaussian distributions
   - Formula: I(X;Y) = 0.5 * log(det(Σ_X)det(Σ_Y)/det(Σ_XY))
   
4. **mutual_information_binning**: Non-parametric MI using histogram binning
   - Formula: I(X;Y) = Σ p(x,y)log(p(x,y)/(p(x)p(y)))
   
5. **gaussian_mi_analytic**: Gaussian MI with Edgeworth expansion corrections
   - Formula: I(X;Y) = I_Gaussian + Σ E_k (order k corrections)
   
6. **conditional_mutual_information**: MI conditioned on third variable
   - Formula: I(X;Y|Z) = Σ p(x,y,z)log(p(x,y|z)/(p(x|z)p(y|z)))
   
7. **average_redundancy**: Average pairwise MI between neurons
   - Formula: R_avg = mean(I(X_i;X_j)) for i≠j
   
8. **layer_redundancy**: Redundancy computed at layer level
   - Formula: Similar to average_redundancy but for entire layers

### Partial Information Decomposition (4)

9. **pid_shared**: Redundant information from BROJA framework
   - Formula: I_shared = min(I(X1→Y), I(X2→Y))
   
10. **pid_unique_x**: Unique information from X1
    - Formula: I_unique(X1) = I(X1→Y) - I_shared
    
11. **pid_unique_y**: Unique information from X2
    - Formula: I_unique(X2) = I(X2→Y) - I_shared
    
12. **pid_synergy**: Synergistic information
    - Formula: I_syn = I(X1,X2→Y) - I_unique(X1) - I_unique(X2) - I_shared

### Similarity Metrics (4)

13. **activation_cosine_similarity**: Cosine similarity between activations
    - Formula: cos(a,b) = (a·b)/(||a|| ||b||)
    
14. **weight_cosine_similarity**: Cosine similarity between weight vectors
    - Formula: cos(w1,w2) = (w1·w2)/(||w1|| ||w2||)
    
15. **node_redundancy**: Redundancy between input features
    - Formula: R = mean(|corr(x_i,x_j)|)
    
16. **weight_activation_alignment**: Alignment between weights and activations
    - Formula: align(w,a) = cos(w,a)

### Spectral Metrics (9)

17. **spectral_gap**: Gap between largest eigenvalues
    - Formula: gap = (λ_1 - λ_2)/λ_1
    
18. **spectral_norm_ratio**: Ratio of spectral to Frobenius norm
    - Formula: ratio = σ_max/||W||_F
    
19. **eigenvalue_entropy**: Entropy of eigenvalue distribution
    - Formula: H(λ) = -Σ p_i log(p_i), p_i = λ_i/Σλ_j
    
20. **spectral_clustering_score**: Quality of spectral clustering
    - Based on eigenspace separation metrics
    
21. **eigenvalue_alignment**: Wasserstein distance between eigenvalue distributions
    - Formula: W_p(λ_1,λ_2) = (Σ|λ_1^i - λ_2^i|^p)^(1/p)
    
22. **spectral_clustering**: Alignment between eigenspaces and data clusters
    - Measures projection variance ratio
    
23. **power_iteration**: Convergence rate of power iteration
    - Formula: rate = 1/iterations_to_converge
    
24. **spectral_alignment**: General spectral alignment metric
    - Combines multiple spectral properties

### Task-Specific Metrics (8)

#### General Task Metrics (4)

25. **task_alignment**: Alignment with task-specific gradients
    - Formula: align(w,g) = |w·∇_x L|
    
26. **class_selectivity**: Fisher discriminant ratio for classification
    - Formula: S = σ_between²/σ_within²
    
27. **feature_importance**: Permutation or gradient-based importance
    - Formula: I_i = E[L(f(X_perm_i)) - L(f(X))]
    
28. **representation_quality**: Quality via linear probe accuracy
    - Formula: Q = R² = 1 - SS_res/SS_tot

#### Domain-Specific Metrics (4)

29. **classification_alignment**: Alignment with decision boundaries
    - Uses entropy near boundaries
    
30. **language_model_alignment**: Alignment for language modeling
    - Next-token prediction accuracy
    
31. **vision_task_alignment**: Alignment for vision tasks
    - Spatial coherence, edge detection
    
32. **reinforcement_learning_alignment**: Alignment for RL
    - Correlation with value functions

### Higher-Order Information Metrics (4)

33. **total_correlation**: Multi-information among variables
    - Formula: TC = Σ H(X_i) - H(X_1,...,X_n)
    
34. **interaction_information**: Three-way information interactions
    - Formula: I(X;Y;Z) = I(X;Y) - I(X;Y|Z)
    
35. **connected_information**: Pure n-way interactions
    - Uses inclusion-exclusion principle
    
36. **synergistic_information**: Information from joint states only
    - Formula: Syn = H_joint - Σ H_i (Gaussian assumption)

## Quick Reference by Category

| Category | Count | Metric Names |
|----------|-------|--------------|
| Rayleigh Quotient | 2 | rayleigh_quotient, rq_patchwise |
| Information Theory | 6 | mutual_information_gaussian, mutual_information_binning, gaussian_mi_analytic, conditional_mutual_information, average_redundancy, layer_redundancy |
| PID | 4 | pid_shared, pid_unique_x, pid_unique_y, pid_synergy |
| Similarity | 4 | activation_cosine_similarity, weight_cosine_similarity, node_redundancy, weight_activation_alignment |
| Spectral | 8 | spectral_gap, spectral_norm_ratio, eigenvalue_entropy, spectral_clustering_score, eigenvalue_alignment, spectral_clustering, power_iteration, spectral_alignment |
| Task-Specific | 8 | task_alignment, class_selectivity, feature_importance, representation_quality, classification_alignment, language_model_alignment, vision_task_alignment, reinforcement_learning_alignment |
| Higher-Order | 4 | total_correlation, interaction_information, connected_information, synergistic_information |
| **Total** | **36** | |

## Alphabetical List

1. activation_cosine_similarity
2. average_redundancy
3. class_selectivity
4. classification_alignment
5. conditional_mutual_information
6. connected_information
7. eigenvalue_alignment
8. eigenvalue_entropy
9. feature_importance
10. gaussian_mi_analytic
11. interaction_information
12. language_model_alignment
13. layer_redundancy
14. mutual_information_binning
15. mutual_information_gaussian
16. node_redundancy
17. pid_shared
18. pid_synergy
19. pid_unique_x
20. pid_unique_y
21. power_iteration
22. rayleigh_quotient
23. reinforcement_learning_alignment
24. representation_quality
25. rq_patchwise
26. spectral_alignment
27. spectral_clustering
28. spectral_clustering_score
29. spectral_gap
30. spectral_norm_ratio
31. synergistic_information
32. task_alignment
33. total_correlation
34. vision_task_alignment
35. weight_activation_alignment
36. weight_cosine_similarity 