# Changelog

## Recent Updates

### Added
- Service layer architecture: `ActivationCaptureService`, `NodeScoringService`, `MaskOperations`
- Information-theoretic metrics: `PairwiseRedundancyGaussian`, `SynergyGaussianMMI`
- Class-conditioned RQ: `RayleighQuotient.compute_class_conditioned()`
- Hook management: `HookManager`, `PersistentHookManager`
- `forward_with_activations()` for safe activation capture

### Changed
- RayleighQuotient: Added regularization parameter for numerical stability
- BaseModelWrapper: Integrated HookManager for automatic cleanup

### Fixed
- Memory leaks from accumulated hooks
- Numerical stability in small batch scenarios

## Core Features

- Rayleigh Quotient and class-conditioned variants
- Mutual Information (Gaussian and binning methods)
- Information-theoretic metrics (redundancy, synergy, PID)
- Similarity metrics (CKA, activation correlation)
- Model wrappers for PyTorch models
- Pruning strategies (magnitude, gradient, alignment-based)
- Service layer for activation capture and scoring
