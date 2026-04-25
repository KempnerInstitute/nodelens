# Architecture

NodeLens is organized around a reusable Python package and a small set of
configuration-driven workflows. The library code stays general; project folders
show how the same components are combined for a concrete study.

## Data Flow

```text
YAML config
    |
    v
Experiment runner
    |
    |-- loads model and dataset
    |-- selects tracked layers
    |-- captures activations, gradients, weights, and masks
    |
    v
Metric and scoring layer
    |
    |-- activation and norm statistics
    |-- Rayleigh quotient and spectral metrics
    |-- mutual information, redundancy, and synergy
    |-- gradient, Taylor, curvature, and loss-proxy scores
    |
    v
Analysis and intervention layer
    |
    |-- clustering and cross-layer analyses
    |-- ablation and sensitivity probes
    |-- structured pruning strategies
    |-- plots, tables, JSON summaries, and reports
```

## Package Layout

```text
src/nodelens/
|-- analysis/        # Aggregation, clustering, visualization, reports
|-- configs/         # Config loading and validation
|-- core/            # Registries, protocols, base abstractions
|-- dataops/         # Dataset loading and tensor preprocessing
|-- experiments/     # Config-driven experiment classes
|-- infrastructure/  # Logging, distributed helpers, storage utilities
|-- metrics/         # Node and channel metrics
|-- models/          # Model wrappers and model factory helpers
|-- pruning/         # Pruning configs, masks, and strategies
|-- services/        # Activation capture, scoring, and mask operations
`-- training/        # Training and evaluation helpers
```

## Design Rules

- Keep reusable metrics, model wrappers, pruning code, and experiment classes
  in `src/nodelens/`.
- Keep runnable experiment settings in `configs/`.
- Keep generated outputs in `outputs/`, which is ignored by git.
- Keep project folders focused on reproducible usage: configs, helper scripts,
  artifact descriptions, and notes that connect a study to the shared library.
- Do not store model weights, raw datasets, checkpoints, scheduler logs, access
  tokens, or private absolute paths in the repository.

## Common Workflows

### Metric Analysis

```text
model + dataloader
    -> activation capture
    -> metric computation
    -> per-layer channel scores
    -> plots or JSON summaries
```

Use this path for activation outliers, Rayleigh quotient scores, information
metrics, redundancy estimates, or loss-proxy ranking.

### Intervention Analysis

```text
channel scores
    -> masks or ablation sets
    -> model evaluation
    -> sensitivity curves
```

Use this path to test whether a metric identifies channels that matter for
accuracy, perplexity, robustness, pruning, or other downstream behavior.

### Project Workflow

```text
shared package + configs
    -> experiment outputs
    -> aggregation scripts
    -> figures, tables, and artifact manifests
```

Project folders under `projects/` should make a study easy to inspect without
turning project-specific scripts into core library code.
