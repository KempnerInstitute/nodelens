# Architecture

LossLens is organized as a reusable library plus paper-specific project
folders. The library code should remain general; each paper folder should only
contain release notes, configs, and artifact packaging scripts for that paper.

```mermaid
flowchart TB
    subgraph Library[src/alignment]
        M[metrics]
        P[pruning]
        E[experiments]
        A[analysis]
        S[services]
    end

    subgraph Inputs[Inputs]
        C[configs]
        D[calibration data]
        N[model checkpoints]
    end

    subgraph Projects[projects]
        R[supernodes_scar]
    end

    C --> E
    D --> S
    N --> S
    S --> M
    M --> P
    M --> A
    P --> E
    A --> E
    E --> R
    R --> H[Hugging Face artifact bundle]
```

## Design Rules

- Keep reusable metrics, services, pruning code, and experiment classes in
  `src/alignment/`.
- Keep paper release instructions and packaging scripts in `projects/`.
- Keep generated outputs in `outputs/`, which is ignored by git.
- Do not store model weights, raw datasets, cluster logs, or private paths in
  the repository.
- Use project manifests and checksums for anything uploaded as an artifact.

## Supernodes and SCAR Flow

```mermaid
sequenceDiagram
    participant Config as YAML config
    participant Runner as run_experiment.py
    participant Capture as activation and gradient capture
    participant Metrics as SCAR metrics
    participant Prune as structured pruning
    participant Artifacts as artifact bundle

    Config->>Runner: choose model, calibration data, sparsity, metrics
    Runner->>Capture: collect layer-wise activations and gradients
    Capture->>Metrics: compute LP, activation, curvature, and Taylor scores
    Metrics->>Prune: protect supernode core and rank remaining channels
    Prune->>Artifacts: write results, figures, tables, and manifests
```
