## Paper reproducibility notes (alignment repo)

This note records **output-affecting** changes observed between the code version used for early “paper runs”
(`009eff7`, 2026-01-20) and the later version (`084b65c`, 2026-01-22), and the additional reproducibility
instrumentation we added afterwards.

### A. Output-affecting algorithm changes (009eff7 → 084b65c)

#### A1) Task MI / Synergy estimation changed (fix pseudo-replication)

- **Old behaviour (009eff7)**: when `activation_samples="flatten_spatial"`, the target \(T\) (logit margin) is
  repeated across spatial positions and treated as if it had \(B \times H \times W\) independent samples. Both
  `MI(T;Y_i)` and the Gaussian synergy approximation were computed from these *spatially-flattened* stats:
  - `mi_t` computed from `cov_ty / sqrt(var_t * var_y)`
  - partner ordering for synergy used the *local* redundancy MI matrix (`mi_matrix`) from `cov_yy`
  - joint MI `I(T; [Y_i, Y_j])` used `var_t, var_y, cov_ty, cov_yy` (local accumulator)

- **New behaviour (084b65c)**: decision-level quantities involving image-level targets are computed from
  **per-image pooled** activations (GAP), regardless of spatial sampling mode, to avoid pseudo-replication:
  - `mi_t` computed from `cov_ty_task / sqrt(var_t_task * var_y_task)`
  - partner ordering for synergy uses `mi_matrix_task` from the **task** covariance `cov_yy_task`
  - joint MI uses task stats `var_t_task, var_y_task, cov_ty_task, cov_yy_task`

This change can materially alter:
- within-layer cluster structure (esp. synergy dimension),
- halo significance tests (if enabled),
- pruning scores for any methods using synergy/red as components.

#### A2) Cluster type mapping changed (reduce “label swapping” across layers)

- **Old behaviour (009eff7)**: greedy assignment
  1) `critical := argmax(log_rq - red)`
  2) `redundant := argmax(red)` among remaining
  3) `synergistic := argmax(syn)` among remaining
  4) leftover is background

- **New behaviour (084b65c)**: **global one-to-one assignment** over all permutations that maximizes a linear
  score for the four semantic types (critical/redundant/synergistic/background). This is specifically intended
  to reduce centroid/label “swaps” across layers that can make depth trends look noisy.

This change is a likely contributor to the “cleaner critical-vs-depth trend” you observed in newer runs.

#### A3) Pruning distribution changed (layer-level safety cap)

- In `global_threshold` distributions, **per-layer sparsity is capped** (previously unbounded), preventing
  pathological cases where a layer is completely removed (a common cause of collapse at high sparsity).

This affects **all pruning methods**, not just cluster-aware ones, and can change both absolute performance and gaps.

#### A4) Optional BN activation point support added

New config knob:
- `activation_point`: `"pre_bn"` (default) vs `"post_bn"` (hook BN outputs).

When using `"post_bn"`, the RQ denominator is adjusted by BN scale so RQ remains comparable.

### B. Extra diagnostics added (primarily additive, but can affect RNG use)

- **Metric ablation**: clustering can be run with metric subsets (`rq_red`, `rq_syn`, `red_syn`, …).
- **Halo permutation baseline**: compute null distributions by shuffling source-layer labels.

These are usually *additive outputs*, but they can change runtime and (if any shared RNG is used) must be handled
carefully for strict reproducibility.

### C. Reproducibility instrumentation added (post 084b65c)

To make paper runs exactly reproducible from “current code”, we added:

- **Deterministic calibration subset**:
  - create a fixed set of `n_calibration` dataset indices using the experiment seed,
  - save to `calibration_indices.json` in the run directory,
  - compute metrics/Taylor/HRank on this deterministic subset via a calibration DataLoader (no shuffle).

- **Run metadata**:
  - write `run_metadata.json` to the run directory (git commit/dirty, python/torch/numpy versions, SLURM IDs),
  - embed the same metadata into `results.json` under `metadata`.

- **Configurable per-layer sparsity cap**:
  - expose `max_per_layer_sparsity_cap` via `PruningPipelineOptions` and `PruningDistributionManager` kwargs.
  - default remains `0.90` (current behaviour); set `1.0` to emulate legacy behaviour.

### D. Paper protocol recommendation

For the paper, we should:
- pick a **single** algorithm version (recommended: the newer task-level synergy + global type mapping),
- run **multi-seed** experiments and report mean ± std,
- generate all figures/tables from an explicit **manifest** of run directories (no mtime heuristics),
- record commit hashes + calibration indices in every run directory.

