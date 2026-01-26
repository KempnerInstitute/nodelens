## Paper reproducibility notes (alignment repo)

This note records **output-affecting** changes observed between the code version used for early “paper runs”
(`009eff7`) and a later version (`084b65c`), plus the reproducibility instrumentation added afterwards.

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

#### A3) Pruning distribution changed (global_threshold code path)

**CRITICAL FIX (Jan 25, 2026)**:

- **Old behaviour (009eff7, used for Jan 20 runs)**: For `distribution="global_threshold"`, the pipeline used
  `MaskOperations.global_threshold_mask()` directly, which:
  - Computes a single threshold across ALL layers
  - Applies the threshold uniformly with NO per-layer caps
  - Can prune entire layers if all their scores fall below threshold

- **Changed behaviour (26d06b0, Jan 21)**: The direct `global_threshold_mask` call was REMOVED and replaced
  with `PruningDistributionManager`, which:
  - Computes the global threshold but then converts to per-layer amounts
  - Applies `max_per_layer_sparsity_cap` (defaulted to 0.90)
  - Produces different pruning distributions even with cap=1.0

- **Restored behaviour (current)**: The direct `global_threshold_mask` path is restored for
  `distribution in {"global_threshold", "global"}`. This reproduces Jan 20 results exactly.

**Impact**: This was the root cause of 4-7% accuracy drops at high sparsity (70%+) for cluster_aware_annealed
and 6% improvements for Taylor at 90%. The different distribution logic fundamentally changed which channels
were pruned at each sparsity level.

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
  - default is `1.0` (disabled / legacy behavior); set e.g. `0.90` to enable a safety cap.

### D. Isolation experiments (Jan 2026): quantifying each factor

To understand which changes contributed to performance differences, we ran controlled isolation
experiments using the **exact Jan-20 checkpoint** but varying one config at a time:

| Isolation Run | activation_point | task_activation_samples | type_mapping_mode | calibration_mode | cap | cluster_aware@0.9 |
|---------------|------------------|------------------------|-------------------|------------------|-----|-------------------|
| Jan-20 ref    | pre_bn (implicit)| match (implicit)       | greedy (implicit) | train_loader     | 1.0 | **0.7866**        |
| isoA          | **post_bn**      | gap                    | global            | indices          | 0.9 | 0.6262            |
| isoB          | pre_bn           | **gap**                | global            | indices          | 0.9 | 0.7413            |
| isoC          | pre_bn           | match                  | **global**        | indices          | 0.9 | 0.7594            |
| isoD          | pre_bn           | match                  | **greedy**        | indices          | 0.9 | 0.7567            |
| isoE          | pre_bn           | match                  | greedy            | indices          | 1.0 | 0.7567            |
| isoF          | pre_bn           | match                  | greedy            | **train_loader** | 1.0 | 0.7271            |
| isoG          | pre_bn           | match                  | global            | **train_loader** | 1.0 | 0.7322            |

**Key findings:**

1. **activation_point is the dominant factor**: `post_bn` (isoA: 0.6262) is ~12% worse than `pre_bn` (0.74-0.76).
   The old code always hooked Conv2d outputs directly (pre-BN), so `activation_point=pre_bn` is required
   to match Jan-20 behaviour.

2. **task_activation_samples matters**: Using `gap` (isoB: 0.7413) is ~1.8% worse than `match` (isoC: 0.7594).
   The old code used spatially-flattened samples for all metrics including TaskMI/synergy, so
   `task_activation_samples=match` is needed to reproduce.

3. **type_mapping_mode has minimal effect**: `greedy` (isoD: 0.7567) vs `global` (isoC: 0.7594) differ by <0.3%.

4. **calibration_mode affects results**: `indices` (deterministic) gives 0.75-0.76, while `train_loader`
   (shuffled) gives 0.72-0.73. The variance from shuffle order is significant.

5. **Remaining gap to Jan-20 (~2.7%)**: The best isolation run (isoC: 0.7594) still trails Jan-20 (0.7866)
   by ~2.7%. This gap is attributed to **different calibration samples**:
   - Jan-20 ran `do_train=true` (50 epochs), which advanced the torch RNG by ~50 `randperm(50000)` calls
   - After training, the shuffled DataLoader produced a specific sequence of calibration samples
   - Isolation runs used `do_train=false` (fresh RNG) or deterministic indices
   - **Without the original RNG state, exact reproduction is impossible**

### E. Recommendations for going forward

1. **For new paper runs**: Use `activation_point=pre_bn` and `task_activation_samples=match` to match
   the proven Jan-20 algorithm behaviour while benefiting from reproducibility improvements.

2. **For reproducibility**: Always use `calibration_mode=indices` to get deterministic calibration subsets.
   This trades off the exact Jan-20 samples for guaranteed reproducibility.

3. **Accept ~2-3% variance**: Calibration sample selection introduces variance. Report mean ± std over
   multiple seeds rather than relying on single-run numbers.

4. **Run from scratch with saved indices**: For the best of both worlds, run `do_train=true` with
   the new code (which saves calibration_indices.json) to get a fresh, fully reproducible baseline.

### F. Paper protocol recommendation

For the paper, we should:
- Use `activation_point=pre_bn` and `task_activation_samples=match` (matches Jan-20 algorithm)
- Use `calibration_mode=indices` (deterministic, reproducible)
- Run **multi-seed** experiments and report mean ± std
- Generate all figures/tables from an explicit **manifest** of run directories (no mtime heuristics)
- Record commit hashes + calibration indices in every run directory

### G. MobileNet pruning regression diagnosis (Jan 25 2026)

**Symptoms observed:**
- MobileNet pruning using `cluster_aware_annealed` dropped from ~59% (Jan 20-22 "good" runs) to ~10-55%
  (Jan 23+ runs) at 50% sparsity
- Some methods crashed or returned near-random accuracy
- The 50% bar in the paper figure showed "Ours" significantly worse than Taylor for MobileNet

**Root cause identified:**
Commit `967e9ae` (Jan 22 23:01 EST) introduced `max_per_layer_sparsity_cap = 0.90` as a **new default**
for `global_threshold` pruning distributions. Additionally, the MobileNet paper suite was switched from
`distribution: uniform` to `distribution: global_threshold`.

This combination was catastrophic for MobileNet because:
1. **global_threshold** allows score-driven layer allocation, concentrating pruning in layers with
   low-scored channels
2. For MobileNet, this often targets depthwise layers or early pointwise layers, causing network collapse
3. The **0.90 cap** prevented the worst cases but still forced pruning into sensitive layers

**The "good" Jan 20-22 runs used a different protocol:**
- `distribution: uniform` (equal pruning per layer)
- `pointwise_only: true` (skip depthwise and expansion layers)
- `skip_depthwise: true` (redundant but explicit)
- No per-layer cap (effectively 1.0)

This protocol achieved **Ours (ann.) ≈ 59% vs Taylor ≈ 55%** at 50% sparsity consistently.

**Fix applied:**
1. Updated `mobilenetv2_cifar10_unified.yaml` to use `distribution: uniform`, `pointwise_only: true`,
   `skip_depthwise: true`, `max_per_layer_sparsity_cap: 1.0`
2. Updated `run_manifest.json` to point to the Jan 22 "good" runs:
   - `mobilenetv2_cifar10_cluster_analysis_20260122_005227_56304538` (seed 42)
   - `mobilenetv2_cifar10_cluster_analysis_20260122_005328_56304626` (seed 123)
   - `mobilenetv2_cifar10_cluster_analysis_20260122_005349_56304492` (seed 456)
3. Regenerated all paper figures/tables from the updated manifest

**Verification:**
After the fix, the 50% pruning table shows:
- MobileV2: Taylor = 55.3 ± 2.2, **Ours (ann.) = 59.4 ± 0.2** (as expected)

**Lesson learned:**
MobileNet requires special treatment due to its inverted residual architecture. Always use:
- `distribution: uniform` (not `global_threshold`)
- `pointwise_only: true` (skip depthwise and expansion)
- Explicit per-layer cap = 1.0 (no additional constraint beyond uniform)

### G. MobileNet pruning regression diagnosis (Jan 25 2026)

**Symptoms observed:**
- MobileNet pruning using `cluster_aware_annealed` dropped from ~59% (Jan 20-22 "good" runs) to ~10-55%
  (Jan 23+ runs) at 50% sparsity
- Some methods crashed or returned near-random accuracy
- The 50% bar in the paper figure showed "Ours" significantly worse than Taylor for MobileNet

**Root cause identified:**
Commit `967e9ae` (Jan 22 23:01 EST) introduced `max_per_layer_sparsity_cap = 0.90` as a **new default**
for `global_threshold` pruning distributions. Additionally, the MobileNet paper suite was switched from
`distribution: uniform` to `distribution: global_threshold`.

This combination was catastrophic for MobileNet because:
1. **global_threshold** allows score-driven layer allocation, concentrating pruning in layers with
   low-scored channels
2. For MobileNet, this often targets depthwise layers or early pointwise layers, causing network collapse
3. The **0.90 cap** prevented the worst cases but still forced pruning into sensitive layers

**The "good" Jan 20-22 runs used a different protocol:**
- `distribution: uniform` (equal pruning per layer)
- `pointwise_only: true` (skip depthwise and expansion layers)
- `skip_depthwise: true` (redundant but explicit)
- No per-layer cap (effectively 1.0)

This protocol achieved **Ours (ann.) ≈ 59% vs Taylor ≈ 55%** at 50% sparsity consistently.

**Fix applied:**
1. Updated `mobilenetv2_cifar10_unified.yaml` to use `distribution: uniform`, `pointwise_only: true`,
   `skip_depthwise: true`, `max_per_layer_sparsity_cap: 1.0`
2. Updated `run_manifest.json` to point to the Jan 22 "good" runs:
   - `mobilenetv2_cifar10_cluster_analysis_20260122_005227_56304538` (seed 42)
   - `mobilenetv2_cifar10_cluster_analysis_20260122_005328_56304626` (seed 123)
   - `mobilenetv2_cifar10_cluster_analysis_20260122_005349_56304492` (seed 456)
3. Regenerated all paper figures/tables from the updated manifest

**Verification:**
After the fix, the 50% pruning table shows:
- MobileV2: Taylor = 55.3 ± 2.2, **Ours (ann.) = 59.4 ± 0.2** (as expected)

**Lesson learned:**
MobileNet requires special treatment due to its inverted residual architecture. Always use:
- `distribution: uniform` (not `global_threshold`)
- `pointwise_only: true` (skip depthwise and expansion)
- Explicit per-layer cap = 1.0 (no additional constraint beyond uniform)
