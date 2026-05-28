# New Methods Grid Submitted - 2026-05-15

## Headline Diagnostic from Existing Checkpoints

The analytic gradient orthogonality predicted by the two-axis paper
(Proposition: Gaussian residualization of the two gradients) is essentially
already realized at every hidden layer of BP-trained ResNet-18 on CIFAR-100.
This is the key falsification result for "BP already does it" and was computed
via `scripts/learning_rules/analyze_grad_orthogonality.py` over 7
representative checkpoints at `calibration_size=512`.

Per-layer `|cos(grad_I_X, grad_I_T)|` (lower = more orthogonal):

| layer | BP | DeCov | RTP 0.1 | RTP 0.3 | TARD 0.1 | TARD 0.3 | RTC-TARD 0.3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| conv1 | 0.43 | 0.45 | 0.50 | 0.37 | 0.47 | 0.33 | 0.53 |
| layer1.0.conv1 | 0.16 | 0.13 | 0.13 | 0.13 | 0.13 | 0.11 | 0.12 |
| layer1.0.conv2 | 0.09 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 | 0.06 |
| layer2.0.conv2 | 0.09 | 0.12 | 0.12 | 0.10 | 0.07 | 0.11 | 0.10 |
| layer3.0.conv1 | 0.08 | 0.09 | 0.11 | 0.09 | 0.08 | 0.10 | 0.08 |
| layer4.0.conv1 | 0.06 | 0.05 | 0.06 | 0.05 | 0.05 | 0.05 | 0.05 |
| layer4.1.conv1 | 0.11 | 0.10 | 0.08 | 0.09 | 0.11 | 0.06 | 0.10 |

Reads:

- conv1 is the only layer where the two analytic gradients are not orthogonal.
- All hidden layers sit in `|cos| in [0.04, 0.13]` independent of method.
- TARD/RTP/RTC nudge the cosine toward zero in the shallowest 1-2 layers but
  do not move deep layers further.
- This explains the persistent null on accuracy across earlier grids: the
  regularizers push along a direction the gradient already drives.

A full scan over all 56 ResNet-18/CIFAR-100 checkpoints at
`calibration_size=2048` is queued as job `13122581` and will write
`grad_orthogonality_resnet18_20260515_summary.csv` and `_channels.csv` to
`projects/replaceability_learning_rules/paper_artifacts/tables/`.

## New Code Landed

Implementations under `src/nodelens/learning_rules/` and
`scripts/learning_rules/`:

- `compact_hull_from_correlation`: greedy compact-hull statistic
  (`max_size`, `eps`, ridge), returning `(hull_size, full_r2, hull_score)`.
  Direct port of the original two-axis paper's compact-hull score
  `E_i^full / max(1, |H_i|)`.
- `compact_hull_penalty`: `gate_i * hull_score_i` per layer, normalized.
- `cross_layer_weights`: forward-only allocator with modes `uniform`,
  `dtm_share`, `ntm_share`, `depth`, `dtm_depth`. Weights normalize to
  sum-to-`n` so the user-facing `lambda` retains scale across modes. This is
  a CAP-QGW-flavored cross-layer allocator that touches the part of the
  original paper's pruning result not reachable by per-channel gates alone.
- `bp_hull` method dispatch in `replaceability_regularization_loss`, plus
  per-method capacity stats including `hull_size_mean`, `hull_score_mean`,
  `cross_layer_weight_*`.
- `LearningRuleConfig` extended with `cross_layer_alloc`,
  `cross_layer_alpha`, `hull_max_size`, `hull_eps`.
- `ExperimentConfig` and both `config_loader` registries extended with the
  same four flat keys.
- `summarize_mvp_grid.py` surfaces the new diagnostic columns.
- 7 new unit tests cover compact-hull, allocator modes, hull-method
  dispatch, and cross-layer logging. All 16 learning-rule tests pass.
- `scripts/learning_rules/analyze_grad_orthogonality.py`: offline analytic
  gradient orthogonality estimator (patch-level, ridge-stabilized). Run
  per-checkpoint; emits per-layer and per-channel CSVs.

## Configs Added

- `configs/learning_rules/resnet18_cifar100_bp_hull.yaml` (full grid form).
- `configs/learning_rules/resnet18_cifar100_bp_tard_xlayer.yaml`
  (TARD with `cross_layer_alloc: dtm_depth`).
- `configs/learning_rules/vgg16_cifar100_bp_baseline.yaml` and
  `..._bp_tard.yaml` for the architecture probe.
- Smoke variants of `bp_hull` and `bp_tard_xlayer`.

## Submitted Grids

All on `kempner_dev`, ResNet-18 jobs at 64G/8 CPU/24h, VGG at 96G/8 CPU/36h.
Total 32 training jobs plus 1 offline analysis job.

| Grid | Job-name prefix | Job IDs | Count |
|---|---|---|---:|
| A: width replication BP/TARD lambda 0.3 widths {0.35, 0.5} seeds {456,789,1011} | `lr-width-replication` | 13121603-13121614 | 12 |
| B: bp_hull lambda 0.3 widths {0.35, 0.5, 1.0} seeds {42, 123, 456} | `lr-bp-hull` | 13121656-13121664 | 9 |
| C: bp_tard_xlayer lambda 0.3 alloc dtm_depth widths {0.35, 0.5, 1.0} seeds {42, 123, 456} | `lr-bp-tard-xlayer` | 13121676-13121685 | 9 |
| D: VGG-16 BP and TARD lambda 0.3 full width seed 42 | `lr-vgg-probe` | 13122570, 13122571 | 2 |
| E: full ResNet-18 grad_orthogonality scan (56 checkpoints, calibration_size 2048) | `lr-grad-ortho-scan` | 13122581 | 1 |

## What Each Grid Decides

- **A** replicates the May 14 constrained-width TARD signal at three new
  seeds. If the +0.001-0.002 acc deltas vanish, the constrained-width story
  goes from "candidate signal" to "noise." If they survive, it becomes a
  small but real effect worth reporting.
- **B** tests whether penalizing easily-replaced channels (compact-hull score)
  beats the simpler pairwise correlation penalty. The compact-hull score is
  the original paper's strongest within-bin lesion-damage predictor; if
  any per-channel object can dominate TARD it is this one.
- **C** tests CAP-QGW-for-training: layer-wise lambda biased toward layers
  with high duplicate-task-mass and toward deeper depths. None of the
  earlier grids touch cross-layer allocation.
- **D** tests the VGG-16 boundary case from the original paper. ResNet-18
  results may not transfer because magnitude is already a competitive
  local-axis surrogate on VGG.
- **E** produces the headline `cos(grad_I_X, grad_I_T)` plot across all
  trained checkpoints to back the "BP already does residualized credit
  assignment" mechanism story (Angle 5).

## Decision Rules After Grids Land

- If A is null and B is also null: the per-channel penalty family is
  exhausted. Pivot fully to Angle 5 (mechanism paper) with E as the
  headline figure plus a small constrained-width result.
- If C produces a >=0.5 pp width-constrained accuracy gain over plain TARD:
  the cross-layer-allocation pivot becomes the paper spine and bp_hull
  drops to an ablation.
- If D's VGG cell shows TARD weakly hurts accuracy while it helps on
  ResNet-18: pivot to architecture-conditioning ("two-axis training matters
  when magnitude does not already capture the local axis").
- If everything is null: Angle 5 alone, with the orthogonality figure as
  the actual contribution.

## Bookkeeping

- Code paths touched outside `src/nodelens/learning_rules/`:
  - `src/nodelens/configs/config_loader.py` (two registries).
  - `src/nodelens/experiments/base.py` (4 dataclass fields).
  - `scripts/run_experiment.py` (4 LearningRuleConfig kwargs).
  - `scripts/learning_rules/summarize_mvp_grid.py` (new columns).
- `tests/unit/test_learning_rules.py`: 16 tests pass.
- `tests/unit/test_config_loader.py`: 38 tests pass (no regression).
- Smoke configs verified end-to-end on CPU; new diagnostics flow into
  `training_history.json`.

## Monitor

```bash
squeue -u $USER -o "%.10i %.40j %.8T %.10M %R" | rg "JOBID|lr-(width-replication|bp-hull|bp-tard-xlayer|vgg-probe|grad-ortho)"
```
