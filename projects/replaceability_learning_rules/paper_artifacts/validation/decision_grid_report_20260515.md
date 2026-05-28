# Decision Grid Report - 2026-05-15

## Executive Take

The decision grid is healthy and mostly complete. The two ridge-RTP jobs were
cancelled because they were computationally intractable in the current online
form. The completed cells support the mechanism story, but they still do not
support a strong performance claim.

The best positive signal is narrow and modest: TARD at `lambda=0.3` gives small
accuracy and pruning-AUC gains at constrained width. The effect size is under
0.2 percentage points in accuracy on two seeds, so it should be treated as a
candidate signal, not a result.

## Job Status

- Completed cleanly: 20/22 jobs.
- Cancelled intentionally: 2/22 jobs.
- Remaining queued/running jobs: none.

Cancelled jobs:

| Job ID | Method | Seed | State | Reason |
|---:|---|---:|---|---|
| 12989066 | ridge-RTP, `lambda=0.3` | 42 | cancelled | too slow; epoch 24 after about 9.9h |
| 12989068 | ridge-RTP, `lambda=0.3` | 123 | cancelled | too slow; epoch 20 after about 9.9h |

No tracebacks, OOMs, CUDA errors, NaNs, or fatal runtime errors were found in
the decision-grid logs.

## Result Summary

Two-seed means. Key AUC is for
`composite_twoaxis_ixy_hybrid_taylor`.

| Cell | Best acc | Key AUC | DTM frac | NTM frac | Weighted reg |
|---|---:|---:|---:|---:|---:|
| BP width 0.35 | 0.7169 | 0.1421 | -- | -- | 0.0000 |
| TARD width 0.35, lambda 0.1 | 0.7178 | 0.1274 | 0.0435 | 0.9565 | 0.0012 |
| TARD width 0.35, lambda 0.3 | 0.7186 | 0.1467 | 0.0392 | 0.9608 | 0.0032 |
| BP width 0.5 | 0.7298 | 0.1875 | -- | -- | 0.0000 |
| TARD width 0.5, lambda 0.1 | 0.7317 | 0.1813 | 0.0423 | 0.9577 | 0.0011 |
| TARD width 0.5, lambda 0.3 | 0.7307 | 0.1944 | 0.0388 | 0.9612 | 0.0031 |
| RTC-TARD lambda 0.1 | 0.7568 | 0.2141 | 0.0395 | 0.9605 | 0.0011 |
| RTC-TARD lambda 0.3 | 0.7582 | 0.2307 | 0.0351 | 0.9649 | 0.0028 |
| TARD always-on lambda 0.1 | 0.7578 | 0.2314 | 0.0394 | 0.9606 | 0.0011 |
| TARD metric-trigger lambda 0.1 | 0.7533 | 0.2105 | 0.0442 | 0.9558 | 0.0012 |

## Paired Width Deltas

Relative to same-width BP:

| Width | Method | Delta acc | Delta key AUC |
|---:|---|---:|---:|
| 0.35 | TARD lambda 0.1 | +0.0010 | -0.0147 |
| 0.35 | TARD lambda 0.3 | +0.0017 | +0.0046 |
| 0.5 | TARD lambda 0.1 | +0.0020 | -0.0062 |
| 0.5 | TARD lambda 0.3 | +0.0010 | +0.0069 |

Interpretation: `lambda=0.3` is the only constrained-width setting with both
accuracy and pruning-AUC deltas in the right direction, but the effect is tiny.
This is worth a paired-seed replication, not a full-paper claim.

## Diagnostics

### D1. Mechanism moved, performance barely moved

In every TARD/RTC-TARD setting, increasing `lambda` lowers DTM and raises NTM.
That part is robust. But the performance readouts do not scale monotonically
with DTM/NTM movement. This means the paper should avoid the simple claim
``less duplicate task mass implies better pruning retention.'' The safer claim
is that the learning rule can control the two-axis geometry; whether that
geometry is useful depends on width, schedule, and pruning protocol.

### D2. Width constraint is the only promising performance regime

Full-width RTC-TARD is stable but not better than task-gated TARD. The strongest
new signal is constrained-width TARD at `lambda=0.3`, where both widths show
small positive accuracy and pruning-AUC deltas. This is exactly where the
capacity story should have the most room to show itself.

### D3. Metric trigger fired too late

The metric-triggered runs fired at epochs 94 and 88. That is effectively a
late-training perturbation, not a real schedule. The observed underperformance
should not be interpreted as evidence against metric-triggering in general; it
is evidence that `rho_cap <= 0.015` is too conservative or the wrong online
proxy.

Next diagnostic: plot `rho_cap` by epoch for fixed-warmup, always-on, and
metric-triggered runs, then choose a threshold that fires around epochs 10--30.
Longer-term, the trigger should use online `corr(I_X,t)` rather than `rho_cap`.

### D4. RTC gate did not buy a clear advantage

RTC-TARD was stable and moved DTM/NTM, so the implementation is viable. But it
did not clearly improve accuracy or AUC over the cheaper task-gated TARD in
this two-seed decision grid.

Next diagnostic: compare channel-level task gate versus RTC gate on the same
minibatches/checkpoints. The useful plot is a layerwise scatter of
`task_relevance` versus `rtc_relevance`, plus gate entropy/sparsity by layer. If
the gates are nearly rank-equivalent, RTC-TARD cannot be expected to differ
from task-TARD.

### D5. Online ridge-RTP is not viable as implemented

Ridge-RTP had:

- seed 42: epoch 24, best acc 0.5702, weighted regularizer about 0.1485;
- seed 123: epoch 20, best acc 0.5683, weighted regularizer about 0.1513;
- peer reconstructability around 0.95;
- estimated remaining time beyond the 24h walltime.

This is both computationally too expensive and dynamically too strong. Full
ridge peer reconstruction should be moved out of the online main path unless it
is made sparse/groupwise/cached.

## Recommendation

Do not launch the 200-epoch, five-seed main grid yet. The right next step is a
small replication/diagnostic package:

1. Replicate only the best constrained-width candidate:
   BP versus TARD `lambda=0.3` at widths `0.35` and `0.5`, adding seeds
   `456, 789, 1011` at 100 epochs.
2. Keep RTC-TARD as a secondary diagnostic, not the main method, unless gate
   diagnostics show it meaningfully differs from task-gated TARD.
3. Drop online full ridge-RTP from the spine. If RTP stays, use `avg_corr2` or a
   cheaper sparse/groupwise ridge approximation.
4. Re-run metric-triggering only after threshold calibration. Target onset:
   epochs 10--30, not 88--94.
5. Update the paper claim to: ``replaceability-aware training can control
   two-axis capacity geometry and may provide small benefits under constrained
   width.'' Avoid claiming a full-width or pruning breakthrough.

## Suggested Follow-Up Commands

Paired width replication:

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --slurm \
  --configs configs/learning_rules/resnet18_cifar100_bp_baseline.yaml \
  --seeds 456 789 1011 \
  --width-multipliers 0.5 0.35 \
  --training-epochs 100 \
  --device cuda \
  --allow-dirty \
  --partition kempner_dev \
  --gres gpu:1 \
  --cpus-per-task 8 \
  --mem 64G \
  --time 1-00:00:00 \
  --job-name-prefix lr-width-repl-bp
```

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --slurm \
  --configs configs/learning_rules/resnet18_cifar100_bp_tard.yaml \
  --seeds 456 789 1011 \
  --width-multipliers 0.5 0.35 \
  --learning-rule-lambda 0.3 \
  --training-epochs 100 \
  --device cuda \
  --allow-dirty \
  --partition kempner_dev \
  --gres gpu:1 \
  --cpus-per-task 8 \
  --mem 64G \
  --time 1-00:00:00 \
  --job-name-prefix lr-width-repl-tard
```

Cheap ridge-RTP diagnostic, if still desired:

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --slurm \
  --configs configs/learning_rules/resnet18_cifar100_bp_rtp.yaml \
  --seeds 42 \
  --learning-rule-lambda 0.03 \
  --training-epochs 30 \
  --override learning_rule.peer_proxy=ridge \
  --override learning_rule.max_layers=2 \
  --device cuda \
  --allow-dirty \
  --partition kempner_dev \
  --gres gpu:1 \
  --cpus-per-task 8 \
  --mem 64G \
  --time 4:00:00 \
  --job-name-prefix lr-rtp-ridge-cheap
```
