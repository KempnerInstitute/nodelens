# Decision Grid Submission - 2026-05-15

## State

Submitted the next decision grid after the lambda pilot. At submission time all
jobs were pending in `kempner_dev`; no logs or result directories had been
created yet.

## Purpose

This grid tests the critiques raised after the lambda pilot:

- Does replaceability-aware training help under constrained width?
- Does RTC-gated TARD work better than the task-gated TARD used in the pilot?
- Does always-on or metric-triggered scheduling change the result?
- Does RTP improve when peer reconstructability uses ridge `R^2` rather than
  average squared correlation?

## Submitted Jobs

### Width-Constrained BP Baselines

| Job ID | Width | Seed |
|---:|---:|---:|
| 12988797 | 0.50 | 42 |
| 12988811 | 0.50 | 123 |
| 12988834 | 0.35 | 42 |
| 12988836 | 0.35 | 123 |

### Width-Constrained BP-TARD

| Job ID | Width | Lambda | Seed |
|---:|---:|---:|---:|
| 12988882 | 0.50 | 0.1 | 42 |
| 12988883 | 0.50 | 0.1 | 123 |
| 12988884 | 0.35 | 0.1 | 42 |
| 12988885 | 0.35 | 0.1 | 123 |
| 12988886 | 0.50 | 0.3 | 42 |
| 12988887 | 0.50 | 0.3 | 123 |
| 12988888 | 0.35 | 0.3 | 42 |
| 12988889 | 0.35 | 0.3 | 123 |

### RTC-Gated TARD

| Job ID | Lambda | Seed |
|---:|---:|---:|
| 12988921 | 0.1 | 42 |
| 12988922 | 0.1 | 123 |
| 12988923 | 0.3 | 42 |
| 12988924 | 0.3 | 123 |

### Scheduling Controls

| Job ID | Schedule | Lambda | Seed |
|---:|---|---:|---:|
| 12988988 | always-on | 0.1 | 42 |
| 12988989 | always-on | 0.1 | 123 |
| 12989035 | metric-triggered (`rho_cap <= 0.015`, min epoch 5) | 0.1 | 42 |
| 12989036 | metric-triggered (`rho_cap <= 0.015`, min epoch 5) | 0.1 | 123 |

The fixed-warmup comparison for full-width TARD at `lambda=0.1` already exists
from the completed lambda pilot.

### Ridge-RTP Check

| Job ID | Peer proxy | Lambda | Seed |
|---:|---|---:|---:|
| 12989066 | ridge | 0.3 | 42 |
| 12989068 | ridge | 0.3 | 123 |

## Next Check

Once jobs start, first inspect the earliest logs for config errors, especially:

- `model.name=cifar_resnet18` and `model.width_multiplier` are preserved.
- RTC runs save `learning_rule_method=bp_rtc_tard` and
  `learning_rule_task_gate_source=rtc`.
- Metric-triggered runs record `learning_rule_metric_trigger_epoch` if the
  trigger fires.

After completion, rebuild the summary tables:

```bash
python scripts/learning_rules/summarize_mvp_grid.py \
  --results-root results/learning_rules/resnet18_cifar100 \
  --out-dir projects/replaceability_learning_rules/paper_artifacts/tables
```
