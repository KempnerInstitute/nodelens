# Decision Grid Status - 2026-05-15

## Job Health

As of this check, 20/22 decision-grid jobs have completed with exit code `0:0`.
The remaining two jobs are the ridge-RTP checks:

- `12989066`: seed 42, running, epoch 24 in `training_history.json`.
- `12989068`: seed 123, running, epoch 20 in `training_history.json`.

No tracebacks, OOMs, CUDA errors, NaNs, or fatal runtime errors were found in
the decision-grid SLURM logs.

## Completed Decision Cells

Values are two-seed means from
`projects/replaceability_learning_rules/paper_artifacts/tables/run_summary.csv`
and `pruning_auc.csv` after regenerating the summary tables.

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

The fixed-warmup full-width TARD lambda pilot from 2026-05-14 remains the
comparison point for scheduling:

- TARD lambda 0.1, fixed warmup: best acc 0.7576, key AUC 0.2333.

## Early Interpretation

- Width-constrained TARD gives tiny accuracy gains over BP at widths 0.35 and
  0.5, but the effect is under 0.2 percentage points and should be treated as
  noise unless it replicates with more seeds.
- Width-constrained pruning AUC is mixed: lambda 0.3 improves over BP at both
  widths, while lambda 0.1 is worse.
- RTC-gated TARD is stable but does not clearly beat task-gated TARD.
- Always-on TARD is stable and roughly tied with fixed warmup in this two-seed
  check.
- Metric-triggered TARD turns on very late (`epoch 94` and `epoch 88`) and
  underperforms. The current `rho_cap <= 0.015` trigger is too conservative or
  is the wrong online proxy.

## Ridge-RTP Issue

Ridge-RTP is technically running but effectively unhealthy for the decision
grid:

- Seed 42: epoch 24 after about 9.8 hours.
- Seed 123: epoch 20 after about 9.8 hours.
- Estimated remaining time is about 28.5 and 40.5 hours, beyond the 24-hour
  walltime.
- The regularizer is very large (`weighted reg` about 0.15), peer
  reconstructability is near 0.95, and early accuracy is poor.

Recommendation: cancel the two ridge-RTP jobs and relaunch only a cheap
diagnostic variant, e.g. `max_layers=2`, a smaller calibration/proxy, or a
post-hoc ridge diagnostic. Ridge-RTP should not be in the main grid in its
current online form.
