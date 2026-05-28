# Early Grid Status - 2026-05-14

## Current State

- BP baseline: 5/5 complete with pruning.
- BP+DeCov: 5/5 complete with pruning.
- BP-TARD: 4/5 complete with pruning; seed 1011 is still in flight.
- BP-RTP: 5/5 in flight.
- Error scan: no tracebacks, OOMs, pruning failures, or NaNs in learning-rule
  SLURM logs.

Durable tables:

- `projects/replaceability_learning_rules/paper_artifacts/tables/run_summary.csv`
- `projects/replaceability_learning_rules/paper_artifacts/tables/pruning_curve.csv`
- `projects/replaceability_learning_rules/paper_artifacts/tables/pruning_auc.csv`

## Completed Accuracy

BP best accuracy over five seeds: mean 0.7668.

BP+DeCov best accuracy over five seeds: mean 0.7651. Paired DeCov minus BP
best-accuracy difference is -0.0017 on average, with seed-level differences:
`+0.0012, -0.0005, -0.0017, -0.0002, -0.0074`.

BP-TARD best accuracy over the four completed seeds: mean 0.7650. Paired TARD
minus BP differences for completed seeds are:
`-0.0034, -0.0025, -0.0016, +0.0044`.

The runbook decision rule ("at least 1pp gain or clear pruning-AUC gain") is
not met by TARD at `lambda=1e-3`.

## Pruning AUC

For `composite_twoaxis_ixy_hybrid_taylor`, the current normalized pruning AUC
means are:

- BP: 0.2265 over 5 seeds.
- BP+DeCov: 0.2303 over 5 seeds.
- BP-TARD: 0.2269 over 4 completed seeds.

TARD is therefore indistinguishable from BP on the current pruning readout, and
DeCov's small positive AUC difference is not paired with an accuracy gain.

## Mechanism Read

The current `lambda=1e-3` is too weak to test the mechanism. TARD's final raw
penalty is around 0.011, so the weighted contribution is about `1e-5` versus
CE around `1.5e-2`. The task gate also makes TARD's raw penalty smaller than
DeCov's raw penalty at the same lambda, so `lambda=1e-3` is not a matched
strength comparison.

This grid should be treated as a weak-regularizer control. It is useful for
showing that the hook does not break optimization, but it is not a serious
negative result for replaceability-aware credit assignment.

## Code Changes For Next Pilot

Future TARD/RTP runs now log per-epoch mechanism diagnostics in
`training_history.json`:

- `regularizer_weighted`
- `duplicate_task_mass`
- `non_replaceable_task_mass`
- `duplicate_task_fraction`
- `non_replaceable_task_fraction`
- `peer_reconstructability_mean`
- `task_relevance_mean`
- `task_gate_mean`
- `rho_cap`

`summarize_mvp_grid.py` carries the final values into `run_summary.csv`.

## Recommended Change

Do not expand DeCov. Do not interpret the current TARD/RTP `lambda=1e-3` grid
as a mechanism test. The next real experiment should be a two-seed strength
pilot for TARD/RTP:

- methods: BP-TARD, BP-RTP
- seeds: 42, 123
- lambdas: `{0.03, 0.1, 0.3, 1.0}`
- budget: 100 epochs for the pilot

Primary decision variables:

- matched-epoch accuracy stability
- final accuracy
- weighted versus raw regularizer contribution
- pruning AUC
- DTM/NTM movement
- `rho_cap` movement

If `0.1` moves DTM/NTM without hurting accuracy, it is the best candidate for
the main Version-B paper grid. If only `1.0` moves the mechanism and hurts
accuracy, the formulation or gate needs revision before a larger grid.

## Submitted Pilot

Submitted on 2026-05-14 09:54 EDT.

- Jobs: `12873738`-`12873752`, `12873754`.
- Status at submission check: all pending; first job pending on resources,
  others pending on priority.
- Methods: BP-TARD and BP-RTP.
- Seeds: 42 and 123.
- Lambdas: `0.03, 0.1, 0.3, 1.0`.
- Budget: 100 epochs.
- Logs: `logs/learning_rules/slurm/lr-pilot-*.out`.
