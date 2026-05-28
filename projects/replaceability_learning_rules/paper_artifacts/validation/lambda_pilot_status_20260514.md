# Lambda Pilot Status - 2026-05-14

## Job State

All 16 lambda-pilot jobs completed with exit code `0:0`.

- Methods: BP-TARD and BP-RTP.
- Seeds: 42 and 123.
- Lambdas: `0.03, 0.1, 0.3, 1.0`.
- Budget: 100 epochs.
- Error scan: no tracebacks, OOMs, pruning failures, or NaNs.

Durable tables:

- `projects/replaceability_learning_rules/paper_artifacts/tables/run_summary.csv`
- `projects/replaceability_learning_rules/paper_artifacts/tables/pruning_auc.csv`
- `projects/replaceability_learning_rules/paper_artifacts/tables/pruning_curve.csv`

## Main Observation

The weak `lambda=1e-3` grid was a neutral control. The strength pilot shows
the mechanism is controllable: increasing lambda reduces duplicate task mass
and peer reconstructability, increases non-replaceable task fraction, and does
not collapse accuracy over 100 epochs.

## BP-TARD Pilot Means

| lambda | best acc | key pruning AUC | weighted reg | DTM frac | NTM frac | rho_cap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.7579 | 0.2139 | 0.00034 | 0.0418 | 0.9582 | 0.0172 |
| 0.1 | 0.7576 | 0.2333 | 0.00107 | 0.0397 | 0.9603 | 0.0152 |
| 0.3 | 0.7618 | 0.2283 | 0.00284 | 0.0350 | 0.9650 | 0.0139 |
| 1.0 | 0.7588 | 0.2292 | 0.00772 | 0.0285 | 0.9715 | 0.0171 |

## BP-RTP Pilot Means

| lambda | best acc | key pruning AUC | weighted reg | DTM frac | NTM frac | rho_cap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.7599 | 0.2181 | 0.00064 | 0.0407 | 0.9593 | 0.0143 |
| 0.1 | 0.7582 | 0.2231 | 0.00195 | 0.0374 | 0.9626 | 0.0148 |
| 0.3 | 0.7572 | 0.2249 | 0.00498 | 0.0319 | 0.9681 | 0.0117 |
| 1.0 | 0.7581 | 0.2049 | 0.01265 | 0.0243 | 0.9757 | 0.0166 |

## Interpretation

The pilot supports the mechanism claim more than the performance claim.

- DTM fraction falls with lambda for both TARD and RTP.
- NTM fraction rises with lambda for both TARD and RTP.
- Accuracy remains stable over the tested range.
- Pruning AUC does not improve monotonically with lambda.
- TARD is the better immediate paper spine than RTP: it has the best pilot
  accuracy at `lambda=0.3` and the best key pruning AUC at `lambda=0.1`.
- RTP at `lambda=1.0` moves the mechanism strongly but hurts key pruning AUC,
  so it is not the first main-grid candidate.
- The `lambda=0.3` accuracy edge is a two-seed selection on a small gap, so it
  should not be treated as a locked winner.
- Pruning AUC being non-monotone is a real diagnostic warning: moving DTM/NTM
  is not automatically the same as improving pruning retention.

## Revised Next Step

Do not immediately expand BP-TARD at `lambda=0.1` and `lambda=0.3` to the full
200-epoch, 5-seed grid. First run a smaller decision grid:

- width-constrained CIFAR-ResNet-18 at width `0.5` and `0.35`;
- RTC-gated TARD, so residualized task credit is an actual training signal;
- always-on versus fixed warmup versus metric-triggered onset;
- ridge-RTP versus the completed average-correlation RTP proxy.

If one of those cells shows a clear accuracy or capacity-normalized signal,
then expand that cell to 200 epochs and five seeds. If none does, the honest
paper is a mechanism/negative-mechanistic paper rather than a performance paper.
