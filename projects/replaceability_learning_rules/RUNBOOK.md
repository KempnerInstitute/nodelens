# Learning-Rules Experiment Runbook

This is the executable plan for the first paper-grade MVP: ResNet-18 on
CIFAR-100 with BP, BP+DeCov, BP-TARD, and BP-RTP.

## 1. Smoke Test

Run this after code changes:

```bash
python scripts/run_experiment.py \
  --config configs/learning_rules/resnet18_cifar100_smoke_bp_tard.yaml \
  --allow-dirty
```

Expected artifacts:

- `results/training_history.json`
- `checkpoints/trained_model.pth`
- `results/results_*.json`

The smoke config has `visualization.enabled: false`, so it should not spend
time generating figures.

## 2. Main Grid

Primary paper grid:

- Methods: BP, BP+DeCov, BP-TARD, BP-RTP
- Seeds: 42, 123, 456, 789, 1011
- Backbone/data: ResNet-18/CIFAR-100
- Training: 200 epochs, SGD, cosine schedule
- Primary readouts: final accuracy, pruning AUC, layerwise
  RQ/redundancy/synergy summaries, halo/cascade diagnostics

The pruning readout includes the older ICML-era baselines plus two-axis-paper
variants:

- `composite` and `cluster_aware`: older RQ/synergy/redundancy baselines.
- `cap_ixy_hybrid_taylor`: Taylor layer allocation with I(X;Y)-first CAP
  channel ranking.
- `composite_twoaxis_ixy_hybrid_taylor`: Taylor allocation with a score-only
  two-axis ranking.
- `cluster_aware_stratified_twoaxis_adaptive_ixy_hybrid_taylor`: Taylor
  allocation with two-axis ranking plus stratified cluster/halo structure.

The `_hybrid_taylor` suffix is local to that pruning method; it does not change
the older `cluster_aware` baseline.

Inspect commands without launching:

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --dry-run \
  --slurm \
  --device cuda \
  --allow-dirty
```

Submit to SLURM:

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --slurm \
  --device cuda \
  --allow-dirty \
  --partition kempner_dev \
  --gres gpu:1 \
  --cpus-per-task 8 \
  --mem 64G \
  --time 2-00:00:00
```

Logs go to `logs/learning_rules/slurm/`.

Monitor the grid:

```bash
squeue -u "$USER" -o "%.18i %.9P %.60j %.8T %.10M %.6D %R" | rg "JOBID|lr-resnet18"
```

Launch a strength pilot without hand-authoring new YAMLs:

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --dry-run \
  --configs configs/learning_rules/resnet18_cifar100_bp_tard.yaml \
            configs/learning_rules/resnet18_cifar100_bp_rtp.yaml \
  --seeds 42 123 \
  --learning-rule-lambdas 0.03 0.1 0.3 1.0 \
  --training-epochs 100 \
  --device cuda \
  --allow-dirty
```

Remove `--dry-run` and add the usual `--slurm` resource flags when ready to
submit.

## 3. Summaries

Once runs finish, build tables for the draft:

```bash
python scripts/learning_rules/summarize_mvp_grid.py \
  --results-root results/learning_rules/resnet18_cifar100 \
  --out-dir projects/replaceability_learning_rules/paper_artifacts/tables
```

Outputs:

- `run_summary.csv`: one row per run
  - `training_epochs` is the planned budget from the config.
  - `actual_training_epochs` and `training_complete` are the reliable fields
    for distinguishing finished runs from in-flight jobs.
  - New TARD/RTP runs also include final DTM/NTM, task fractions,
    `rho_cap`, and weighted regularizer contribution.
- `pruning_curve.csv`: one row per pruning method and sparsity
- `pruning_auc.csv`: normalized accuracy AUC per run and pruning method

For runs that finished before the two-axis pruning methods were added, extend
them without retraining:

```bash
python scripts/extend_run.py \
  --run-dir <run-dir> \
  --tasks pruning \
  --device cuda \
  --methods cap_ixy_hybrid_taylor,composite_twoaxis_ixy_hybrid_taylor,cluster_aware_stratified_twoaxis_adaptive_ixy_hybrid_taylor \
  --ratios 0.1,0.3,0.5,0.7,0.8,0.9 \
  --backup
```

## 4. Decision Criteria

Version B is the paper spine if BP-TARD or BP-RTP shows either:

- at least a 1 percentage-point gain in final accuracy at matched training
  budget, or
- a clear pruning-AUC gain at matched unpruned accuracy.

If the BP result is weak but the representational diagnostics move in the
predicted direction, keep this as a mechanism paper and add the warmup and
lambda sweeps before adding DFA.

Current status: the lambda pilot satisfies the mechanism criterion but not the
performance criterion. Do not launch a plain 200-epoch five-seed expansion just
because `lambda=0.3` had the best two-seed pilot accuracy; that is too small a
selection margin. First run the decision grid below.

## 5. Immediate Decision Grid

The next runs should test whether the mechanism matters under constrained
capacity and whether the paper's central RTC object helps as a training signal.

### 5.1 Width-constrained TARD pilot

Use the CIFAR-ResNet-18 width-scaled model added for this project:

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --dry-run \
  --configs configs/learning_rules/resnet18_cifar100_bp_baseline.yaml \
            configs/learning_rules/resnet18_cifar100_bp_tard.yaml \
  --seeds 42 123 \
  --width-multipliers 0.5 0.35 \
  --learning-rule-lambdas 0.1 0.3 \
  --training-epochs 100 \
  --device cuda \
  --allow-dirty
```

For BP baselines, launch them separately without `--learning-rule-lambdas` to
avoid duplicate baseline jobs per lambda. The TARD cells should use the lambda
sweep.

### 5.2 RTC-gated TARD pilot

This closes the manuscript/code gap: RTC becomes the gate used by the online
objective, not only a diagnostic.

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --dry-run \
  --configs configs/learning_rules/resnet18_cifar100_bp_tard.yaml \
  --seeds 42 123 \
  --learning-rule-lambdas 0.1 0.3 \
  --training-epochs 100 \
  --override learning_rule.method=bp_rtc_tard \
  --override learning_rule.task_gate_source=rtc \
  --override learning_rule.rtc_ridge=0.001 \
  --device cuda \
  --allow-dirty
```

### 5.3 Scheduling pilot

Compare always-on, fixed warmup, and metric-triggered onset before committing
full-grid compute.

```bash
# Always-on
python scripts/learning_rules/launch_mvp_grid.py --dry-run \
  --configs configs/learning_rules/resnet18_cifar100_bp_tard.yaml \
  --seeds 42 123 --learning-rule-lambda 0.1 --training-epochs 100 \
  --override learning_rule.warmup_epochs=0 --override learning_rule.ramp_epochs=0 \
  --device cuda --allow-dirty

# Metric-triggered onset, using rho_cap as the currently logged proxy.
python scripts/learning_rules/launch_mvp_grid.py --dry-run \
  --configs configs/learning_rules/resnet18_cifar100_bp_tard.yaml \
  --seeds 42 123 --learning-rule-lambda 0.1 --training-epochs 100 \
  --override learning_rule.schedule=metric_triggered \
  --override learning_rule.trigger_metric=rho_cap \
  --override learning_rule.trigger_threshold=0.015 \
  --override learning_rule.trigger_min_epoch=5 \
  --device cuda --allow-dirty
```

The exact two-axis trigger should eventually use `corr(I_X,t)` per layer. The
current hook can trigger on logged training diagnostics; `rho_cap` is only a
proxy until `I_X` is logged online.

### 5.4 RTP ridge check

The completed RTP pilot used average squared correlation. Before expanding RTP,
run the ridge peer-reconstructability implementation:

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --dry-run \
  --configs configs/learning_rules/resnet18_cifar100_bp_rtp.yaml \
  --seeds 42 123 \
  --learning-rule-lambda 0.3 \
  --training-epochs 100 \
  --override learning_rule.peer_proxy=ridge \
  --device cuda \
  --allow-dirty
```

## 6. Later Ablations

Run these only after the decision grid has at least one promising method:

- Warmup: always-on versus 10-epoch warmup versus metric-triggered onset.
- Strength: lambda in `{0.03, 0.1, 0.3}` around the winning method.
  Early ResNet-18/CIFAR-100 runs showed that `lambda=1e-3` makes the TARD
  contribution about `1e-5`, far below CE during training, so the original
  `{3e-4, 1e-3, 3e-3}` grid is too weak to test the mechanism.
- Peer set: full layer versus top-k correlated peers.
- Breadth: VGG-16 and MobileNetV2 as secondary backbones.
