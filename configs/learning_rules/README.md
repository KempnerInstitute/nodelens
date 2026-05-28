# Learning-Rules Configs

This directory is reserved for reusable configs for the
replaceability-aware-learning project.

Planned first configs:

- `resnet18_cifar100_smoke_bp_tard.yaml`: fast wiring check, not a paper result.
- `resnet18_cifar100_bp_baseline.yaml`: standard BP.
- `resnet18_cifar100_bp_decov.yaml`: ungated covariance/decorrelation baseline.
- `resnet18_cifar100_bp_tard.yaml`: task-aware redundancy decorrelation.
- `resnet18_cifar100_bp_rtp.yaml`: task-gated peer-reconstructability penalty.

The full configs include both older pruning baselines (`composite`,
`cluster_aware`) and the two-axis-paper readouts with Taylor layer allocation:
`cap_ixy_hybrid_taylor`, `composite_twoaxis_ixy_hybrid_taylor`, and
`cluster_aware_stratified_twoaxis_adaptive_ixy_hybrid_taylor`.

Do not add large sweep grids here. Keep sweep-only material in private run
directories or generate it from a small locked template.

Run the smoke check with:

```bash
python scripts/run_experiment.py \
  --config configs/learning_rules/resnet18_cifar100_smoke_bp_tard.yaml \
  --allow-dirty
```

For paper-grade runs, override the seed:

```bash
python scripts/run_experiment.py \
  --config configs/learning_rules/resnet18_cifar100_bp_tard.yaml \
  --seed 123
```

For the full 4 method x 5 seed grid, use:

```bash
python scripts/learning_rules/launch_mvp_grid.py \
  --slurm \
  --device cuda \
  --allow-dirty
```
