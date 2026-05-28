# Learning-Rules Scripts

Small orchestration helpers for the replaceability-aware-learning project live
here.

Current entry points:

- `launch_mvp_grid.py`: run locally or submit the ResNet-18/CIFAR-100 MVP grid
  to SLURM.
- `summarize_mvp_grid.py`: collect raw run directories into durable CSV tables
  under `projects/replaceability_learning_rules/paper_artifacts/tables/` for
  the manuscript.
  The run table includes both the planned epoch budget and the actual epochs
  observed so in-flight runs are not mistaken for completed experiments.

Reusable implementation should stay in `src/nodelens/learning_rules/` or the
existing `nodelens` training/analysis modules.

Current launcher:

```bash
python scripts/learning_rules/launch_mvp_grid.py --dry-run
python scripts/learning_rules/launch_mvp_grid.py --slurm --device cuda --allow-dirty
python scripts/learning_rules/launch_mvp_grid.py \
  --dry-run \
  --configs configs/learning_rules/resnet18_cifar100_bp_tard.yaml \
            configs/learning_rules/resnet18_cifar100_bp_rtp.yaml \
  --seeds 42 123 \
  --learning-rule-lambdas 0.03 0.1 0.3 1.0 \
  --training-epochs 100
```
