# Replaceability-Aware Learning Rules

This project is the in-repo home for the learning-rules follow-up to the
two-axis channel-information paper.

The working thesis is:

> Credit should be routed toward task-relevant channel content that is not
> already recoverable from same-layer peers.

## Layout

- Manuscript draft: `drafts/learning_rules/replaceability_learning_rules.tex`
- Reusable code: `src/nodelens/learning_rules/`
- Future runnable configs: `configs/learning_rules/`
- Future orchestration scripts: `scripts/learning_rules/`
- Raw run directories: `results/learning_rules/`
- Paper-ready derived tables: `projects/replaceability_learning_rules/paper_artifacts/tables/`

The draft directory is currently under the repo-wide ignored `drafts/` tree and
also has its own nested `.git` metadata. Keep writing there for now. When the
paper direction stabilizes, either keep only the source `.tex`/`.bib` files
tracked in the parent repo or move the manuscript into this project directory.

## MVP Decision

Use BP-TARD/RTC-TARD as the first paper spine:

1. ResNet-18/CIFAR-100 full diagnostic depth.
2. BP, BP+DeCov or covariance penalty, BP-TARD, RTC-gated TARD, and ridge-RTP.
3. Constrained-width training plus warmup versus always-on/metric-triggered onset
   as headline ablations.
4. Accuracy at fixed width, FLOPs-matched pruning AUC, duplicate task mass,
   non-replaceable task mass, RTC, and lesion/replacement metrics.

Treat RA-DFA as an upside mechanism probe until the BP regularizer produces a
clear capacity-use signal.

## Current Code Seed

`nodelens.learning_rules` now contains tensor utilities for:

- channel correlation matrices;
- Gaussian-MI transforms from correlation;
- peer reconstructability from a correlation matrix;
- task-aware redundancy loss;
- RTC-gated task-aware redundancy loss;
- task-gated peer-reconstructability penalty;
- variance-floor regularization;
- duplicate and non-replaceable task masses.

These are intentionally small building blocks and are now wired into the
existing cluster-analysis training path through the `learning_rule:` config
block.

## First Executable Grid

The first runnable grid is in `configs/learning_rules/`:

- BP baseline
- BP+DeCov-style ungated covariance penalty
- BP-TARD
- BP-RTP

Use the smoke config before launching long runs:

```bash
python scripts/run_experiment.py \
  --config configs/learning_rules/resnet18_cifar100_smoke_bp_tard.yaml \
  --allow-dirty
```

Then inspect the full command grid:

```bash
python scripts/learning_rules/launch_mvp_grid.py --dry-run
```

The executable run plan lives in
`projects/replaceability_learning_rules/RUNBOOK.md`.

As of the lambda pilot, the project has a mechanism result but not a
performance result. The immediate next experiments are the decision-grid pilots
in the runbook, especially constrained-width CIFAR-ResNet-18 and RTC-gated
TARD, before a full 200-epoch five-seed expansion.
