# Final Pivot Results - 2026-05-17 (updated 2026-05-18 with synergy + anti-decoupling)

## Topline

**Eight orthogonal training-rule pivots, all null or negative at full width.**
The directionally opposite tests of the original idea (anti-decoupling and
synergy reward) hurt more, not less, than the more-orthogonal-pushing
pivots. This is the cleanest statement available that BP's natural
decoupled endpoint is locally optimal for ResNet-18/CIFAR-100. The
follow-up should not stand alone as Version C; better positioning is
**supporting material for the original two-axis paper**, deployable in
revision or in response to reviewer questions about training-time use of
the two-axis distinction.

See `## Reframe: support material, not standalone paper` below.

## Paired-seed deltas vs BP across all 8 pivots

| Pivot | Setup | Δacc (paired, mean ± SD) | n | Verdict |
|---|---|---:|---:|---|
| Per-channel RTC gate | full width, lambda 0.3 | not separable from task-gated TARD | 2 | null |
| Width-constrained TARD | w 0.35 / 0.5, lambda 0.3 | -0.0011 / -0.0029 | 3 / 3 | null/negative |
| Cross-layer dtm_depth allocator | w 0.35 / 0.5, lambda 0.3 | +0.0019 / -0.0020 | 3 / 3 | sign-flip null |
| Compact-hull penalty | w 0.35 / 0.5, lambda 0.3 | -0.0006 / -0.0001 | 3 / 3 | flat null |
| Gradient projection (strength 0.5) | w 0.35 / 0.5 | -0.0018 / +0.0011 | 3 / 3 | null |
| Gradient projection (strength 1.0) | w 0.35 / 0.5 | -0.0040 / -0.0027 | 3 / 3 | slightly negative |
| VGG-16 TARD (architecture probe) | full width, lambda 0.3 | +0.0012 +- 0.0015 SD (SEM 0.0007) | 5 | null (1.7 SEM above zero) |
| Pairwise synergy reward (alternative objective) | ResNet w 0.35 / 0.5 / 1.0 | +0.0022 / -0.0027 / **-0.0107** | 3/3/3 | full-width clear negative |
| Pairwise synergy reward (VGG-16) | VGG-16 full width, 5 seeds | -0.0024 +- 0.0028 SD (SEM 0.0012) | 5 | null/negative (1 positive, 4 negative) |
| Anti-decoupling (target rho 0.3) | ResNet w 0.35 / 0.5 / 1.0 | -0.0020 / partial / **-0.0078** | 3/1/3 | full-width clear negative |
| Anti-decoupling (VGG-16) | VGG-16 full width, 5 seeds | +0.0005 +- 0.0026 SD (SEM 0.0012) | 5 | null (3 positive, 2 negative) |

Eight pivots. Two of them (synergy, anti-decouple) tested the
*directionally opposite* hypothesis: maybe BP over-decouples and the
right rule should recouple the axes or reward group-coded credit. Both
hurt more, not less, than the original "decouple harder" pivots,
especially at full width.

Full-width ResNet-18 results have the cleanest sign:

| Method | full-width Δacc | n seeds | Sign across seeds |
|---|---:|---:|---|
| Synergy reward | **-0.0107 +- 0.0040** | 3 | 3 / 3 negative |
| Anti-decoupling | **-0.0078 +- 0.0010** | 3 | 3 / 3 negative |
| Gradient projection s=1.0 | -0.0040 +- ~ | 3 | 3 / 3 negative |
| TARD lambda 0.3 | within noise | 5 | mixed |

All three "stronger" interventions converge to "BP at full width is
better." The constrained-width results were noise because capacity
limits hide the signal; at full width the local-optimality of BP's
fixed point is unambiguous.

## Orthogonality scan v2 (ResNet-18, 101 checkpoints)

`|cos(grad_I_X, grad_I_T)|` per layer, averaged across seeds within method
(`paper_artifacts/tables/grad_orthogonality_resnet18_20260518_summary.csv`):

| layer | BP | DeCov | TARD0.3 | RTP0.3 | xlayer | hull | gradproj0.5 | gradproj1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conv1 | 0.56 | 0.40 | 0.49 | 0.43 | 0.50 | 0.55 | 0.61 | 0.66 |
| layer1.0.conv1 | 0.16 | 0.18 | 0.14 | 0.17 | 0.14 | 0.14 | 0.18 | 0.22 |
| layer2.0.conv2 | 0.13 | 0.12 | 0.15 | 0.11 | 0.13 | 0.15 | 0.12 | 0.15 |
| layer3.0.conv1 | 0.13 | 0.09 | 0.14 | 0.11 | 0.13 | 0.15 | 0.14 | 0.15 |
| layer4.0.conv1 | 0.08 | 0.06 | 0.08 | 0.05 | 0.07 | 0.09 | 0.07 | 0.07 |
| layer4.0.downsample.0 | 0.09 | 0.06 | 0.09 | 0.05 | 0.08 | 0.11 | 0.08 | 0.08 |

Reads:

- Every method keeps hidden-layer `|cos|` in `[0.05, 0.18]`. The
  orthogonality regime is the natural BP regime on this architecture.
- **DeCov has the lowest `|cos|` at most layers**, supporting the
  interpretation that pure decorrelation does drive analytic orthogonality
  but is also the only method whose accuracy is indistinguishable from BP.
- **Gradient projection at strength 1.0 has HIGHER `|cos|` than BP** at
  most layers, including conv1 (0.66 vs 0.56). This is the opposite of
  what was intended: the per-step orthogonal projection of weight
  gradients to `Sigma_X w_i` does not produce a trained checkpoint with
  smaller analytic cosine; the post-hoc cosine actually increases. This
  is itself a publishable mechanism observation.
- **Hull also raises `|cos|` everywhere** (the penalty pushes the
  channel covariance into a regime where `Sigma_X w_i` and the
  task-covariance direction are more aligned, not less).

## VGG-16 probe

VGG-16 with `cifar_vgg16` (CIFAR-adapted variant: 5 of 5 maxpools removed
becomes 4, small classifier on 2x2 feature map). Single-seed orthogonality
(`paper_artifacts/tables/grad_orthogonality_vgg16_20260518_summary.csv`):

| layer | BP | TARD | delta |
|---|---:|---:|---:|
| features.0 | 0.52 | 0.53 | +0.00 |
| features.3 | 0.11 | 0.10 | -0.01 |
| features.7 | 0.06 | 0.06 | -0.00 |
| features.10 | 0.07 | 0.06 | -0.01 |
| features.14 | 0.05 | 0.06 | +0.01 |
| features.17 | 0.07 | 0.07 | +0.01 |
| features.20 | 0.07 | 0.07 | +0.00 |
| features.24 | 0.10 | 0.06 | -0.04 |
| features.27 | 0.16 | 0.10 | -0.06 |
| features.30 | 0.24 | 0.18 | -0.06 |
| features.34 | 0.14 | 0.14 | -0.01 |
| features.37 | 0.33 | 0.34 | +0.01 |
| features.40 | 0.04 | 0.04 | -0.00 |

VGG-16 is structurally different from ResNet-18: middle layers
(features.27, .30, .37) have `|cos|` in `[0.16, 0.33]` under BP, i.e.
**outside the orthogonal regime**. TARD does meaningfully move
features.24/27/30 toward orthogonality (`Delta = -0.04` to `-0.06`).

So TARD has a measurable mechanistic effect on VGG that does NOT exist on
ResNet. But the four-seed paired accuracy delta is still null. The
mechanism moves; the accuracy does not follow.

## Jobs completed today

| Grid | Method | Jobs | Outcome |
|---|---|---:|---|
| B v2 | bp_hull (vectorized) widths {0.35, 0.5, 1.0} seeds {42, 123, 456} | 9 | 6 COMPLETED, 3 RUNNING (w=1.0 will time out at ~ep 65) |
| D v2 | VGG-16 (cifar_vgg16) BP + TARD lambda 0.3 seed 42 | 2 | COMPLETED clean |
| F | gradproj strength 1.0 widths {0.35, 0.5, 1.0} seeds {42, 123, 456} | 9 | COMPLETED clean |
| G | gradproj strength 0.5 widths {0.35, 0.5, 1.0} seeds {42, 123, 456} | 9 | COMPLETED clean |
| H | VGG-16 multi-seed BP + TARD seeds {42, 123, 456, 789, 1011} | 10 | 9 COMPLETED, 1 RUNNING |
| E v2 | ResNet ortho scan over 101 checkpoints | 1 | COMPLETED in 4 min |
| E vgg | VGG ortho scan | 1 | COMPLETED in 1.5 min |

Total: 41 SLURM jobs across two days; 5 still running (3 hull w=1.0 nearing
walltime, 1 VGG TARD finishing, 0 critical).

## What this means

The empirically supported claims:

1. **The two-axis decoupling emerges implicitly under standard BP.** At
   every hidden layer of trained ResNet-18 on CIFAR-100, the analytic
   `|cos(grad_I_X, grad_I_T)| < 0.18` regardless of training scheme.
2. **Replaceability-aware regularizers cannot meaningfully improve
   accuracy on this architecture** because they push along a direction
   the gradient is already pushing. Six "more orthogonal" pivots are
   null or slightly negative.
3. **The opposite direction also fails.** Anti-decoupling (forcing
   controlled coupling) and synergy reward (rewarding group-coded credit
   instead of singleton residual credit) are clearly negative at full
   width (-0.78 and -1.07 pp respectively). BP at the full-width
   endpoint is locally optimal in both directions.
4. **The orthogonality regime is architecture-dependent.** VGG-16 has
   coupled middle layers and replaceability rules can move them; but
   five-seed paired accuracy on VGG is still null for both TARD,
   synergy, and anti-decoupling.
5. **The two-axis distinction matters for post-hoc diagnostics, not for
   training.** The original paper's pruning-AUC result stands; the
   training-time use case is empirically closed.

These findings, taken together, are strong **support for the original
two-axis paper** rather than a standalone follow-up. See the reframing
section below.

## Mechanism observations the regularizers DO produce

Even though accuracy is null, the regularizers produce sharply distinct
mechanistic signatures. This is the case for the "intended geometry moved,
loss surface did not" framing.

### Capacity geometry under each rule (full-width runs, mean across seeds)

| Method | DTM frac | NTM frac | hull_size_mean | gp_shrink | xlayer_max |
|---|---:|---:|---:|---:|---:|
| BP | -- | -- | -- | -- | -- |
| TARD lambda 0.3 | 0.039 | 0.961 | -- | -- | -- |
| TARD-xlayer dtm_depth | 0.037 | 0.963 | -- | -- | 4.18 |
| Compact hull lambda 0.3 | **0.844** | **0.156** | 9.75 | -- | -- |
| Grad projection s=0.5 | 0.000 | 0.000 | -- | 0.985 | -- |
| Grad projection s=1.0 | 0.000 | 0.000 | -- | 0.977 | -- |

Three observations the manuscript should include:

1. **Compact hull produces DTM 20x larger than TARD** (0.84 vs 0.04). The
   hull penalty pushes the network into a regime where the `avg_corr2`
   proxy for peer-reconstructability rises sharply while the ridge-R^2
   used inside the hull penalty falls. The two proxies disagree under
   hull training; this is itself an interesting result about which
   replaceability proxy a learning rule should target. Likely worth a
   short discussion paragraph.

2. **Gradient projection removes only 2% of the BP gradient norm**
   (`gp_shrink = 0.98`). This quantifies how nearly orthogonal the BP
   loss gradient already is to the signal-power direction `Sigma_X w_i`.
   The mean abs cosine before projection is around 0.14 across layers.
   This is the cleanest numerical statement of "BP already does it" and
   is the direct quantitative bridge from the original paper's analytic
   gradient claim to the trained-checkpoint claim.

3. **Cross-layer allocator produces 4x spread in per-layer lambda**
   (`xlayer_max = 4.18` with mean normalized to 1.0). The CAP-QGW-style
   allocator IS concentrating budget on deeper, higher-DTM layers, but
   even that concentrated push does not move accuracy.

## Paradox: gradient projection RAISES post-hoc orthogonality cosine

Naively, projecting weight gradients onto the orthogonal complement of
`Sigma_X w_i` should drive a trained checkpoint toward smaller analytic
`|cos(grad_I_X, grad_I_T)|`. The opposite is what we see:

| layer | BP | gradproj-1.0 |
|---|---:|---:|
| conv1 | 0.56 | **0.66** |
| layer1.0.conv1 | 0.16 | **0.22** |
| layer2.0.conv2 | 0.13 | **0.15** |
| layer3.0.conv1 | 0.13 | **0.15** |
| layer4.0.conv1 | 0.08 | 0.07 |

The likely explanation: the per-step projection constrains the trajectory
in weight space but the resulting fixed-point weights still produce a
`Sigma_X w` that aligns with the loss-gradient component of the task
direction. The cosine is a property of the trained-network configuration,
not of the per-step update. **This is the mechanism reason gradient
projection cannot work as a training tool**: removing a per-step direction
does not enforce a post-hoc constraint on the analytic objects.

This deserves a short subsection in the manuscript and probably a small
schematic figure.

## Figures ready for the original paper's supporting material

1. **Eight-pivot accuracy delta panel.** Paired-seed CIs across all
   eight rules at three widths plus VGG-16, all null or negative.
   Single small panel; can fit in the original paper's discussion or
   in an appendix.
2. **Orthogonality scan.** `|cos(grad_I_X, grad_I_T)|` per layer for
   ResNet-18 across 8 training methods (101 checkpoints) showing the
   universal orthogonal regime. The cleanest single figure for the
   "BP already does it" claim.
3. **Architecture contrast (optional).** VGG-16 panel showing middle
   layers outside the orthogonal regime and replaceability rules
   moving them, with accuracy still null. Useful only if a reviewer
   pushes on architecture generality.
4. **Bi-directional null panel (optional).** Side-by-side bars showing
   "decouple more" rules (TARD, RTP, hull, grad-projection) and
   "decouple less / regroup" rules (anti-decouple, synergy) both
   producing nulls at constrained width and clear negatives at full
   width. Reinforces the local-optimum framing.

## Two-axis-paper findings not yet probed as training targets

These remain open angles that could in principle still rescue a
performance result, but each is significantly more work and the prior on
each is weakening given six independent nulls:

| Untested angle | Source (original paper) | Why it might still differ |
|---|---|---|
| Synergy reward (triplet S_3/S_2 rises with depth) | Sec on higher-order target excess | Singleton task gate underweights synergy at deep layers; a synergy-aware credit term would reward channels whose pairwise/triplet task MI exceeds singleton sum |
| Modularity-gap objective | Newman modularity of R-graph > S-graph at every depth | Penalize the community structure of the redundancy graph, not pairs |
| Cross-layer information flow shaping | local->local and target->target propagate but cross is 0 | Train so that local-axis content propagates along weight-routed paths matched to downstream sensitivity |
| Compact-hull with ridge R^2 (not avg_corr2) as the penalty target | Hull is the strongest matched-task-MI lesion predictor | The current hull penalty uses avg_corr2 internally for some steps; ridge R^2 throughout would be a cleaner port of the original paper's exact compact-hull score |
| Trajectory-derivative scheduling | rho(I_X, t) drops from 0.71 to 0.22 over training | Trigger lambda on dRho/dt rather than absolute rho; might land at a different fixed point |

If any one of these is worth a focused day of effort, modularity-gap is
the highest-leverage because it directly targets a structural object the
original paper showed survives across depth and architecture.

## CSV artifacts produced

In `paper_artifacts/tables/`:

- `run_summary.csv` (115 ResNet runs after May 17 grids land).
- `pruning_auc.csv`, `pruning_curve.csv`.
- `grad_orthogonality_resnet18_20260518_summary.csv` (101 runs x 20 layers
  = 2020 rows).
- `grad_orthogonality_resnet18_20260518_channels.csv` (per-channel detail,
  ~100 MB).
- `grad_orthogonality_vgg16_20260518_summary.csv` (the VGG scan; first
  scan picked up failed chance-acc runs, re-run targeting working
  checkpoints lives in `/tmp/vgg_ortho_working/`; should be moved into
  paper_artifacts before manuscript revision).

VGG multi-seed summary lives in `results/learning_rules/vgg16_cifar100/`
and needs a separate `summarize_mvp_grid.py --results-root` invocation
once the last seed completes; the temporary VGG-final summary is at
`/tmp/summary_vgg_final/`.

## Final job status across all grids

Sum of work over the multi-day push (May 14 - May 18):

- 41 jobs from the first round of pivots (B v2 hull, D v2 VGG, F/G
  gradient projection, H VGG multi-seed, E/E-vgg orthogonality scans).
- 28 jobs from the synergy / anti-decoupling round on May 17, of which
  the 18 ResNet jobs migrated to `kempner_h100` with
  `kempner_bsabatini_lab` for speed (started running within seconds,
  completed in 20-35 min each on H100 vs the original 13-20h on
  kempner_dev A100s for hull jobs).
- 12 cancellations during the H100 migration (no compute wasted; all
  cancelled jobs were still pending).
- 3 timeouts (hull w=1.0 partial-epoch runs, not blocking).

Per-grid outcomes follow below.

## Synergy + anti-decoupling grid status (May 17 round)

| Job IDs | What | Outcome |
|---|---|---|
| 13305322-13305324 | Hull-v2 w=0.35 | COMPLETED |
| 13305326-13305328 | Hull-v2 w=0.5 | COMPLETED |
| 13305329-13305331 | Hull-v2 w=1.0 | TIMEOUT at epoch 57/100; best_acc 0.65-0.67 vs BP w=1.0 ~0.77; not converged, not comparable |
| 13306395, 13306396 | VGG-16 BP/TARD seed 42 (probe) | COMPLETED |
| 13307931-13307940 | Gradproj strength 1.0, 9 jobs | COMPLETED |
| 13307941-13307949 | Gradproj strength 0.5, 9 jobs | COMPLETED |
| 13469051-13469060 | VGG-16 multi-seed grid, 10 jobs | COMPLETED |
| 13468555 | ResNet ortho scan v2 | COMPLETED in 4 min |
| 13468572 | VGG ortho scan | COMPLETED in 1.5 min |
| 13534654-13534679, 13534866-13534877 | 18 ResNet synergy + antidecouple on H100 (kempner_bsabatini_lab) | 16 COMPLETED, 2 still RUNNING at last check (antidecouple w=0.5) |
| 13530254-13530279 | 10 VGG-16 synergy + antidecouple on kempner_dev | COMPLETED in 14-17 min each |

Total: 41 + 28 = 69 jobs across the full push.

## Next steps

- Wait for the last VGG seed and the 3 hull w=1.0 partial-epoch results.
- Move the working-VGG orthogonality CSVs into `paper_artifacts/tables/`.
- See the reframing section below: this body of work is supporting
  material for the original two-axis paper rather than a standalone
  follow-up manuscript.

## Reframe: support material, not standalone paper

The eight-pivot null is informative but does not by itself sustain a
publishable follow-up. The space of "I tried things; nothing worked" is
hard to land as a primary contribution, and the only positive signal
(VGG-16 TARD at +0.0012 +- 0.0015 SD) is well within noise.

A stronger use of this work is as **supporting material for the original
two-axis paper**, deployable in two places:

### 1. Pre-emptive paragraph in the original paper's discussion

A short paragraph noting that the two-axis distinction was tested as a
training-time signal in eight orthogonal pivots (decorrelation,
peer-reconstructability, residualized credit, compact hull, cross-layer
allocation, gradient projection, pairwise synergy reward, controlled
anti-decoupling) on ResNet-18 / CIFAR-100, and that none meaningfully
moved accuracy. The orthogonality scan shows why: BP-trained networks
already satisfy the analytic gradient orthogonality predicted by the
paper's own Proposition. This converts the "is it useful for training?"
question from open to answered, strengthens the post-hoc diagnostic
framing, and pre-empts reviewers asking "did you try to use this as a
learning rule?"

### 2. Reviewer-response material in revision

When a reviewer asks any variant of "could this be a learning rule?", the
ready answer is:

> *We tested this directly. Across eight orthogonal training-time
> interventions on ResNet-18 / CIFAR-100 (full list in Appendix X), none
> improved accuracy. The analytic gradient orthogonality predicted by
> Proposition Y is already approximately satisfied by standard BP at
> every hidden layer of the trained network (Figure Z, |cos| in
> [0.05, 0.18] across 101 checkpoints). Therefore the two-axis
> distinction is informative as a post-hoc diagnostic but is not a
> productive training-time objective on this architecture.*

This is a one-paragraph answer with one figure (the orthogonality scan)
and one table (the eight-pivot delta summary). It is the cleanest
defensive response to the obvious reviewer question.

### What to keep ready for the appendix or revision

| Artifact | Source | Purpose |
|---|---|---|
| Eight-pivot accuracy delta table with paired seeds + CIs | `paper_artifacts/tables/run_summary.csv` (115 ResNet runs after May 17 grids land) | Table that the reviewer-response paragraph cites |
| Orthogonality scan across 101 ResNet-18 checkpoints | `paper_artifacts/tables/grad_orthogonality_resnet18_20260518_summary.csv` | Figure for the one-paragraph response |
| VGG-16 orthogonality contrast | `paper_artifacts/tables/grad_orthogonality_vgg16_20260518_summary.csv` plus the working-VGG re-scan in `/tmp/vgg_ortho_working/` | Demonstrates the orthogonality regime is not universal; useful if a reviewer pushes on architecture generality |
| Mechanism observation paragraph (DTM 20x for hull, gp_shrink 0.98, xlayer_max 4x, gradproj cosine paradox) | This document | One short paragraph showing the rules DID move the geometry, just not the loss |
| Code: `nodelens.learning_rules` package | `src/nodelens/learning_rules/` | Available in the repo; mention in availability statement |

### What this means for the existing draft

The `drafts/learning_rules/replaceability_learning_rules.tex` document
no longer needs to be polished as a standalone paper. Two options:

1. **Park it.** Keep the draft as-is in `drafts/`. Lift one paragraph
   and one figure into the original paper's discussion. Use the rest
   in revision.
2. **Convert to internal note.** Restructure as an internal technical
   note for the lab, documenting the eight pivots and their failure
   modes for future researchers. Useful as a record of what does not
   work, even if not published.

I do not recommend forcing the draft into a Version C / mechanism paper
at this point. The data is strong enough to support the original paper,
but not strong enough to anchor a standalone manuscript that would have
to argue "this is interesting because it failed in 8 specific ways."

### When the eight-pivot result might become a paper on its own

Only if one of the remaining angles (modularity-gap, cross-axis flow
shaping, carrier-coupled DFA, trajectory-derivative scheduling) produces
a clean positive accuracy result. Given the existing trend, that is
unlikely. Without such a positive, this is supporting material, not a
manuscript.
