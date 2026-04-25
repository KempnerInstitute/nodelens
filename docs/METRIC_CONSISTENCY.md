# Metric Definitions & Sign Conventions (Theory <-> Code)

This document is a **codebase-facing** reference for the core metrics used throughout `src/nodelens/`.
It exists to prevent subtle drift in:
- **Formulas** (what is computed),
- **Keys** (how values are named/stored),
- **Sign conventions** (what "high" means when used for pruning/scoring).

It intentionally avoids referencing any paper draft; the canonical sources are the implementations under `src/nodelens/metrics/` and the experiment pipeline that stores per-layer metric arrays.

---

## Conventions (important)

### "Metric value" vs "importance score"

Many metrics are naturally "larger = more of something" (e.g., more redundancy).
But pruning code often needs an **importance score** with the convention:

- **Higher score = more important (keep)**
- **Lower score = less important (prune)**

Therefore:
- **Redundancy is typically used as a penalty** (we negate it or apply a negative weight).
- "High redundancy" ~ "more replaceable" => **more prunable**.

### Single-metric pruning directions (sanity controls)

In vision pruning experiments we often include both directions for a metric:
- `*_high`: **prune high values** (sometimes meaningful, sometimes an inverse control)
- `*_low`: **prune low values**

For redundancy specifically:
- **Meaningful**: `redundancy_high` (prune high redundancy)
- **Inverse control**: `redundancy_low` (prune low redundancy; usually worse)

---

## Metric definitions (core)

### 1) Rayleigh Quotient (RQ)

**Definition**
\[
\mathrm{RQ}(w;\Sigma_X) = \frac{w^\top \Sigma_X w}{w^\top w}
\]

**Implementation**
- `src/nodelens/metrics/rayleigh/rayleigh_quotient.py`
  - Computes covariance \(\Sigma_X\) from inputs (optionally class-conditioned) and returns per-output-channel RQ.

**Notes**
- RQ can span orders of magnitude; downstream code often uses \(\log(\mathrm{RQ})\).

---

### 2) Redundancy (Gaussian MI via correlation)

**Definition (Gaussian approximation)**
For scalar Gaussian variables \(Y_i,Y_j\) with correlation \(\rho\):
\[
I(Y_i;Y_j) = -\tfrac12 \log(1-\rho^2)
\]

We typically summarize "redundancy of channel \(i\)" as an **average MI** to other channels (or sampled references).

**Implementation**
- `src/nodelens/metrics/information/redundancy.py`
  - Computes correlations between projected outputs and converts to MI using the formula above.
  - Returns **nonnegative** redundancy values (more redundancy => larger).

**Pruning sign**
- When converted into an importance score: **use `-redundancy`** (or a negative weight).

---

### 3) Synergy (Gaussian PID, MMI axiom)

We use an MMI-based Gaussian PID synergy with respect to a target \(Z\) (e.g., a task signal):

**Definition**
\[
S(Z;Y_i,Y_j)= I(Z;[Y_i,Y_j]) - I(Z;Y_i) - I(Z;Y_j) + \min\{I(Z;Y_i),I(Z;Y_j)\}
\]
This simplifies to:
\[
S(Z;Y_i,Y_j)= I(Z;[Y_i,Y_j]) - \max\{I(Z;Y_i),I(Z;Y_j)\}
\]

Per-channel synergy is commonly computed as an average over a sampled set of partner channels.

**Implementation**
- `src/nodelens/metrics/information/gaussian_pid.py`

**Interpretation**
- Synergy is a **pair-structure descriptor**, not a scalar importance proxy; it is often weakly correlated with loss sensitivity within layers.

---

## Composite scoring (example)

A common composite importance score combines multiple signals:
- increase with alignment / task relevance,
- decrease with redundancy.

**Implementation**
- `src/nodelens/metrics/composite.py`

**Typical sign pattern**
- `+ logRQ`
- `+ synergy`
- `- redundancy`

---

## Where metric arrays live in experiment outputs

For vision runs, per-layer metric arrays are usually stored under (names may vary by experiment):
- `results.json["layer_metrics"][layer_name]["rq"]`
- `results.json["layer_metrics"][layer_name]["redundancy"]`
- `results.json["layer_metrics"][layer_name]["synergy"]`
- (optionally) `mi_in_proxy`, `task_mi`, etc.

Pruning strategies may consume these via "precomputed metrics" dicts.

---

## Quick verification snippet

```python
from nodelens.metrics import get_metric

rq = get_metric("rayleigh_quotient")         # RQ(w; Sigma_X)
red = get_metric("average_redundancy")      # -0.5 log(1-rho^2) aggregated per neuron
syn = get_metric("gaussian_pid_synergy_mmi")# MMI Gaussian PID synergy
```

---

## Why keep this doc?

- It prevents **silent sign flips** (especially for redundancy).
- It keeps metric naming/keys stable across refactors.
- It gives reviewers and future contributors a single, repo-local "what exactly is computed?" reference.
