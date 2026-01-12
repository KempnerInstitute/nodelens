## Reference: External Wanda Implementation (Vendored)

This directory vendors a reference implementation of **Wanda** (Sun et al., 2023) used as a baseline for LLM pruning.

### Purpose

- **Reference-only**: this code is kept to make it easy to audit our internal Wanda baseline against a known implementation.
- Our paper’s comparisons use **channel-adapted baselines** implemented in `src/alignment/pruning/strategies/llm_baselines.py`.
- When we run the paper-faithful *unstructured* Wanda reproduction baseline, we also use the internal implementation (for integration/consistency), but keep this reference code for cross-checking.

### Provenance

This code was merged via `origin/iss117_acllm_v3` (see merge commit on the target branch) and corresponds to the files:

- `src/alignment/pruning/strategies/external/wanda/data.py`
- `src/alignment/pruning/strategies/external/wanda/layerwrapper.py`
- `src/alignment/pruning/strategies/external/wanda/prune.py`

### Key details to match

- The running activation statistic:
  - `scaler_row` update uses the expected **sum of squared activations** (per feature) accumulated sequentially.
  - Pruning uses `W_metric = |W| * sqrt(scaler_row)`.
- Row-wise, stable sorting:
  - `sort_res = torch.sort(W_metric, dim=-1, stable=True)`.

