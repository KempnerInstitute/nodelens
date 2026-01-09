### SCAR paper experiment suite (batch + collection)

This folder contains **SLURM batch scripts** that run a complete ICML-style paper suite:

- **Main results + generalization** (4 models)
- **Key controls / ablations** on Llama-3.1-8B:
  - **LP-no-protect** + **remove-supernodes-early** (mode=high) control
  - **Protect+Wanda** and **Protect+Magnitude** (baseline + supernode protection)
  - **Positive-only redundancy** ablation (anti-correlation does NOT count as redundancy)
  - **Calibration sensitivity** sweep (dataset + sample-count)

All jobs write to a single `OUTPUT_BASE` using the unified job directory structure:

`{OUTPUT_BASE}/{experiment_name}_{timestamp}_{job_id}/`

### How to run

- **Set output base** (or let scripts use the default in each file):

```bash
export OUTPUT_BASE="/n/holylfs06/LABS/kempner_project_b/Lab/alignment/Prune_LLM"
```

- **Submit the full suite**:

```bash
bash slurm_jobs/prune_llm/paper/submit_suite.sh
```

### How to collect artifacts (tables + placeholder figures)

After jobs finish:

```bash
python scripts/collect_paper_artifacts.py \
  --results-base "$OUTPUT_BASE" \
  --draft-dir /n/holylabs/kempner_dev/Users/hsafaai/Code/alignment/drafts/LLM_prune
```

This will:
- write LaTeX snippets to `drafts/LLM_prune/paper_artifacts/tables/*.tex`
- copy key plots into `drafts/LLM_prune/placeholder_*.png` so the draft auto-fills.

