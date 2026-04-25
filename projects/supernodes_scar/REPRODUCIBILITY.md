# Reproducibility Notes

This page describes the local workflow used for the paper release. It separates
three different tasks: rerunning experiments, rebuilding figures and tables from
locked outputs, and rebuilding the arXiv PDF.

## 1. Rerun Experiments

Install the code:

```bash
conda env create -f environment.yml
conda activate alignment
pip install -e .
```

Run the main 8B config:

```bash
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_unified.yaml \
  --base-output-dir /path/to/results/Prune_LLM
```

The main paper configs are listed in `projects/supernodes_scar/README.md`.
Large runs, especially the 70B validation, require substantial GPU memory and
should usually be launched through the local cluster workflow.

## 2. Rebuild Figures And Tables From Locked Outputs

The paper figures and tables are regenerated from locked result JSON files. The
release bundle stores those JSON files under `raw_results/` as sanitized
`.json.gz` files and records their public names in:

```text
metadata/result_sources.json
```

The local working copy keeps the original output folders so the active paper
scripts can be rerun without downloading anything:

```bash
python drafts/LLM_prune/paper/scripts/regenerate_fig1_overview.py
python drafts/LLM_prune/paper/scripts/regenerate_fig2_halo.py
python drafts/LLM_prune/paper/scripts/generate_70b_scale_figures.py
python drafts/LLM_prune/paper/scripts/generate_lp_vs_activation_overlap_figure.py
python drafts/LLM_prune/paper/scripts/generate_lp_vs_activation_supernode_figure.py
python drafts/LLM_prune/paper/scripts/collect_paper_artifacts.py \
  --results-base /path/to/results/Prune_LLM/PAPER \
  --draft-dir drafts/LLM_prune
```

The public artifact bundle also includes the active paper scripts under
`paper_scripts/`. Some scripts use path constants because they were designed for
the locked local paper tree; update those constants or run the script from this
repository with the original output folders available.

## 3. Rebuild The Paper

The paper has one shared body file:

```text
drafts/LLM_prune/paper_body.tex
```

Build the arXiv and anonymous versions:

```bash
cd drafts/LLM_prune
./compile_pdf.sh paper_arxiv.tex
./compile_pdf.sh paper_icml.tex
```

## 4. Build And Verify The Hugging Face Bundle

```bash
python projects/supernodes_scar/scripts/prepare_hf_artifacts.py \
  --output-dir outputs/supernodes_scar_hf \
  --clean

python projects/supernodes_scar/scripts/verify_hf_artifacts.py \
  outputs/supernodes_scar_hf
```

The verifier checks:

- `MANIFEST.sha256`
- absence of Python caches, LaTeX build files, PDFs, checkpoints, model weights,
  and raw datasets
- absence of private local paths in plain text and compressed `.json.gz` files

## 5. Local Storage Policy

Uploading to Hugging Face is not a replacement for local retention. Keep:

- the frozen HF bundle under `outputs/supernodes_scar_hf`
- the original locked result folders used to regenerate paper figures
- the arXiv source bundle under `drafts/LLM_prune/arxiv_bundle.tar.gz`
- the Git commit or release tag associated with the upload

This lets future work continue from the exact paper state while the public HF
repo remains a clean, portable snapshot.
