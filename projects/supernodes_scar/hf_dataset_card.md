---
license: mit
pretty_name: "Supernodes and Halos Reproducibility Artifacts"
task_categories:
- text-generation
tags:
- large-language-models
- pruning
- structured-pruning
- mechanistic-interpretability
- loss-sensitivity
- reproducibility
size_categories:
- n<1K
---

# Supernodes and Halos Reproducibility Artifacts

This dataset repository contains derived artifacts for the paper
"Supernodes and Halos: Loss-Critical Hubs in LLM Feed-Forward Layers".

It is not a training dataset and does not include model weights. The files here
are intended to make the paper results inspectable and reproducible: compressed
result JSON files, generated figures, LaTeX tables, experiment configs, active
paper scripts, and checksums.

## Contents

```text
MANIFEST.json
MANIFEST.sha256
metadata/release_metadata.json
configs/
paper_artifacts/
paper_scripts/
raw_results/
docs/
```

## How To Use

Download the artifact bundle and inspect the manifest:

```bash
huggingface-cli download hsafaai/supernodes-scar-artifacts \
  --repo-type dataset \
  --local-dir supernodes_scar_artifacts

cd supernodes_scar_artifacts
sha256sum -c MANIFEST.sha256
```

The corresponding code release is available at:

```text
https://github.com/KempnerInstitute/alignment
```

Use the configs in `configs/` with `scripts/run_experiment.py` from the code
repo to rerun the experiments.

The file `metadata/result_sources.json` lists the public artifact path for each
locked result JSON used in the paper.

## Data And Model Sources

The experiments use public model families and public evaluation/calibration
datasets through their original providers and licenses. This artifact repository
does not redistribute those raw assets.

## Limitations

Some full reruns require substantial GPU memory and time, especially the 70B
validation. The artifact bundle is meant to support inspection and targeted
reproduction without requiring every reader to rerun all large-model jobs.
