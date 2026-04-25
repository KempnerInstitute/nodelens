# Reproducibility Notes

This page describes how to rerun the Supernodes and SCAR workflow with
NodeLens and how to inspect the derived artifacts. It focuses on public inputs:
repository configs, public model identifiers, public datasets, and the artifact
bundle.

## 1. Install NodeLens

From the repository root:

```bash
conda env create -f environment.yml
conda activate nodelens
pip install -e .
```

Install optional dependencies when building documentation, running large LLM
experiments, or using all plotting utilities:

```bash
pip install -e .[all]
```

## 2. Run A Paper Config

Run the main Llama-3.1-8B workflow:

```bash
python scripts/run_experiment.py \
  --config configs/prune_llm/llama3_8b_unified.yaml \
  --base-output-dir outputs/supernodes_scar_runs
```

Useful related configs are listed in `projects/supernodes_scar/README.md`.
The 70B configs are targeted validation runs and require a large-memory or
parallel model-loading setup.

## 3. Inspect Outputs

Each experiment writes a timestamped job directory under the selected
`--base-output-dir`. A typical run contains:

```text
experiment_config.yaml
logs/
results/
figures/
analysis/
```

The most important outputs are the per-layer metric arrays, pruning summaries,
ablation results, and generated figure inputs under `results/` and `analysis/`.

## 4. Use The Public Artifact Dataset

The artifact dataset provides derived outputs from the runs used in the paper.
Download it with:

```bash
huggingface-cli download hsafaai/supernodes-scar-artifacts \
  --repo-type dataset \
  --local-dir supernodes_scar_artifacts
```

Verify the downloaded files:

```bash
cd supernodes_scar_artifacts
sha256sum -c MANIFEST.sha256
python -m json.tool MANIFEST.json | head
```

`metadata/result_sources.json` maps paper-facing result names to public
artifact paths. See `ARTIFACTS.md` for the full layout.

## 5. Build A Local Derived-Artifact Bundle

If compatible result folders are available locally, build a clean bundle with:

```bash
python projects/supernodes_scar/scripts/prepare_hf_artifacts.py \
  --output-dir outputs/supernodes_scar_hf \
  --clean

python projects/supernodes_scar/scripts/verify_hf_artifacts.py \
  outputs/supernodes_scar_hf
```

The verification step checks the manifest, checksums, and exclusion rules for
public derived artifacts.

## Notes On External Inputs

Model weights and benchmark datasets are not redistributed here. Users should
download them from their original providers and follow the relevant licenses.
Calibration and evaluation choices are encoded in the YAML configs whenever
they are needed for reproduction.
