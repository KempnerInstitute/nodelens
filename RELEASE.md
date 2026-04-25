# Release Checklist

This checklist keeps a public release reproducible without exposing private
draft files, cluster paths, logs, or model weights.

## Before Tagging

- Confirm that paper-facing code lives in `src/alignment/` and
  `projects/supernodes_scar/`.
- Confirm that private draft material under `drafts/` is not tracked for the
  public release.
- Run the test and documentation checks listed below.
- Rebuild the Hugging Face artifact bundle and verify checksums.
- Record the code tag, Hugging Face revision, and arXiv version together.
- Rebuild the artifact bundle after the final commit so
  `metadata/release_metadata.json` records `repo_dirty: false`.

## Local Checks

```bash
python -m pip install -e .
PYTHONPATH=src python -c "import alignment; print(alignment.__version__)"
pytest tests/unit -q
cd docs && make html
```

## Artifact Bundle

```bash
python projects/supernodes_scar/scripts/prepare_hf_artifacts.py \
  --output-dir outputs/supernodes_scar_hf \
  --clean

python projects/supernodes_scar/scripts/verify_hf_artifacts.py \
  outputs/supernodes_scar_hf
```

Upload the verified folder to:

```text
https://huggingface.co/datasets/hsafaai/supernodes-scar-artifacts
```

## GitHub Tag

Use an annotated tag for the public paper release:

```bash
git tag -a v0.2.0-supernodes-scar -m "Supernodes and SCAR public release"
git push origin v0.2.0-supernodes-scar
```

The current release workflow builds and checks the package, then creates a
GitHub release. It does not publish to PyPI.

## Suggested GitHub Release Notes

```text
Supernodes and SCAR public release.

Includes:
- loss-proxy and SCAR pruning code
- paper configs
- Supernodes and SCAR release docs
- scripts for staging Hugging Face artifacts

Artifacts:
https://huggingface.co/datasets/hsafaai/supernodes-scar-artifacts
```
