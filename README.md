# NodeLens

Node and channel metrics for neural network interpretability, importance, and interventions.

[![Tests](https://github.com/KempnerInstitute/NodeLens/actions/workflows/test.yml/badge.svg)](https://github.com/KempnerInstitute/NodeLens/actions/workflows/test.yml)
[![Lint](https://github.com/KempnerInstitute/NodeLens/actions/workflows/lint.yml/badge.svg)](https://github.com/KempnerInstitute/NodeLens/actions/workflows/lint.yml)
[![Documentation](https://github.com/KempnerInstitute/NodeLens/actions/workflows/docs.yml/badge.svg)](https://github.com/KempnerInstitute/NodeLens/actions/workflows/docs.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

NodeLens is a research codebase for studying which channels, neurons, and
features matter most for model behavior. It combines activation capture,
importance metrics, redundancy and information measures, structured
interventions, and report generation in one configuration-driven workflow. The
Python package is imported as `nodelens`.

## What The Code Does

```text
Model + data
    |
    v
Activation and gradient capture
    |
    v
Channel and node metrics
    |-- activation statistics
    |-- Rayleigh quotient and spectral alignment
    |-- mutual information, redundancy, and synergy
    |-- gradients, curvature, Taylor scores, and loss proxies
    |
    v
Analysis and interventions
    |-- identify outliers or loss-critical cores
    |-- cluster channels by metric profile
    |-- test ablations, pruning, and sensitivity probes
    |-- generate figures, tables, summaries, and manifests
```

Core capabilities:

- Metric analysis for MLPs, CNNs, transformers, and Hugging Face causal LMs.
- Node and channel scoring with activation, alignment, information,
  redundancy, gradient, curvature, and loss-sensitive metrics.
- Structured pruning and ablation tools for testing whether high-scoring
  channels are functionally important.
- Clustering and cross-layer analyses for studying local organization,
  redundancy, and downstream dependence.
- Project workflows under `projects/` that show how to reproduce concrete
  analyses with the shared library.

## Installation

```bash
git clone https://github.com/KempnerInstitute/NodeLens.git
cd NodeLens
conda env create -f environment.yml
conda activate nodelens
pip install -e .
```

For documentation and optional analysis tools:

```bash
pip install -e .[all]
```

## Quick Start

Run experiments from YAML configs:

```bash
# Small vision smoke test
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# CNN pruning and clustering
python scripts/run_experiment.py --config configs/vision_prune/resnet18_cifar10_full.yaml

# LLM channel analysis and structured FFN pruning
python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_unified.yaml
```

Use metrics directly from Python:

```python
from nodelens.metrics import get_metric, list_metrics

print(list_metrics())

metric = get_metric("rayleigh_quotient")
scores = metric.compute(inputs=layer_inputs, weights=layer_weights)
```

## Project Workflows

Reusable library code lives in `src/nodelens`. Project folders contain the
configs, small helper scripts, and artifact descriptions needed to reproduce a
specific analysis with the shared package.

Current project:

- `projects/supernodes_scar/`: workflow for the Supernodes and SCAR study of
  loss-sensitive FFN channels in LLMs.

The Supernodes and SCAR project also has a public derived-artifact dataset:

- `https://huggingface.co/datasets/hsafaai/supernodes-scar-artifacts`

## Main Concepts

| Area | Examples |
|------|----------|
| Activation metrics | `activation_l2_norm`, `activation_variance`, `activation_outlier_index` |
| Alignment metrics | `rayleigh_quotient`, `delta_alignment` |
| Information metrics | `mutual_information_gaussian`, `pairwise_redundancy_gaussian`, `gaussian_pid_synergy_mmi` |
| Loss-sensitive metrics | `scar_activation_power`, `scar_taylor`, `scar_curvature`, `scar_loss_proxy` |
| Pruning strategies | `magnitude`, `alignment`, `composite`, `cluster_aware`, `random` |

## Repository Layout

```text
NodeLens/
|-- configs/
|   |-- examples/           # Small runnable configs
|   |-- prune_llm/          # LLM channel-analysis and pruning configs
|   `-- vision_prune/       # Vision pruning and clustering configs
|-- projects/               # Reproducible project workflows
|-- scripts/
|   |-- run_experiment.py   # Main experiment entry point
|   `-- run_analysis.py     # Post-hoc analysis entry point
|-- src/nodelens/
|   |-- analysis/           # Visualization, clustering, cascade analysis
|   |-- experiments/        # Experiment classes
|   |-- metrics/            # Importance and information metrics
|   |-- models/             # Model wrappers
|   |-- pruning/            # Pruning strategies
|   `-- services/           # Activation capture, scoring, and mask utilities
|-- tests/                  # Unit and integration tests
`-- docs/                   # Documentation
```

## Documentation

- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
- [LLM Guide](docs/llm_guide.md)
- [Metric Consistency](docs/METRIC_CONSISTENCY.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Supernodes and SCAR Workflow](projects/supernodes_scar/README.md)

Build the Sphinx docs locally:

```bash
cd docs
make html
```

## Testing

```bash
pytest tests/
pytest tests/unit/ -v
```

## Citation

If you use NodeLens, cite the repository metadata in `CITATION.cff`. If you use
a project workflow or public artifact dataset, also cite the associated paper
and artifact record.

## License

This repository is released under the MIT license. See [LICENSE](LICENSE).
