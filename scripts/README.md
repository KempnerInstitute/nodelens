# Scripts

Entry points for running experiments and analysis.

## Main Scripts

### run_experiment.py

Run experiments from YAML configuration files.

```bash
# Basic usage
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# With overrides
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml \
    --device cuda:0 \
    --seed 42 \
    --output-dir results/my_run

# Regenerate plots from existing results
python scripts/run_experiment.py \
    --config configs/examples/resnet_pruning.yaml \
    --analysis-only \
    --experiment-dir results/existing_run
```

Options:
- `--config PATH`: YAML configuration file (required)
- `--device STRING`: Override device (cuda:0, cpu)
- `--seed INT`: Override random seed
- `--output-dir PATH`: Override output directory
- `--analysis-only`: Regenerate plots from existing results
- `--experiment-dir PATH`: Existing experiment directory (with --analysis-only)

### run_analysis.py

Generate visualizations from experiment results.

```bash
# Run analysis from config
python scripts/run_analysis.py --config configs/analysis/vision_figures.yaml

# Quick analysis
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots --quick

# Specific analyses
python scripts/run_analysis.py --results-dir ./results \
    --analyses histograms scatter_plots pruning_curves \
    --output-dir ./custom_plots
```

Options:
- `--config PATH`: Analysis config YAML file
- `--results-dir PATH`: Directory with experiment result JSONs
- `--results-file PATH`: Single result JSON file
- `--output-dir PATH`: Output directory for plots
- `--format {png,pdf,svg}`: Output image format
- `--analyses LIST`: Which analyses to run
- `--quick`: Run all analyses with default settings

## Configuration Files

### Experiment Configs

| Directory | Description |
|-----------|-------------|
| `configs/template.yaml` | Full parameter reference |
| `configs/examples/` | Example experiment configs |
| `configs/projects/` | Project-specific configs |

### Analysis Configs

| Config | Description |
|--------|-------------|
| `configs/analysis/llm_paper_figures.yaml` | LLM paper figures |
| `configs/analysis/vision_figures.yaml` | Vision experiment figures |
| `configs/analysis/llm_layer_analysis.yaml` | LLM layer analysis |

## Shell Scripts

Example shell scripts for common workflows:

- `run_mnist_basic.sh` - Basic MNIST experiment
- `run_llm_supernode.sh` - LLM supernode analysis
- `run_vision_synergy.sh` - Vision redundancy/synergy analysis
