# Scripts

Entry points for experiments and analysis.

## run_experiment.py

Run experiments from YAML configuration:

```bash
# Vision analysis
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml

# LLM analysis  
python scripts/run_experiment.py --config configs/prune_llm/llama3_8b_full.yaml

# Cluster-based analysis
python scripts/run_experiment.py --config configs/cluster_analysis/resnet18_cifar10_full.yaml
```

Options:
- `--config PATH` - Configuration file (required)
- `--device STRING` - Override device (e.g., cuda:0)
- `--seed INT` - Override random seed
- `--output-dir PATH` - Override output directory
- `--analysis-only` - Regenerate plots from existing results
- `--experiment-dir PATH` - Existing experiment directory (with --analysis-only)

## run_analysis.py

Generate visualizations from results:

```bash
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots --quick
python scripts/run_analysis.py --config configs/analysis_template.yaml
```

Options:
- `--config PATH` - Analysis config file
- `--results-dir PATH` - Results directory
- `--output-dir PATH` - Output directory
- `--analyses LIST` - Specific analyses to run
- `--quick` - Run all analyses with defaults


