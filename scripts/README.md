# Scripts

Entry points for experiments and analysis.

## run_experiment.py

Run experiments from YAML configuration:

```bash
python scripts/run_experiment.py --config configs/examples/mnist_basic.yaml
python scripts/run_experiment.py --config configs/examples/resnet_pruning.yaml
python scripts/run_experiment.py --config configs/examples/llm_alignment.yaml
```

Options:
- `--config PATH` - Configuration file (required)
- `--device STRING` - Override device
- `--seed INT` - Override random seed
- `--output-dir PATH` - Override output directory
- `--analysis-only` - Regenerate plots from existing results
- `--experiment-dir PATH` - Existing experiment directory

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
