# Analysis Module

Tools for analyzing experiment results, aggregating metrics, and creating visualizations.

## Components

- **AnalysisRunner**: Unified entry point for all analysis and visualization tasks
- **Aggregation**: Collect and summarize results across experiments
- **UnifiedVisualizer**: Create plots and visualizations
- **UnifiedReporter**: Generate reports in HTML, Markdown, and JSON formats

## Quick Usage

### Using the Analysis Runner (Recommended)

```python
from alignment.analysis import AnalysisRunner, AnalysisConfig

# From a config file
runner = AnalysisRunner("configs/analysis_template.yaml")
outputs = runner.run()

# Or programmatically
config = AnalysisConfig(
    results_dir="./results",
    output_dir="./plots",
    analyses=["histograms", "scatter_plots", "pruning_curves"],
)
runner = AnalysisRunner(config)
outputs = runner.run()
```

### Command-line Interface

```bash
# Run all analyses from config
python -m alignment.analysis.analysis_runner --config configs/analysis_template.yaml

# Quick analysis without config
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots --quick

# Run specific analyses
python scripts/run_analysis.py --config configs/analysis_template.yaml \
    --analyses histograms scatter_plots
```

### Direct Component Usage

```python
from alignment.analysis import ResultAggregator, UnifiedVisualizer, UnifiedReporter

# Aggregate results
aggregator = ResultAggregator()
aggregator.load_from_directory("./results/")

# Create visualizations
visualizer = UnifiedVisualizer()
visualizer.plot_metric_evolution(steps, values, save_path="plot.png")
visualizer.plot_pruning_before_after(sparsities, before, after, save_dir="./plots")
visualizer.plot_scar_layer_scores(scar_scores, metric_name="scar_loss_proxy")

# Generate reports
reporter = UnifiedReporter(title="Experiment Results")
reporter.add_dataframe("Metrics", results_df)
reporter.generate("report.html")
```

## Available Analyses

| Analysis | Description |
|----------|-------------|
| `histograms` | Importance score distributions per layer/metric |
| `scatter_plots` | Metric correlation scatter plots |
| `heatmaps` | Layer-metric heatmaps of mean scores |
| `layer_distributions` | Violin/box plots of scores across layers |
| `pruning_curves` | Sparsity vs performance curves |
| `scar_analysis` | SCAR-specific layer plots and heatmaps |

## Configuration

See `configs/analysis_template.yaml` for a complete configuration reference.

## Integration

The analysis module integrates with the experiment framework:
- Experiments automatically call visualization methods when `generate_plots` is enabled
- Results are saved in JSON format compatible with the analysis runner
- Use `--analysis-only` mode to regenerate plots from existing results
