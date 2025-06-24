# Analysis Module

The analysis module provides comprehensive tools for analyzing experiment results, aggregating metrics, generating reports, and creating visualizations.

## Overview

The module is organized into three main components:

1. **Aggregation**: Tools for collecting and summarizing results across experiments
2. **Reporting**: Utilities for generating reports in various formats (HTML, Markdown, JSON)
3. **Visualization**: Functions for creating plots and interactive visualizations

## Module Structure

```
analysis/
├── aggregation/
│   ├── results.py      # ResultAggregator for experiment results
│   ├── metrics.py      # MetricAggregator for time-series metrics
│   └── layers.py       # LayerAggregator for layer-wise analysis
├── reporting/
│   ├── html.py         # HTMLReporter for web reports
│   ├── markdown.py     # MarkdownReporter for documentation
│   └── json_reporter.py # JSONReporter for data export
└── visualization/
    └── plots.py        # Plotting functions
```

## Aggregation

### ResultAggregator

Aggregates results from multiple experiments for comparison and analysis.

```python
from alignment.analysis import ResultAggregator

aggregator = ResultAggregator()

# Load results from files
aggregator.load_from_directory("./results/")

# Or add results manually
aggregator.add_results("exp1", results_dict, metadata)

# Get specific metrics
values = aggregator.get_metric_values("rayleigh_quotient", layer_name="fc1")

# Compute statistics
stats = aggregator.compute_statistics("rayleigh_quotient", "fc1")
print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

# Convert to DataFrame for analysis
df = aggregator.to_dataframe()
```

### MetricAggregator

Tracks metrics over time during training or experiments.

```python
from alignment.analysis import MetricAggregator

aggregator = MetricAggregator()

# Add metrics at each step
for step in range(100):
    metrics = compute_metrics()  # Your metric computation
    aggregator.add_step(step, metrics)

# Get metric evolution
steps, values = aggregator.get_metric_evolution("loss", "layer1")

# Analyze trends
trends = aggregator.compute_trends("accuracy", "layer2", window_size=10)
print(f"Slope: {trends['slope']:.4f}, Change: {trends['percent_change']:.2f}%")
```

### LayerAggregator

Analyzes patterns across network layers.

```python
from alignment.analysis import LayerAggregator

aggregator = LayerAggregator()

# Add metrics from multiple evaluations
for metrics in evaluation_results:
    aggregator.add_metrics(metrics)

# Get layer statistics
summary = aggregator.get_layer_summary("conv1")

# Rank layers by metric
rankings = aggregator.rank_layers("gradient_norm", criterion="mean")

# Find anomalous layers
anomalous = aggregator.find_anomalous_layers("activation_norm", threshold_std=2.0)
```

## Reporting

### HTMLReporter

Generate interactive HTML reports with tables and visualizations.

```python
from alignment.analysis import HTMLReporter

reporter = HTMLReporter(title="Experiment Results")

# Add sections
reporter.add_section("Summary", "<p>Experiment completed successfully</p>")

# Add data tables
reporter.add_dataframe("Metrics", metrics_df)

# Add figures
reporter.add_figure("plots/accuracy.png", caption="Training Accuracy")

# Generate report
reporter.generate("report.html")
```

### MarkdownReporter

Create markdown reports for documentation or GitHub.

```python
from alignment.analysis import MarkdownReporter

reporter = MarkdownReporter(title="Experiment Analysis")

# Add content
reporter.add_section("Results", "The experiment showed...")

# Add tables
reporter.add_table("Performance Metrics", results_df)

# Generate report
reporter.generate("analysis.md")
```

### JSONReporter

Export results in JSON format for further processing.

```python
from alignment.analysis import JSONReporter

reporter = JSONReporter(title="Experiment Data")

# Add data sections
reporter.add_section("metrics", metrics_dict)
reporter.add_section("config", config_dict)

# Generate JSON file
reporter.generate("results.json")
```

## Visualization

### Plotting Functions

The visualization module provides various plotting utilities:

```python
from alignment.analysis.visualization import (
    plot_metric_evolution,
    plot_layer_comparison,
    plot_correlation_matrix,
    create_interactive_dashboard
)

# Plot metric over time
plot_metric_evolution(
    steps, values, 
    title="Loss Evolution",
    save_path="loss_plot.png"
)

# Compare metrics across layers
plot_layer_comparison(
    layer_metrics,
    metric_name="rayleigh_quotient",
    save_path="layer_comparison.png"
)

# Correlation matrix
plot_correlation_matrix(
    metrics_df,
    save_path="correlations.png"
)

# Interactive dashboard
create_interactive_dashboard(
    results,
    output_dir="./dashboard/"
)
```

## Best Practices

1. **Organize Results**: Use consistent naming and directory structure for experiments
2. **Add Metadata**: Include configuration and context with results
3. **Use DataFrames**: Convert to pandas DataFrames for easier analysis
4. **Automate Reports**: Generate reports automatically after experiments
5. **Version Results**: Track experiment versions and configurations

## Examples

### Complete Analysis Pipeline

```python
from alignment.analysis import ResultAggregator, HTMLReporter
from alignment.analysis.visualization import plot_metric_evolution

# Aggregate results
aggregator = ResultAggregator()
aggregator.load_from_directory("./experiments/pruning/")

# Analyze
df = aggregator.to_dataframe()
best_exp = df.loc[df['final_accuracy'].idxmax()]

# Create report
reporter = HTMLReporter("Pruning Experiment Analysis")
reporter.add_section("Best Configuration", f"<p>{best_exp['experiment']}</p>")
reporter.add_dataframe("All Results", df)

# Add visualizations
for metric in ['accuracy', 'sparsity']:
    plot_metric_evolution(
        df['epoch'], df[metric],
        title=f"{metric.title()} Evolution",
        save_path=f"{metric}_plot.png"
    )
    reporter.add_figure(f"{metric}_plot.png", f"{metric.title()} over training")

reporter.generate("pruning_analysis.html")
```

### Comparing Multiple Experiments

```python
from alignment.analysis import ResultAggregator
import matplotlib.pyplot as plt

aggregator = ResultAggregator()

# Load different experiment types
for exp_type in ['magnitude', 'gradient', 'random']:
    aggregator.load_from_file(f"results/{exp_type}_pruning.json", name=exp_type)

# Compare final accuracies
accuracies = {}
for exp_name in aggregator.results:
    acc = aggregator.get_metric_values('accuracy', experiment_names=[exp_name])
    accuracies[exp_name] = list(acc.values())[0]

# Plot comparison
plt.bar(accuracies.keys(), accuracies.values())
plt.ylabel('Final Accuracy')
plt.title('Pruning Strategy Comparison')
plt.savefig('strategy_comparison.png')
```

## Integration with Experiments

The analysis module integrates seamlessly with the experiment framework:

```python
from alignment.experiments import GeneralAlignmentExperiment
from alignment.analysis import ResultAggregator, HTMLReporter

# Run experiment
experiment = GeneralAlignmentExperiment.from_yaml("config.yaml")
results = experiment.run()

# Automatic analysis
aggregator = ResultAggregator()
aggregator.add_results(experiment.config.name, results)

# Generate report
reporter = HTMLReporter(f"{experiment.config.name} Analysis")
reporter.add_section("Configuration", str(experiment.config))
reporter.add_section("Results", aggregator.to_dataframe().to_html())
reporter.generate(f"{experiment.config.name}_report.html")
``` 