# Analysis Module

Tools for analyzing experiment results, aggregating metrics, and creating visualizations.

## Components

- **Aggregation**: Collect and summarize results across experiments
- **Unified Visualizer**: Create plots and interactive visualizations  
- **Unified Reporter**: Generate reports in HTML, Markdown, and JSON formats

## Quick Usage

```python
from alignment.analysis import ResultAggregator, UnifiedVisualizer, UnifiedReporter

# Aggregate results
aggregator = ResultAggregator()
aggregator.load_from_directory("./results/")

# Create visualizations
visualizer = UnifiedVisualizer()
visualizer.plot_metric_evolution(steps, values, save_path="plot.png")

# Generate reports
reporter = UnifiedReporter(title="Experiment Results")
reporter.add_dataframe("Metrics", results_df)
reporter.generate("report.html")
```

## Integration

Works seamlessly with the experiment framework for automatic analysis and reporting. 