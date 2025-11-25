# Analysis Module

Result analysis, visualization, and reporting.

## Components

- `AnalysisRunner` - Unified entry point for analysis tasks
- `UnifiedVisualizer` - Plot generation
- `UnifiedReporter` - Report generation (HTML, Markdown, JSON)
- `ResultAggregator` - Result collection and summarization

## Usage

```python
from alignment.analysis import AnalysisRunner, AnalysisConfig

config = AnalysisConfig(
    results_dir="./results",
    output_dir="./plots",
    analyses=["histograms", "pruning_curves"],
)
runner = AnalysisRunner(config)
runner.run()
```

```bash
python scripts/run_analysis.py --results-dir ./results --output-dir ./plots --quick
```

## Available Analyses

- `histograms` - Importance score distributions
- `scatter_plots` - Metric correlations
- `heatmaps` - Layer-metric heatmaps
- `pruning_curves` - Sparsity vs performance
- `scar_analysis` - SCAR metrics (LLM)
