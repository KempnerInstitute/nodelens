"""
Visualization components for alignment analysis.

Structure:
- UnifiedVisualizer: Primary interface for all visualization needs
- PruningVisualizer: Specialized pruning analysis plots (advanced use)
- AlignmentVisualizer: Specialized alignment plots (advanced use)
- HaloPlots: Halo redundancy visualization (by layer depth)
- MetricPlots: Histogram/distribution plots for metrics (RQ, Redundancy, Synergy)
- ClusterPlots: Cluster-specific visualizations (scatter, evolution, cascade)

For most use cases, use UnifiedVisualizer:

    from alignment.analysis.visualization import UnifiedVisualizer
    
    viz = UnifiedVisualizer()
    viz.plot_layer_scores(scores, "Rayleigh Quotient")
    viz.plot_pruning_before_after(sparsities, before, after)

For metric histograms and distributions:

    from alignment.analysis.visualization import (
        plot_metric_histogram,
        plot_metric_violin,
        plot_multi_metric_histogram,
    )
    
    plot_metric_histogram(rq_values, "rq", layer_name="conv1", highlight_percentile=95)

For pruning comparisons (unified for both vision and LLM):

    from alignment.analysis.visualization import plot_unified_pruning_comparison
    
    plot_unified_pruning_comparison(results, baseline_value=0.92, metric='accuracy')

For halo redundancy analysis:

    from alignment.analysis.visualization import (
        plot_halo_redundancy_by_depth,
        plot_halo_redundancy_comprehensive
    )
    
    plot_halo_redundancy_by_depth(layers, halo_means, non_halo_means, cross_means)
"""

# Primary interface
from .unified_visualizer import UnifiedVisualizer, plot_quick_summary, generate_experiment_visualizations

# Specialized visualizers (for advanced use cases)
from .pruning_plots import (
    PruningVisualizer,
    # Unified pruning functions (work for both vision and LLM)
    plot_unified_pruning_comparison,
    plot_pruning_accuracy_loss_grid,
    plot_pruning_recovery_chart,
    PRUNING_METHOD_COLORS,
)
from .alignment_plots import AlignmentVisualizer

# Metric distribution plots (histograms, violins, correlations)
from .metric_plots import (
    plot_metric_histogram,
    plot_metric_violin,
    plot_metric_boxplot,
    plot_multi_metric_histogram,
    plot_metric_scatter_matrix,
    plot_metric_correlation_heatmap,
    plot_layer_metric_heatmap,
    plot_top_neurons_bar,
    METRIC_COLORS,
)

# Halo redundancy plots
from .halo_plots import (
    plot_halo_redundancy_by_depth,
    plot_halo_redundancy_comprehensive,
    plot_halo_redundancy_heatmap,
)

# Paper-specific plots (SCAR draft)
from .paper_plots import (
    plot_loss_proxy_concentration,
    plot_halo_structure,
    plot_supernode_halo_summary,
    plot_scar_schematic,
)

# Cluster visualization plots
from .cluster_plots import (
    plot_metric_scatter,
    plot_cluster_evolution,
    plot_influence_matrix,
    plot_cascade_test,
    plot_halo_properties,
    plot_pruning_comparison,
    plot_metric_distributions_for_layer,
    plot_layer_metric_summary,
    plot_layer_metric_trends,
    plot_metric_statistics_table,
    plot_centroid_evolution,
    plot_centroid_depth_profiles,
    CLUSTER_COLORS,
)

# New bar chart functions for pruning
from .pruning_plots import (
    plot_pruning_bar_comparison,
    plot_pruning_heatmap,
    plot_pruning_ranking,
)

__all__ = [
    # Primary
    "UnifiedVisualizer",
    "plot_quick_summary",
    "generate_experiment_visualizations",
    # Specialized visualizers
    "PruningVisualizer",
    "AlignmentVisualizer",
    # Unified pruning plots (vision + LLM)
    "plot_unified_pruning_comparison",
    "plot_pruning_accuracy_loss_grid",
    "plot_pruning_recovery_chart",
    "PRUNING_METHOD_COLORS",
    # Metric distribution plots
    "plot_metric_histogram",
    "plot_metric_violin",
    "plot_metric_boxplot",
    "plot_multi_metric_histogram",
    "plot_metric_scatter_matrix",
    "plot_metric_correlation_heatmap",
    "plot_layer_metric_heatmap",
    "plot_top_neurons_bar",
    "METRIC_COLORS",
    # Halo plots
    "plot_halo_redundancy_by_depth",
    "plot_halo_redundancy_comprehensive",
    "plot_halo_redundancy_heatmap",
    # Paper plots
    "plot_loss_proxy_concentration",
    "plot_halo_structure",
    "plot_supernode_halo_summary",
    "plot_scar_schematic",
    # Cluster plots
    "plot_metric_scatter",
    "plot_cluster_evolution",
    "plot_influence_matrix",
    "plot_cascade_test",
    "plot_halo_properties",
    "plot_pruning_comparison",
    "plot_metric_distributions_for_layer",
    "plot_layer_metric_summary",
    "plot_layer_metric_trends",
    "plot_metric_statistics_table",
    "plot_centroid_evolution",
    "plot_centroid_depth_profiles",
    "CLUSTER_COLORS",
    # New bar chart functions
    "plot_pruning_bar_comparison",
    "plot_pruning_heatmap",
    "plot_pruning_ranking",
]
