"""
Visualization components for alignment analysis.

Structure:
- UnifiedVisualizer: Primary interface for all visualization needs
- PruningVisualizer: Specialized pruning analysis plots (advanced use)
- AlignmentVisualizer: Specialized alignment plots (advanced use)
- HaloPlots: Halo redundancy visualization (by layer depth)

For most use cases, use UnifiedVisualizer:

    from alignment.analysis.visualization import UnifiedVisualizer
    
    viz = UnifiedVisualizer()
    viz.plot_layer_scores(scores, "Rayleigh Quotient")
    viz.plot_pruning_before_after(sparsities, before, after)

For advanced pruning visualizations (multi-seed, ablations):

    from alignment.analysis.visualization import PruningVisualizer
    
    viz = PruningVisualizer()
    viz.plot_sparsity_perplexity_curves(df, save_path="curves.png")

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
from .pruning_plots import PruningVisualizer
from .alignment_plots import AlignmentVisualizer

# Halo redundancy plots
from .halo_plots import (
    plot_halo_redundancy_by_depth,
    plot_halo_redundancy_comprehensive,
    plot_halo_redundancy_heatmap,
)

# Cluster visualization plots
from .cluster_plots import (
    plot_metric_scatter,
    plot_cluster_evolution,
    plot_influence_matrix,
    plot_cascade_test,
    plot_halo_properties,
)

__all__ = [
    # Primary
    "UnifiedVisualizer",
    "plot_quick_summary",
    "generate_experiment_visualizations",
    # Specialized
    "PruningVisualizer",
    "AlignmentVisualizer",
    # Halo plots
    "plot_halo_redundancy_by_depth",
    "plot_halo_redundancy_comprehensive",
    "plot_halo_redundancy_heatmap",
    # Cluster plots
    "plot_metric_scatter",
    "plot_cluster_evolution",
    "plot_influence_matrix",
    "plot_cascade_test",
    "plot_halo_properties",
]
