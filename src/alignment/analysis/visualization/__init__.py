"""
Visualization components for alignment analysis.

Structure:
- UnifiedVisualizer: Primary interface for all visualization needs
- PruningVisualizer: Specialized pruning analysis plots (advanced use)
- AlignmentVisualizer: Specialized alignment plots (advanced use)

For most use cases, use UnifiedVisualizer:

    from alignment.analysis.visualization import UnifiedVisualizer
    
    viz = UnifiedVisualizer()
    viz.plot_layer_scores(scores, "Rayleigh Quotient")
    viz.plot_pruning_before_after(sparsities, before, after)

For advanced pruning visualizations (multi-seed, ablations):

    from alignment.analysis.visualization import PruningVisualizer
    
    viz = PruningVisualizer()
    viz.plot_sparsity_perplexity_curves(df, save_path="curves.png")
"""

# Primary interface
from .unified_visualizer import UnifiedVisualizer, plot_quick_summary, generate_experiment_visualizations

# Specialized visualizers (for advanced use cases)
from .pruning_plots import PruningVisualizer
from .alignment_plots import AlignmentVisualizer

__all__ = [
    # Primary
    "UnifiedVisualizer",
    "plot_quick_summary",
    "generate_experiment_visualizations",
    # Specialized
    "PruningVisualizer",
    "AlignmentVisualizer",
]
