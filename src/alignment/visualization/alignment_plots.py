"""
Visualization utilities for alignment metrics analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path
import logging

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except (ImportError, AttributeError):
    HAS_SEABORN = False
    import warnings
    warnings.warn("Seaborn not available or incompatible version. Some visualization features may be limited.")

logger = logging.getLogger(__name__)


class AlignmentVisualizer:
    """
    Create various visualizations for alignment metrics analysis.
    """
    
    def __init__(self, style: str = 'seaborn', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        self.figsize = figsize
        if HAS_SEABORN:
            sns.set_palette("husl")
    
    def plot_layer_scores(
        self,
        scores: Dict[str, torch.Tensor],
        metric_name: str,
        save_path: Optional[str] = None,
        show_statistics: bool = True
    ) -> plt.Figure:
        """
        Plot alignment scores across layers.
        
        Args:
            scores: Dictionary of layer_name -> scores
            metric_name: Name of the metric
            save_path: Path to save the plot
            show_statistics: Whether to show mean/std statistics
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        layer_names = list(scores.keys())
        data = []
        positions = []
        
        for i, (layer_name, layer_scores) in enumerate(scores.items()):
            if isinstance(layer_scores, torch.Tensor):
                layer_scores = layer_scores.cpu().numpy()
            data.append(layer_scores)
            positions.append(i)
        
        # Create violin plot
        parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)
        
        # Add layer names
        ax.set_xticks(positions)
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        
        # Labels and title
        ax.set_xlabel('Layer')
        ax.set_ylabel(f'{metric_name} Score')
        ax.set_title(f'{metric_name} Distribution Across Layers')
        
        # Add statistics if requested
        if show_statistics:
            stats_text = []
            for i, layer_scores in enumerate(data):
                mean = np.mean(layer_scores)
                std = np.std(layer_scores)
                stats_text.append(f"{layer_names[i]}: μ={mean:.3f}, σ={std:.3f}")
            
            # Add text box
            textstr = '\n'.join(stats_text)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def plot_score_heatmap(
        self,
        scores: Dict[str, Dict[str, torch.Tensor]],
        save_path: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ) -> plt.Figure:
        """
        Create heatmap of scores across layers and metrics.
        
        Args:
            scores: Nested dict of layer -> metric -> scores
            save_path: Path to save the plot
            vmin: Minimum value for color scale
            vmax: Maximum value for color scale
            
        Returns:
            Matplotlib figure
        """
        # Prepare data for heatmap
        layers = sorted(scores.keys())
        metrics = sorted(set(metric for layer_scores in scores.values() 
                           for metric in layer_scores.keys()))
        
        # Create matrix
        matrix = np.zeros((len(layers), len(metrics)))
        
        for i, layer in enumerate(layers):
            for j, metric in enumerate(metrics):
                if metric in scores[layer]:
                    layer_scores = scores[layer][metric]
                    if isinstance(layer_scores, torch.Tensor):
                        layer_scores = layer_scores.cpu().numpy()
                    matrix[i, j] = np.mean(layer_scores)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(len(metrics) * 1.5, len(layers) * 0.8))
        
        if HAS_SEABORN:
            sns.heatmap(
                matrix,
                xticklabels=metrics,
                yticklabels=layers,
                cmap='coolwarm',
                center=0,
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Mean Score'},
                vmin=vmin,
                vmax=vmax,
                ax=ax
            )
        else:
            # Fallback to matplotlib heatmap
            im = ax.imshow(matrix, cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Score')
            
            # Set ticks
            ax.set_xticks(range(len(metrics)))
            ax.set_yticks(range(len(layers)))
            ax.set_xticklabels(metrics)
            ax.set_yticklabels(layers)
            
            # Add text annotations
            for i in range(len(layers)):
                for j in range(len(metrics)):
                    text = ax.text(j, i, f'{matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Alignment Metrics Heatmap')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Layer')
        
        # Rotate metric names for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {save_path}")
        
        return fig
    
    def plot_pruning_analysis(
        self,
        pruning_results: Dict[float, Dict[str, float]],
        metrics: List[str] = ['accuracy', 'loss'],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot pruning analysis results.
        
        Args:
            pruning_results: Dict of pruning_ratio -> metric -> value
            metrics: List of metrics to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(
            1, len(metrics), 
            figsize=(len(metrics) * 5, 5)
        )
        
        if len(metrics) == 1:
            axes = [axes]
        
        pruning_ratios = sorted(pruning_results.keys())
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            values = [pruning_results[ratio].get(metric, 0) for ratio in pruning_ratios]
            
            # Plot line with markers
            ax.plot(pruning_ratios, values, 'o-', linewidth=2, markersize=8)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Labels
            ax.set_xlabel('Pruning Ratio')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} vs Pruning Ratio')
            
            # Add percentage labels
            ax.set_xticklabels([f'{r*100:.0f}%' for r in pruning_ratios])
        
        plt.suptitle('Pruning Analysis Results', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pruning analysis to {save_path}")
        
        return fig
    
    def plot_neuron_importance(
        self,
        layer_name: str,
        scores: torch.Tensor,
        top_k: int = 20,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot top-k most and least important neurons.
        
        Args:
            layer_name: Name of the layer
            scores: Neuron importance scores
            top_k: Number of top/bottom neurons to show
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        
        # Get indices of top and bottom neurons
        sorted_indices = np.argsort(scores)
        top_indices = sorted_indices[-top_k:][::-1]
        bottom_indices = sorted_indices[:top_k]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot top neurons
        ax1.bar(range(top_k), scores[top_indices], color='green', alpha=0.7)
        ax1.set_xticks(range(top_k))
        ax1.set_xticklabels([f'N{idx}' for idx in top_indices], rotation=45)
        ax1.set_ylabel('Score')
        ax1.set_title(f'Top {top_k} Most Important Neurons in {layer_name}')
        ax1.grid(True, alpha=0.3)
        
        # Plot bottom neurons
        ax2.bar(range(top_k), scores[bottom_indices], color='red', alpha=0.7)
        ax2.set_xticks(range(top_k))
        ax2.set_xticklabels([f'N{idx}' for idx in bottom_indices], rotation=45)
        ax2.set_ylabel('Score')
        ax2.set_title(f'Top {top_k} Least Important Neurons in {layer_name}')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved neuron importance plot to {save_path}")
        
        return fig
    
    def create_report(
        self,
        results: Dict[str, Any],
        output_dir: str,
        experiment_name: str = "alignment_analysis"
    ):
        """
        Create a comprehensive visual report.
        
        Args:
            results: Dictionary containing all results
            output_dir: Directory to save the report
            experiment_name: Name of the experiment
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate various plots
        if 'layer_scores' in results:
            for metric_name, scores in results['layer_scores'].items():
                self.plot_layer_scores(
                    scores,
                    metric_name,
                    save_path=plots_dir / f"{metric_name}_distribution.png"
                )
        
        if 'heatmap_data' in results:
            self.plot_score_heatmap(
                results['heatmap_data'],
                save_path=plots_dir / "metrics_heatmap.png"
            )
        
        if 'pruning_results' in results:
            self.plot_pruning_analysis(
                results['pruning_results'],
                save_path=plots_dir / "pruning_analysis.png"
            )
        
        # Create summary statistics
        summary = []
        if 'layer_scores' in results:
            for metric_name, layer_scores in results['layer_scores'].items():
                for layer_name, scores in layer_scores.items():
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    summary.append({
                        'Metric': metric_name,
                        'Layer': layer_name,
                        'Mean': np.mean(scores),
                        'Std': np.std(scores),
                        'Min': np.min(scores),
                        'Max': np.max(scores)
                    })
        
        # Save summary as CSV
        if summary:
            df = pd.DataFrame(summary)
            df.to_csv(output_dir / "summary_statistics.csv", index=False)
        
        # Create README
        readme_content = f"""# {experiment_name} Report

Generated visualization report for alignment analysis.

## Contents

- `plots/`: Visualization plots
  - `*_distribution.png`: Score distributions for each metric
  - `metrics_heatmap.png`: Heatmap of all metrics across layers
  - `pruning_analysis.png`: Pruning experiment results
- `summary_statistics.csv`: Summary statistics for all metrics

## Experiment Details

Date: {pd.Timestamp.now()}
Number of metrics: {len(results.get('layer_scores', {}))}
Number of layers: {len(set(layer for scores in results.get('layer_scores', {}).values() for layer in scores))}
"""
        
        with open(output_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Report saved to {output_dir}")


def plot_quick_summary(scores: Dict[str, torch.Tensor], title: str = "Alignment Scores"):
    """
    Quick plotting function for immediate visualization.
    
    Args:
        scores: Dictionary of layer_name -> scores
        title: Plot title
    """
    visualizer = AlignmentVisualizer()
    fig = visualizer.plot_layer_scores(scores, title)
    plt.show()
    return fig 