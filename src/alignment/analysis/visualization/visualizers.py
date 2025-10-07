"""
Visualization utilities for alignment metrics and analysis.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd
import logging
import torch

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except (ImportError, AttributeError):
    HAS_SEABORN = False

logger = logging.getLogger(__name__)


class MetricVisualizer:
    """
    Visualizes alignment metrics in various formats.
    
    This class provides methods for creating:
    - Line plots for metric evolution
    - Bar plots for comparisons
    - Heatmaps for layer-metric matrices
    - Distribution plots
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            # Fallback to default if style not available
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except:
                plt.style.use('default')
        self.figsize = figsize
        # Use seaborn colors if available, otherwise use matplotlib defaults
        if HAS_SEABORN:
            self.colors = sns.color_palette("husl", 10)
        else:
            # Use matplotlib's tab10 colormap as fallback
            import matplotlib.cm as cm
            self.colors = [cm.tab10(i) for i in range(10)]
    
    def plot_metric_evolution(
        self,
        steps: List[int],
        values: Dict[str, List[float]],
        title: str = "Metric Evolution",
        xlabel: str = "Step",
        ylabel: str = "Value",
        legend_title: str = "Series",
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Plot the evolution of metrics over time.
        
        Args:
            steps: List of step numbers
            values: Dictionary mapping series names to value lists
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            legend_title: Legend title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (name, vals) in enumerate(values.items()):
            color = self.colors[i % len(self.colors)]
            ax.plot(steps[:len(vals)], vals, label=name, color=color, linewidth=2)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(title=legend_title, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_layer_comparison(
        self,
        data: Dict[str, float],
        title: str = "Layer-wise Metric Comparison",
        xlabel: str = "Layer",
        ylabel: str = "Value",
        rotation: int = 45,
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Create a bar plot comparing metrics across layers.
        
        Args:
            data: Dictionary mapping layer names to values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            rotation: X-tick label rotation
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        layers = list(data.keys())
        values = list(data.values())
        
        bars = ax.bar(range(len(layers)), values, color=self.colors[0])
        
        # Color bars by value
        norm = plt.Normalize(min(values), max(values))
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        
        for bar, val in zip(bars, values):
            bar.set_color(sm.to_rgba(val))
        
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=rotation, ha='right')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metric_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Metric Heatmap",
        cmap: str = "coolwarm",
        annotate: bool = True,
        fmt: str = ".3f",
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Create a heatmap of metrics.
        
        Args:
            data: DataFrame with metrics
            title: Plot title
            cmap: Colormap to use
            annotate: Whether to annotate cells with values
            fmt: Format string for annotations
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if HAS_SEABORN:
            sns.heatmap(
                data,
                ax=ax,
                cmap=cmap,
                center=0,
                annot=annotate,
                fmt=fmt,
                cbar_kws={'label': 'Value'},
                linewidths=0.5
            )
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(data.values, cmap=cmap, aspect='auto')
            
            # Set ticks
            ax.set_xticks(np.arange(len(data.columns)))
            ax.set_yticks(np.arange(len(data.index)))
            ax.set_xticklabels(data.columns)
            ax.set_yticklabels(data.index)
            
            # Rotate the tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Value', rotation=270, labelpad=15)
            
            # Add annotations if requested
            if annotate:
                for i in range(len(data.index)):
                    for j in range(len(data.columns)):
                        text = ax.text(j, i, format(data.iloc[i, j], fmt),
                                     ha="center", va="center", color="black")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distribution(
        self,
        data: Dict[str, List[float]],
        title: str = "Metric Distribution",
        xlabel: str = "Value",
        ylabel: str = "Density",
        bins: int = 30,
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Plot distribution of metric values.
        
        Args:
            data: Dictionary mapping series names to value lists
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of histogram bins
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (name, values) in enumerate(data.items()):
            color = self.colors[i % len(self.colors)]
            ax.hist(values, bins=bins, alpha=0.6, label=name, color=color, density=True)
            
            # Add KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class LayerVisualizer:
    """
    Specialized visualizations for layer-wise analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize layer visualizer."""
        self.figsize = figsize
    
    def plot_layer_evolution(
        self,
        layer_data: Dict[str, Dict[int, float]],
        title: str = "Layer Metric Evolution",
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Plot how metrics evolve for each layer.
        
        Args:
            layer_data: Nested dict: {layer_name: {step: value}}
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (layer_name, step_values) in enumerate(layer_data.items()):
            steps = sorted(step_values.keys())
            values = [step_values[step] for step in steps]
            
            ax.plot(steps, values, label=layer_name, marker='o', markersize=4)
        
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Metric Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_layer_correlation(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Layer Correlation Matrix",
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Plot correlation between layers.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        
        if HAS_SEABORN:
            sns.heatmap(
                correlation_matrix,
                mask=mask,
                ax=ax,
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                annot=True,
                fmt='.2f',
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Correlation'}
            )
        else:
            # Fallback to matplotlib
            # Apply mask
            masked_data = np.ma.masked_where(mask, correlation_matrix.values)
            
            im = ax.imshow(masked_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
            
            # Set ticks
            ax.set_xticks(np.arange(len(correlation_matrix.columns)))
            ax.set_yticks(np.arange(len(correlation_matrix.index)))
            ax.set_xticklabels(correlation_matrix.columns)
            ax.set_yticklabels(correlation_matrix.index)
            
            # Rotate the tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation', rotation=270, labelpad=15)
            
            # Add annotations
            for i in range(len(correlation_matrix.index)):
                for j in range(len(correlation_matrix.columns)):
                    if not mask[i, j]:
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class ComparisonVisualizer:
    """
    Visualizations for comparing multiple experiments or models.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8)):
        """Initialize comparison visualizer."""
        self.figsize = figsize
    
    def plot_experiment_comparison(
        self,
        data: pd.DataFrame,
        metric_columns: List[str],
        experiment_column: str = "experiment",
        title: str = "Experiment Comparison",
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Create a multi-metric comparison plot.
        
        Args:
            data: DataFrame with experiment results
            metric_columns: Columns containing metrics to plot
            experiment_column: Column containing experiment names
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        n_metrics = len(metric_columns)
        fig, axes = plt.subplots(1, n_metrics, figsize=self.figsize, sharey=True)
        
        if n_metrics == 1:
            axes = [axes]
        
        experiments = data[experiment_column].unique()
        x_pos = np.arange(len(experiments))
        
        for idx, (metric, ax) in enumerate(zip(metric_columns, axes)):
            values = [data[data[experiment_column] == exp][metric].mean() 
                     for exp in experiments]
            
            bars = ax.bar(x_pos, values)
            
            # Color bars
            colors = plt.cm.viridis(np.linspace(0, 1, len(experiments)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(experiments, rotation=45, ha='right')
            ax.set_title(metric, fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, values)):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        axes[0].set_ylabel("Value", fontsize=12)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_metric_radar(
        self,
        data: Dict[str, Dict[str, float]],
        title: str = "Metric Radar Chart",
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """
        Create a radar chart comparing multiple metrics.
        
        Args:
            data: Nested dict: {experiment_name: {metric_name: value}}
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure
        """
        # Get metrics and experiments
        experiments = list(data.keys())
        metrics = list(next(iter(data.values())).keys())
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for i, exp_name in enumerate(experiments):
            values = [data[exp_name][metric] for metric in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=exp_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, None)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig 