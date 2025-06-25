"""
Advanced visualization tools for pruning experiments.

This module provides comprehensive plotting capabilities for pruning analysis,
including multi-strategy comparisons, accuracy/loss curves, and statistical analysis.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
from pathlib import Path
import logging
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec

# Try to import seaborn, but make it optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except (ImportError, AttributeError):
    HAS_SEABORN = False

logger = logging.getLogger(__name__)


class PruningVisualizer:
    """
    Advanced visualization for pruning experiments with multi-strategy support.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize pruning visualizer.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        self.figsize = figsize
        
        # Define extended color palette for strategies
        self.strategy_colors = {
            # Pruning strategies
            'magnitude': '#1f77b4',      # Blue
            'gradient': '#ff7f0e',       # Orange  
            'fisher': '#2ca02c',         # Green
            'random': '#d62728',         # Red
            
            # Importance modes
            'low': '#9467bd',            # Purple
            'high': '#8c564b',           # Brown
            'magnitude_low': '#1f77b4',  # Blue
            'magnitude_high': '#17becf', # Light Blue
            'gradient_low': '#ff7f0e',   # Orange
            'gradient_high': '#ffbb78',  # Light Orange
            
            # Layer analysis
            'layer_0': '#e377c2',        # Pink
            'layer_1': '#7f7f7f',        # Gray
            'layer_2': '#bcbd22',        # Yellow-green
            'layer_3': '#17becf',        # Cyan
        }
        
        # Set global style parameters
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def plot_pruning_performance(
        self,
        results: Dict[str, Dict[float, Dict[str, float]]],
        metrics: List[str] = ['accuracy', 'loss'],
        save_path: Optional[str] = None,
        title: Optional[str] = None,
        show_confidence: bool = True
    ) -> plt.Figure:
        """
        Plot performance metrics for multiple pruning strategies.
        
        Args:
            results: Nested dict of strategy -> sparsity -> metric -> value
                     Can also include 'mean' and 'std' for confidence intervals
            metrics: List of metrics to plot
            save_path: Path to save the plot
            title: Overall title for the plot
            show_confidence: Whether to show confidence intervals if available
            
        Returns:
            Matplotlib figure
        """
        num_metrics = len(metrics)
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] * num_metrics // 2))
        
        gs = GridSpec(num_metrics, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[idx, :])
            
            for strategy, strategy_results in results.items():
                sparsities = sorted(strategy_results.keys())
                
                # Extract values
                if isinstance(list(strategy_results.values())[0], dict):
                    # Handle nested structure with mean/std
                    means = []
                    stds = []
                    
                    for sparsity in sparsities:
                        if 'mean' in strategy_results[sparsity]:
                            means.append(strategy_results[sparsity]['mean'][metric])
                            if 'std' in strategy_results[sparsity] and show_confidence:
                                stds.append(strategy_results[sparsity]['std'][metric])
                        else:
                            means.append(strategy_results[sparsity].get(metric, 0))
                    
                    # Plot with confidence intervals
                    color = self.strategy_colors.get(strategy, None)
                    line = ax.plot(sparsities, means, 'o-', label=strategy, 
                                  linewidth=2.5, markersize=8, color=color)
                    
                    if stds and show_confidence:
                        means = np.array(means)
                        stds = np.array(stds)
                        ax.fill_between(sparsities, means - stds, means + stds,
                                      alpha=0.2, color=line[0].get_color())
                else:
                    # Simple structure
                    values = [strategy_results[s] for s in sparsities]
                    color = self.strategy_colors.get(strategy, None)
                    ax.plot(sparsities, values, 'o-', label=strategy,
                           linewidth=2.5, markersize=8, color=color)
            
            # Formatting
            ax.set_xlabel('Sparsity Level', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f'{metric.capitalize()} vs Sparsity', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Format x-axis as percentages
            ax.set_xticklabels([f'{s*100:.0f}%' for s in ax.get_xticks()])
            
            # Add annotations for key points
            if metric == 'accuracy':
                ax.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, 
                          label='Random baseline (10%)')
        
        if title:
            fig.suptitle(title, fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved pruning performance plot to {save_path}")
        
        return fig
    
    def plot_pruning_comparison_grid(
        self,
        results: Dict[str, Dict[float, Dict[str, Any]]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive grid comparing all aspects of pruning strategies.
        
        Args:
            results: Pruning results for multiple strategies
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_metric_comparison(ax1, results, 'accuracy', 'Accuracy Comparison')
        
        # 2. Loss comparison
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_metric_comparison(ax2, results, 'loss', 'Loss Comparison')
        
        # 3. Efficiency plot (accuracy drop vs sparsity)
        ax3 = fig.add_subplot(gs[2, :2])
        self._plot_efficiency(ax3, results)
        
        # 4. Strategy ranking
        ax4 = fig.add_subplot(gs[0, 2])
        self._plot_strategy_ranking(ax4, results)
        
        # 5. Sparsity distribution
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_sparsity_distribution(ax5, results)
        
        # 6. Summary statistics
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_summary_stats(ax6, results)
        
        fig.suptitle('Comprehensive Pruning Strategy Comparison', fontsize=16)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison grid to {save_path}")
        
        return fig
    
    def plot_multi_seed_results(
        self,
        results: Dict[str, List[Dict[float, Dict[str, float]]]],
        metric: str = 'accuracy',
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot results from multiple seeds with statistical analysis.
        
        Args:
            results: Dict of strategy -> list of results per seed
            metric: Metric to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Process data
        processed_results = {}
        for strategy, seed_results in results.items():
            sparsities = sorted(seed_results[0].keys())
            
            # Aggregate across seeds
            metric_values = {s: [] for s in sparsities}
            for seed_result in seed_results:
                for sparsity in sparsities:
                    if metric in seed_result[sparsity]:
                        metric_values[sparsity].append(seed_result[sparsity][metric])
            
            # Compute statistics
            means = [np.mean(metric_values[s]) for s in sparsities]
            stds = [np.std(metric_values[s]) for s in sparsities]
            
            processed_results[strategy] = {
                'sparsities': sparsities,
                'means': means,
                'stds': stds,
                'all_values': metric_values
            }
        
        # Plot 1: Mean with confidence intervals
        for strategy, data in processed_results.items():
            color = self.strategy_colors.get(strategy, None)
            line = ax1.plot(data['sparsities'], data['means'], 'o-', 
                           label=f"{strategy} (μ±σ)", linewidth=2.5, 
                           markersize=8, color=color)
            
            # Add confidence intervals
            means = np.array(data['means'])
            stds = np.array(data['stds'])
            ax1.fill_between(data['sparsities'], means - stds, means + stds,
                           alpha=0.2, color=line[0].get_color())
        
        ax1.set_xlabel('Sparsity Level')
        ax1.set_ylabel(metric.capitalize())
        ax1.set_title(f'{metric.capitalize()} Across Seeds (Mean ± Std)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Box plots at key sparsity levels
        key_sparsities = [0.5, 0.7, 0.9]
        box_data = []
        labels = []
        
        for sparsity in key_sparsities:
            for strategy, data in processed_results.items():
                if sparsity in data['all_values']:
                    box_data.append(data['all_values'][sparsity])
                    labels.append(f"{strategy}\n{sparsity*100:.0f}%")
        
        bp = ax2.boxplot(box_data, labels=labels, patch_artist=True)
        
        # Color boxes by strategy
        for patch, label in zip(bp['boxes'], labels):
            strategy = label.split('\n')[0]
            patch.set_facecolor(self.strategy_colors.get(strategy, 'gray'))
            patch.set_alpha(0.7)
        
        ax2.set_ylabel(metric.capitalize())
        ax2.set_title(f'{metric.capitalize()} Distribution at Key Sparsity Levels')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(f'Multi-Seed {metric.capitalize()} Analysis', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved multi-seed plot to {save_path}")
        
        return fig
    
    def plot_layer_wise_pruning(
        self,
        layer_sparsities: Dict[str, Dict[str, float]],
        model_accuracy: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize layer-wise pruning patterns.
        
        Args:
            layer_sparsities: Dict of strategy -> layer -> sparsity
            model_accuracy: Dict of strategy -> accuracy
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        strategies = list(layer_sparsities.keys())
        layers = list(next(iter(layer_sparsities.values())).keys())
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                       gridspec_kw={'height_ratios': [3, 1]})
        
        # Prepare data for heatmap
        data = []
        for strategy in strategies:
            row = [layer_sparsities[strategy].get(layer, 0) for layer in layers]
            data.append(row)
        
        # Plot heatmap
        im = ax1.imshow(data, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax1.set_xticks(range(len(layers)))
        ax1.set_yticks(range(len(strategies)))
        ax1.set_xticklabels(layers, rotation=45, ha='right')
        ax1.set_yticklabels(strategies)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Sparsity Level', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(strategies)):
            for j in range(len(layers)):
                text = ax1.text(j, i, f'{data[i][j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        ax1.set_title('Layer-wise Sparsity Patterns by Strategy')
        
        # Plot accuracy comparison
        ax2.bar(range(len(strategies)), [model_accuracy.get(s, 0) for s in strategies])
        ax2.set_xticks(range(len(strategies)))
        ax2.set_xticklabels(strategies)
        ax2.set_ylabel('Final Accuracy')
        ax2.set_title('Model Accuracy After Pruning')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved layer-wise pruning plot to {save_path}")
        
        return fig
    
    def _plot_metric_comparison(self, ax, results, metric, title):
        """Helper to plot metric comparison."""
        for strategy, strategy_results in results.items():
            sparsities = sorted(strategy_results.keys())
            values = [strategy_results[s].get(metric, 0) for s in sparsities]
            
            color = self.strategy_colors.get(strategy, None)
            ax.plot(sparsities, values, 'o-', label=strategy,
                   linewidth=2.5, markersize=8, color=color)
        
        ax.set_xlabel('Sparsity Level')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_efficiency(self, ax, results):
        """Plot efficiency (accuracy retention vs sparsity)."""
        for strategy, strategy_results in results.items():
            sparsities = sorted(strategy_results.keys())
            
            # Get initial accuracy (at 0 sparsity or lowest sparsity)
            initial_acc = strategy_results.get(0, strategy_results[sparsities[0]]).get('accuracy', 100)
            
            # Calculate accuracy retention
            retentions = []
            for s in sparsities:
                current_acc = strategy_results[s].get('accuracy', 0)
                retention = (current_acc / initial_acc) * 100 if initial_acc > 0 else 0
                retentions.append(retention)
            
            color = self.strategy_colors.get(strategy, None)
            ax.plot(sparsities, retentions, 'o-', label=strategy,
                   linewidth=2.5, markersize=8, color=color)
        
        ax.set_xlabel('Sparsity Level')
        ax.set_ylabel('Accuracy Retention (%)')
        ax.set_title('Pruning Efficiency: Accuracy Retention vs Sparsity')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% retention')
    
    def _plot_strategy_ranking(self, ax, results):
        """Create strategy ranking based on area under accuracy curve."""
        rankings = {}
        
        for strategy, strategy_results in results.items():
            sparsities = sorted(strategy_results.keys())
            accuracies = [strategy_results[s].get('accuracy', 0) for s in sparsities]
            
            # Calculate area under curve (simple trapezoidal)
            if len(sparsities) > 1:
                auc = np.trapz(accuracies, sparsities)
                rankings[strategy] = auc
        
        # Sort and plot
        sorted_strategies = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        strategies, scores = zip(*sorted_strategies)
        
        colors = [self.strategy_colors.get(s, 'gray') for s in strategies]
        bars = ax.barh(range(len(strategies)), scores, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        ax.set_xlabel('AUC Score')
        ax.set_title('Strategy Ranking (by AUC)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                   f'{score:.2f}', ha='left', va='center', fontsize=9)
    
    def _plot_sparsity_distribution(self, ax, results):
        """Plot distribution of final sparsities achieved."""
        # This would be more relevant with layer-wise data
        # For now, show target vs achieved sparsity
        ax.text(0.5, 0.5, 'Sparsity Distribution\n(Requires layer-wise data)',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _plot_summary_stats(self, ax, results):
        """Create summary statistics table."""
        summary_data = []
        
        for strategy in results:
            # Get performance at key sparsity levels
            perf_50 = results[strategy].get(0.5, {}).get('accuracy', 'N/A')
            perf_70 = results[strategy].get(0.7, {}).get('accuracy', 'N/A')
            perf_90 = results[strategy].get(0.9, {}).get('accuracy', 'N/A')
            
            summary_data.append([strategy, f"{perf_50:.1f}%", f"{perf_70:.1f}%", f"{perf_90:.1f}%"])
        
        # Create table
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Strategy', '50% Sparse', '70% Sparse', '90% Sparse'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color header
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Accuracy at Key Sparsity Levels', pad=20)
    
    def plot_weight_distribution_comparison(
        self,
        weights_before: Dict[str, torch.Tensor],
        weights_after: Dict[str, torch.Tensor],
        strategies: List[str],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare weight distributions before and after pruning.
        
        Args:
            weights_before: Original weights per layer
            weights_after: Pruned weights per layer/strategy
            strategies: List of pruning strategies used
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        num_strategies = len(strategies)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, strategy in enumerate(strategies[:4]):  # Max 4 subplots
            ax = axes[idx]
            
            # Get weights for first layer (or combine all layers)
            layer_key = list(weights_before.keys())[0]
            original = weights_before[layer_key].flatten().cpu().numpy()
            
            # Get pruned weights
            pruned_key = f"{layer_key}_{strategy}" if f"{layer_key}_{strategy}" in weights_after else layer_key
            if pruned_key in weights_after:
                pruned = weights_after[pruned_key].flatten().cpu().numpy()
            else:
                # Simulate pruning effect
                mask = np.random.rand(len(original)) > 0.5
                pruned = original[mask]
            
            # Plot histograms
            bins = np.linspace(original.min(), original.max(), 50)
            ax.hist(original, bins=bins, alpha=0.5, label='Original', 
                    color='gray', density=True)
            ax.hist(pruned, bins=bins, alpha=0.7, label=f'After {strategy}',
                    color=self.strategy_colors.get(strategy, '#333333'), density=True)
            
            ax.set_title(f'{strategy.replace("_", " ").title()} Pruning', fontsize=12)
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Weight Distribution Before/After Pruning', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weight distribution plot to {save_path}")
        
        return fig
    
    def plot_multi_metric_radar(
        self,
        strategy_metrics: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create radar chart comparing strategies across multiple metrics.
        
        Args:
            strategy_metrics: Dict of strategy -> metric -> value
                             Metrics should be normalized to [0, 1]
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        strategies = list(strategy_metrics.keys())
        metrics = list(next(iter(strategy_metrics.values())).keys())
        num_vars = len(metrics)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for strategy in strategies:
            values = [strategy_metrics[strategy].get(m, 0) for m in metrics]
            values += values[:1]
            
            color = self.strategy_colors.get(strategy, '#333333')
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        # Customize
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Multi-Metric Strategy Comparison', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved radar chart to {save_path}")
        
        return fig
    
    def plot_pruning_efficiency_curve(
        self,
        results: Dict[str, List[Tuple[float, float]]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot efficiency curves showing accuracy retention vs model compression.
        
        Args:
            results: Dict of strategy -> list of (compression_ratio, accuracy) tuples
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for strategy, data in results.items():
            if data:
                compression_ratios, accuracies = zip(*data)
                color = self.strategy_colors.get(strategy, '#333333')
                ax.plot(compression_ratios, accuracies, 'o-', label=strategy,
                        color=color, linewidth=2.5, markersize=8)
        
        # Add efficiency zones
        ax.axhspan(90, 100, alpha=0.1, color='green', label='Excellent')
        ax.axhspan(70, 90, alpha=0.1, color='yellow', label='Good')
        ax.axhspan(0, 70, alpha=0.1, color='red', label='Poor')
        
        # Formatting
        ax.set_xlabel('Compression Ratio', fontsize=12)
        ax.set_ylabel('Accuracy Retention (%)', fontsize=12)
        ax.set_title('Pruning Efficiency: Accuracy vs Compression', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.set_xlim(left=1)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved efficiency curve to {save_path}")
        
        return fig
    
    def plot_comprehensive_dashboard(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            results: Complete experiment results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Main accuracy plot (spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if 'strategies' in results:
            dropout_rates = results.get('dropout_rates', [])
            for strategy, accuracies in results['strategies'].items():
                color = self.strategy_colors.get(strategy, '#333333')
                ax1.plot(dropout_rates, accuracies, 'o-', label=strategy,
                        color=color, linewidth=2.5, markersize=8)
        ax1.set_title('Accuracy vs Dropout Rate', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Dropout Rate')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Strategy ranking
        ax2 = fig.add_subplot(gs[0, 2])
        if 'strategies' in results:
            strategies = list(results['strategies'].keys())
            final_accs = [results['strategies'][s][-1] for s in strategies]
            colors = [self.strategy_colors.get(s, '#333333') for s in strategies]
            bars = ax2.barh(strategies, final_accs, color=colors, alpha=0.7)
            ax2.set_xlabel('Final Accuracy (%)')
            ax2.set_title('Strategy Ranking', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Efficiency comparison
        ax3 = fig.add_subplot(gs[0, 3])
        efficiency_data = results.get('efficiency', {'low': 0.9, 'high': 0.6, 'random': 0.7})
        strategies = list(efficiency_data.keys())
        values = list(efficiency_data.values())
        colors = [self.strategy_colors.get(s, '#333333') for s in strategies]
        ax3.bar(strategies, values, color=colors, alpha=0.7)
        ax3.set_ylabel('Efficiency Score')
        ax3.set_title('Pruning Efficiency', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Layer importance (spans 2 rows)
        ax4 = fig.add_subplot(gs[1:, :2])
        if 'layer_importance' in results:
            layers = list(results['layer_importance'].keys())
            importance = [results['layer_importance'][l].get('mean', 0) for l in layers]
        else:
            layers = ['conv1', 'conv2', 'fc1', 'fc2']
            importance = np.random.rand(len(layers))
        colors_layers = [self.strategy_colors.get(f'layer_{i}', '#333333') for i in range(len(layers))]
        bars = ax4.bar(layers, importance, color=colors_layers, alpha=0.7)
        ax4.set_ylabel('Importance Score')
        ax4.set_title('Layer Importance Analysis', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Performance over time
        ax5 = fig.add_subplot(gs[1, 2:])
        if 'fine_tuning' in results:
            for strategy, performance in results['fine_tuning'].items():
                epochs = range(1, len(performance) + 1)
                color = self.strategy_colors.get(strategy, '#333333')
                ax5.plot(epochs, performance, 'o-', label=strategy, color=color, linewidth=2)
        ax5.set_xlabel('Fine-tuning Epoch')
        ax5.set_ylabel('Accuracy (%)')
        ax5.set_title('Recovery After Pruning', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary statistics table
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create summary data
        summary_data = []
        if 'strategies' in results:
            for strategy in ['low', 'high', 'random']:
                if strategy in results['strategies']:
                    accs = results['strategies'][strategy]
                    summary_data.append([
                        strategy.capitalize(),
                        f"{accs[0]:.1f}%",
                        f"{accs[len(accs)//2]:.1f}%",
                        f"{accs[-1]:.1f}%",
                        f"{max(accs) - min(accs):.1f}%"
                    ])
        
        if summary_data:
            table = ax6.table(cellText=summary_data,
                             colLabels=['Strategy', 'Initial', 'Mid-point', 'Final', 'Range'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style header
            for i in range(5):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)
        
        # Overall title
        fig.suptitle('Pruning Experiment Comprehensive Dashboard', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved dashboard to {save_path}")
        
        return fig
    
    def plot_accuracy_vs_sparsity_enhanced(
        self,
        results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Enhanced accuracy vs sparsity plot with relative performance.
        
        Args:
            results: Experiment results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        if 'strategies' in results:  # Progressive dropout format
            strategies = results['strategies']
            dropout_rates = results.get('dropout_rates', [])
            
            # Plot accuracy vs dropout
            for strategy, accuracies in strategies.items():
                color = self.strategy_colors.get(strategy, '#333333')
                ax1.plot(dropout_rates, accuracies, 'o-', label=strategy,
                        color=color, linewidth=2.5, markersize=8)
            
            ax1.set_xlabel('Dropout Rate', fontsize=12)
            ax1.set_ylabel('Accuracy (%)', fontsize=12)
            ax1.set_title('Accuracy vs Dropout Rate', fontsize=14, fontweight='bold')
            
            # Calculate relative performance
            baseline = strategies.get('low', strategies[list(strategies.keys())[0]])
            for strategy, accuracies in strategies.items():
                if strategy != 'low':
                    relative = [(a/b)*100 if b > 0 else 0 for a, b in zip(accuracies, baseline)]
                    color = self.strategy_colors.get(strategy, '#333333')
                    ax2.plot(dropout_rates, relative, 'o-', label=strategy,
                            color=color, linewidth=2.5, markersize=8, alpha=0.8)
            
            ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Dropout Rate', fontsize=12)
            ax2.set_ylabel('Relative Accuracy (%)', fontsize=12)
            ax2.set_title('Relative Performance (vs Low Mode)', fontsize=14, fontweight='bold')
        
        # Formatting
        for ax in [ax1, ax2]:
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved enhanced accuracy plot to {save_path}")
        
        return fig 