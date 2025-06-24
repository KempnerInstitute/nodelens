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
import seaborn as sns

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
        
        # Define color palette for strategies
        self.strategy_colors = {
            'magnitude': '#1f77b4',
            'gradient': '#ff7f0e', 
            'fisher': '#2ca02c',
            'random': '#d62728',
            'low': '#9467bd',
            'high': '#8c564b',
            'magnitude_low': '#1f77b4',
            'magnitude_high': '#17becf',
            'gradient_low': '#ff7f0e',
            'gradient_high': '#ffbb78',
        }
    
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