"""
Unified Analysis Runner for alignment experiments.

This module provides a single entry point for all analysis and visualization tasks,
driven by configuration. It consolidates scattered plotting code from experiments,
scripts, and other modules into a unified interface.

Usage:
    from alignment.analysis import AnalysisRunner
    
    runner = AnalysisRunner(config)
    runner.run()
    
Or via CLI:
    python -m alignment.analysis.analysis_runner --config analysis_config.yaml
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml

from .visualization import UnifiedVisualizer

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for analysis and visualization tasks."""
    
    # Input sources
    results_dir: Optional[str] = None  # Directory with experiment result JSONs
    results_file: Optional[str] = None  # Single result file
    importance_scores_file: Optional[str] = None  # Pre-computed importance scores
    
    # Output
    output_dir: str = "./analysis_output"
    
    # Visualization style
    style: str = "seaborn-v0_8-paper"
    figsize: tuple = (10, 6)
    dpi: int = 300
    format: str = "png"
    
    # Which analyses to run
    analyses: List[str] = field(default_factory=lambda: ["all"])
    
    # Per-analysis configuration
    histograms: Dict[str, Any] = field(default_factory=dict)
    scatter_plots: Dict[str, Any] = field(default_factory=dict)
    heatmaps: Dict[str, Any] = field(default_factory=dict)
    pruning_curves: Dict[str, Any] = field(default_factory=dict)
    layer_distributions: Dict[str, Any] = field(default_factory=dict)
    scar_analysis: Dict[str, Any] = field(default_factory=dict)
    advanced_pruning: Dict[str, Any] = field(default_factory=dict)  # Uses PruningVisualizer
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnalysisConfig":
        """Create config from dictionary."""
        # Handle nested analysis block
        if "analysis" in d:
            d = {**d, **d.pop("analysis")}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AnalysisConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


class AnalysisRunner:
    """
    Unified runner for all analysis and visualization tasks.
    
    This class consolidates analysis functionality from:
    - LLMAlignmentExperiment plotting methods
    - GeneralAlignmentExperiment visualization
    - scripts/generate_paper_figures.py
    - scripts/generate_llm_layer_figures.py
    - scripts/generate_vision_figures.py
    
    All analyses are driven by configuration, making it easy to reproduce
    figures and customize outputs.
    """
    
    def __init__(self, config: Union[AnalysisConfig, Dict[str, Any], str, Path]):
        """
        Initialize analysis runner.
        
        Args:
            config: AnalysisConfig, dict, or path to YAML config file.
        """
        if isinstance(config, (str, Path)):
            self.config = AnalysisConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = AnalysisConfig.from_dict(config)
        else:
            self.config = config
        
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.viz = UnifiedVisualizer(
            style=self.config.style,
            figsize=self.config.figsize
        )
        
        # Loaded data
        self.results: List[Dict[str, Any]] = []
        self.importance_scores: Dict[str, Dict[str, torch.Tensor]] = {}
        self.scar_scores: Dict[str, Dict[str, torch.Tensor]] = {}
    
    def load_data(self) -> None:
        """Load experiment results and scores from configured sources."""
        # Load from results directory
        if self.config.results_dir:
            results_path = Path(self.config.results_dir)
            for json_file in results_path.glob("**/*.json"):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        self.results.append(data)
                        logger.info(f"Loaded results from {json_file}")
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")
        
        # Load single results file
        if self.config.results_file:
            try:
                with open(self.config.results_file, "r") as f:
                    data = json.load(f)
                    self.results.append(data)
                    logger.info(f"Loaded results from {self.config.results_file}")
            except Exception as e:
                logger.error(f"Failed to load {self.config.results_file}: {e}")
        
        # Load pre-computed importance scores
        if self.config.importance_scores_file:
            try:
                scores_data = torch.load(self.config.importance_scores_file)
                if isinstance(scores_data, dict):
                    for layer, metrics in scores_data.items():
                        self.importance_scores[layer] = {}
                        for metric, values in metrics.items():
                            if isinstance(values, torch.Tensor):
                                self.importance_scores[layer][metric] = values
                            else:
                                self.importance_scores[layer][metric] = torch.tensor(values)
                    logger.info(f"Loaded importance scores from {self.config.importance_scores_file}")
            except Exception as e:
                logger.error(f"Failed to load importance scores: {e}")
        
        # Extract importance scores from results if not loaded separately
        if not self.importance_scores:
            for result in self.results:
                if "importance_scores" in result:
                    for layer, metrics in result["importance_scores"].items():
                        if layer not in self.importance_scores:
                            self.importance_scores[layer] = {}
                        for metric, values in metrics.items():
                            if isinstance(values, dict):
                                # Summary stats format
                                continue
                            if isinstance(values, list):
                                self.importance_scores[layer][metric] = torch.tensor(values)
                            elif isinstance(values, torch.Tensor):
                                self.importance_scores[layer][metric] = values
        
        # Extract SCAR scores
        for result in self.results:
            if "scar_scores" in result:
                for layer, metrics in result["scar_scores"].items():
                    if layer not in self.scar_scores:
                        self.scar_scores[layer] = {}
                    for metric, values in metrics.items():
                        if isinstance(values, dict):
                            continue
                        if isinstance(values, list):
                            self.scar_scores[layer][metric] = torch.tensor(values)
                        elif isinstance(values, torch.Tensor):
                            self.scar_scores[layer][metric] = values
    
    def run(self) -> Dict[str, Any]:
        """
        Run all configured analyses.
        
        Returns:
            Dictionary with paths to generated outputs.
        """
        logger.info(f"Starting analysis, output to {self.output_dir}")
        
        self.load_data()
        
        outputs = {}
        analyses = self.config.analyses
        run_all = "all" in analyses
        
        # Importance score histograms
        if run_all or "histograms" in analyses:
            outputs["histograms"] = self._run_histograms()
        
        # Scatter plots
        if run_all or "scatter_plots" in analyses:
            outputs["scatter_plots"] = self._run_scatter_plots()
        
        # Heatmaps
        if run_all or "heatmaps" in analyses:
            outputs["heatmaps"] = self._run_heatmaps()
        
        # Layer distributions
        if run_all or "layer_distributions" in analyses:
            outputs["layer_distributions"] = self._run_layer_distributions()
        
        # Pruning curves
        if run_all or "pruning_curves" in analyses:
            outputs["pruning_curves"] = self._run_pruning_curves()
        
        # SCAR analysis
        if run_all or "scar_analysis" in analyses:
            outputs["scar_analysis"] = self._run_scar_analysis()
        
        # Advanced pruning analysis (uses PruningVisualizer)
        if "advanced_pruning" in analyses:
            outputs["advanced_pruning"] = self._run_advanced_pruning()
        
        # Generate summary report
        self._generate_summary(outputs)
        
        logger.info(f"Analysis complete. Outputs saved to {self.output_dir}")
        return outputs
    
    def _run_histograms(self) -> List[str]:
        """Generate importance score histograms."""
        outputs = []
        cfg = self.config.histograms
        
        bins = cfg.get("bins", 100)
        top_k = cfg.get("top_k", 5)
        metrics = cfg.get("metrics", None)  # None = all available
        layers = cfg.get("layers", None)  # None = all available
        
        hist_dir = self.output_dir / "histograms"
        hist_dir.mkdir(exist_ok=True)
        
        for layer_name, layer_scores in self.importance_scores.items():
            if layers and layer_name not in layers:
                continue
            
            for metric_name, scores in layer_scores.items():
                if metrics and metric_name not in metrics:
                    continue
                
                if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
                    continue
                
                try:
                    save_path = self.viz.plot_importance_histogram(
                        scores=scores,
                        layer_name=layer_name,
                        metric_name=metric_name,
                        plots_dir=hist_dir,
                        top_k=top_k,
                    )
                    outputs.append(str(save_path))
                    logger.info(f"Generated histogram: {save_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate histogram for {layer_name}/{metric_name}: {e}")
        
        return outputs
    
    def _run_scatter_plots(self) -> List[str]:
        """Generate scatter plots comparing metrics."""
        outputs = []
        cfg = self.config.scatter_plots
        
        pairs = cfg.get("pairs", [
            ("activation_l2_norm", "rayleigh_quotient"),
            ("activation_variance", "rayleigh_quotient"),
            ("rayleigh_quotient", "pairwise_redundancy_gaussian"),
        ])
        alpha = cfg.get("alpha", 0.5)
        point_size = cfg.get("point_size", 10)
        
        scatter_dir = self.output_dir / "scatter_plots"
        scatter_dir.mkdir(exist_ok=True)
        
        for layer_name, layer_scores in self.importance_scores.items():
            for x_metric, y_metric in pairs:
                if x_metric not in layer_scores or y_metric not in layer_scores:
                    continue
                
                x_scores = layer_scores[x_metric]
                y_scores = layer_scores[y_metric]
                
                if not isinstance(x_scores, torch.Tensor) or not isinstance(y_scores, torch.Tensor):
                    continue
                
                try:
                    safe_layer = layer_name.replace(".", "_").replace("/", "_")
                    save_path = scatter_dir / f"{safe_layer}_{x_metric}_vs_{y_metric}.{self.config.format}"
                    
                    self.viz.plot_scatter_2d(
                        x=x_scores,
                        y=y_scores,
                        xlabel=x_metric,
                        ylabel=y_metric,
                        title=f"{layer_name}: {x_metric} vs {y_metric}",
                        save_path=save_path,
                        alpha=alpha,
                        s=point_size,
                    )
                    outputs.append(str(save_path))
                    logger.info(f"Generated scatter plot: {save_path}")
                except Exception as e:
                    logger.warning(f"Failed to generate scatter for {layer_name}: {e}")
        
        return outputs
    
    def _run_heatmaps(self) -> List[str]:
        """Generate heatmaps of metrics across layers."""
        outputs = []
        cfg = self.config.heatmaps
        
        metrics = cfg.get("metrics", None)
        
        heatmap_dir = self.output_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        
        # Build layer x metric matrix of mean values
        layer_metric_means: Dict[str, Dict[str, float]] = {}
        
        for layer_name, layer_scores in self.importance_scores.items():
            layer_metric_means[layer_name] = {}
            for metric_name, scores in layer_scores.items():
                if metrics and metric_name not in metrics:
                    continue
                if isinstance(scores, torch.Tensor) and scores.numel() > 0:
                    layer_metric_means[layer_name][metric_name] = float(scores.mean().item())
        
        if layer_metric_means:
            try:
                save_path = heatmap_dir / f"importance_metrics_heatmap.{self.config.format}"
                self.viz.plot_heatmap(
                    data=layer_metric_means,
                    title="Mean Importance Scores by Layer and Metric",
                    save_path=save_path,
                )
                outputs.append(str(save_path))
                logger.info(f"Generated heatmap: {save_path}")
            except Exception as e:
                logger.warning(f"Failed to generate importance heatmap: {e}")
        
        return outputs
    
    def _run_layer_distributions(self) -> List[str]:
        """Generate layer-wise score distribution plots."""
        outputs = []
        cfg = self.config.layer_distributions
        
        plot_type = cfg.get("plot_type", "violin")
        metrics = cfg.get("metrics", None)
        
        dist_dir = self.output_dir / "layer_distributions"
        dist_dir.mkdir(exist_ok=True)
        
        # Group scores by metric across layers
        metric_to_layers: Dict[str, Dict[str, torch.Tensor]] = {}
        
        for layer_name, layer_scores in self.importance_scores.items():
            for metric_name, scores in layer_scores.items():
                if metrics and metric_name not in metrics:
                    continue
                if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
                    continue
                
                if metric_name not in metric_to_layers:
                    metric_to_layers[metric_name] = {}
                metric_to_layers[metric_name][layer_name] = scores
        
        for metric_name, layer_scores in metric_to_layers.items():
            try:
                save_path = dist_dir / f"{metric_name}_layer_distribution.{self.config.format}"
                self.viz.plot_layer_scores(
                    scores=layer_scores,
                    metric_name=metric_name,
                    plot_type=plot_type,
                    save_path=str(save_path),
                )
                outputs.append(str(save_path))
                logger.info(f"Generated layer distribution: {save_path}")
            except Exception as e:
                logger.warning(f"Failed to generate layer distribution for {metric_name}: {e}")
        
        return outputs
    
    def _run_pruning_curves(self) -> List[str]:
        """Generate pruning performance curves."""
        outputs = []
        cfg = self.config.pruning_curves
        
        curves_dir = self.output_dir / "pruning_curves"
        curves_dir.mkdir(exist_ok=True)
        
        # Extract pruning results from loaded data
        for result in self.results:
            pruning_results = result.get("pruning_results", {})
            if not pruning_results:
                continue
            
            # Check for sparsity-perplexity data
            sparsities = []
            perplexities = []
            
            # Handle different result formats
            if "strategies" in pruning_results:
                # Multi-strategy format
                for strategy, data in pruning_results["strategies"].items():
                    strat_sparsities = data.get("sparsities", [])
                    losses = data.get("losses_after_finetune", []) or data.get("losses_before_finetune", [])
                    strat_ppls = [np.exp(loss) for loss in losses]
                    
                    if strat_sparsities and strat_ppls:
                        try:
                            save_path = curves_dir / f"pruning_{strategy}.{self.config.format}"
                            self.viz.plot_sparsity_performance(
                                sparsities=strat_sparsities,
                                perplexities=strat_ppls,
                                strategy_name=strategy,
                                title=f"Pruning Performance: {strategy}",
                                save_path=save_path,
                            )
                            outputs.append(str(save_path))
                            logger.info(f"Generated pruning curve: {save_path}")
                        except Exception as e:
                            logger.warning(f"Failed to generate pruning curve for {strategy}: {e}")
            else:
                # Single-run format (sparsity_X keys)
                baseline_ppl = result.get("evaluation", {}).get("baseline_perplexity")
                
                for key, data in pruning_results.items():
                    if key.startswith("sparsity_"):
                        sparsity = float(key.replace("sparsity_", ""))
                        ppl = data.get("perplexity")
                        if ppl is not None:
                            sparsities.append(sparsity)
                            perplexities.append(ppl)
                
                if sparsities and perplexities:
                    # Sort by sparsity
                    sorted_pairs = sorted(zip(sparsities, perplexities))
                    sparsities, perplexities = zip(*sorted_pairs)
                    
                    try:
                        exp_name = result.get("config", {}).get("name", "experiment")
                        save_path = curves_dir / f"pruning_{exp_name}.{self.config.format}"
                        self.viz.plot_sparsity_performance(
                            sparsities=list(sparsities),
                            perplexities=list(perplexities),
                            strategy_name=exp_name,
                            baseline_ppl=baseline_ppl,
                            title=f"Sparsity vs Perplexity",
                            save_path=save_path,
                        )
                        outputs.append(str(save_path))
                        logger.info(f"Generated pruning curve: {save_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate pruning curve: {e}")
        
        return outputs
    
    def _run_scar_analysis(self) -> List[str]:
        """Generate SCAR-specific visualizations."""
        outputs = []
        cfg = self.config.scar_analysis
        
        scar_dir = self.output_dir / "scar_analysis"
        scar_dir.mkdir(exist_ok=True)
        
        if not self.scar_scores:
            logger.info("No SCAR scores available for analysis")
            return outputs
        
        # SCAR layer scores (violin/box plots)
        for metric_name in ["scar_loss_proxy", "scar_activation_power", "scar_curvature", "scar_taylor"]:
            try:
                save_path = scar_dir / f"{metric_name}_layers.{self.config.format}"
                fig = self.viz.plot_scar_layer_scores(
                    scar_scores=self.scar_scores,
                    metric_name=metric_name,
                    plot_type=cfg.get("plot_type", "violin"),
                    save_path=str(save_path),
                )
                if fig:
                    outputs.append(str(save_path))
                    logger.info(f"Generated SCAR layer plot: {save_path}")
            except Exception as e:
                logger.warning(f"Failed to generate SCAR layer plot for {metric_name}: {e}")
        
        # SCAR heatmap
        try:
            save_path = scar_dir / f"scar_metrics_heatmap.{self.config.format}"
            self.viz.plot_scar_heatmap(
                scar_scores=self.scar_scores,
                metrics=["scar_activation_power", "scar_taylor", "scar_curvature", "scar_loss_proxy"],
                title="SCAR Metrics per Layer",
                save_path=str(save_path),
            )
            outputs.append(str(save_path))
            logger.info(f"Generated SCAR heatmap: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to generate SCAR heatmap: {e}")
        
        return outputs
    
    def _run_advanced_pruning(self) -> List[str]:
        """Generate advanced pruning visualizations using PruningVisualizer."""
        outputs = []
        cfg = self.config.advanced_pruning
        
        try:
            from .visualization.pruning_plots import PruningVisualizer
        except ImportError:
            logger.warning("PruningVisualizer not available for advanced pruning analysis")
            return outputs
        
        adv_dir = self.output_dir / "advanced_pruning"
        adv_dir.mkdir(exist_ok=True)
        
        pruning_viz = PruningVisualizer(style=self.config.style, figsize=self.config.figsize)
        
        # Extract alignment history for metric distributions
        for result in self.results:
            alignment_history = result.get("train_results", {}).get("alignment", {})
            
            if alignment_history:
                # Metric distributions
                try:
                    pruning_viz.plot_metric_distributions_from_alignment(
                        alignment_history,
                        save_dir=adv_dir,
                        metric_mapping=cfg.get("metric_mapping", {
                            "activation_outlier_index": "Outlier Index (OI)",
                            "scar_taylor": "Taylor Saliency (T)",
                            "scar_curvature": "Curvature (R)",
                            "rayleigh_quotient": "Rayleigh Quotient",
                        }),
                    )
                    outputs.append(str(adv_dir / "metric_distributions.png"))
                    logger.info("Generated metric distributions plot")
                except Exception as e:
                    logger.warning(f"Failed to generate metric distributions: {e}")
                
                # Redundancy-synergy scatter
                redundancy = alignment_history.get("pairwise_redundancy_gaussian")
                synergy = alignment_history.get("gaussian_pid_synergy_mmi")
                outlier = alignment_history.get("activation_outlier_index")
                
                if redundancy and synergy:
                    try:
                        red_snap = redundancy[0] if isinstance(redundancy, list) else redundancy
                        syn_snap = synergy[0] if isinstance(synergy, list) else synergy
                        oi_snap = outlier[0] if isinstance(outlier, list) else outlier if outlier else None
                        
                        save_path = adv_dir / f"redundancy_synergy_scatter.{self.config.format}"
                        pruning_viz.plot_redundancy_synergy_scatter(
                            red_snap, syn_snap,
                            outlier_snapshot=oi_snap,
                            save_path=save_path,
                        )
                        outputs.append(str(save_path))
                        logger.info(f"Generated redundancy-synergy scatter: {save_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate redundancy-synergy scatter: {e}")
            
            # Sparsity-perplexity curves from pruning results
            pruning_results = result.get("pruning_results", {})
            if "strategies" in pruning_results:
                import pandas as pd
                
                rows = []
                for strategy, data in pruning_results["strategies"].items():
                    sparsities = data.get("sparsities", [])
                    losses = data.get("losses_after_finetune", []) or data.get("losses_before_finetune", [])
                    ppls = [np.exp(loss) for loss in losses]
                    for s, p in zip(sparsities, ppls):
                        rows.append({"Method": strategy, "Sparsity": s, "Perplexity": p})
                
                if rows:
                    try:
                        df = pd.DataFrame(rows)
                        save_path = adv_dir / f"sparsity_perplexity_comparison.{self.config.format}"
                        pruning_viz.plot_sparsity_perplexity_curves(
                            df,
                            save_path=save_path,
                            title="Sparsity vs Perplexity Comparison",
                        )
                        outputs.append(str(save_path))
                        logger.info(f"Generated sparsity-perplexity comparison: {save_path}")
                    except Exception as e:
                        logger.warning(f"Failed to generate sparsity-perplexity curves: {e}")
        
        return outputs
    
    def _generate_summary(self, outputs: Dict[str, List[str]]) -> None:
        """Generate a summary of all outputs."""
        summary_path = self.output_dir / "analysis_summary.json"
        
        summary = {
            "output_dir": str(self.output_dir),
            "num_results_loaded": len(self.results),
            "num_layers_with_scores": len(self.importance_scores),
            "num_layers_with_scar": len(self.scar_scores),
            "generated_outputs": outputs,
            "total_files_generated": sum(len(v) for v in outputs.values()),
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis summary saved to {summary_path}")


def run_analysis_from_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to run analysis from a config file.
    
    Args:
        config_path: Path to YAML config file.
    
    Returns:
        Dictionary with paths to generated outputs.
    """
    runner = AnalysisRunner(config_path)
    return runner.run()


def main():
    """CLI entry point for analysis runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run analysis and visualization from configuration."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to analysis config YAML file",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Override results directory from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--analyses",
        type=str,
        nargs="+",
        default=None,
        help="Override which analyses to run (e.g., histograms scatter_plots)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = AnalysisConfig.from_yaml(args.config)
    
    # Apply overrides
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.analyses:
        config.analyses = args.analyses
    
    # Run analysis
    runner = AnalysisRunner(config)
    outputs = runner.run()
    
    print(f"\nAnalysis complete. Generated {sum(len(v) for v in outputs.values())} files.")
    print(f"Output directory: {config.output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()

