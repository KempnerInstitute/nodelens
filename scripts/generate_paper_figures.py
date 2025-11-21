import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from alignment.analysis.visualization.pruning_plots import PruningVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_experiment_data(results_path: Path) -> Dict[str, Any]:
    """Load experiment results from a JSON file."""
    try:
        with open(results_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results from {results_path}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Generate figures for SCAR paper")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory containing experiment results JSONs")
    parser.add_argument("--output_dir", type=str, default="paper_figures", help="Directory to save generated figures")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Gather all result files
    result_files = list(results_path.glob("**/*.json"))
    if not result_files:
        logger.error(f"No JSON result files found in {results_path}")
        return

    all_results = []
    for f in result_files:
        data = load_experiment_data(f)
        if data:
            all_results.append(data)

    # Generate Figures
    
    # For detailed metric plots (Figs 1 & 2), we need a run that has detailed alignment history.
    # We pick the most recent one that has "alignment" data.
    detailed_run = None
    for res in sorted(all_results, key=lambda x: x.get("config", {}).get("name", ""), reverse=True):
        if "alignment" in res.get("train_results", {}):
            detailed_run = res
            break
    
    visualizer = PruningVisualizer(style="seaborn-v0_8-paper", figsize=(10, 5))

    if detailed_run:
        alignment_history = detailed_run.get("train_results", {}).get("alignment", {})
        visualizer.plot_metric_distributions_from_alignment(
            alignment_history,
            save_dir=output_path,
            metric_mapping={
                "activation_outlier_index": "Outlier Index (OI)",
                "scar_taylor": "Taylor Saliency (T)",
                "scar_curvature": "Curvature (R)",
                "rayleigh_quotient": "Rayleigh Quotient",
            },
        )

        redundancy_history = alignment_history.get("pairwise_redundancy_gaussian")
        synergy_history = alignment_history.get("gaussian_pid_synergy_mmi")
        outlier_history = alignment_history.get("activation_outlier_index")

        if redundancy_history and synergy_history:
            red_snapshot = redundancy_history[0] if isinstance(redundancy_history, list) else redundancy_history
            syn_snapshot = synergy_history[0] if isinstance(synergy_history, list) else synergy_history
            oi_snapshot = outlier_history[0] if isinstance(outlier_history, list) else outlier_history
            visualizer.plot_redundancy_synergy_scatter(
                red_snapshot,
                syn_snapshot,
                outlier_snapshot=oi_snapshot,
                save_path=output_path / "fig2_redundancy_halos.png",
            )
    else:
        logger.warning("Could not find a run with alignment history for Figs 1 & 2.")

    # Build dataframe for Figure 3
    rows = []
    for res in all_results:
        exp_name = res.get("config", {}).get("name", "unknown")
        strategies = res.get("pruning_results", {}).get("strategies", {})
        for strategy, data in strategies.items():
            sparsities = data.get("sparsities", [])
            losses = data.get("losses_after_finetune", []) or data.get("losses_before_finetune", [])
            ppls = [np.exp(loss) for loss in losses]
            for s, p in zip(sparsities, ppls):
                rows.append({"Method": strategy, "Sparsity": s, "Perplexity": p, "Experiment": exp_name})

    if rows:
        df_curves = pd.DataFrame(rows)
        visualizer.plot_sparsity_perplexity_curves(
            df_curves, save_path=output_path / "fig3_sparsity_perplexity.png", title="Sparsity vs Perplexity (Wikitext-2)"
        )
    else:
        logger.warning("No pruning results found for Figure 3.")

    # Build dataframe for Figure 4 (ablation)
    ablation_rows = []
    for res in all_results:
        strategies = res.get("pruning_results", {}).get("strategies", {})
        for strategy, data in strategies.items():
            # Focus on SCAR variants
            if "scar" not in strategy.lower():
                continue
            sparsities = data.get("sparsities", [])
            losses = data.get("losses_after_finetune", []) or data.get("losses_before_finetune", [])
            ppls = [np.exp(loss) for loss in losses]
            for s, p in zip(sparsities, ppls):
                ablation_rows.append({"Variant": strategy, "Sparsity": s, "Perplexity": p})

    if ablation_rows:
        df_ablations = pd.DataFrame(ablation_rows)
        visualizer.plot_ablation_summary(
            df_ablations,
            save_path=output_path / "fig4_ablation.png",
            title="Ablation: Impact of SCAR Components",
        )
    else:
        logger.warning("No SCAR ablation results found for Figure 4.")

    logger.info(f"All figures generated in {output_path}")


if __name__ == "__main__":
    main()

