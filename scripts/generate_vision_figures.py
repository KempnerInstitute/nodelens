import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from alignment.analysis.visualization.pruning_plots import PruningVisualizer


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_results(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return {}


def is_vision_experiment(cfg: Dict[str, Any]) -> bool:
    dataset = (cfg.get("dataset_name") or "").lower()
    exp_type = (cfg.get("experiment_type") or "").lower()
    return dataset in {"mnist", "cifar10", "cifar100"} or exp_type in {"vision_synergy", "general_alignment"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures for the vision redundancy/synergy paper.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing vision experiment JSON result files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper_figures_vision",
        help="Directory where vision figures will be written.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all JSONs under results_dir
    result_files = list(results_root.glob("**/*.json"))
    if not result_files:
        logger.error(f"No JSON results found under {results_root}")
        return

    all_results: List[Dict[str, Any]] = []
    for path in result_files:
        data = load_results(path)
        if not data:
            continue
        cfg = data.get("config", {})
        if is_vision_experiment(cfg):
            all_results.append(data)

    if not all_results:
        logger.error("Found JSON files but none matched vision experiments (MNIST / CIFAR / vision_synergy).")
        return

    # Pick a detailed run with alignment history for structural figures
    detailed_run: Dict[str, Any] = {}
    for res in sorted(all_results, key=lambda x: x.get("config", {}).get("name", ""), reverse=True):
        if "alignment" in res.get("train_results", {}):
            detailed_run = res
            break

    viz = PruningVisualizer(style="seaborn-v0_8-paper", figsize=(10, 5))

    if detailed_run:
        alignment_history = detailed_run.get("train_results", {}).get("alignment", {})

        # Fig. 1: layer-wise distributions for RQ, redundancy, synergy
        metric_mapping = {
            "rayleigh_quotient": "Rayleigh Quotient",
            "pairwise_redundancy_gaussian": "Gaussian Redundancy",
            "gaussian_pid_synergy_mmi": "Gaussian PID Synergy",
        }
        viz.plot_metric_distributions_from_alignment(
            alignment_history,
            metric_mapping=metric_mapping,
            save_dir=output_dir,
        )

        # Fig. 2: redundancy–synergy scatter for selected layers
        redundancy_history = alignment_history.get("pairwise_redundancy_gaussian")
        synergy_history = alignment_history.get("gaussian_pid_synergy_mmi")
        if redundancy_history and synergy_history:
            red_snapshot = redundancy_history[0] if isinstance(redundancy_history, list) else redundancy_history
            syn_snapshot = synergy_history[0] if isinstance(synergy_history, list) else synergy_history
            viz.plot_redundancy_synergy_scatter(
                red_snapshot,
                syn_snapshot,
                outlier_snapshot=None,
                save_path=output_dir / "vision_redundancy_synergy_scatter.png",
            )

        # Fig. 3a-style: alignment vs synergy scatter
        rq_history = alignment_history.get("rayleigh_quotient")
        if rq_history and synergy_history:
            rq_snapshot = rq_history[0] if isinstance(rq_history, list) else rq_history
            syn_snapshot = synergy_history[0] if isinstance(synergy_history, list) else synergy_history
            viz.plot_alignment_vs_synergy_scatter(
                rq_snapshot,
                syn_snapshot,
                save_path=output_dir / "vision_alignment_vs_synergy.png",
            )
    else:
        logger.warning("No vision run with alignment history found; structural figures will be skipped.")

    # Build pruning DataFrames for accuracy–sparsity curves
    mnist_rows: List[Dict[str, Any]] = []
    cifar_rows: List[Dict[str, Any]] = []

    for res in all_results:
        cfg = res.get("config", {})
        dataset = (cfg.get("dataset_name") or "").lower()
        exp_name = cfg.get("name", "unknown")

        pruning = res.get("pruning_results", {}).get("strategies", {})
        if not pruning:
            continue

        for strategy, data in pruning.items():
            sparsities = data.get("sparsities", []) or data.get("pruning_amounts", [])
            accs = data.get("accuracies_after_finetune") or data.get("accuracies_before_finetune") or []
            if not sparsities or not accs:
                continue

            for s, acc in zip(sparsities, accs):
                row = {"Method": strategy, "Sparsity": s, "Accuracy": acc, "Experiment": exp_name}
                if dataset == "mnist":
                    mnist_rows.append(row)
                elif dataset in {"cifar10", "cifar100"}:
                    cifar_rows.append(row)

    # Fig. MNIST pruning curves
    if mnist_rows:
        df_mnist = pd.DataFrame(mnist_rows)
        viz.plot_sparsity_perplexity_curves(
            df_mnist,
            x_col="Sparsity",
            y_col="Accuracy",
            hue="Method",
            save_path=output_dir / "vision_mnist_pruning_curves.png",
            title="MNIST MLP: Accuracy vs Sparsity",
            y_label="Accuracy (%)",
        )
    else:
        logger.warning("No MNIST pruning results found for vision figures.")

    # Fig. CIFAR pruning curves
    if cifar_rows:
        df_cifar = pd.DataFrame(cifar_rows)
        viz.plot_sparsity_perplexity_curves(
            df_cifar,
            x_col="Sparsity",
            y_col="Accuracy",
            hue="Method",
            save_path=output_dir / "vision_cifar_pruning_curves.png",
            title="CIFAR: Accuracy vs Sparsity",
            y_label="Accuracy (%)",
        )

        # Fig. ablation-style summary at high sparsity
        viz.plot_ablation_summary(
            df_cifar,
            hue="Method",
            value_col="Accuracy",
            sparsity_col="Sparsity",
            min_sparsity=0.5,
            save_path=output_dir / "vision_pruning_ablation.png",
            title="Pruning Strategies at High Sparsity (CIFAR)",
        )
    else:
        logger.warning("No CIFAR pruning results found for vision figures.")

    logger.info(f"Vision figures written to {output_dir}")


if __name__ == "__main__":
    main()


