"""
Enhanced Pruning Visualization Demo

This script demonstrates the enhanced visualization capabilities of the PruningVisualizer,
including all the new plotting methods added to the framework.

Usage:
    python visualization_demo.py

This demo showcases:
    - Enhanced accuracy vs sparsity plots with relative performance
    - Weight distribution comparison before/after pruning
    - Multi-metric radar charts for strategy comparison
    - Pruning efficiency curves
    - Comprehensive dashboards
    - All existing visualization methods

Output:
    Results are saved to: results/enhanced_visualizations/
"""

from pathlib import Path

import numpy as np
import torch

from alignment.analysis.visualization import PruningVisualizer


def create_demo_results():
    """Create comprehensive demo results for visualization."""

    # Progressive dropout results
    dropout_results = {
        "dropout_rates": [0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
        "strategies": {
            "low": [95.2, 93.8, 90.5, 85.3, 75.2, 65.8],
            "high": [95.2, 88.3, 78.5, 65.2, 45.3, 25.8],
            "random": [95.2, 91.5, 85.2, 75.8, 60.2, 45.5],
        },
        "layer_importance": {
            "conv1": {"mean": 0.85, "std": 0.05},
            "conv2": {"mean": 0.75, "std": 0.08},
            "fc1": {"mean": 0.65, "std": 0.10},
            "fc2": {"mean": 0.45, "std": 0.12},
        },
        "efficiency": {"low": 0.9, "high": 0.6, "random": 0.7},
        "fine_tuning": {
            "low": [66, 72, 78, 82, 85, 87, 88, 89, 90, 90],
            "high": [26, 35, 42, 48, 52, 55, 57, 58, 59, 60],
            "random": [46, 55, 62, 67, 71, 74, 76, 77, 78, 79],
        },
    }

    # Standard pruning results (sparsity-based)
    pruning_results = {}
    for strategy in ["magnitude", "gradient", "fisher", "random"]:
        pruning_results[strategy] = {}
        for sparsity in [0.1, 0.3, 0.5, 0.7, 0.9]:
            base_acc = 95.0
            decay_factors = {"magnitude": 0.15, "gradient": 0.20, "fisher": 0.18, "random": 0.35}
            accuracy = base_acc * (1 - sparsity * decay_factors[strategy])
            accuracy += np.random.normal(0, 1.0)

            pruning_results[strategy][sparsity] = {
                "accuracy": max(10, min(100, accuracy)),
                "loss": 0.1 + sparsity * 0.8 * (1 + decay_factors[strategy]),
            }

    # Multi-seed results
    multi_seed_results = {}
    for strategy in ["magnitude", "gradient", "random"]:
        multi_seed_results[strategy] = []
        for seed in range(5):
            np.random.seed(seed)
            seed_data = {}
            for sparsity in [0.1, 0.3, 0.5, 0.7, 0.9]:
                base = pruning_results[strategy][sparsity]
                seed_data[sparsity] = {"accuracy": base["accuracy"] + np.random.normal(0, 2), "loss": base["loss"] + np.random.normal(0, 0.1)}
            multi_seed_results[strategy].append(seed_data)

    # Layer-wise sparsity patterns
    layer_sparsities = {
        "magnitude": {"conv1": 0.3, "conv2": 0.5, "fc1": 0.7, "fc2": 0.9},
        "gradient": {"conv1": 0.4, "conv2": 0.5, "fc1": 0.6, "fc2": 0.8},
        "random": {"conv1": 0.5, "conv2": 0.5, "fc1": 0.5, "fc2": 0.5},
    }

    model_accuracy = {"magnitude": 85.3, "gradient": 82.1, "random": 68.5}

    # Weight distributions (simulated)
    np.random.seed(42)
    weights_before = {"conv1": torch.randn(64, 3, 3, 3), "fc1": torch.randn(128, 512)}

    weights_after = {}
    for layer_name, weights in weights_before.items():
        # Simulate pruning effects
        mask_magnitude = torch.abs(weights) > torch.quantile(torch.abs(weights), 0.5)
        weights_after[f"{layer_name}_magnitude"] = weights * mask_magnitude

        mask_random = torch.rand_like(weights) > 0.5
        weights_after[f"{layer_name}_random"] = weights * mask_random

    # Multi-metric comparison data
    strategy_metrics = {
        "magnitude_low": {"Accuracy": 0.95, "Efficiency": 0.90, "Stability": 0.85, "Speed": 0.80, "Memory": 0.85},
        "magnitude_high": {"Accuracy": 0.70, "Efficiency": 0.60, "Stability": 0.75, "Speed": 0.85, "Memory": 0.80},
        "gradient_low": {"Accuracy": 0.85, "Efficiency": 0.80, "Stability": 0.80, "Speed": 0.70, "Memory": 0.75},
        "random": {"Accuracy": 0.60, "Efficiency": 0.95, "Stability": 0.50, "Speed": 0.95, "Memory": 0.90},
    }

    # Efficiency curve data
    efficiency_data = {}
    for strategy in ["magnitude_low", "magnitude_high", "gradient_low", "random"]:
        compression_ratios = np.linspace(1, 10, 10)
        efficiency_factors = {"magnitude_low": 0.95, "magnitude_high": 0.70, "gradient_low": 0.85, "random": 0.60}
        factor = efficiency_factors[strategy]
        accuracies = 100 * np.exp(-0.3 * (compression_ratios - 1) / factor)
        accuracies = np.clip(accuracies, 10, 100)

        efficiency_data[strategy] = list(zip(compression_ratios, accuracies))

    return {
        "dropout_results": dropout_results,
        "pruning_results": pruning_results,
        "multi_seed_results": multi_seed_results,
        "layer_sparsities": layer_sparsities,
        "model_accuracy": model_accuracy,
        "weights_before": weights_before,
        "weights_after": weights_after,
        "strategy_metrics": strategy_metrics,
        "efficiency_data": efficiency_data,
    }


def main():
    """Run the enhanced visualization demonstration."""
    print("=" * 60)
    print("Enhanced Pruning Visualization Demo")
    print("=" * 60)

    # Create output directory
    output_dir = Path("results/enhanced_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create demo data
    print("\nGenerating demo data...")
    demo_data = create_demo_results()

    # Initialize visualizer
    visualizer = PruningVisualizer()

    # 1. Enhanced accuracy vs sparsity plot
    print("\n1. Creating enhanced accuracy vs sparsity plot...")
    visualizer.plot_accuracy_vs_sparsity_enhanced(demo_data["dropout_results"], save_path=output_dir / "enhanced_accuracy_vs_sparsity.png")

    # 2. Standard pruning performance
    print("2. Creating standard pruning performance plot...")
    visualizer.plot_pruning_performance(
        demo_data["pruning_results"],
        metrics=["accuracy", "loss"],
        save_path=output_dir / "pruning_performance.png",
        title="Pruning Strategy Performance Comparison",
    )

    # 3. Weight distribution comparison
    print("3. Creating weight distribution comparison...")
    visualizer.plot_weight_distribution_comparison(
        demo_data["weights_before"], demo_data["weights_after"], strategies=["magnitude", "random"], save_path=output_dir / "weight_distributions.png"
    )

    # 4. Multi-metric radar chart
    print("4. Creating multi-metric radar chart...")
    visualizer.plot_multi_metric_radar(demo_data["strategy_metrics"], save_path=output_dir / "multi_metric_radar.png")

    # 5. Pruning efficiency curves
    print("5. Creating pruning efficiency curves...")
    visualizer.plot_pruning_efficiency_curve(demo_data["efficiency_data"], save_path=output_dir / "efficiency_curves.png")

    # 6. Comprehensive dashboard
    print("6. Creating comprehensive dashboard...")
    visualizer.plot_comprehensive_dashboard(demo_data["dropout_results"], save_path=output_dir / "comprehensive_dashboard.png")

    # 7. Comparison grid
    print("7. Creating comparison grid...")
    visualizer.plot_pruning_comparison_grid(demo_data["pruning_results"], save_path=output_dir / "comparison_grid.png")

    # 8. Multi-seed analysis
    print("8. Creating multi-seed analysis...")
    visualizer.plot_multi_seed_results(demo_data["multi_seed_results"], metric="accuracy", save_path=output_dir / "multi_seed_analysis.png")

    # 9. Layer-wise pruning visualization
    print("9. Creating layer-wise pruning visualization...")
    visualizer.plot_layer_wise_pruning(demo_data["layer_sparsities"], demo_data["model_accuracy"], save_path=output_dir / "layer_wise_pruning.png")

    # Summary
    print("\n" + "=" * 60)
    print("Enhanced Visualization Demo Complete!")
    print("=" * 60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  1. enhanced_accuracy_vs_sparsity.png - Accuracy with relative performance")
    print("  2. pruning_performance.png - Standard performance metrics")
    print("  3. weight_distributions.png - Weight distribution comparison")
    print("  4. multi_metric_radar.png - Multi-metric strategy comparison")
    print("  5. efficiency_curves.png - Accuracy vs compression ratio")
    print("  6. comprehensive_dashboard.png - Complete experiment overview")
    print("  7. comparison_grid.png - 6-panel strategy comparison")
    print("  8. multi_seed_analysis.png - Statistical analysis across seeds")
    print("  9. layer_wise_pruning.png - Layer-specific sparsity patterns")

    print("\nVisualization Features:")
    print("  - Consistent color scheme across all plots")
    print("  - High-resolution output (300 DPI)")
    print("  - Professional styling with seaborn theme")
    print("  - Comprehensive analysis from multiple perspectives")
    print("  - Support for different result formats (dropout/sparsity)")


if __name__ == "__main__":
    main()
