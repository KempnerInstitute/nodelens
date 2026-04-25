#!/usr/bin/env python3
"""
Unified Analysis Script

This script provides a single entry point for all analysis and visualization tasks.
It replaces the scattered scripts:
- generate_paper_figures.py
- generate_llm_layer_figures.py
- generate_vision_figures.py

Usage:
    # Run all analyses from a config file
    python scripts/run_analysis.py --config configs/analysis_template.yaml

    # Run specific analyses with overrides
    python scripts/run_analysis.py --config configs/analysis_template.yaml \
        --analyses histograms scatter_plots \
        --output-dir ./custom_output

    # Run on a specific results directory
    python scripts/run_analysis.py --results-dir ./results/llm_experiment \
        --output-dir ./llm_plots

    # Quick analysis without config file
    python scripts/run_analysis.py --results-dir ./results --quick

For more control, use the AnalysisRunner class directly:
    from nodelens.analysis import AnalysisRunner, AnalysisConfig

    config = AnalysisConfig(
        results_dir="./results",
        output_dir="./plots",
        analyses=["histograms", "pruning_curves"],
    )
    runner = AnalysisRunner(config)
    outputs = runner.run()
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    from nodelens.analysis import AnalysisConfig, AnalysisRunner
except ImportError:
    # Add src to path for development (repo-local runs without installing the package)
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from nodelens.analysis import AnalysisConfig, AnalysisRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Unified analysis and visualization runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to analysis config YAML file",
    )

    # Input sources (can override config)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing experiment result JSONs",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Single result JSON file",
    )
    parser.add_argument(
        "--scores-file",
        type=str,
        default=None,
        help="Pre-computed importance scores (.pt file)",
    )

    # Output (can override config)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg"],
        default=None,
        help="Output image format",
    )

    # Analysis selection (can override config)
    parser.add_argument(
        "--analyses",
        type=str,
        nargs="+",
        default=None,
        help="Which analyses to run (e.g., histograms scatter_plots pruning_curves)",
    )

    # Quick mode (no config file needed)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run all analyses with default settings",
    )

    # Style options
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Matplotlib style (e.g., seaborn-v0_8-paper)",
    )

    args = parser.parse_args()

    # Build config
    if args.config:
        config = AnalysisConfig.from_yaml(args.config)
    elif args.quick or args.results_dir or args.results_file:
        config = AnalysisConfig()
    else:
        parser.error("Either --config or --results-dir/--results-file is required")

    # Apply overrides
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.results_file:
        config.results_file = args.results_file
    if args.scores_file:
        config.importance_scores_file = args.scores_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.format:
        config.format = args.format
    if args.analyses:
        config.analyses = args.analyses
    if args.style:
        config.style = args.style

    # Validate we have input
    if not config.results_dir and not config.results_file and not config.importance_scores_file:
        parser.error("No input source specified. Use --results-dir, --results-file, or --scores-file")

    # Run analysis
    logger.info("Starting analysis...")
    runner = AnalysisRunner(config)
    outputs = runner.run()

    # Print summary
    total_files = sum(len(v) for v in outputs.values())
    print("\nAnalysis complete!")
    print(f"Generated {total_files} files in {config.output_dir}")

    for analysis_type, files in outputs.items():
        if files:
            print(f"  {analysis_type}: {len(files)} files")


if __name__ == "__main__":
    main()
