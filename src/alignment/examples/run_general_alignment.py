#!/usr/bin/env python3
"""
Example script for running a general alignment experiment.

This script demonstrates how to:
1. Configure a complete alignment analysis pipeline
2. Train a model on MNIST
3. Compute alignment metrics
4. Apply pruning based on the analysis
5. Generate comprehensive reports
"""

import argparse
from pathlib import Path
import logging

from alignment.experiments import GeneralAlignmentExperiment, GeneralAlignmentConfig
from alignment.analysis import ResultAggregator, HTMLReporter


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Run general alignment experiment with MNIST."""
    parser = argparse.ArgumentParser(
        description="Run general alignment analysis experiment"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "cifar100"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="Model architecture"
    )
    parser.add_argument(
        "--pruning-strategy",
        type=str,
        default="magnitude",
        choices=["magnitude", "gradient", "random", "fisher"],
        help="Pruning strategy to use"
    )
    parser.add_argument(
        "--pruning-amount",
        type=float,
        default=0.5,
        help="Amount to prune (0-1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/general_alignment",
        help="Output directory"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.config:
        # Load from config file
        logging.info(f"Loading configuration from {args.config}")
        experiment = GeneralAlignmentExperiment.from_yaml(args.config)
    else:
        # Create configuration programmatically
        logging.info("Creating configuration from command line arguments")
        
        # Model configuration based on dataset
        if args.dataset == "mnist":
            model_config = {
                "input_dim": 784,
                "hidden_dims": [256, 128],
                "output_dim": 10
            }
        elif args.dataset == "cifar10":
            model_config = {
                "input_channels": 3,
                "num_classes": 10
            }
        else:  # cifar100
            model_config = {
                "input_channels": 3,
                "num_classes": 100
            }
        
        config = GeneralAlignmentConfig(
            name=f"{args.dataset}_{args.model}_{args.pruning_strategy}",
            description=f"General alignment analysis on {args.dataset}",
            
            # Model
            model_name=args.model,
            model_config=model_config,
            
            # Dataset
            dataset_name=args.dataset,
            dataset_config={
                "data_path": "./data",
                "download": True,
                "normalize": True
            },
            
            # Training
            training_config={
                "epochs": args.epochs,
                "learning_rate": 0.001,
                "batch_size": 64,
                "optimizer": "adam",
                "scheduler": "cosine"
            },
            
            # Metrics
            alignment_metrics=[
                "rayleigh_quotient",
                "mutual_information_gaussian",
                "weight_cosine_similarity",
                "spectral_gap"
            ],
            
            # Pruning
            pruning_strategy=args.pruning_strategy,
            pruning_config={
                "amount": args.pruning_amount,
                "structured": False
            },
            
            # Experiment flow
            train_model=True,
            compute_initial_metrics=True,
            apply_pruning=True,
            fine_tune_after_pruning=True,
            fine_tune_epochs=5,
            
            # System
            device=args.device,
            log_dir=str(output_dir),
            checkpoint_dir=str(output_dir / "checkpoints")
        )
        
        experiment = GeneralAlignmentExperiment(config)
    
    # Run experiment
    logging.info("Starting experiment...")
    results = experiment.run()
    
    # Generate report
    logging.info("Generating analysis report...")
    
    # Create aggregator with results
    aggregator = ResultAggregator()
    aggregator.add_results(experiment.config.name, results)
    
    # Create HTML report
    reporter = HTMLReporter(f"{experiment.config.name} - Analysis Report")
    
    # Add configuration section
    config_html = f"""
    <h3>Configuration</h3>
    <ul>
        <li><b>Dataset:</b> {experiment.config.dataset_name}</li>
        <li><b>Model:</b> {experiment.config.model_name}</li>
        <li><b>Pruning Strategy:</b> {experiment.config.pruning_strategy}</li>
        <li><b>Pruning Amount:</b> {experiment.config.pruning_config['amount']}</li>
        <li><b>Training Epochs:</b> {experiment.config.training_config['epochs']}</li>
    </ul>
    """
    reporter.add_section("Configuration", config_html)
    
    # Add metrics comparison
    if "initial_metrics" in results and "final_metrics" in results:
        comparison_html = "<h3>Metric Changes</h3><table class='metric-table'>"
        comparison_html += "<tr><th>Metric</th><th>Layer</th><th>Initial</th><th>Final</th><th>Change (%)</th></tr>"
        
        for metric_name in experiment.config.alignment_metrics:
            if metric_name in results["initial_metrics"] and metric_name in results["final_metrics"]:
                for layer_name in results["initial_metrics"][metric_name]:
                    if layer_name in results["final_metrics"][metric_name]:
                        initial = results["initial_metrics"][metric_name][layer_name]
                        final = results["final_metrics"][metric_name][layer_name]
                        change = ((final - initial) / (initial + 1e-8)) * 100
                        
                        comparison_html += f"""
                        <tr>
                            <td>{metric_name}</td>
                            <td>{layer_name}</td>
                            <td>{initial:.4f}</td>
                            <td>{final:.4f}</td>
                            <td>{change:+.2f}%</td>
                        </tr>
                        """
        
        comparison_html += "</table>"
        reporter.add_section("Metric Comparison", comparison_html)
    
    # Add sparsity information
    if "pruning_results" in results and "sparsity" in results["pruning_results"]:
        sparsity_html = "<h3>Achieved Sparsity</h3><ul>"
        for layer, sparsity in results["pruning_results"]["sparsity"].items():
            sparsity_html += f"<li><b>{layer}:</b> {sparsity:.2%}</li>"
        sparsity_html += "</ul>"
        reporter.add_section("Sparsity", sparsity_html)
    
    # Generate report
    report_path = output_dir / "analysis_report.html"
    reporter.generate(report_path)
    logging.info(f"Report saved to {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Dataset: {experiment.config.dataset_name}")
    print(f"Model: {experiment.config.model_name}")
    print(f"Pruning: {experiment.config.pruning_strategy} ({experiment.config.pruning_config['amount']:.0%})")
    
    if "pruning_results" in results and "sparsity" in results["pruning_results"]:
        print(f"Overall Sparsity: {results['pruning_results']['sparsity']['overall']:.2%}")
    
    if "analysis" in results and "sparsity_impact" in results["analysis"]:
        retention = results["analysis"]["sparsity_impact"].get("performance_retention", 0)
        print(f"Performance Retention: {retention:.2f}%")
    
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"Report available at: {report_path}")


if __name__ == "__main__":
    main() 