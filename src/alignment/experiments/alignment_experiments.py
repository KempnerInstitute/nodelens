"""
Alignment experiment implementations.

This module contains experiment classes for neural network alignment studies,
focusing on alignment metrics, dropout impacts, and training analysis.
"""

import logging
import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Any, Union
import copy
import pickle

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from alignment.config import ExperimentConfig
from alignment.experiments.experiment import Experiment
from alignment.metrics import AlignmentMetric, get_metric
from alignment.models.registry import create_model
from alignment.dropout import progressive_dropout, eigenvector_dropout
from alignment.training import train_model, evaluate_model
from alignment.utils.core import setup_logging
from alignment.utils.plotting import plot_dropout_results, plot_experiment_summary
from alignment.datasets import get_dataset, load_dataset

logger = logging.getLogger(__name__)


class AlignmentExperiment(Experiment):
    """
    Experiment class for studying neural network alignment properties.
    
    This class implements experiments that assess alignment between layers
    in neural networks, with support for different dropout strategies,
    multiple metrics, and visualization.
    """
    
    def __init__(self, config: Dict) -> None:
        """Initialize the experiment with the given config.

        Args:
            config: Experiment configuration.
        """
        super().__init__(config)
        
        # Make sure dataset_config has required fields to prevent loading errors
        if isinstance(self.config.dataset, dict):
            dataset_config = self.config.dataset
            
            # Ensure dataset_config has both "name" and "dataset_name" fields for compatibility
            if "name" in dataset_config and "dataset_name" not in dataset_config:
                dataset_config["dataset_name"] = dataset_config["name"]
            elif "dataset_name" in dataset_config and "name" not in dataset_config:
                dataset_config["name"] = dataset_config["dataset_name"]
            elif "name" not in dataset_config and "dataset_name" not in dataset_config:
                if isinstance(self.config.dataset, str):
                    # If config.dataset is a string, use it as the dataset name
                    dataset_config["name"] = self.config.dataset
                    dataset_config["dataset_name"] = self.config.dataset
                    logger.info(f"Using dataset name from string: {self.config.dataset}")
                else:
                    # Default to CIFAR10 if no dataset name provided
                    logger.warning("No dataset name provided in configuration. Using default: cifar10")
                    dataset_config["name"] = "cifar10"
                    dataset_config["dataset_name"] = "cifar10"
            
            logger.info(f"Using dataset: {dataset_config.get('name', 'unknown')}")
        
        # Set up experiment-specific paths
        self.figure_path = None
        self.weights_path = None
        self.setup_paths()
        
        logger.debug(f"Initialized alignment experiment")
        
        # Add a sanity check
        # Ensure checkpoint parameters exist in the config
        if not hasattr(self.config, "checkpoint"):
            self.config.checkpoint = {}
            logger.warning("Checkpoint configuration not found, using defaults")
        
        self.metric = get_metric(config.alignment.metric)
        
    def get_basename(self) -> str:
        """
        Get the base name for the experiment.
        
        Returns:
            Base name string
        """
        return f"alignment_{self.config.model.model_name}_{self.config.dataset.dataset_name}"
    
    def prepare_path(self) -> List[str]:
        """
        Prepare the experiment path components.
        
        Returns:
            List of path components
        """
        return [
            "alignment",
            self.config.model.model_name,
            self.config.dataset.dataset_name,
            f"metric_{self.config.alignment.metric}"
        ]
    
    def create_networks(self) -> List[nn.Module]:
        """
        Create multiple neural networks for the experiment, each with different initialization.
        
        Following the alignment_v2 approach, this creates multiple independent networks
        rather than copies of a single network.
        
        Returns:
            List containing multiple independently initialized networks
        """
        # Get the number of network replicates from config
        num_replicates = self.config.training.replicates if hasattr(self.config.training, "replicates") else 5
        
        # Create multiple models based on configuration, each with different initialization
        networks = []
        for i in range(num_replicates):
            # Set a different seed for each network to ensure different initializations
            if hasattr(self.config, 'seed') and self.config.seed is not None:
                # Use the base seed plus the replicate index to get different but reproducible initializations
                torch.manual_seed(self.config.seed + i)
                torch.cuda.manual_seed_all(self.config.seed + i)
                np.random.seed(self.config.seed + i)
            
            # Create a new model
            model = create_model(self.config.model)
            model.to(self.device)
            networks.append(model)
            
        logger.info(f"Created {len(networks)} models with independent initializations: {self.config.model.model_name}")
        
        return networks
    
    def main(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Main experiment execution method.
        
        Returns:
            Tuple of (results dictionary, list of networks)
        """
        # Additional initialization
        self.setup_paths()
        logger.info(f"Set up paths. Results will be saved to {self.results_path}")

        # Create network models for experiment
        networks = self.create_networks()
        
        # Run the experiment based on the specified type
        experiment_type = getattr(self.config, 'experiment_type', 'alignment_analysis')
        
        if experiment_type == "alignment_analysis" or experiment_type == "alignment":
            # Run alignment analysis and store results
            results = self.run_alignment_analysis(networks)
            
        elif experiment_type == "progressive_dropout":
            # Run progressive dropout experiment for all networks
            results = self._run_progressive_dropout(networks)
            
            # Generate and log visualization
            self._plot_dropout_results(
                results, 
                self.figure_path, 
                title_prefix=f"{getattr(self.config, 'experiment_name', 'Progressive Dropout')}"
            )
            
            # Save results for later reference
            with open(os.path.join(self.results_path, "progressive_dropout_results.pkl"), "wb") as f:
                pickle.dump(results, f)
            
        elif experiment_type == "eigenvector_dropout":
            # Run eigenvector dropout experiment for one network
            results = self._run_eigenvector_dropout(networks[0])
            
            # Generate and log visualization (this has a different format)
            self._generate_dropout_plots(
                results,
                "Eigenvector Dropout",
                "eigenvector_dropout"
            )
            
            # Save results for later reference
            with open(os.path.join(self.results_path, "eigenvector_dropout_results.pkl"), "wb") as f:
                pickle.dump(results, f)
                
        elif experiment_type == "training_alignment":
            # Run training alignment experiment
            # This is handled separately because it needs different structure
            results = self._run_training_alignment()
            
            # Generate and log visualization
            self._generate_training_alignment_plots(results)
            
            # Save results for later reference
            with open(os.path.join(self.results_path, "training_alignment_results.pkl"), "wb") as f:
                pickle.dump(results, f)
        
        else:
            raise ValueError(f"Unsupported experiment type: {experiment_type}")
            
        logger.info(f"Completed {getattr(self.config, 'experiment_name', experiment_type)} experiment")
        
        return results, networks
    
    def _run_progressive_dropout(self, networks: List[nn.Module]) -> Dict:
        """
        Run progressive dropout experiment with multiple networks.
        
        Uses the progressive_dropout function from alignment.dropout module 
        instead of reimplementing the logic directly.

        Args:
            networks: List of networks to run the experiment on.

        Returns:
            Dictionary with results.
        """
        # Get dropout fractions from config
        dropin_min = self.config.alignment.dropout_min
        dropin_max = self.config.alignment.dropout_max
        num_dropout_fractions = self.config.alignment.dropout_steps
        dropout_fractions = np.linspace(dropin_min, dropin_max, num_dropout_fractions).tolist()
        
        # Get pruning mode and dropout mode from config
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
        dropout_mode = getattr(self.config.extra, "dropout_mode", "scaled")
        
        logger.info(f"Running progressive dropout with pruning_mode={pruning_mode}, dropout_mode={dropout_mode}")
        
        # Prepare dataset
        try:
            batch_size = getattr(self.config.training, "batch_size", 128)
            dataset = load_dataset(self.config.dataset, batch_size=batch_size)
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return {"error": f"Dataset configuration error: {str(e)}", "dropout_fractions": dropout_fractions}
        
        # Initialize results structure
        aggregate_results = {
            "dropout_fractions": dropout_fractions,
            "accuracies": {"high_rq": [], "low_rq": [], "random": []},
            "losses": {"high_rq": [], "low_rq": [], "random": []},
            "alignment_values": {"high_rq": [], "low_rq": [], "random": []}
        }
        
        # Process each network
        all_network_results = []
        for i, network in enumerate(tqdm(networks, desc="Running progressive dropout")):
            try:
                # Use the progressive_dropout function directly
                network_result = progressive_dropout(
                    [network],  # Function expects a list of networks
                    dataset,
                    dropout_fractions,
                    self.metric,
                    device=self.device,
                    pruning_mode=pruning_mode,
                    dropout_mode=dropout_mode,
                    use_tensorized=getattr(self.config.extra, "use_tensorized", True)
                )
                
                # Store results for this network
                all_network_results.append(network_result)
                
            except Exception as e:
                logger.error(f"Error processing network {i}: {str(e)}")
        
        # If no results were successfully computed, return error
        if not all_network_results:
            return {"error": "No networks were successfully processed", "dropout_fractions": dropout_fractions}
        
        # Aggregate results across networks
        for strategy in ["high_rq", "low_rq", "random"]:
            # Collect accuracies from all networks for this strategy
            strategy_accuracies = []
            strategy_losses = []
            
            for net_result in all_network_results:
                if hasattr(net_result, "network_accuracies") and strategy in net_result.network_accuracies:
                    accuracies = net_result.network_accuracies[strategy]
                    strategy_accuracies.append(accuracies)
                    
                    # Calculate losses as (100 - accuracy)
                    losses = [100.0 - acc for acc in accuracies]
                    strategy_losses.append(losses)
            
            # Calculate mean accuracies and losses across networks
            if strategy_accuracies:
                mean_accuracies = np.mean(strategy_accuracies, axis=0).tolist()
                mean_losses = np.mean(strategy_losses, axis=0).tolist()
                
                aggregate_results["accuracies"][strategy] = mean_accuracies
                aggregate_results["losses"][strategy] = mean_losses
            
            # Store alignment values from the first network (if available)
            if all_network_results and hasattr(all_network_results[0], "alignment_values"):
                for net_result in all_network_results:
                    if hasattr(net_result, "alignment_values") and strategy in net_result.alignment_values:
                        aggregate_results["alignment_values"][strategy] = net_result.alignment_values[strategy]
                        break
        
        return aggregate_results
    
    def _run_eigenvector_dropout(self, net: nn.Module) -> Dict:
        """
        Run eigenvector dropout experiments.
        
        Uses the eigenvector_dropout function from alignment.dropout module 
        instead of reimplementing the logic directly.
        
        Args:
            net: Neural network to analyze
            
        Returns:
            Dictionary of eigenvector dropout results
        """
        # Get dropout fractions from config
        dropout_fractions = np.linspace(
            self.config.alignment.dropout_min,
            self.config.alignment.dropout_max,
            self.config.alignment.dropout_steps
        ).tolist()
        
        # Initialize result structure
        results = {
            "dropout_fractions": dropout_fractions,
            "accuracies": {"eigenvector": []},
            "losses": {"eigenvector": []},
            "alignment_values": {"eigenvector": []}
        }
        
        # Prepare dataset config
        dataset_config = self.config.dataset
        
        # Process each dropout fraction
        for dropout_fraction in tqdm(dropout_fractions, desc="Eigenvector Dropout"):
            try:
                # Call the eigenvector_dropout function directly
                accuracy, alignment_values = eigenvector_dropout(
                    net,
                    dataset_config,
                    dropout_fraction=dropout_fraction,
                    metric=self.metric,
                    device=self.device,
                    dropout_mode=getattr(self.config.extra, "dropout_mode", "scaled"),
                    dropout_pruning_mode=getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
                )
                
                # Store results
                results["accuracies"]["eigenvector"].append(accuracy)
                results["losses"]["eigenvector"].append(100.0 - accuracy)
                results["alignment_values"]["eigenvector"].append(alignment_values)
                
            except Exception as e:
                logger.error(f"Error in eigenvector dropout at fraction {dropout_fraction}: {str(e)}")
                # Add placeholder values to maintain result structure
                results["accuracies"]["eigenvector"].append(0.0)
                results["losses"]["eigenvector"].append(100.0)
                results["alignment_values"]["eigenvector"].append(None)
        
        return results
    
    def plot(self, results: Dict) -> Dict[str, List[str]]:
        """
        Plot experiment results using enhanced visualization methods.
        
        Args:
            results: Dictionary containing experiment results
            
        Returns:
            Dictionary of plot paths by type
        """
        logger.info("Generating experiment visualizations")
        
        # Check for results to plot
        if not results:
            logger.warning("No results to plot")
            return {}
            
        plot_paths = {}
        
        # Generate experiment summary
        try:
            logger.info("Creating experiment summary visualization")
            summary_path = self._generate_summary_plots(results)
            if summary_path:
                plot_paths["summary"] = summary_path
            logger.info("Successfully created experiment summary visualization")
        except Exception as e:
            logger.error(f"Error creating summary visualization: {str(e)}", exc_info=True)
        
        # Generate progressive dropout plots
        if "progressive_dropout" in results:
            try:
                logger.info("Generating progressive dropout visualizations")
                prog_paths = self._plot_dropout_results(
                    results["progressive_dropout"], 
                    self.figure_path,
                    "Progressive Dropout"
                )
                if prog_paths:
                    plot_paths["progressive_dropout"] = prog_paths
                logger.info("Successfully generated progressive dropout visualizations")
            except Exception as e:
                logger.error(f"Error generating progressive dropout visualizations: {str(e)}", exc_info=True)
        
        # Generate eigenvector dropout plots
        if "eigenvector_dropout" in results:
            try:
                logger.info("Generating eigenvector dropout visualizations")
                eig_paths = self._plot_dropout_results(
                    results["eigenvector_dropout"], 
                    self.figure_path,
                    "Eigenvector Dropout"
                )
                if eig_paths:
                    plot_paths["eigenvector_dropout"] = eig_paths
                logger.info("Successfully generated eigenvector dropout visualizations")
            except Exception as e:
                logger.error(f"Error generating eigenvector dropout visualizations: {str(e)}", exc_info=True)
        
        # Generate combined visualizations if both types of results exist
        if "progressive_dropout" in results and "eigenvector_dropout" in results:
            try:
                logger.info("Generating combined dropout comparison visualization")
                comparison_path = self._generate_dropout_comparison(
                    results["progressive_dropout"],
                    results["eigenvector_dropout"]
                )
                if comparison_path:
                    plot_paths["comparison"] = comparison_path
                logger.info("Successfully generated combined dropout comparison")
            except Exception as e:
                logger.error(f"Error generating dropout comparison: {str(e)}", exc_info=True)
        
        logger.info(f"Completed experiment visualization generation: {len(plot_paths)} plot types")
        return plot_paths
    
    def _plot_dropout_results(self, results, figure_path=None, title_prefix="Dropout"):
        """
        Plot dropout experiment results using the plotting module.
        
        Args:
            results: Results dictionary from progressive_dropout
            figure_path: Directory to save plots to
            title_prefix: Prefix for plot titles
            
        Returns:
            List of saved plot files
        """
        # Get pruning mode and dropout mode
        pruning_mode = getattr(self.config.extra, "dropout_pruning_mode", "global_joint")
        dropout_mode = getattr(self.config.extra, "dropout_mode", "scaled")
        
        # Ensure figure path exists
        if figure_path is None:
            figure_path = self.figure_path
            
        # Generate plots
        try:
            from alignment.utils.plotting import plot_dropout_results
            
            saved_figures = plot_dropout_results(
                results,
                figure_path,
                pruning_mode=pruning_mode,
                dropout_mode=dropout_mode,
                title_prefix=title_prefix
            )
            
            # Log plots to wandb if configured
            if saved_figures and hasattr(self.config.checkpointing, "use_wandb") and self.config.checkpointing.use_wandb:
                try:
                    from alignment.utils.plotting import log_plots_to_wandb
                    log_plots_to_wandb(saved_figures)
                    logger.info(f"Logged {len(saved_figures)} dropout plots to wandb")
                except Exception as e:
                    logger.warning(f"Failed to log dropout plots to wandb: {str(e)}")
                    
            return saved_figures
            
        except Exception as e:
            logger.error(f"Error generating dropout plots: {str(e)}")
            return []
        
    def _generate_dropout_plots(self, results, title, filename=None):
        """
        Generate enhanced dropout plots using the plotting module.
        
        Args:
            results: Dropout experiment results
            title: Plot title
            filename: Base filename (not used, kept for backward compatibility)
            
        Returns:
            List of saved plot files
        """
        # Just use the standard plot_dropout_results function
        return self._plot_dropout_results(results, self.figure_path, title)
        
    def _generate_summary_plots(self, results):
        """
        Generate comprehensive summary plots for the experiment using the plotting module.
        
        Args:
            results: Experiment results dictionary
            
        Returns:
            Path to the saved summary plot
        """
        try:
            from alignment.utils.plotting import plot_experiment_summary
            
            # Generate the summary plot
            experiment_name = getattr(self.config, "experiment_name", self.get_basename())
            filepath = plot_experiment_summary(
                results, 
                self.figure_path,
                experiment_name=experiment_name
            )
            
            # Log to wandb if configured
            if filepath and hasattr(self.config.checkpointing, "use_wandb") and self.config.checkpointing.use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"experiment_summary": wandb.Image(filepath)})
                        logger.info("Logged experiment summary to wandb")
                except Exception as e:
                    logger.warning(f"Failed to log experiment summary to wandb: {str(e)}")
                    
            return filepath
                    
        except Exception as e:
            logger.error(f"Error creating summary visualization: {str(e)}")
            return None
            
    def _generate_dropout_comparison(self, progressive_results, eigenvector_results):
        """
        Generate comparison plots between progressive and eigenvector dropout using the plotting module.
        
        Args:
            progressive_results: Progressive dropout results
            eigenvector_results: Eigenvector dropout results
            
        Returns:
            Path to the saved comparison plot
        """
        try:
            from alignment.utils.plotting import plot_dropout_comparison
            
            # Generate the comparison plot
            experiment_name = getattr(self.config, "experiment_name", "Experiment")
            filepath = plot_dropout_comparison(
                progressive_results,
                eigenvector_results,
                self.figure_path,
                title=f"Dropout Comparison: {experiment_name}"
            )
            
            # Log to wandb if configured
            if filepath and hasattr(self.config.checkpointing, "use_wandb") and self.config.checkpointing.use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({"dropout_comparison": wandb.Image(filepath)})
                        logger.info("Logged dropout comparison to wandb")
                except Exception as e:
                    logger.warning(f"Failed to log dropout comparison to wandb: {str(e)}")
                    
            return filepath
                    
        except Exception as e:
            logger.error(f"Error creating dropout comparison: {str(e)}")
            return None

    def _initialize_wandb(self) -> Optional[Any]:
        """
        Initialize wandb for experiment tracking.
        
        Returns:
            wandb module if initialized successfully, None otherwise
        """
        if not getattr(self.config.checkpointing, "use_wandb", False):
            return None
            
        try:
            import wandb
            
            # Get wandb configuration
            entity = getattr(self.config, "wandb_entity", None)
            project = getattr(self.config, "wandb_project", "neural_alignment")
            
            # Create a unique run name
            if hasattr(self.config, "experiment_name") and self.config.experiment_name:
                run_name = f"{self.config.experiment_name}"
            else:
                run_name = f"{self.config.model.model_name}_{self.config.dataset.dataset_name}_{self.timestamp}"
                
            # Convert the config to a dictionary for wandb
            config_dict = self.config.to_dict() if hasattr(self.config, "to_dict") else vars(self.config)
            
            # Initialize wandb run
            wandb.init(
                project=project,
                entity=entity,
                name=run_name,
                config=config_dict,
                reinit=True  # Allow reinitializing in the same process
            )
            
            # Set up wandb to watch models if available
            if hasattr(self, "networks") and self.networks:
                # Watch the first network
                wandb.watch(self.networks[0], log="all", log_freq=10)
                
            logger.info(f"Initialized Weights & Biases tracking for run '{run_name}'")
            return wandb
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {str(e)}")
            return None
    
    def _log_to_wandb(self, data: Dict) -> bool:
        """
        Log data to wandb.
        
        Args:
            data: Data to log to wandb (dict with keys corresponding to metric names)
            
        Returns:
            True if logging was successful, False otherwise
        """
        if not getattr(self.config.checkpointing, "use_wandb", False):
            return False
            
        try:
            import wandb
            if wandb.run is None:
                wandb_module = self._initialize_wandb()
                if wandb_module is None:
                    return False
                    
            # Log scalar metrics
            metrics_to_log = {}
            for k, v in data.items():
                # Skip non-scalar values
                if isinstance(v, (int, float)):
                    metrics_to_log[k] = v
                elif isinstance(v, (list, tuple)) and len(v) == 1:
                    # Single-element list or tuple with a scalar
                    if isinstance(v[0], (int, float)):
                        metrics_to_log[k] = v[0]
                        
            # Log the metrics to wandb
            if metrics_to_log:
                wandb.log(metrics_to_log)
                
            # Check if data contains progressive dropout results that we should plot
            if "progressive_dropout" in data and "plot_files" in data and getattr(self.config.extra, "log_images", True):
                # Find all plot files and log them to wandb
                for plot_file in data["plot_files"]:
                    if os.path.exists(plot_file):
                        plot_name = os.path.basename(plot_file).replace(".png", "")
                        wandb.log({plot_name: wandb.Image(plot_file)})
                
                # If 'accuracies' exists in the progressive_dropout results, log them as a line plot
                if "accuracies" in data["progressive_dropout"] and "dropout_fractions" in data["progressive_dropout"]:
                    # Create a table to log
                    accuracies = np.array(data["progressive_dropout"]["accuracies"])
                    dropout_fractions = data["progressive_dropout"]["dropout_fractions"]
                    
                    # Calculate mean and std
                    mean_accuracies = np.mean(accuracies, axis=0)
                    std_accuracies = np.std(accuracies, axis=0)
                    
                    # Prepare data table for the custom plot
                    table_data = [[x, y, y_std] for x, y, y_std in zip(dropout_fractions, mean_accuracies, std_accuracies)]
                    
                    # Log the data as a custom plot
                    wandb.log({
                        "dropout_curve": wandb.Table(
                            data=table_data,
                            columns=["dropout_fraction", "mean_accuracy", "std_accuracy"]
                        )
                    })
                    
                    # Create interactive plot
                    dropout_data = []
                    for i, frac in enumerate(dropout_fractions):
                        dropout_data.append([frac, mean_accuracies[i], mean_accuracies[i]-std_accuracies[i], mean_accuracies[i]+std_accuracies[i]])
                    
                    wandb.log({
                        "dropout_plot_interactive": wandb.plot.line_series(
                            xs=[x[0] for x in dropout_data],
                            ys=[[x[1] for x in dropout_data],  # mean
                                [x[2] for x in dropout_data],  # mean-std
                                [x[3] for x in dropout_data]],  # mean+std
                            keys=["Mean Accuracy", "Lower Bound", "Upper Bound"],
                            title=f"Progressive Dropout ({self.config.extra.dropout_pruning_mode})",
                            xname="Dropout Fraction",
                            yname="Accuracy"
                        )
                    })
                
            return True
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {str(e)}")
            return False

    def run(self) -> Tuple[Dict, List[nn.Module]]:
        """
        Run the experiment with custom logging and wandb setup.
        
        Returns:
            Tuple containing (results, networks)
        """
        # Setup paths for experiment
        self.setup_paths()
        self.prepare_path()
        
        # Set up logging and suppress warnings
        set_logging_level(logging.INFO)
        
        # Set random seed if configured
        if hasattr(self.config, 'seed') and self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            logger.info(f"Set random seed to {self.config.seed}")
        
        # Initialize wandb explicitly before starting experiment
        if hasattr(self.config, 'checkpointing') and self.config.checkpointing.use_wandb:
            self._initialize_wandb()
        
        # Run the main experiment
        results, networks = self.main()
        
        # Ensure all plots are saved and logged
        self.plot(results)
        
        # Save experiment state
        self.save()
        
        # Close wandb run if it exists
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
                logger.info("Closed wandb run")
        except Exception as e:
            logger.warning(f"Error closing wandb run: {str(e)}")
        
        return results, networks

    def run_alignment_analysis(self, networks: List[nn.Module]) -> Dict:
        """
        Run alignment analysis experiment.
        
        This experiment analyzes alignment metrics across networks and can run
        both progressive dropout and eigenvector dropout experiments based on
        configuration.
        
        Args:
            networks: List of neural networks to analyze
            
        Returns:
            Results dictionary
        """
        logger.info("Running alignment analysis experiment")
        
        # Prepare the results structure
        results = {
            "progressive_dropout": {},
            "eigenvector_dropout": {},
            "config": self.config,
        }
        
        # Progressive dropout experiment
        if self.config.alignment.run_progressive:
            logger.info(f"Running progressive dropout experiment (mode: {getattr(self.config.extra, 'dropout_pruning_mode', 'global')})")
            try:
                prog_results = self._run_progressive_dropout(networks)
                results["progressive_dropout"] = prog_results
                
                # Add debug logging to inspect results structure
                if prog_results:
                    logger.info(f"Progressive dropout results keys: {list(prog_results.keys())}")
                    if 'accuracies' in prog_results:
                        logger.info(f"Progressive dropout accuracies keys: {list(prog_results['accuracies'].keys())}")
                
                # Check if there was an error in the dataset configuration
                if "error" in prog_results:
                    logger.error(f"Progressive dropout experiment failed: {prog_results['error']}")
                else:
                    # Generate plots right after getting results
                    try:
                        logger.info("Generating progressive dropout visualizations")
                        # Add debug print before and after plotting
                        figure_path = self._plot_dropout_results(
                            results["progressive_dropout"],
                            self.figure_path,
                            title_prefix="Progressive Dropout"
                        )
                        logger.info(f"Progressive dropout plot generated at {figure_path}")
                    except Exception as e:
                        logger.error(f"Error generating progressive dropout visualizations: {str(e)}", exc_info=True)
            except Exception as e:
                logger.error(f"Error running progressive dropout experiment: {str(e)}")
                results["progressive_dropout"] = {"error": str(e)}
        
        # Eigenvector dropout experiment
        if self.config.alignment.run_eigenvector:
            logger.info("Running eigenvector dropout experiment")
            try:
                # Note: For eigenvector dropout, we use just the first network
                eig_results = self._run_eigenvector_dropout(networks[0])
                results["eigenvector_dropout"] = eig_results
                
                # Add debug logging
                if eig_results:
                    logger.info(f"Eigenvector dropout results keys: {list(eig_results.keys())}")
                    if 'accuracies' in eig_results:
                        logger.info(f"Eigenvector dropout accuracies keys: {list(eig_results['accuracies'].keys())}")
                
                # Generate plots right after getting results
                try:
                    logger.info("Generating eigenvector dropout visualizations")
                    # Use the specialized method for eigenvector dropout
                    self._generate_dropout_plots(
                        results["eigenvector_dropout"],
                        "Eigenvector Dropout",
                        "eigenvector_dropout"
                    )
                    logger.info("Successfully generated eigenvector dropout visualizations")
                except Exception as e:
                    logger.error(f"Error generating eigenvector dropout visualizations: {str(e)}", exc_info=True)
            except Exception as e:
                logger.error(f"Error running eigenvector dropout experiment: {str(e)}")
                results["eigenvector_dropout"] = {"error": str(e)}
        
        # Generate and log summary plots for all experiments
        try:
            logger.info("Generating experiment summary")
            self._generate_summary_plots(results)
            logger.info("Successfully generated experiment summary")
        except Exception as e:
            logger.error(f"Error generating experiment summary: {str(e)}", exc_info=True)
        
        return results


def set_logging_level(level=logging.INFO):
    """
    Set logging level for all loggers in the alignment package.
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    # Set root logger level
    logging.getLogger().setLevel(level)
    
    # Set level for alignment package loggers
    logging.getLogger('alignment').setLevel(level)
    
    # Reduce verbosity for other loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.WARNING)
    
    # Suppress common warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pruning fraction.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Combined pruning across layers.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*dropout of the entire tensor.*")

def cli_main():
    """Command-line interface for running alignment experiments."""
    parser = argparse.ArgumentParser(description="Neural network alignment experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    args = parser.parse_args()

    # Set logging level based on quiet flag
    if args.quiet:
        set_logging_level(logging.WARNING)
    else:
        set_logging_level(logging.INFO)

    # Load configuration
    config = ExperimentConfig.load(args.config)
    
    # Initialize and run experiment
    experiment = AlignmentExperiment(config)
    results, networks = experiment.run()
    
    return results, networks


if __name__ == "__main__":
    cli_main()