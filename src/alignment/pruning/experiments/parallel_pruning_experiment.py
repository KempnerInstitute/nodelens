"""
Parallel pruning experiment for multi-network analysis.

This module provides a wrapper around general_alignment.py's multi-network
functionality for backward compatibility and convenience.
"""

from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, field
from pathlib import Path

from alignment.experiments.general_alignment import (
    GeneralAlignmentExperiment,
    GeneralAlignmentConfig
)
from alignment.experiments.config_components import (
    PruningConfig,
    MultiNetworkConfig
)

logger = logging.getLogger(__name__)


@dataclass
class ParallelExperimentConfig(GeneralAlignmentConfig):
    """Configuration for parallel pruning experiments.
    
    This is a convenience wrapper around GeneralAlignmentConfig
    with sensible defaults for multi-network pruning experiments.
    """
    
    # Override defaults for multi-network experiments
    num_networks: int = 5
    
    # Convenience aliases
    num_seeds: Optional[int] = None  # Alias for num_networks
    
    def __post_init__(self):
        """Handle aliases and set defaults."""
        # Handle num_seeds alias
        if self.num_seeds is not None:
            self.num_networks = self.num_seeds
            
        # Call parent post_init
        super().__post_init__()
        
        # Set default pruning strategies if not specified
        if self.pruning_strategies is None:
            self.pruning_strategies = ['magnitude', 'gradient', 'random']
        
        # Set default sparsity levels if not specified  
        if self.sparsity_levels is None:
            self.sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]


class ParallelPruningExperiment(GeneralAlignmentExperiment):
    """
    Parallel pruning experiment using multi-network support.
    
    This is a convenience wrapper around GeneralAlignmentExperiment
    that provides backward compatibility and simplified interface
    for multi-network pruning experiments.
    """
    
    def __init__(self, config: ParallelExperimentConfig):
        """Initialize parallel experiment."""
        # Ensure multi-network mode is enabled
        if config.num_networks < 2:
            logger.warning(
                f"ParallelPruningExperiment created with num_networks={config.num_networks}. "
                "Consider using GeneralAlignmentExperiment directly for single networks."
            )
        
        # Initialize parent with config
        super().__init__(config)
        
    def run(self) -> Dict[str, Any]:
        """
        Run the parallel pruning experiment.
        
        This delegates to the parent class which handles multi-network
        experiments automatically when num_networks > 1.
        """
        logger.info(
            f"Running parallel pruning experiment with {self.config.num_networks} networks"
        )
        
        # Run the experiment using parent implementation
        results = super().run()
        
        # Log summary of multi-network results
        if 'dropout_analysis' in results and self.config.num_networks > 1:
            dropout_data = results['dropout_analysis']
            
            # Log variance information
            for strategy in dropout_data.get('strategies', []):
                if f'{strategy}_variance' in dropout_data:
                    variances = dropout_data[f'{strategy}_variance']
                    logger.info(
                        f"Strategy {strategy} - Mean variance across dropout rates: "
                        f"{sum(variances) / len(variances):.4f}"
                    )
        
        return results


def run_parallel_pruning_experiment(
    model_class: type,
    model_kwargs: Dict[str, Any],
    num_networks: int = 5,
    dataset_name: str = 'mnist',
    output_dir: str = 'results/parallel_pruning',
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run a parallel pruning experiment.
    
    Args:
        model_class: Model class to instantiate
        model_kwargs: Arguments for model construction
        num_networks: Number of networks to train
        dataset_name: Name of dataset to use
        output_dir: Directory to save results
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary containing all results
    """
    # Create config
    config = ParallelExperimentConfig(
        num_networks=num_networks,
        model_class=model_class.__name__,
        model_kwargs=model_kwargs,
        dataset_name=dataset_name,
        output_dir=output_dir,
        **kwargs
    )
    
    # Create and run experiment
    experiment = ParallelPruningExperiment(config)
    return experiment.run()


# For backward compatibility
def create_parallel_experiment(config_dict: Dict[str, Any]) -> ParallelPruningExperiment:
    """Create a parallel pruning experiment from a config dictionary."""
    config = ParallelExperimentConfig(**config_dict)
    return ParallelPruningExperiment(config) 