"""
Result aggregation utilities for analyzing experiment outputs.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates results from multiple experiments or runs.
    
    This class provides utilities for:
    - Loading results from multiple sources
    - Computing statistics across runs
    - Extracting specific metrics
    - Comparing experiments
    """
    
    def __init__(self):
        """Initialize result aggregator."""
        self.results = {}
        self.metadata = {}
    
    def add_results(
        self,
        name: str,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add results from an experiment.
        
        Args:
            name: Experiment name/identifier
            results: Experiment results dictionary
            metadata: Optional metadata
        """
        self.results[name] = results
        if metadata:
            self.metadata[name] = metadata
        logger.info(f"Added results for experiment: {name}")
    
    def load_from_file(self, path: Union[str, Path], name: Optional[str] = None):
        """
        Load results from a JSON file.
        
        Args:
            path: Path to results file
            name: Name to use (defaults to filename)
        """
        path = Path(path)
        if not name:
            name = path.stem
        
        with open(path, 'r') as f:
            results = json.load(f)
        
        self.add_results(name, results)
    
    def load_from_directory(self, directory: Union[str, Path], pattern: str = "*_results.json"):
        """
        Load all matching result files from a directory.
        
        Args:
            directory: Directory containing result files
            pattern: Glob pattern for result files
        """
        directory = Path(directory)
        for result_file in directory.glob(pattern):
            self.load_from_file(result_file)
        
        logger.info(f"Loaded {len(self.results)} result files from {directory}")
    
    def get_metric_values(
        self,
        metric_name: str,
        layer_name: Optional[str] = None,
        experiment_names: Optional[List[str]] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Extract specific metric values across experiments.
        
        Args:
            metric_name: Name of the metric
            layer_name: Specific layer (None for all layers)
            experiment_names: Experiments to include (None for all)
            
        Returns:
            Dictionary mapping experiment names to metric values
        """
        if experiment_names is None:
            experiment_names = list(self.results.keys())
        
        metric_values = {}
        
        for exp_name in experiment_names:
            if exp_name not in self.results:
                continue
            
            exp_results = self.results[exp_name]
            
            # Navigate to metrics
            if 'metrics' in exp_results:
                metrics = exp_results['metrics']
                
                # Get final metrics (last step)
                if metrics:
                    last_step = max(int(k) for k in metrics.keys())
                    step_metrics = metrics[str(last_step)]
                    
                    if metric_name in step_metrics:
                        if layer_name:
                            value = step_metrics[metric_name].get(layer_name)
                            if value is not None:
                                metric_values[exp_name] = value
                        else:
                            metric_values[exp_name] = step_metrics[metric_name]
        
        return metric_values
    
    def compute_statistics(
        self,
        metric_name: str,
        layer_name: str,
        experiment_pattern: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute statistics for a metric across experiments.
        
        Args:
            metric_name: Name of the metric
            layer_name: Layer to analyze
            experiment_pattern: Pattern to filter experiments
            
        Returns:
            Dictionary with statistics (mean, std, min, max, etc.)
        """
        # Filter experiments
        if experiment_pattern:
            exp_names = [name for name in self.results.keys() 
                        if experiment_pattern in name]
        else:
            exp_names = list(self.results.keys())
        
        # Get metric values
        values_dict = self.get_metric_values(metric_name, layer_name, exp_names)
        values = list(values_dict.values())
        
        if not values:
            return {}
        
        values_array = np.array(values)
        
        return {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'q1': float(np.percentile(values_array, 25)),
            'q3': float(np.percentile(values_array, 75)),
            'count': len(values)
        }
    
    def to_dataframe(
        self,
        metrics: Optional[List[str]] = None,
        layers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.
        
        Args:
            metrics: Metrics to include (None for all)
            layers: Layers to include (None for all)
            
        Returns:
            DataFrame with experiments as rows and metric/layer combinations as columns
        """
        data = []
        
        for exp_name, results in self.results.items():
            row = {'experiment': exp_name}
            
            # Add metadata
            if exp_name in self.metadata:
                row.update(self.metadata[exp_name])
            
            # Add metrics
            if 'metrics' in results and results['metrics']:
                # Get final metrics
                last_step = max(int(k) for k in results['metrics'].keys())
                step_metrics = results['metrics'][str(last_step)]
                
                for metric_name, layer_values in step_metrics.items():
                    if metrics and metric_name not in metrics:
                        continue
                    
                    if isinstance(layer_values, dict):
                        for layer_name, value in layer_values.items():
                            if layers and layer_name not in layers:
                                continue
                            
                            col_name = f"{metric_name}_{layer_name}"
                            row[col_name] = value
                    else:
                        row[metric_name] = layer_values
            
            data.append(row)
        
        return pd.DataFrame(data) 