"""
Result aggregation utilities for analyzing experiment outputs.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging
from collections import defaultdict

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


class MetricAggregator:
    """
    Aggregates metrics across time steps or iterations.
    
    Useful for analyzing metric evolution during training or experiments.
    """
    
    def __init__(self):
        """Initialize metric aggregator."""
        self.metrics_over_time = defaultdict(lambda: defaultdict(list))
        self.steps = []
    
    def add_step(self, step: int, metrics: Dict[str, Dict[str, float]]):
        """
        Add metrics from a single step.
        
        Args:
            step: Step/iteration number
            metrics: Dictionary of metrics by name and layer
        """
        self.steps.append(step)
        
        for metric_name, layer_values in metrics.items():
            if isinstance(layer_values, dict):
                for layer_name, value in layer_values.items():
                    self.metrics_over_time[metric_name][layer_name].append(value)
            else:
                self.metrics_over_time[metric_name]['value'].append(layer_values)
    
    def get_metric_evolution(
        self,
        metric_name: str,
        layer_name: str
    ) -> Tuple[List[int], List[float]]:
        """
        Get the evolution of a metric over time.
        
        Args:
            metric_name: Name of the metric
            layer_name: Name of the layer
            
        Returns:
            Tuple of (steps, values)
        """
        if metric_name not in self.metrics_over_time:
            return [], []
        
        if layer_name not in self.metrics_over_time[metric_name]:
            return [], []
        
        values = self.metrics_over_time[metric_name][layer_name]
        return self.steps[:len(values)], values
    
    def compute_trends(
        self,
        metric_name: str,
        layer_name: str,
        window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Compute trend statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            layer_name: Name of the layer
            window_size: Window size for moving average
            
        Returns:
            Dictionary with trend statistics
        """
        steps, values = self.get_metric_evolution(metric_name, layer_name)
        
        if len(values) < 2:
            return {}
        
        values_array = np.array(values)
        
        # Compute moving average
        if len(values) >= window_size:
            moving_avg = np.convolve(values_array, 
                                    np.ones(window_size) / window_size, 
                                    mode='valid')
        else:
            moving_avg = values_array
        
        # Compute linear trend
        coeffs = np.polyfit(range(len(values)), values_array, 1)
        slope = coeffs[0]
        
        # Find change points
        if len(values) > 2:
            diffs = np.diff(values_array)
            change_points = np.where(np.abs(diffs) > 2 * np.std(diffs))[0]
        else:
            change_points = []
        
        return {
            'initial_value': float(values[0]),
            'final_value': float(values[-1]),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'slope': float(slope),
            'percent_change': float((values[-1] - values[0]) / (values[0] + 1e-8) * 100),
            'moving_average': moving_avg.tolist() if len(moving_avg) > 0 else [],
            'change_points': change_points.tolist()
        }


class LayerAggregator:
    """
    Aggregates metrics by layer to analyze layer-wise patterns.
    """
    
    def __init__(self):
        """Initialize layer aggregator."""
        self.layer_metrics = defaultdict(lambda: defaultdict(list))
    
    def add_metrics(self, metrics: Dict[str, Dict[str, float]]):
        """
        Add metrics from a single evaluation.
        
        Args:
            metrics: Dictionary of metrics by name and layer
        """
        for metric_name, layer_values in metrics.items():
            if isinstance(layer_values, dict):
                for layer_name, value in layer_values.items():
                    self.layer_metrics[layer_name][metric_name].append(value)
    
    def get_layer_summary(self, layer_name: str) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dictionary mapping metric names to statistics
        """
        if layer_name not in self.layer_metrics:
            return {}
        
        summary = {}
        for metric_name, values in self.layer_metrics[layer_name].items():
            if values:
                values_array = np.array(values)
                summary[metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'count': len(values)
                }
        
        return summary
    
    def rank_layers(
        self,
        metric_name: str,
        criterion: str = 'mean',
        ascending: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Rank layers by a specific metric.
        
        Args:
            metric_name: Metric to rank by
            criterion: Statistic to use ('mean', 'max', 'min', 'std')
            ascending: Whether to sort in ascending order
            
        Returns:
            List of (layer_name, value) tuples
        """
        layer_values = []
        
        for layer_name, metrics in self.layer_metrics.items():
            if metric_name in metrics and metrics[metric_name]:
                values_array = np.array(metrics[metric_name])
                
                if criterion == 'mean':
                    value = np.mean(values_array)
                elif criterion == 'max':
                    value = np.max(values_array)
                elif criterion == 'min':
                    value = np.min(values_array)
                elif criterion == 'std':
                    value = np.std(values_array)
                else:
                    raise ValueError(f"Unknown criterion: {criterion}")
                
                layer_values.append((layer_name, float(value)))
        
        # Sort by value
        layer_values.sort(key=lambda x: x[1], reverse=not ascending)
        
        return layer_values
    
    def find_anomalous_layers(
        self,
        metric_name: str,
        threshold_std: float = 2.0
    ) -> List[str]:
        """
        Find layers with anomalous metric values.
        
        Args:
            metric_name: Metric to analyze
            threshold_std: Number of standard deviations for anomaly detection
            
        Returns:
            List of anomalous layer names
        """
        # Collect all values
        all_values = []
        layer_means = {}
        
        for layer_name, metrics in self.layer_metrics.items():
            if metric_name in metrics and metrics[metric_name]:
                mean_value = np.mean(metrics[metric_name])
                layer_means[layer_name] = mean_value
                all_values.append(mean_value)
        
        if len(all_values) < 3:
            return []
        
        # Compute global statistics
        global_mean = np.mean(all_values)
        global_std = np.std(all_values)
        
        # Find anomalous layers
        anomalous = []
        for layer_name, mean_value in layer_means.items():
            if abs(mean_value - global_mean) > threshold_std * global_std:
                anomalous.append(layer_name)
        
        return anomalous 