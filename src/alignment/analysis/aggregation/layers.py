"""
Layer-wise metric aggregation for analyzing patterns across network layers.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


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