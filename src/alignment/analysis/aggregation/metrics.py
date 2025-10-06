"""
Metric aggregation utilities for tracking metrics over time.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
