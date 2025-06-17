"""Base metrics module."""

from alignment.core.base import BaseMetric

# MetricComputer doesn't seem to exist in core, so we'll create a placeholder or remove it
class MetricComputer:
    """Placeholder for MetricComputer - to be implemented if needed."""
    pass


class BaseInformationMetric(BaseMetric):
    """Base class for information-theoretic metrics."""
    
    def __init__(self, **kwargs):
        """Initialize information metric."""
        super().__init__(**kwargs)
    
    # Information metrics typically need both inputs and outputs
    requires_outputs = True


__all__ = ['BaseMetric', 'MetricComputer', 'BaseInformationMetric']
