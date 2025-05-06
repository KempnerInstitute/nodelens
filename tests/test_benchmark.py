"""Simple script to test the alignment metrics."""

import os
import sys
import torch
import numpy as np

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our metrics utility functions
from alignment.utils.metrics_utils import (
    AlignmentMetricBase,
    RQMetric,
    MIMetric,
    WeightSimilarityMetric,
    NodeRedundancyMetric,
    AlignmentMetricsFactory,
    alignment
)

def test_rq_metric():
    # Create random inputs and weights
    inputs = torch.randn(100, 10)
    weights = torch.randn(5, 10)
    
    # Compute RQ alignment
    rq_values = RQMetric.measure(inputs, weights)
    print(f"RQ values: {rq_values}")
    
    # Test via factory
    rq_factory = AlignmentMetricsFactory.measure(inputs, weights, method="RQ")
    print(f"RQ via factory: {rq_factory}")
    
    # Test via alignment function
    rq_align = alignment(inputs, weights, method="RQ")
    print(f"RQ via alignment function: {rq_align}")
    
    # Verify all approaches give same results
    assert torch.allclose(rq_values, rq_factory)
    assert torch.allclose(rq_values, rq_align)
    print("All RQ implementations match!")

if __name__ == "__main__":
    print("Testing RQ Metric...")
    test_rq_metric()
    print("All tests passed!") 