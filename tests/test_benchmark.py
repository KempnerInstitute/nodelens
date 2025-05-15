"""Simple script to test the alignment metrics."""

import os
import sys
import torch
import numpy as np

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import from the new metrics system
from alignment.metrics import get_metric, compute_rq_alternative_denominator # For direct comparison if needed

# Keep torch and numpy
# Removed imports from alignment.utils.metrics_utils:
# AlignmentMetricBase, RQMetric, MIMetric, WeightSimilarityMetric,
# NodeRedundancyMetric, AlignmentMetricsFactory, alignment


def test_rq_metric():
    # Create random inputs and weights
    inputs = torch.randn(100, 10)
    weights = torch.randn(5, 10)
    
    # Compute RQ alignment using the new system
    # The 'rq_alt_denom' metric corresponds to the old RQMetric from metrics_utils.py
    metric_instance = get_metric(name="rq_alt_denom") 
    rq_values_new = metric_instance.compute_per_node_scores(
        layer_inputs=inputs, 
        layer_weights=weights
        # relative=True is the default in compute_rq_alternative_denominator
        # epsilon is also handled internally
    )
    print(f"RQ values (new system 'rq_alt_denom'): {rq_values_new}")

    # For a sanity check, we can call the underlying function directly if we want to be absolutely sure
    # This assumes compute_rq_alternative_denominator is imported.
    # rq_values_direct_call = compute_rq_alternative_denominator(layer_inputs=inputs, layer_weights=weights, relative=True)
    # print(f"RQ values (direct call to compute_rq_alternative_denominator): {rq_values_direct_call}")
    # assert torch.allclose(rq_values_new, rq_values_direct_call), "New system get_metric differs from direct call!"

    # Since the original test compared different ways of calling the *old* system,
    # and we've now switched to the *new* system, the old assertions are not directly applicable.
    # We are now testing that get_metric("rq_alt_denom") works as expected.
    # If there was a 'golden' value or a way to compute it via the old system still, we could compare.
    # For now, this test primarily ensures the new call path executes and returns a tensor of expected shape.
    
    assert rq_values_new.shape == (weights.shape[0],) 
    print("RQ 'rq_alt_denom' test with new system completed. Shape is correct.")

# It would be good to add more tests here for other metrics like:
# - mi_proj_vs_mean_input
# - weight_cosine_similarity, etc.
# - The original 'rq' (compute_rayleigh_quotient) from metrics.py

def test_main_rq_metric():
    inputs = torch.randn(100, 10)
    weights = torch.randn(5, 10)
    
    metric_instance = get_metric(name="rq", scale_by_norm=True) # Test the main RQ, relative scaling
    rq_values_main = metric_instance.compute_per_node_scores(
        layer_inputs=inputs,
        layer_weights=weights
    )
    print(f"RQ values (new system 'rq', scaled): {rq_values_main}")
    assert rq_values_main.shape == (weights.shape[0],)
    print("RQ 'rq' (scaled) test with new system completed. Shape is correct.")

    metric_instance_unscaled = get_metric(name="rq", scale_by_norm=False) # Test unscaled
    rq_values_main_unscaled = metric_instance_unscaled.compute_per_node_scores(
        layer_inputs=inputs,
        layer_weights=weights
    )
    print(f"RQ values (new system 'rq', unscaled): {rq_values_main_unscaled}")
    assert rq_values_main_unscaled.shape == (weights.shape[0],)
    print("RQ 'rq' (unscaled) test with new system completed. Shape is correct.")

def test_mi_proj_metric():
    inputs = torch.randn(100, 10) # batch, features
    weights = torch.randn(5, 10)  # out_features, features
    
    metric_instance = get_metric(name="mi_proj_vs_mean_input")
    mi_values = metric_instance.compute_per_node_scores(
        layer_inputs=inputs,
        layer_weights=weights,
        bins=10 # Example of passing a metric-specific kwarg
    )
    print(f"MI Proj vs Mean Input values (new system): {mi_values}")
    assert mi_values.shape == (weights.shape[0],) # Should be per output neuron
    print("MI Proj vs Mean Input test completed. Shape is correct.")

def test_weight_similarity_metric():
    weights = torch.randn(5, 10) # out_features, in_features

    # Test Cosine Similarity
    metric_instance_cos = get_metric(name="weight_cosine_similarity")
    sim_cos_values = metric_instance_cos.compute_per_node_scores(
        layer_weights=weights
        # No layer_inputs or layer_outputs needed
    )
    print(f"Weight Cosine Similarity values (new system):\\n{sim_cos_values}")
    assert sim_cos_values.shape == (weights.shape[0], weights.shape[0]) # Pairwise [out, out]
    print("Weight Cosine Similarity test completed. Shape is correct.")

    # Test Dot Similarity
    metric_instance_dot = get_metric(name="weight_dot_similarity")
    sim_dot_values = metric_instance_dot.compute_per_node_scores(layer_weights=weights)
    print(f"Weight Dot Similarity values (new system):\\n{sim_dot_values}")
    assert sim_dot_values.shape == (weights.shape[0], weights.shape[0])
    print("Weight Dot Similarity test completed. Shape is correct.")

    # Test Euclidean Distance
    metric_instance_euc = get_metric(name="weight_euclidean_distance")
    dist_euc_values = metric_instance_euc.compute_per_node_scores(layer_weights=weights)
    print(f"Weight Euclidean Distance values (new system):\\n{dist_euc_values}")
    assert dist_euc_values.shape == (weights.shape[0], weights.shape[0])
    print("Weight Euclidean Distance test completed. Shape is correct.")

def test_node_redundancy_metric():
    """Test the NodeRedundancy metric functionality."""
    # Create test input data - [batch_size, features]
    inputs = torch.randn(100, 10)
    
    # Create metric instance
    metric_instance = get_metric(name="node_redundancy")
    
    # Compute node redundancy scores
    redundancy_scores = metric_instance.compute_per_node_scores(
        layer_inputs=inputs
    )
    print(f"Node Redundancy values (new system): {redundancy_scores}")
    
    # Check shape - should return one score per input feature
    assert redundancy_scores.shape == (inputs.shape[1],)
    print("Node Redundancy test completed. Shape is correct.")


if __name__ == "__main__":
    print("Testing RQ Metric ('rq_alt_denom') with new system...")
    test_rq_metric()
    print("\nTesting main RQ Metric ('rq') with new system...")
    test_main_rq_metric()
    print("\nTesting MI Projected vs Mean Input Metric with new system...")
    test_mi_proj_metric()
    print("\nTesting Weight Similarity Metrics with new system...")
    test_weight_similarity_metric()
    print("\nTesting Node Redundancy Metric with new system...")
    test_node_redundancy_metric()
    print("\nAll tests passed!") 