"""Standalone test script for alignment metrics."""

import os
import sys
import torch
import numpy as np

# Add the project root to the Python path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Direct implementation of RQMetric
def compute_rq_metric(inputs, weights, relative=True, epsilon=1e-8):
    """Compute Representation Quality metric."""
    # Ensure inputs have at least 2 dimensions
    if inputs.dim() < 2:
        inputs = inputs.unsqueeze(0)
        
    # Move weights to same device as inputs
    weights = weights.to(inputs.device)
    
    # Center the inputs
    X = inputs - inputs.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    cov = torch.matmul(X.t(), X) / (X.size(0) - 1)
    
    # Add small value to diagonal for stability
    cov = cov + torch.eye(cov.size(0), device=cov.device) * epsilon
    
    # Compute the RQ values
    numerator = torch.sum(weights * torch.matmul(weights, cov), dim=1)
    denominator = (torch.norm(weights, dim=1) ** 2) * (torch.norm(weights @ cov, dim=1) + epsilon)
    
    # Calculate RQ as cosine similarity between weight vectors and weight @ covariance
    rq = numerator / denominator
    
    if relative:
        # Make RQ values relative to random vectors in high dimensions (expected value is 1/sqrt(d))
        d = weights.size(1)
        rq = rq * np.sqrt(d)
        
    return rq

def test_rq_metric():
    """Test the RQ metric implementation."""
    # Create random inputs and weights
    inputs = torch.randn(100, 10)
    weights = torch.randn(5, 10)
    
    # Compute RQ alignment
    rq_values = compute_rq_metric(inputs, weights)
    print(f"RQ values: {rq_values}")
    
    # Test non-relative RQ
    rq_nonrel = compute_rq_metric(inputs, weights, relative=False)
    print(f"Non-relative RQ: {rq_nonrel}")
    
    # Compare with theoretical expectation
    # For random data, RQ should be approximately 1.0 when relative=True
    avg_rq = rq_values.mean().item()
    print(f"Average RQ: {avg_rq:.4f} (should be close to 1.0 for random data)")
    
    # Check if the relative scaling works as expected
    d = weights.size(1)
    scaled_nonrel = rq_nonrel * np.sqrt(d)
    print(f"Manually scaled non-relative RQ: {scaled_nonrel}")
    
    # Verify relative scaling
    assert torch.allclose(rq_values, scaled_nonrel)
    print("Relative scaling works correctly!")
    
if __name__ == "__main__":
    print("Testing standalone RQ Metric implementation...")
    test_rq_metric()
    print("All tests passed!") 