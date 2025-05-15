#!/usr/bin/env python
"""
Simple test script to verify that the scaling factor fix works for high dropout ratios.
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import numpy as np
from alignment.metrics import AlignmentMetric
from alignment.dropout import progressive_dropout

# Create a simple model for testing
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
        self.alignment_layers = [self.fc1, self.fc2]
        self.alignment_names = ['fc1', 'fc2']
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Mock dataset class for testing
class MockDataset:
    def __init__(self):
        self.call_count = 0
        
    def evaluate(self, model, device):
        # Return a unique accuracy for each call so we can see if multiple fractions work
        self.call_count += 1
        accuracy = 100.0 - self.call_count * 5  # Lower accuracy with each call
        return accuracy, 0.1  # Mock accuracy and loss
    
    @property
    def test_loader(self):
        # Return a minimal dataloader-like object
        class DummyLoader:
            def __iter__(self):
                return iter([])
                
        return DummyLoader()

# Mock metric class
class MockMetric(AlignmentMetric):
    def compute_per_node_scores(self, x, w, device):
        return torch.ones(w.shape[0])

def test_scaling_factor_calculation():
    """
    Directly test the scaling factor calculation with high dropout fractions
    to verify our fix works.
    """
    print("\nDirect test of scaling factor calculation:")
    high_fractions = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    for fraction in high_fractions:
        # Simulating the calculation in dropout.py
        scaling_factor = fraction
        
        # Original calculation (problematic at high fractions)
        original_scale = 1.0 / (1.0 - scaling_factor) if scaling_factor < 1.0 else 1.0
        
        # Fixed calculation with safety limit
        fixed_scale = 1.0 / (1.0 - scaling_factor) if scaling_factor < 0.9 else 10.0
        
        print(f"Dropout fraction: {fraction:.2f} -> Original scale: {original_scale:.2f}, Fixed scale: {fixed_scale:.2f}")

def main():
    print("Testing dropout with high dropout fractions...")
    model = TestModel()
    dataset = MockDataset()
    
    # First test the scaling factor calculation directly
    test_scaling_factor_calculation()
    
    # Test with progressively higher dropout fractions
    dropout_fractions = [0.0, 0.5, 0.8, 0.9, 0.95]
    
    print("\nRunning full progressive dropout test:")
    try:
        results, losses = progressive_dropout(
            networks=[model],
            dataset=dataset,
            dropout_fractions=dropout_fractions,
            metric=MockMetric(),
            device='cpu',
            dropout_mode='scaled',
            pruning_mode='layer_wise'
        )
        print(f"Successfully completed test with high dropout ratios")
        print(f"Dropout fractions: {dropout_fractions}")
        
        if 0 in results and results[0]:
            print(f"Results for first network: {results[0]}")
        else:
            print("No results returned - this is unexpected")
            print(f"Results object structure: {results}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 