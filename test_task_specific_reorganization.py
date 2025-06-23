#!/usr/bin/env python3
"""Test task-specific metrics reorganization."""

import torch
from src.alignment.core.registry import METRIC_REGISTRY, get_metric

def test_task_specific_metrics():
    """Test that all task-specific metrics are properly registered and work."""
    # List all available metrics
    all_metrics = METRIC_REGISTRY.list()
    print(f"Total metrics available: {len(all_metrics)}")
    
    # Task-specific metrics we expect
    task_specific_metrics = [
        'task_alignment',
        'class_selectivity',
        'feature_importance',
        'representation_quality',
        'classification_alignment',
        'language_model_alignment',
        'vision_task_alignment',
        'reinforcement_learning_alignment'
    ]
    
    print("\nTesting task-specific metrics:")
    print("=" * 50)
    
    # Test data
    batch_size = 32
    input_dim = 64
    output_dim = 128
    
    inputs = torch.randn(batch_size, input_dim)
    weights = torch.randn(output_dim, input_dim)
    outputs = inputs @ weights.T
    
    success_count = 0
    
    for metric_name in task_specific_metrics:
        try:
            # Get metric class
            metric_class = METRIC_REGISTRY.get(metric_name)
            
            # Initialize based on metric type
            if metric_name == 'classification_alignment':
                metric = metric_class(n_classes=10)
            elif metric_name == 'language_model_alignment':
                metric = metric_class(vocab_size=1000)
            elif metric_name == 'vision_task_alignment':
                metric = metric_class()
            elif metric_name == 'reinforcement_learning_alignment':
                metric = metric_class()
            else:
                metric = metric_class()
            
            # Compute metric
            scores = metric.compute(inputs=inputs, weights=weights, outputs=outputs)
            
            # Check output
            assert scores.shape == (output_dim,), f"Expected shape ({output_dim},), got {scores.shape}"
            assert not torch.isnan(scores).any(), f"NaN values in {metric_name}"
            
            print(f"✓ {metric_name:35} - Mean: {scores.mean().item():.4f}, Std: {scores.std().item():.4f}")
            success_count += 1
            
        except Exception as e:
            print(f"✗ {metric_name:35} - Error: {str(e)}")
    
    print(f"\nSuccess: {success_count}/{len(task_specific_metrics)} metrics")
    
    # Test imports from new structure
    print("\nTesting direct imports:")
    try:
        from src.alignment.metrics.task_specific import (
            TaskAlignment,
            ClassSelectivity,
            FeatureImportance,
            RepresentationQuality,
            ClassificationAlignment,
            LanguageModelAlignment,
            VisionTaskAlignment,
            ReinforcementLearningAlignment
        )
        print("✓ All imports successful from task_specific module")
    except ImportError as e:
        print(f"✗ Import error: {e}")
    
    # Test metric registration in categories
    print("\nMetrics by category:")
    categories = {
        'General Task': ['task_alignment', 'class_selectivity', 'feature_importance', 'representation_quality'],
        'Classification': ['classification_alignment'],
        'Language Model': ['language_model_alignment'],
        'Vision': ['vision_task_alignment'],
        'RL': ['reinforcement_learning_alignment']
    }
    
    for category, metrics in categories.items():
        available = [m for m in metrics if m in all_metrics]
        print(f"{category:15} - {len(available)}/{len(metrics)} available")

if __name__ == "__main__":
    test_task_specific_metrics() 