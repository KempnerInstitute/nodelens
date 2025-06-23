"""
Quick demo of the alignment module functionality.
"""

import torch
import torch.nn as nn
from alignment.models import ModelWrapper
from alignment.metrics import METRIC_REGISTRY


def main():
    # 1. Create a simple neural network
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    print("Model created with 3 linear layers")
    
    # 2. Wrap the model to track activations
    wrapped_model = ModelWrapper(model)
    print(f"\nTracked layers: {wrapped_model.tracked_layers}")
    
    # 3. Generate some dummy data
    batch_size = 32
    inputs = torch.randn(batch_size, 784)
    
    # 4. Forward pass with activation tracking
    outputs, activations = wrapped_model.forward_with_activations(inputs)
    print(f"\nCollected activations from {len(activations)} points")
    
    # 5. Get layer weights
    weights = wrapped_model.get_layer_weights()
    print(f"Extracted weights from {len(weights)} layers")
    
    # 6. Compute alignment metrics
    print("\n" + "="*60)
    print("Computing Alignment Metrics")
    print("="*60)
    
    # Rayleigh Quotient
    rq_metric = METRIC_REGISTRY['rayleigh_quotient']()
    
    for layer_name in wrapped_model.tracked_layers:
        # Get inputs and weights for this layer
        layer_inputs = activations[f"{layer_name}_input"]
        layer_weights = weights[layer_name]
        
        # Compute RQ scores
        scores = rq_metric.compute(inputs=layer_inputs, weights=layer_weights)
        
        print(f"\nLayer: {layer_name}")
        print(f"  Input shape: {layer_inputs.shape}")
        print(f"  Weight shape: {layer_weights.shape}")
        print(f"  RQ scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
        print(f"  Min neuron score: {scores.min():.4f}")
        print(f"  Max neuron score: {scores.max():.4f}")
    
    # 7. Try other metrics
    print("\n" + "="*60)
    print("Trying Other Metrics")
    print("="*60)
    
    # Node correlation on outputs
    node_corr = METRIC_REGISTRY['node_correlation']()
    # Use the actual output of the model
    corr_scores = node_corr.compute(outputs=outputs)
    print(f"\nNode Correlation (final layer): mean={corr_scores.mean():.4f}")
    
    # Weight cosine similarity
    weight_sim = METRIC_REGISTRY['weight_cosine_similarity']()
    for layer_name in wrapped_model.tracked_layers:
        layer_weights = weights[layer_name]
        sim_scores = weight_sim.compute(weights=layer_weights)
        print(f"Weight Cosine Similarity ({layer_name}): mean={sim_scores.mean():.4f}")
    
    # 8. Demonstrate pruning
    print("\n" + "="*60)
    print("Pruning Demo")
    print("="*60)
    
    # Create pruning masks based on low RQ scores
    first_layer = wrapped_model.tracked_layers[0]
    layer_inputs = activations[f"{first_layer}_input"]
    layer_weights = weights[first_layer]
    
    scores = rq_metric.compute(inputs=layer_inputs, weights=layer_weights)
    threshold = scores.median()
    
    # Create mask (True = keep, False = prune)
    masks = {first_layer: scores > threshold}
    
    print(f"\nPruning neurons with RQ < {threshold:.4f} in layer {first_layer}")
    print(f"Keeping {masks[first_layer].sum()}/{len(scores)} neurons")
    
    # Apply structured dropout
    wrapped_model.apply_structured_dropout(masks)
    
    # Test forward pass after pruning
    pruned_outputs, _ = wrapped_model.forward_with_activations(inputs)
    print(f"\nOutput shape after pruning: {pruned_outputs.shape}")
    print("Pruning applied successfully!")
    
    print("\n Demo completed successfully!")


if __name__ == "__main__":
    main() 