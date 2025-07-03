"""
Practical example: Using single-layer alignment-based pruning for model compression.
This shows how to analyze and prune a pre-trained model for deployment.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import time
from collections import OrderedDict

from src.alignment.pruning.base import PruningConfig
from src.alignment.pruning.strategies import AlignmentPruning
from src.alignment.metrics import get_metric


def analyze_model_layers(model, sample_input):
    """Analyze all layers in a model to identify pruning candidates."""
    layer_analysis = OrderedDict()
    
    # Hook to capture inputs and outputs
    activations = {}
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = {
                'input': input[0].detach() if isinstance(input, tuple) else input.detach(),
                'output': output.detach() if not isinstance(output, tuple) else output[0].detach()
            }
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            h = module.register_forward_hook(get_activation(name))
            hooks.append(h)
    
    # Forward pass
    with torch.no_grad():
        _ = model(sample_input)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Analyze each layer
    metric = get_metric('rayleigh_quotient')()
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and name in activations:
            layer_input = activations[name]['input']
            weights = module.weight.data
            
            # Compute alignment scores
            try:
                scores = metric.compute(inputs=layer_input, weights=weights)
                
                layer_analysis[name] = {
                    'type': type(module).__name__,
                    'weight_shape': list(weights.shape),
                    'num_params': weights.numel(),
                    'input_shape': list(layer_input.shape),
                    'alignment_mean': scores.mean().item(),
                    'alignment_std': scores.std().item(),
                    'alignment_min': scores.min().item(),
                    'alignment_max': scores.max().item(),
                }
            except Exception as e:
                layer_analysis[name] = {
                    'type': type(module).__name__,
                    'weight_shape': list(weights.shape),
                    'num_params': weights.numel(),
                    'error': str(e)
                }
    
    return layer_analysis


def selective_pruning_strategy(layer_analysis, model_size_target=0.5):
    """
    Determine which layers to prune and by how much to reach target model size.
    
    Strategy: Prune layers with low average alignment more aggressively.
    """
    # Calculate total parameters
    total_params = sum(info['num_params'] for info in layer_analysis.values() 
                      if 'num_params' in info)
    target_params = int(total_params * model_size_target)
    params_to_prune = total_params - target_params
    
    # Sort layers by average alignment (ascending)
    prunable_layers = [(name, info) for name, info in layer_analysis.items() 
                      if 'alignment_mean' in info]
    prunable_layers.sort(key=lambda x: x[1]['alignment_mean'])
    
    pruning_plan = OrderedDict()
    pruned_so_far = 0
    
    for name, info in prunable_layers:
        if pruned_so_far >= params_to_prune:
            break
        
        # More aggressive pruning for layers with lower alignment
        base_pruning_rate = 0.7  # Maximum pruning rate
        alignment_factor = info['alignment_mean'] / max(l[1]['alignment_mean'] 
                                                       for l in prunable_layers)
        
        # Reduce pruning rate for better-aligned layers
        pruning_rate = base_pruning_rate * (1 - alignment_factor * 0.5)
        
        # Don't prune classification layer too much
        if 'classifier' in name or 'fc' in name:
            pruning_rate = min(pruning_rate, 0.5)
        
        params_in_layer = info['num_params']
        params_to_prune_in_layer = int(params_in_layer * pruning_rate)
        
        if params_to_prune_in_layer > 0:
            pruning_plan[name] = {
                'pruning_rate': pruning_rate,
                'params_to_prune': params_to_prune_in_layer,
                'alignment_mean': info['alignment_mean']
            }
            pruned_so_far += params_to_prune_in_layer
    
    return pruning_plan


def apply_pruning_plan(model, pruning_plan, sample_input):
    """Apply the pruning plan to the model."""
    from test_single_layer_pruning import SingleLayerPruner
    
    total_pruned = 0
    
    for layer_name, plan in pruning_plan.items():
        print(f"\nPruning {layer_name}:")
        print(f"  Target sparsity: {plan['pruning_rate']*100:.1f}%")
        print(f"  Alignment score: {plan['alignment_mean']:.4f}")
        
        try:
            pruner = SingleLayerPruner(model, layer_name)
            pruner.capture_layer_input(sample_input)
            
            # Apply pruning with hooks for potential fine-tuning
            mask = pruner.prune_layer(
                amount=plan['pruning_rate'],
                metric_name='rayleigh_quotient',
                structured=True,
                make_permanent=False
            )
            
            stats = pruner.get_pruning_stats()
            print(f"  Actual sparsity: {stats['sparsity']*100:.1f}%")
            print(f"  Pruned params: {stats['zero_parameters']:,}")
            
            total_pruned += stats['zero_parameters']
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return total_pruned


def measure_inference_speed(model, input_shape, num_runs=100):
    """Measure average inference time."""
    model.eval()
    x = torch.randn(1, *input_shape)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)
    
    # Measure
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(x)
    
    avg_time = (time.time() - start_time) / num_runs
    return avg_time * 1000  # Convert to milliseconds


def main():
    """Demonstrate practical model compression workflow."""
    print("Practical Model Compression with Alignment-Based Pruning")
    print("="*60)
    
    # Load pre-trained model (using MobileNetV2 as it's designed for efficiency)
    print("\n1. Loading pre-trained MobileNetV2...")
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    
    # Count original parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Measure original inference speed
    print("\n2. Measuring original inference speed...")
    orig_time = measure_inference_speed(model, (3, 224, 224))
    print(f"   Average inference time: {orig_time:.2f} ms")
    
    # Analyze layers
    print("\n3. Analyzing layer alignments...")
    sample_input = torch.randn(8, 3, 224, 224)
    layer_analysis = analyze_model_layers(model, sample_input)
    
    print(f"   Analyzed {len(layer_analysis)} layers")
    print("\n   Top 5 best-aligned layers:")
    sorted_layers = sorted(layer_analysis.items(), 
                          key=lambda x: x[1].get('alignment_mean', 0), 
                          reverse=True)
    for name, info in sorted_layers[:5]:
        if 'alignment_mean' in info:
            print(f"   - {name}: {info['alignment_mean']:.4f}")
    
    print("\n   Top 5 worst-aligned layers (pruning candidates):")
    for name, info in sorted_layers[-5:]:
        if 'alignment_mean' in info:
            print(f"   - {name}: {info['alignment_mean']:.4f}")
    
    # Create pruning plan
    print("\n4. Creating pruning plan (target: 50% model size)...")
    pruning_plan = selective_pruning_strategy(layer_analysis, model_size_target=0.5)
    
    print(f"   Planning to prune {len(pruning_plan)} layers")
    total_planned_pruning = sum(plan['params_to_prune'] for plan in pruning_plan.values())
    print(f"   Total parameters to prune: {total_planned_pruning:,}")
    
    # Apply pruning
    print("\n5. Applying pruning plan...")
    total_pruned = apply_pruning_plan(model, pruning_plan, sample_input)
    
    # Calculate final model size
    remaining_params = total_params - total_pruned
    compression_ratio = remaining_params / total_params
    
    print(f"\n6. Results:")
    print(f"   Original parameters: {total_params:,}")
    print(f"   Pruned parameters: {total_pruned:,}")
    print(f"   Remaining parameters: {remaining_params:,}")
    print(f"   Compression ratio: {compression_ratio:.2%}")
    
    # Measure new inference speed
    print("\n7. Measuring pruned model inference speed...")
    pruned_time = measure_inference_speed(model, (3, 224, 224))
    speedup = orig_time / pruned_time
    
    print(f"   Original time: {orig_time:.2f} ms")
    print(f"   Pruned time: {pruned_time:.2f} ms")
    print(f"   Speedup: {speedup:.2f}x")
    
    # Test accuracy (simplified - just checking outputs are valid)
    print("\n8. Validating pruned model...")
    test_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Contains NaN: {torch.isnan(output).any().item()}")
    print(f"   Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    print("\n" + "="*60)
    print("Summary:")
    print(f"- Compressed model to {compression_ratio:.1%} of original size")
    print(f"- Achieved {speedup:.2f}x speedup")
    print("- Model remains functional with hooks for fine-tuning")
    print("\nNext steps:")
    print("- Fine-tune the pruned model on your dataset")
    print("- Export to ONNX or TorchScript for deployment")
    print("- Consider quantization for further compression")


if __name__ == "__main__":
    main() 