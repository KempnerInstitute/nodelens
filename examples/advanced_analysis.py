"""
Advanced alignment analysis example demonstrating:
- Batch processing for efficiency
- Visualization tools
- Experiment tracking with WandB/TensorBoard
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

# Import alignment modules
from alignment.models import ModelWrapper
from alignment.metrics import METRIC_REGISTRY
from alignment.utils.batch_processing import BatchMetricProcessor
from alignment.utils.experiment_tracking import create_tracker, DummyTracker
from alignment.visualization import AlignmentVisualizer


def create_dummy_dataset(num_samples=1000, input_dim=784, num_classes=10):
    """Create a dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def create_model(input_dim=784, hidden_dims=[512, 256, 128], num_classes=10):
    """Create a simple feedforward model."""
    layers = []
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        ])
        prev_dim = hidden_dim
    
    layers.append(nn.Linear(prev_dim, num_classes))
    
    return nn.Sequential(*layers)


def main():
    print("="*80)
    print("ADVANCED ALIGNMENT ANALYSIS DEMO")
    print("="*80)
    
    # Configuration
    config = {
        'model': 'feedforward',
        'hidden_dims': [512, 256, 128],
        'metrics': ['rayleigh_quotient', 'mutual_information_gaussian', 
                   'node_correlation', 'weight_cosine_similarity'],
        'batch_size': 64,
        'num_batches': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create output directory
    output_dir = Path("results/advanced_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Initialize experiment tracking
    print("\n1. Setting up experiment tracking...")
    try:
        # Try to use WandB, fall back to TensorBoard, then to dummy
        try:
            tracker = create_tracker(
                'wandb',
                experiment_name="alignment_advanced_demo",
                config=config,
                project="alignment-demos"
            )
            print("   ✓ Using Weights & Biases tracking")
        except:
            try:
                tracker = create_tracker(
                    'tensorboard',
                    experiment_name="alignment_advanced_demo",
                    config=config,
                    log_dir=str(output_dir / "tensorboard")
                )
                print("   ✓ Using TensorBoard tracking")
            except:
                tracker = DummyTracker()
                print("   ! No tracking backend available, using dummy tracker")
    except Exception as e:
        print(f"   ! Error setting up tracking: {e}")
        tracker = DummyTracker()
    
    # 2. Create model and dataset
    print("\n2. Creating model and dataset...")
    model = create_model()
    wrapped_model = ModelWrapper(model)
    print(f"   ✓ Model created with layers: {wrapped_model.tracked_layers}")
    
    dataset = create_dummy_dataset()
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    print(f"   ✓ Dataset created with {len(dataset)} samples")
    
    # 3. Initialize metrics
    print("\n3. Initializing metrics...")
    metrics = {}
    for metric_name in config['metrics']:
        metric_class = METRIC_REGISTRY.get(metric_name)
        if metric_class:
            metrics[metric_name] = metric_class()
            print(f"   ✓ {metric_name}")
        else:
            print(f"   ✗ {metric_name} not found")
    
    # 4. Batch processing with progress tracking
    print("\n4. Computing metrics with batch processing...")
    processor = BatchMetricProcessor(
        device=torch.device(config['device']),
        max_memory_gb=4.0,
        show_progress=True
    )
    
    # Move model to device
    wrapped_model.model.to(config['device'])
    
    # Process dataset
    results = processor.process_dataset(
        wrapped_model,
        dataloader,
        metrics,
        num_batches=config['num_batches'],
        accumulation_strategy='average'
    )
    
    # 5. Log results to experiment tracker
    print("\n5. Logging results to experiment tracker...")
    tracker.log_alignment_scores(results, step=0)
    
    # Log some summary metrics
    total_neurons = 0
    total_score = 0
    for layer_name, layer_metrics in results.items():
        if 'rayleigh_quotient' in layer_metrics:
            scores = layer_metrics['rayleigh_quotient']
            total_neurons += len(scores)
            total_score += scores.sum().item()
    
    if total_neurons > 0:
        tracker.log_metrics({
            'average_alignment_score': total_score / total_neurons,
            'total_neurons': total_neurons
        }, step=0)
    
    # 6. Create visualizations
    print("\n6. Creating visualizations...")
    visualizer = AlignmentVisualizer(figsize=(12, 8))
    
    # Prepare data for visualization
    viz_results = {
        'layer_scores': {},
        'heatmap_data': results
    }
    
    # Reorganize data by metric
    for layer_name, layer_metrics in results.items():
        for metric_name, scores in layer_metrics.items():
            if metric_name not in viz_results['layer_scores']:
                viz_results['layer_scores'][metric_name] = {}
            viz_results['layer_scores'][metric_name][layer_name] = scores
    
    # Create visualization report
    visualizer.create_report(
        viz_results,
        output_dir=str(output_dir),
        experiment_name="Advanced Alignment Analysis"
    )
    print(f"   ✓ Visualizations saved to {output_dir}")
    
    # Log a sample visualization to tracker
    if 'rayleigh_quotient' in viz_results['layer_scores']:
        import matplotlib.pyplot as plt
        fig = visualizer.plot_layer_scores(
            viz_results['layer_scores']['rayleigh_quotient'],
            'Rayleigh Quotient'
        )
        
        # Convert figure to numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        tracker.log_image('rayleigh_quotient_distribution', img_array, step=0)
        plt.close(fig)
    
    # 7. Demonstrate pruning with visualization
    print("\n7. Running pruning experiment...")
    pruning_results = {}
    pruning_ratios = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    
    for ratio in pruning_ratios:
        # Get scores for first layer
        layer_name = wrapped_model.tracked_layers[0]
        if 'rayleigh_quotient' in results[layer_name]:
            scores = results[layer_name]['rayleigh_quotient']
            
            # Create mask based on scores
            threshold = torch.quantile(scores, ratio)
            mask = scores > threshold
            
            # Simulate accuracy (in real scenario, you'd evaluate the model)
            simulated_accuracy = 0.95 * (1 - ratio * 0.8)  # Degrades with pruning
            simulated_loss = 0.05 + ratio * 0.5
            
            pruning_results[ratio] = {
                'accuracy': simulated_accuracy,
                'loss': simulated_loss,
                'neurons_kept': mask.sum().item(),
                'neurons_total': len(mask)
            }
            
            # Log to tracker
            tracker.log_metrics({
                f'pruning/accuracy_ratio_{ratio}': simulated_accuracy,
                f'pruning/loss_ratio_{ratio}': simulated_loss,
                f'pruning/kept_ratio_{ratio}': mask.sum().item() / len(mask)
            }, step=int(ratio * 10))
    
    # Visualize pruning results
    fig = visualizer.plot_pruning_analysis(pruning_results)
    fig.savefig(output_dir / "pruning_analysis_custom.png")
    plt.close(fig)
    
    # 8. Performance comparison
    print("\n8. Performance Summary:")
    print(f"   - Total samples processed: {config['num_batches'] * config['batch_size']}")
    print(f"   - Number of metrics computed: {len(metrics)}")
    print(f"   - Number of layers analyzed: {len(wrapped_model.tracked_layers)}")
    print(f"   - Output directory: {output_dir}")
    
    # Finish tracking
    tracker.finish()
    
    print("\n✅ Advanced analysis complete!")
    print(f"   Results saved to: {output_dir}")
    print(f"   To view TensorBoard logs: tensorboard --logdir {output_dir}/tensorboard")


if __name__ == "__main__":
    main() 