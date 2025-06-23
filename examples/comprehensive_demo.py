#!/usr/bin/env python3
"""
Comprehensive demonstration of the alignment module features.

This script demonstrates:
1. All available metrics
2. Pruning strategies
3. Batch processing for large datasets
4. Visualization capabilities
5. Experiment tracking integration
6. Advanced analysis techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import alignment modules
from alignment.core import ModelWrapper
from alignment.metrics import METRIC_REGISTRY
from alignment.utils.batch_processing import BatchMetricProcessor, compute_metrics_parallel
from alignment.utils.pruning import PruningUtilities, PruningConfig, create_pruning_schedule
from alignment.utils.experiment_tracking import create_tracker
from alignment.visualization.alignment_plots import AlignmentVisualizer


class DemoModel(nn.Module):
    """Simple model for demonstration."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_synthetic_dataset(n_samples=1000, input_dim=10, n_classes=5):
    """Create synthetic dataset for demonstration."""
    # Generate random data
    X = torch.randn(n_samples, input_dim)
    # Create labels with some structure
    y = torch.zeros(n_samples, dtype=torch.long)
    for i in range(n_classes):
        mask = torch.rand(n_samples) < (1.0 / n_classes)
        y[mask] = i
    
    dataset = TensorDataset(X, y)
    return dataset


def train_model(model, dataloader, epochs=10, device='cpu'):
    """Train the model."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 2 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model


def evaluate_model(model, dataloader, device='cpu'):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = correct / total
    return accuracy


def demonstrate_metrics(model, dataloader, device='cpu'):
    """Demonstrate all available metrics."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING ALL METRICS")
    logger.info("=" * 60)
    
    # Wrap model
    wrapper = ModelWrapper(
        model,
        tracked_layers=['fc1', 'fc2'],
        device=device
    )
    
    # Create metric instances
    metrics = {}
    for metric_name in METRIC_REGISTRY:
        try:
            metrics[metric_name] = METRIC_REGISTRY[metric_name]()
            logger.info(f"Created metric: {metric_name}")
        except Exception as e:
            logger.warning(f"Could not create metric {metric_name}: {e}")
    
    # Compute metrics for a single batch
    inputs, _ = next(iter(dataloader))
    inputs = inputs.to(device)
    
    outputs, activations = wrapper.forward_with_activations(inputs)
    weights = wrapper.get_layer_weights()
    
    logger.info("\nComputing metrics for each layer:")
    for layer_name in wrapper.tracked_layers:
        logger.info(f"\nLayer: {layer_name}")
        
        layer_inputs = activations.get(f"{layer_name}_input")
        layer_weights = weights.get(layer_name)
        layer_outputs = activations.get(f"{layer_name}_output", outputs)
        
        for metric_name, metric in metrics.items():
            try:
                scores = metric.compute(
                    inputs=layer_inputs,
                    weights=layer_weights,
                    outputs=layer_outputs
                )
                
                if scores.numel() == 1:
                    logger.info(f"  {metric_name}: {scores.item():.4f}")
                else:
                    logger.info(f"  {metric_name}: mean={scores.mean().item():.4f}, std={scores.std().item():.4f}")
            except Exception as e:
                logger.debug(f"  {metric_name}: Error - {e}")


def demonstrate_pruning(model, train_loader, test_loader, device='cpu'):
    """Demonstrate various pruning strategies."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATING PRUNING STRATEGIES")
    logger.info("=" * 60)
    
    # Clone model for each pruning strategy
    strategies = ['magnitude', 'random', 'structured']
    results = {}
    
    for strategy in strategies:
        logger.info(f"\n--- {strategy.upper()} PRUNING ---")
        
        # Clone model
        pruned_model = DemoModel().to(device)
        pruned_model.load_state_dict(model.state_dict())
        
        # Initial accuracy
        initial_acc = evaluate_model(pruned_model, test_loader, device)
        logger.info(f"Initial accuracy: {initial_acc:.4f}")
        
        # Apply pruning to each layer
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                if strategy == 'magnitude':
                    mask = PruningUtilities.get_pruning_mask_magnitude(
                        module.weight.data, amount=0.5
                    )
                elif strategy == 'random':
                    mask = PruningUtilities.get_pruning_mask_random(
                        module.weight.data, amount=0.5
                    )
                elif strategy == 'structured':
                    PruningUtilities.structured_pruning(
                        module, amount=0.3, dim=0
                    )
                    continue  # structured pruning applies its own mask
                
                PruningUtilities.apply_pruning_mask(module, mask)
                sparsity = PruningUtilities.get_sparsity(module)
                logger.info(f"  {name}: sparsity = {sparsity:.2%}")
        
        # Final accuracy
        final_acc = evaluate_model(pruned_model, test_loader, device)
        logger.info(f"Final accuracy: {final_acc:.4f} (drop: {initial_acc - final_acc:.4f})")
        
        # Store results
        results[strategy] = {
            'initial_accuracy': initial_acc,
            'final_accuracy': final_acc,
            'sparsity': PruningUtilities.get_model_sparsity(pruned_model)
        }
    
    # Demonstrate iterative pruning
    logger.info("\n--- ITERATIVE MAGNITUDE PRUNING ---")
    iterative_model = DemoModel().to(device)
    iterative_model.load_state_dict(model.state_dict())
    
    accuracies = PruningUtilities.iterative_magnitude_pruning(
        iterative_model,
        amount=0.8,
        iterations=5,
        dataloader=train_loader,
        fine_tune_epochs=2,
        optimizer=optim.Adam(iterative_model.parameters(), lr=0.001),
        loss_fn=nn.CrossEntropyLoss()
    )
    
    logger.info(f"Iterative pruning accuracies: {accuracies}")
    
    return results


def demonstrate_batch_processing(model, dataset, device='cpu'):
    """Demonstrate batch processing capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATING BATCH PROCESSING")
    logger.info("=" * 60)
    
    # Create large dataset
    large_dataset = TensorDataset(
        torch.cat([dataset.tensors[0] for _ in range(10)]),  # 10x larger
        torch.cat([dataset.tensors[1] for _ in range(10)])
    )
    
    dataloader = DataLoader(large_dataset, batch_size=32, shuffle=False)
    
    # Wrap model
    wrapper = ModelWrapper(model, tracked_layers=['fc1', 'fc2'], device=device)
    
    # Select a few metrics
    metrics = {
        'rayleigh_quotient': METRIC_REGISTRY['rayleigh_quotient'](),
        'mutual_information': METRIC_REGISTRY['mutual_information'](),
        'weight_cosine_similarity': METRIC_REGISTRY['weight_cosine_similarity']()
    }
    
    # Regular batch processing
    logger.info("\nRegular batch processing:")
    processor = BatchMetricProcessor(device=device, show_progress=True)
    results = processor.process_dataset(
        wrapper, dataloader, metrics, num_batches=10
    )
    
    for layer_name, layer_results in results.items():
        logger.info(f"\n{layer_name}:")
        for metric_name, scores in layer_results.items():
            logger.info(f"  {metric_name}: shape={scores.shape}, mean={scores.mean():.4f}")
    
    # Parallel processing (if multiple GPUs available)
    if torch.cuda.device_count() > 1:
        logger.info("\nParallel processing across GPUs:")
        parallel_results = compute_metrics_parallel(
            wrapper, dataloader, metrics, num_workers=2
        )
        logger.info("Parallel processing completed")
    else:
        logger.info("\nParallel processing skipped (single GPU/CPU)")
    
    return results


def demonstrate_visualization(results, output_dir):
    """Demonstrate visualization capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATING VISUALIZATIONS")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    visualizer = AlignmentVisualizer()
    
    # Create sample data for visualizations
    # 1. Score distributions
    score_data = {
        'Layer1': np.random.randn(100) + 2,
        'Layer2': np.random.randn(100) + 1,
        'Layer3': np.random.randn(100)
    }
    
    visualizer.plot_score_distribution(
        score_data,
        title="Sample Score Distributions",
        save_path=output_dir / "score_distributions.png"
    )
    logger.info("Created score distribution plot")
    
    # 2. Metric heatmap
    metric_matrix = np.random.rand(5, 10)
    visualizer.plot_metric_heatmap(
        metric_matrix,
        title="Sample Metric Heatmap",
        xlabel="Neurons",
        ylabel="Samples",
        save_path=output_dir / "metric_heatmap.png"
    )
    logger.info("Created metric heatmap")
    
    # 3. Pruning analysis
    pruning_results = {
        0.0: {'accuracy': 0.95, 'loss': 0.15},
        0.1: {'accuracy': 0.94, 'loss': 0.18},
        0.3: {'accuracy': 0.91, 'loss': 0.25},
        0.5: {'accuracy': 0.85, 'loss': 0.35},
        0.7: {'accuracy': 0.75, 'loss': 0.50},
        0.9: {'accuracy': 0.60, 'loss': 0.75}
    }
    
    visualizer.plot_pruning_analysis(
        pruning_results,
        metrics=['accuracy', 'loss'],
        save_path=output_dir / "pruning_analysis.png"
    )
    logger.info("Created pruning analysis plot")
    
    # 4. Generate comprehensive report
    all_results = {
        'metrics': results,
        'pruning_results': pruning_results,
        'model_info': {
            'architecture': 'DemoModel',
            'parameters': 285,
            'layers': 2
        }
    }
    
    report_path = visualizer.generate_report(
        all_results,
        output_dir=output_dir,
        title="Alignment Analysis Report"
    )
    logger.info(f"Generated comprehensive report: {report_path}")


def demonstrate_experiment_tracking(model, dataloader, device='cpu'):
    """Demonstrate experiment tracking integration."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATING EXPERIMENT TRACKING")
    logger.info("=" * 60)
    
    # Create tracker (using base tracker for demo)
    tracker = create_tracker(
        'tensorboard',
        experiment_name='alignment_demo',
        config={
            'model': 'DemoModel',
            'metrics': ['rayleigh_quotient', 'mutual_information'],
            'device': str(device)
        },
        log_dir='./demo_runs'
    )
    
    # Wrap model
    wrapper = ModelWrapper(model, tracked_layers=['fc1', 'fc2'], device=device)
    
    # Track metrics over multiple steps
    metrics = {
        'rayleigh_quotient': METRIC_REGISTRY['rayleigh_quotient'](),
        'mutual_information': METRIC_REGISTRY['mutual_information']()
    }
    
    for step in range(5):
        # Get batch
        inputs, _ = next(iter(dataloader))
        inputs = inputs.to(device)
        
        # Compute metrics
        outputs, activations = wrapper.forward_with_activations(inputs)
        weights = wrapper.get_layer_weights()
        
        step_metrics = {}
        for layer_name in wrapper.tracked_layers:
            layer_inputs = activations.get(f"{layer_name}_input")
            layer_weights = weights.get(layer_name)
            layer_outputs = activations.get(f"{layer_name}_output", outputs)
            
            for metric_name, metric in metrics.items():
                scores = metric.compute(
                    inputs=layer_inputs,
                    weights=layer_weights,
                    outputs=layer_outputs
                )
                
                key = f"{layer_name}/{metric_name}"
                step_metrics[key] = scores.mean().item()
        
        # Log metrics
        tracker.log_metrics(step_metrics, step=step)
        logger.info(f"Step {step}: logged {len(step_metrics)} metrics")
    
    # Log final scores as histograms
    if hasattr(tracker, 'log_alignment_scores'):
        layer_scores = {}
        for layer_name in wrapper.tracked_layers:
            layer_scores[layer_name] = {
                'rayleigh_quotient': torch.randn(100),  # Demo scores
                'mutual_information': torch.rand(100) * 2
            }
        tracker.log_alignment_scores(layer_scores)
    
    tracker.finish()
    logger.info("Experiment tracking completed")


def demonstrate_advanced_features():
    """Demonstrate advanced features like pruning schedules."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATING ADVANCED FEATURES")
    logger.info("=" * 60)
    
    # Create pruning schedules
    schedules = {
        'linear': create_pruning_schedule(0.0, 0.9, 0, 100, 10, 'linear'),
        'polynomial': create_pruning_schedule(0.0, 0.9, 0, 100, 10, 'polynomial'),
        'exponential': create_pruning_schedule(0.0, 0.9, 0, 100, 10, 'exponential')
    }
    
    logger.info("\nPruning schedules at different steps:")
    steps = [0, 25, 50, 75, 100]
    for schedule_type, schedule_fn in schedules.items():
        sparsities = [schedule_fn(step) for step in steps]
        logger.info(f"{schedule_type}: {[f'{s:.2f}' for s in sparsities]}")
    
    # Demonstrate GPU memory monitoring
    if torch.cuda.is_available():
        logger.info("\nGPU Memory usage:")
        logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def main():
    """Run comprehensive demonstration."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset and model
    logger.info("\nCreating synthetic dataset and model...")
    dataset = create_synthetic_dataset(n_samples=1000)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = DemoModel()
    logger.info(f"Model architecture: {model}")
    
    # Train model
    logger.info("\nTraining model...")
    model = train_model(model, train_loader, epochs=10, device=device)
    
    # Run demonstrations
    demonstrate_metrics(model, train_loader, device)
    pruning_results = demonstrate_pruning(model, train_loader, test_loader, device)
    batch_results = demonstrate_batch_processing(model, dataset, device)
    demonstrate_visualization(batch_results, output_dir='./demo_outputs')
    demonstrate_experiment_tracking(model, train_loader, device)
    demonstrate_advanced_features()
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 60)
    logger.info("\nKey takeaways:")
    logger.info("1. The alignment module provides 17+ metrics for analyzing neural networks")
    logger.info("2. Multiple pruning strategies are available with fine-grained control")
    logger.info("3. Batch processing enables efficient computation on large datasets")
    logger.info("4. Visualization tools help interpret results")
    logger.info("5. Experiment tracking integrations support reproducible research")
    logger.info("\nCheck ./demo_outputs/ for generated visualizations")
    logger.info("Check ./demo_runs/ for TensorBoard logs (run: tensorboard --logdir ./demo_runs)")


if __name__ == "__main__":
    main() 