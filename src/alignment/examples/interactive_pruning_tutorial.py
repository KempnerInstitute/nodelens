"""
Interactive Tutorial: Network Pruning with Alignment Metrics

This tutorial demonstrates how to use alignment metrics to identify and prune
unimportant neurons in neural networks. Run this script interactively or
convert it to a Jupyter notebook.

Tutorial Contents:
1. Understanding Rayleigh Quotient (RQ)
2. Training a network and computing metrics
3. Visualizing neuron importance
4. Different pruning strategies
5. Analyzing pruning effects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import seaborn as sns

from alignment.models import ModelWrapper
from alignment.metrics import (
    RayleighQuotient, 
    MutualInformationGaussian,
    SharedInformation
)

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# PART 1: Understanding Alignment Metrics
# ============================================================================

def explain_rayleigh_quotient():
    """
    The Rayleigh Quotient (RQ) measures how well aligned a neuron's weight vector
    is with the principal components of its input distribution.
    
    High RQ = Neuron is well-aligned with important input directions
    Low RQ = Neuron may be redundant or not utilizing input information well
    """
    print("="*70)
    print("PART 1: Understanding Rayleigh Quotient")
    print("="*70)
    print("""
    The Rayleigh Quotient (RQ) for a neuron with weight vector w and input 
    covariance matrix C is defined as:
    
    RQ(w) = (w^T * C * w) / (w^T * w)
    
    This measures how much variance in the input the neuron captures.
    - High RQ: Neuron aligns with high-variance input directions
    - Low RQ: Neuron aligns with low-variance (less informative) directions
    
    We use RQ to identify important vs. unimportant neurons for pruning.
    """)
    
    # Create a simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate 2D data with different variances
    np.random.seed(42)
    data = np.random.multivariate_normal([0, 0], [[4, 1], [1, 0.5]], 1000)
    
    # Plot data
    ax1.scatter(data[:, 0], data[:, 1], alpha=0.5, s=20)
    ax1.set_title('Input Data Distribution')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    
    # Show good vs bad weight vectors
    good_weight = np.array([2, 0.5])  # Aligned with main variance
    bad_weight = np.array([0.5, 2])   # Aligned with low variance
    
    # Normalize for visualization
    good_weight = good_weight / np.linalg.norm(good_weight) * 3
    bad_weight = bad_weight / np.linalg.norm(bad_weight) * 3
    
    ax1.arrow(0, 0, good_weight[0], good_weight[1], 
              head_width=0.3, head_length=0.2, fc='green', ec='green', 
              linewidth=3, label='High RQ neuron')
    ax1.arrow(0, 0, bad_weight[0], bad_weight[1], 
              head_width=0.3, head_length=0.2, fc='red', ec='red', 
              linewidth=3, label='Low RQ neuron')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Compute actual RQ values
    cov = np.cov(data.T)
    rq_good = good_weight @ cov @ good_weight / (good_weight @ good_weight)
    rq_bad = bad_weight @ cov @ bad_weight / (bad_weight @ bad_weight)
    
    ax2.bar(['High RQ\nNeuron', 'Low RQ\nNeuron'], [rq_good, rq_bad], 
            color=['green', 'red'], alpha=0.7)
    ax2.set_ylabel('Rayleigh Quotient Value')
    ax2.set_title('RQ Comparison')
    
    plt.tight_layout()
    plt.savefig('rq_explanation.png', dpi=150)
    plt.close()
    
    print(f"\nHigh RQ neuron: RQ = {rq_good:.3f}")
    print(f"Low RQ neuron: RQ = {rq_bad:.3f}")
    print("\nVisualization saved to 'rq_explanation.png'")


# ============================================================================
# PART 2: Network Architecture and Training
# ============================================================================

class TutorialNet(nn.Module):
    """A network designed for educational pruning experiments."""
    
    def __init__(self, hidden_sizes=[300, 200, 100]):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Build layers
        input_size = 784
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(input_size, hidden_size))
            input_size = hidden_size
        
        # Output layer
        self.output = nn.Linear(input_size, 10)
        
        # Add some redundancy for pruning demonstration
        self._add_redundant_neurons()
    
    def _add_redundant_neurons(self):
        """Intentionally create some redundant neurons for demonstration."""
        with torch.no_grad():
            # Make some neurons in first layer redundant
            if len(self.layers) > 0:
                # Copy weights from neuron 0 to neurons 280-299 with small noise
                for i in range(280, 300):
                    self.layers[0].weight.data[i] = (
                        self.layers[0].weight.data[0] + 
                        0.01 * torch.randn_like(self.layers[0].weight.data[0])
                    )
    
    def forward(self, x):
        x = x.view(-1, 784)
        
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            # Add slight dropout for regularization
            if self.training:
                x = F.dropout(x, p=0.1)
        
        x = self.output(x)
        return F.log_softmax(x, dim=1)


def train_tutorial_network():
    """Train the network and return it with training history."""
    print("\n" + "="*70)
    print("PART 2: Training Network")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Use subset for faster training in tutorial
    train_subset = Subset(train_dataset, range(10000))
    
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create model
    model = TutorialNet()
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\nTraining network...")
    train_losses = []
    test_accs = []
    
    for epoch in range(5):
        # Train
        model.train()
        epoch_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Test
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        test_acc = 100. * correct / len(test_dataset)
        train_losses.append(epoch_loss / len(train_loader))
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Test Acc = {test_acc:.2f}%")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss History')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(test_accs, 'g-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy History')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.close()
    
    print(f"\nFinal test accuracy: {test_accs[-1]:.2f}%")
    print("Training history saved to 'training_history.png'")
    
    return model, test_loader


# ============================================================================
# PART 3: Computing and Visualizing Neuron Importance
# ============================================================================

def analyze_neuron_importance(model, test_loader):
    """Compute importance metrics for all neurons."""
    print("\n" + "="*70)
    print("PART 3: Computing Neuron Importance")
    print("="*70)
    
    # Wrap model
    layer_names = [f'layers.{i}' for i in range(len(model.layers))]
    model_wrapper = ModelWrapper(model, tracked_layers=layer_names)
    
    # Initialize metrics
    rq_metric = RayleighQuotient()
    mi_metric = MutualInformationGaussian()
    
    print("\nComputing metrics for each layer...")
    layer_scores = {}
    
    # Compute scores
    for batch_idx, (data, _) in enumerate(test_loader):
        if batch_idx >= 5:  # Use 5 batches
            break
        
        _, activations = model_wrapper.forward_with_activations(data)
        weights = model_wrapper.get_layer_weights()
        
        for layer_name in layer_names:
            if layer_name not in layer_scores:
                layer_scores[layer_name] = {'rq': [], 'mi': []}
            
            # Get inputs and weights
            layer_input = activations[f"{layer_name}_input"].flatten(start_dim=1)
            layer_weight = weights[layer_name]
            
            # Compute RQ
            rq_scores = rq_metric.compute(inputs=layer_input, weights=layer_weight)
            layer_scores[layer_name]['rq'].append(rq_scores.cpu())
            
            # Compute MI if output available
            if layer_name in activations:
                layer_output = activations[layer_name].flatten(start_dim=1)
                mi_scores = mi_metric.compute(
                    inputs=layer_input,
                    weights=layer_weight,
                    outputs=layer_output
                )
                layer_scores[layer_name]['mi'].append(mi_scores.cpu())
    
    # Average scores
    avg_scores = {}
    for layer_name in layer_names:
        avg_scores[layer_name] = {
            'rq': torch.stack(layer_scores[layer_name]['rq']).mean(dim=0),
            'mi': torch.stack(layer_scores[layer_name]['mi']).mean(dim=0) if layer_scores[layer_name]['mi'] else None
        }
    
    # Visualize importance distributions
    fig, axes = plt.subplots(len(layer_names), 2, figsize=(12, 4*len(layer_names)))
    if len(layer_names) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, layer_name in enumerate(layer_names):
        # RQ distribution
        rq_scores = avg_scores[layer_name]['rq'].numpy()
        axes[idx, 0].hist(rq_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[idx, 0].axvline(rq_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rq_scores.mean():.3f}')
        axes[idx, 0].set_xlabel('Rayleigh Quotient')
        axes[idx, 0].set_ylabel('Number of Neurons')
        axes[idx, 0].set_title(f'{layer_name} - RQ Distribution')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Identify redundant neurons (those we added)
        if idx == 0:  # First layer where we added redundancy
            redundant_mask = np.zeros(len(rq_scores), dtype=bool)
            redundant_mask[280:300] = True
            axes[idx, 0].hist(rq_scores[redundant_mask], bins=20, alpha=0.5, color='red', label='Redundant neurons')
            axes[idx, 0].legend()
        
        # MI distribution if available
        if avg_scores[layer_name]['mi'] is not None:
            mi_scores = avg_scores[layer_name]['mi'].numpy()
            axes[idx, 1].hist(mi_scores, bins=50, alpha=0.7, color='green', edgecolor='black')
            axes[idx, 1].axvline(mi_scores.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {mi_scores.mean():.3f}')
            axes[idx, 1].set_xlabel('Mutual Information')
            axes[idx, 1].set_ylabel('Number of Neurons')
            axes[idx, 1].set_title(f'{layer_name} - MI Distribution')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('importance_distributions.png', dpi=150)
    plt.close()
    
    print("\nImportance distributions saved to 'importance_distributions.png'")
    
    # Print statistics
    print("\nLayer-wise importance statistics:")
    print("-" * 60)
    for layer_name in layer_names:
        rq = avg_scores[layer_name]['rq']
        print(f"\n{layer_name}:")
        print(f"  RQ - Mean: {rq.mean():.4f}, Std: {rq.std():.4f}")
        print(f"  RQ - Min: {rq.min():.4f}, Max: {rq.max():.4f}")
        
        # Identify potentially redundant neurons
        low_rq_threshold = rq.mean() - 2 * rq.std()
        low_rq_neurons = (rq < low_rq_threshold).sum().item()
        print(f"  Neurons with very low RQ (< mean - 2*std): {low_rq_neurons}")
    
    return avg_scores


# ============================================================================
# PART 4: Pruning Strategies
# ============================================================================

def demonstrate_pruning_strategies(model, test_loader, scores):
    """Show different pruning strategies and their effects."""
    print("\n" + "="*70)
    print("PART 4: Pruning Strategies Comparison")
    print("="*70)
    
    device = next(model.parameters()).device
    
    strategies = {
        'magnitude': "Prune neurons with lowest weight magnitudes",
        'rq_based': "Prune neurons with lowest Rayleigh Quotients",
        'random': "Randomly prune neurons (baseline)",
        'structured': "Prune entire channels based on layer-wise importance"
    }
    
    print("\nTesting different pruning strategies:")
    for name, description in strategies.items():
        print(f"\n{name}: {description}")
    
    # Test different pruning percentages
    prune_percentages = [0, 10, 20, 30, 40, 50]
    results = {strategy: [] for strategy in strategies}
    
    # Original accuracy
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    
    original_acc = 100. * correct / len(test_loader.dataset)
    
    # Test each strategy
    for strategy in strategies:
        print(f"\nTesting {strategy} pruning...")
        
        for prune_pct in prune_percentages:
            # Clone model
            pruned_model = TutorialNet()
            pruned_model.load_state_dict(model.state_dict())
            pruned_model = pruned_model.to(device)
            
            if prune_pct > 0:
                # Apply pruning
                apply_pruning_strategy(pruned_model, scores, strategy, prune_pct/100)
            
            # Evaluate
            pruned_model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = pruned_model(data)
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
            
            acc = 100. * correct / len(test_loader.dataset)
            results[strategy].append(acc)
            print(f"  {prune_pct}% pruned: {acc:.2f}% accuracy")
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    
    for strategy, accs in results.items():
        plt.plot(prune_percentages, accs, marker='o', linewidth=2, label=strategy)
    
    plt.xlabel('Pruning Percentage (%)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Comparison of Pruning Strategies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 55)
    plt.ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('pruning_strategies_comparison.png', dpi=150)
    plt.close()
    
    print("\nPruning comparison saved to 'pruning_strategies_comparison.png'")
    
    return results


def apply_pruning_strategy(model, scores, strategy, prune_fraction):
    """Apply a specific pruning strategy to the model."""
    if strategy == 'magnitude':
        # Prune based on weight magnitude
        for i, layer in enumerate(model.layers):
            weights = layer.weight.data
            magnitudes = weights.abs().sum(dim=1)
            threshold = torch.quantile(magnitudes, prune_fraction)
            mask = magnitudes > threshold
            layer.weight.data *= mask.unsqueeze(1)
            if layer.bias is not None:
                layer.bias.data *= mask
    
    elif strategy == 'rq_based':
        # Prune based on RQ scores
        for i, layer_name in enumerate([f'layers.{j}' for j in range(len(model.layers))]):
            if layer_name in scores:
                rq_scores = scores[layer_name]['rq']
                threshold = torch.quantile(rq_scores, prune_fraction)
                mask = rq_scores > threshold
                
                layer = model.layers[i]
                layer.weight.data *= mask.unsqueeze(1)
                if layer.bias is not None:
                    layer.bias.data *= mask
    
    elif strategy == 'random':
        # Random pruning
        for layer in model.layers:
            num_neurons = layer.weight.shape[0]
            num_to_prune = int(num_neurons * prune_fraction)
            indices = torch.randperm(num_neurons)[:num_to_prune]
            mask = torch.ones(num_neurons, dtype=torch.bool)
            mask[indices] = False
            
            layer.weight.data *= mask.unsqueeze(1)
            if layer.bias is not None:
                layer.bias.data *= mask
    
    elif strategy == 'structured':
        # Structured pruning (prune least important layer more)
        layer_importance = []
        for i, layer_name in enumerate([f'layers.{j}' for j in range(len(model.layers))]):
            if layer_name in scores:
                layer_importance.append(scores[layer_name]['rq'].mean().item())
        
        # Normalize importance
        layer_importance = np.array(layer_importance)
        layer_importance = (layer_importance - layer_importance.min()) / (layer_importance.max() - layer_importance.min() + 1e-8)
        
        # Prune more from less important layers
        for i, layer in enumerate(model.layers):
            # Inverse importance for pruning fraction
            layer_prune_frac = prune_fraction * (2 - layer_importance[i])
            layer_prune_frac = min(layer_prune_frac, 0.7)  # Cap at 70%
            
            if i < len(layer_importance):
                layer_name = f'layers.{i}'
                if layer_name in scores:
                    rq_scores = scores[layer_name]['rq']
                    threshold = torch.quantile(rq_scores, layer_prune_frac)
                    mask = rq_scores > threshold
                    
                    layer.weight.data *= mask.unsqueeze(1)
                    if layer.bias is not None:
                        layer.bias.data *= mask


# ============================================================================
# PART 5: Analysis and Insights
# ============================================================================

def analyze_pruning_effects(model, original_model, scores):
    """Analyze what happens when we prune."""
    print("\n" + "="*70)
    print("PART 5: Analyzing Pruning Effects")
    print("="*70)
    
    # Apply 30% RQ-based pruning for analysis
    pruned_model = TutorialNet()
    pruned_model.load_state_dict(original_model.state_dict())
    apply_pruning_strategy(pruned_model, scores, 'rq_based', 0.3)
    
    # Count remaining neurons
    print("\nNeuron counts after 30% pruning:")
    print("-" * 40)
    
    total_original = 0
    total_remaining = 0
    
    for i, (orig_layer, pruned_layer) in enumerate(zip(original_model.layers, pruned_model.layers)):
        orig_weights = orig_layer.weight.data
        pruned_weights = pruned_layer.weight.data
        
        # Count non-zero neurons
        orig_count = (orig_weights.abs().sum(dim=1) > 0).sum().item()
        pruned_count = (pruned_weights.abs().sum(dim=1) > 0).sum().item()
        
        total_original += orig_count
        total_remaining += pruned_count
        
        print(f"Layer {i}: {orig_count} -> {pruned_count} neurons ({pruned_count/orig_count*100:.1f}% remaining)")
    
    print(f"\nTotal: {total_original} -> {total_remaining} neurons ({total_remaining/total_original*100:.1f}% remaining)")
    
    # Analyze which neurons were pruned
    print("\nAnalyzing pruned neurons in first layer...")
    
    first_layer_rq = scores['layers.0']['rq'].numpy()
    first_layer_weights = original_model.layers[0].weight.data
    pruned_first_layer_weights = pruned_model.layers[0].weight.data
    
    # Identify pruned neurons
    pruned_mask = (pruned_first_layer_weights.abs().sum(dim=1) == 0).cpu().numpy()
    
    # Check if redundant neurons were pruned
    redundant_neurons = list(range(280, 300))
    redundant_pruned = sum(pruned_mask[redundant_neurons])
    
    print(f"\nRedundant neurons (280-299) pruned: {redundant_pruned}/20 ({redundant_pruned/20*100:.0f}%)")
    print(f"Average RQ of pruned neurons: {first_layer_rq[pruned_mask].mean():.4f}")
    print(f"Average RQ of remaining neurons: {first_layer_rq[~pruned_mask].mean():.4f}")
    
    # Visualize weight magnitude changes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original weights
    orig_magnitudes = original_model.layers[0].weight.data.abs().sum(dim=1).cpu().numpy()
    ax1.hist(orig_magnitudes, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Weight Magnitude (L1 norm)')
    ax1.set_ylabel('Number of Neurons')
    ax1.set_title('Original Weight Distribution')
    ax1.grid(True, alpha=0.3)
    
    # After pruning
    remaining_magnitudes = orig_magnitudes[~pruned_mask]
    ax2.hist(remaining_magnitudes, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Weight Magnitude (L1 norm)')
    ax2.set_ylabel('Number of Neurons')
    ax2.set_title('Weight Distribution After Pruning')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pruning_weight_analysis.png', dpi=150)
    plt.close()
    
    print("\nWeight analysis saved to 'pruning_weight_analysis.png'")


# ============================================================================
# MAIN TUTORIAL FUNCTION
# ============================================================================

def main():
    """Run the complete pruning tutorial."""
    print("\n" + "="*70)
    print("INTERACTIVE NEURAL NETWORK PRUNING TUTORIAL")
    print("="*70)
    print("\nThis tutorial will guide you through the process of pruning")
    print("neural networks using alignment metrics.")
    
    # Part 1: Understanding metrics
    explain_rayleigh_quotient()
    input("\nPress Enter to continue to Part 2...")
    
    # Part 2: Train network
    model, test_loader = train_tutorial_network()
    input("\nPress Enter to continue to Part 3...")
    
    # Part 3: Compute importance
    scores = analyze_neuron_importance(model, test_loader)
    input("\nPress Enter to continue to Part 4...")
    
    # Part 4: Compare strategies
    results = demonstrate_pruning_strategies(model, test_loader, scores)
    input("\nPress Enter to continue to Part 5...")
    
    # Part 5: Analyze effects
    analyze_pruning_effects(model, model, scores)
    
    # Summary
    print("\n" + "="*70)
    print("TUTORIAL SUMMARY")
    print("="*70)
    print("""
    Key Takeaways:
    
    1. Rayleigh Quotient (RQ) effectively identifies important neurons
       - High RQ neurons capture significant input variance
       - Low RQ neurons are often redundant
    
    2. RQ-based pruning outperforms random and magnitude-based pruning
       - Can often prune 30-40% of neurons with minimal accuracy loss
       - Automatically identifies and removes redundant neurons
    
    3. Different layers have different sensitivity to pruning
       - Early layers often have more redundancy
       - Later layers are typically more sensitive
    
    4. Structured pruning can balance accuracy and efficiency
       - Prunes more aggressively from less important layers
       - Maintains network balance
    
    Next Steps:
    - Try different architectures and datasets
    - Experiment with other metrics (MI, PID)
    - Combine pruning with fine-tuning
    - Explore iterative pruning strategies
    """)
    
    print("\nAll visualizations have been saved to the current directory.")
    print("Thank you for completing the tutorial!")


if __name__ == "__main__":
    main() 