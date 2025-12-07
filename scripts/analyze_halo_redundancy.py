#!/usr/bin/env python3
"""
Analyze redundancy patterns between halo and non-halo neurons.

This script computes pairwise correlations/redundancy between:
1. Both neurons in halo (halo-halo)
2. Both neurons NOT in halo (non-halo-non-halo)  
3. One neuron in halo, one not (halo-non-halo / cross-group)

This helps understand whether:
- Halo neurons are indeed redundant with each other
- Non-halo neurons are also redundant (suggesting they too could be pruned safely)
- Halo vs non-halo neurons carry independent information
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compute_redundancy_matrix(activations: torch.Tensor) -> torch.Tensor:
    """Compute pairwise |correlation| matrix."""
    # Center
    centered = activations - activations.mean(dim=0, keepdim=True)
    # Normalize
    std = centered.std(dim=0, keepdim=True)
    std = torch.where(std > 1e-8, std, torch.ones_like(std))
    normalized = centered / std
    # Correlation
    corr = (normalized.T @ normalized) / (activations.shape[0] - 1)
    corr = torch.clamp(corr, -1, 1)
    return torch.abs(corr)


def analyze_layer_redundancy(
    model,
    tokenizer,
    layer_idx: int,
    calibration_texts: list,
    supernode_fraction: float = 0.01,
    halo_fraction: float = 0.1,
    device: str = "cuda",
    sample_size: int = 1000,  # Sample for efficiency
):
    """Analyze redundancy patterns for a single layer."""
    
    # Find the layer
    hf_model = model
    layer_name = f"model.layers.{layer_idx}.mlp.down_proj"
    
    # Get down_proj weight
    down_proj = None
    for name, module in hf_model.named_modules():
        if f"layers.{layer_idx}.mlp.down_proj" in name:
            down_proj = module.weight.data.float()  # [hidden_dim, intermediate_dim]
            break
    
    if down_proj is None:
        print(f"Could not find down_proj for layer {layer_idx}")
        return None
    
    hidden_dim, intermediate_dim = down_proj.shape
    
    # Step 1: Identify supernodes (by weight magnitude as proxy for activation power)
    neuron_magnitude = down_proj.abs().sum(dim=0)  # [intermediate_dim]
    num_supernodes = max(1, int(supernode_fraction * intermediate_dim))
    _, supernode_indices = torch.topk(neuron_magnitude, num_supernodes)
    supernode_mask = torch.zeros(intermediate_dim, dtype=torch.bool)
    supernode_mask[supernode_indices] = True
    
    # Step 2: Compute connection strength to supernodes
    supernode_weights = down_proj[:, supernode_indices]  # [hidden_dim, num_supernodes]
    
    # For intermediate neurons: connection = how much their outputs influence hidden dims that supernodes also influence
    # Simpler: use sum of absolute weights in down_proj row as proxy
    connection_strength = down_proj.abs().sum(dim=0)  # [intermediate_dim]
    
    # Step 3: Define halo (excluding supernodes)
    non_supernode_indices = (~supernode_mask).nonzero(as_tuple=True)[0]
    non_supernode_connection = connection_strength[non_supernode_indices]
    
    num_halo = max(1, int(halo_fraction * len(non_supernode_indices)))
    _, halo_relative_indices = torch.topk(non_supernode_connection, num_halo)
    halo_indices = non_supernode_indices[halo_relative_indices]
    
    halo_mask = torch.zeros(intermediate_dim, dtype=torch.bool)
    halo_mask[halo_indices] = True
    
    # Non-halo = not supernode and not halo
    non_halo_mask = ~supernode_mask & ~halo_mask
    non_halo_indices = non_halo_mask.nonzero(as_tuple=True)[0]
    
    print(f"Layer {layer_idx}: {num_supernodes} supernodes, {len(halo_indices)} halo, {len(non_halo_indices)} non-halo")
    
    # Step 4: Capture activations
    activations = []
    
    def capture_hook(module, inputs, outputs):
        if inputs and inputs[0] is not None:
            inp = inputs[0].detach().float()
            if inp.ndim == 3:
                inp = inp.reshape(-1, inp.shape[-1])
            activations.append(inp.cpu())
    
    hook_handle = None
    for name, module in hf_model.named_modules():
        if f"layers.{layer_idx}.mlp.down_proj" in name:
            hook_handle = module.register_forward_hook(capture_hook)
            break
    
    if hook_handle is None:
        print(f"Could not hook layer {layer_idx}")
        return None
    
    hf_model.eval()
    with torch.no_grad():
        for text in tqdm(calibration_texts[:20], desc=f"Layer {layer_idx} activations"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                hf_model(**inputs)
            except Exception as e:
                pass
    
    hook_handle.remove()
    
    if not activations:
        print(f"No activations captured for layer {layer_idx}")
        return None
    
    all_acts = torch.cat(activations, dim=0)  # [N, intermediate_dim]
    print(f"  Captured {all_acts.shape[0]} activation samples")
    
    # Step 5: Sample for efficiency
    if all_acts.shape[0] > sample_size:
        indices = torch.randperm(all_acts.shape[0])[:sample_size]
        all_acts = all_acts[indices]
    
    # Step 6: Compute full correlation matrix
    print("  Computing correlation matrix...")
    corr_matrix = compute_redundancy_matrix(all_acts)
    corr_matrix.fill_diagonal_(0)  # Exclude self-correlations
    
    # Step 7: Extract correlations for different groups
    halo_idx_np = halo_indices.numpy()
    non_halo_idx_np = non_halo_indices.numpy()
    
    # Sample indices for efficiency (if too many)
    max_per_group = 500
    if len(halo_idx_np) > max_per_group:
        halo_idx_np = np.random.choice(halo_idx_np, max_per_group, replace=False)
    if len(non_halo_idx_np) > max_per_group:
        non_halo_idx_np = np.random.choice(non_halo_idx_np, max_per_group, replace=False)
    
    # Halo-Halo correlations
    halo_halo_corr = corr_matrix[np.ix_(halo_idx_np, halo_idx_np)]
    halo_halo_values = halo_halo_corr[torch.triu(torch.ones_like(halo_halo_corr), diagonal=1).bool()].numpy()
    
    # Non-halo - Non-halo correlations
    non_halo_non_halo_corr = corr_matrix[np.ix_(non_halo_idx_np, non_halo_idx_np)]
    non_halo_values = non_halo_non_halo_corr[torch.triu(torch.ones_like(non_halo_non_halo_corr), diagonal=1).bool()].numpy()
    
    # Cross-group: halo - non-halo correlations
    cross_corr = corr_matrix[np.ix_(halo_idx_np, non_halo_idx_np)]
    cross_values = cross_corr.flatten().numpy()
    
    results = {
        "layer_idx": layer_idx,
        "num_supernodes": num_supernodes,
        "num_halo": len(halo_idx_np),
        "num_non_halo": len(non_halo_idx_np),
        "halo_halo": {
            "mean": float(np.mean(halo_halo_values)),
            "std": float(np.std(halo_halo_values)),
            "median": float(np.median(halo_halo_values)),
            "values": halo_halo_values.tolist()[:5000],  # Limit for storage
        },
        "non_halo_non_halo": {
            "mean": float(np.mean(non_halo_values)),
            "std": float(np.std(non_halo_values)),
            "median": float(np.median(non_halo_values)),
            "values": non_halo_values.tolist()[:5000],
        },
        "cross_group": {
            "mean": float(np.mean(cross_values)),
            "std": float(np.std(cross_values)),
            "median": float(np.median(cross_values)),
            "values": cross_values.tolist()[:5000],
        },
    }
    
    return results


def plot_redundancy_comparison(results_list: list, output_path: str):
    """Create comparison plots for redundancy patterns."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Histogram comparison for all layers combined
    ax = axes[0, 0]
    all_halo_halo = []
    all_non_halo = []
    all_cross = []
    
    for r in results_list:
        all_halo_halo.extend(r["halo_halo"]["values"])
        all_non_halo.extend(r["non_halo_non_halo"]["values"])
        all_cross.extend(r["cross_group"]["values"])
    
    bins = np.linspace(0, 1, 50)
    ax.hist(all_halo_halo, bins=bins, alpha=0.5, label=f'Halo-Halo (μ={np.mean(all_halo_halo):.3f})', density=True, color='#e74c3c')
    ax.hist(all_non_halo, bins=bins, alpha=0.5, label=f'Non-halo (μ={np.mean(all_non_halo):.3f})', density=True, color='#3498db')
    ax.hist(all_cross, bins=bins, alpha=0.5, label=f'Cross-group (μ={np.mean(all_cross):.3f})', density=True, color='#2ecc71')
    ax.set_xlabel("|Correlation|")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise |Correlation| Distribution (All Layers)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Mean redundancy by layer
    ax = axes[0, 1]
    layers = [r["layer_idx"] for r in results_list]
    halo_means = [r["halo_halo"]["mean"] for r in results_list]
    non_halo_means = [r["non_halo_non_halo"]["mean"] for r in results_list]
    cross_means = [r["cross_group"]["mean"] for r in results_list]
    
    x = np.arange(len(layers))
    width = 0.25
    ax.bar(x - width, halo_means, width, label='Halo-Halo', color='#e74c3c', alpha=0.8)
    ax.bar(x, non_halo_means, width, label='Non-halo', color='#3498db', alpha=0.8)
    ax.bar(x + width, cross_means, width, label='Cross-group', color='#2ecc71', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean |Correlation|")
    ax.set_title("Mean Redundancy by Layer")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Box plots for each group
    ax = axes[1, 0]
    data_to_plot = [
        all_halo_halo[:2000],  # Sample for visibility
        all_non_halo[:2000],
        all_cross[:2000],
    ]
    bp = ax.boxplot(data_to_plot, labels=['Halo-Halo', 'Non-halo', 'Cross-group'], patch_artist=True)
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("|Correlation|")
    ax.set_title("Redundancy Distribution by Group")
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = """
REDUNDANCY ANALYSIS SUMMARY
==========================

Group             | Mean    | Std     | Median
------------------|---------|---------|--------
Halo-Halo         | {:.4f}  | {:.4f}  | {:.4f}
Non-halo-Non-halo | {:.4f}  | {:.4f}  | {:.4f}
Cross (Halo-NonH) | {:.4f}  | {:.4f}  | {:.4f}

KEY INSIGHTS:
-------------
• If Halo-Halo >> Non-halo: Halo neurons are more redundant
  → Current approach is correct
  
• If Non-halo ~ Halo: Non-halo neurons also redundant
  → Could prune more aggressively
  
• If Cross << within-group: Groups carry different info
  → Important to distinguish halo vs non-halo

• If Halo-Halo high, Cross low: Halo is an "echo chamber"
  → Safe to prune redundant halo members
""".format(
        np.mean(all_halo_halo), np.std(all_halo_halo), np.median(all_halo_halo),
        np.mean(all_non_halo), np.std(all_non_halo), np.median(all_non_halo),
        np.mean(all_cross), np.std(all_cross), np.median(all_cross),
    )
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze halo vs non-halo redundancy patterns")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help="Model name or path")
    parser.add_argument("--layers", type=str, default="8,12,16,20,24",
                        help="Comma-separated layer indices to analyze")
    parser.add_argument("--supernode-fraction", type=float, default=0.01,
                        help="Fraction of neurons to consider as supernodes")
    parser.add_argument("--halo-fraction", type=float, default=0.1,
                        help="Fraction of non-supernodes to consider as halo")
    parser.add_argument("--output-dir", type=str, default="./halo_analysis",
                        help="Output directory for plots and results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    
    # Calibration texts
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require careful tuning of hyperparameters.",
        "In the beginning, there was darkness, and then there was light.",
        "The stock market experienced significant volatility today.",
        "Scientists have discovered a new species of deep-sea fish.",
        "The conference will be held in San Francisco next month.",
        "Programming languages continue to evolve with new features.",
        "Climate change poses significant challenges for future generations.",
        "The recipe calls for two cups of flour and one cup of sugar.",
        "Historical documents reveal interesting details about past civilizations.",
    ] * 5  # Repeat for more samples
    
    # Analyze layers
    layers = [int(l.strip()) for l in args.layers.split(",")]
    results_list = []
    
    for layer_idx in layers:
        print(f"\nAnalyzing layer {layer_idx}...")
        result = analyze_layer_redundancy(
            model, tokenizer, layer_idx, calibration_texts,
            supernode_fraction=args.supernode_fraction,
            halo_fraction=args.halo_fraction,
            device=args.device,
        )
        if result:
            results_list.append(result)
    
    # Save results
    results_path = os.path.join(args.output_dir, "halo_redundancy_results.json")
    with open(results_path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Create plots
    plot_path = os.path.join(args.output_dir, "halo_redundancy_comparison.png")
    plot_redundancy_comparison(results_list, plot_path)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_halo = [v for r in results_list for v in r["halo_halo"]["values"]]
    all_non_halo = [v for r in results_list for v in r["non_halo_non_halo"]["values"]]
    all_cross = [v for r in results_list for v in r["cross_group"]["values"]]
    
    print(f"Halo-Halo redundancy:     mean={np.mean(all_halo):.4f}, std={np.std(all_halo):.4f}")
    print(f"Non-halo redundancy:      mean={np.mean(all_non_halo):.4f}, std={np.std(all_non_halo):.4f}")
    print(f"Cross-group redundancy:   mean={np.mean(all_cross):.4f}, std={np.std(all_cross):.4f}")
    
    # Interpretation
    print("\nINTERPRETATION:")
    if np.mean(all_halo) > np.mean(all_non_halo) * 1.2:
        print("✓ Halo neurons are MORE redundant than non-halo → Current approach valid")
    elif np.mean(all_halo) < np.mean(all_non_halo) * 0.8:
        print("✗ Non-halo neurons are MORE redundant → Consider revising halo definition")
    else:
        print("≈ Similar redundancy in both groups → May need different selection criteria")
    
    if np.mean(all_cross) < np.mean(all_halo) * 0.8:
        print("✓ Cross-group correlation LOW → Groups carry different information")
    else:
        print("≈ Cross-group correlation similar → Information not well separated by halo")


if __name__ == "__main__":
    main()
