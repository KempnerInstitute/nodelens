"""
Attention Layer Pruning: Neuron-Level vs Head-Level

This example demonstrates TWO ways to prune attention layers:
1. NEURON-LEVEL: Prune individual neurons in Q/K/V/O projections (fine-grained)
2. HEAD-LEVEL: Prune entire heads (coarse-grained)

Key insight: Attention projections are Linear layers with neurons!
- Q projection: 4,096 neurons (organized as 32 heads × 128 dims)
- K projection: 4,096 neurons
- V projection: 4,096 neurons
- O projection: 4,096 neurons

You can prune at either granularity!
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from alignment.metrics import get_metric
from alignment.services import MaskOperations


def analyze_attention_structure(model):
    """Show the structure of attention layers."""
    print("=" * 80)
    print("Attention Layer Structure")
    print("=" * 80)

    # Get first attention layer
    attn_layer = model.transformer.h[0].attn if hasattr(model, 'transformer') else model.model.layers[0].self_attn

    # Find projection layers
    for name, module in attn_layer.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\n{name}:")
            print(f"  Weight shape: {module.weight.shape}")
            print(f"  Output neurons: {module.weight.shape[0]}")
            print(f"  Input dimension: {module.weight.shape[1]}")

            if module.weight.shape[0] % 32 == 0:  # Assuming 32 heads
                neurons_per_head = module.weight.shape[0] // 32
                print(f"  → Organized as: 32 heads × {neurons_per_head} dims/head")
                print(f"  → {module.weight.shape[0]} neurons total (can prune individually!)")


def neuron_level_pruning_demo(model, layer_idx=0):
    """
    Demonstrate NEURON-LEVEL pruning in attention.

    Treats Q/K/V/O projections as Linear layers and prunes individual neurons.
    """
    print("\n" + "=" * 80)
    print("NEURON-LEVEL Attention Pruning")
    print("=" * 80)

    # Get attention layer
    try:
        attn = model.transformer.h[layer_idx].attn  # GPT-2 style
        q_proj = attn.c_attn  # GPT-2 combines Q/K/V
        print("  Model type: GPT-2 (combined QKV projection)")
        separate_projections = False
    except:
        attn = model.model.layers[layer_idx].self_attn  # LLaMA style
        q_proj = attn.q_proj
        print("  Model type: LLaMA (separate Q/K/V/O projections)")
        separate_projections = True

    if not separate_projections:
        print("  Note: GPT-2 combines Q/K/V, will demonstrate on combined projection")
        target_proj = q_proj
        proj_name = "QKV combined"
    else:
        target_proj = q_proj
        proj_name = "Q projection"

    num_neurons = target_proj.weight.shape[0]
    hidden_dim = target_proj.weight.shape[1]

    print(f"\n{proj_name}:")
    print(f"  Shape: {target_proj.weight.shape}")
    print(f"  Total neurons: {num_neurons}")
    print(f"  Input dimension: {hidden_dim}")

    if num_neurons % 32 == 0:
        print(f"  Organized as: {num_neurons // 128} heads × 128 dims")

    # Create dummy inputs for scoring
    print("\nComputing per-neuron importance...")
    inputs = torch.randn(16, hidden_dim)  # [B, hidden_dim]
    outputs = target_proj(inputs)  # [B, num_neurons]

    # Compute per-neuron scores
    rq = get_metric('rayleigh_quotient')
    rq_scores = rq.compute(inputs, target_proj.weight)

    redundancy = get_metric('pairwise_redundancy_gaussian', mode='output_based', num_pairs=20)
    redundancy_scores = redundancy.compute(outputs=outputs)

    # Composite score
    composite = torch.log(rq_scores + 1e-8) - 0.4 * redundancy_scores

    # Show individual neuron analysis
    print(f"\n  Individual neuron breakdown (first 10 of {num_neurons}):")
    print(f"    {'Neuron':<8} {'RQ':<10} {'Redundancy':<12} {'Composite':<12}")
    print(f"    {'-'*50}")

    for i in range(min(10, num_neurons)):
        print(f"    {i:<8} {rq_scores[i]:.4f}    {redundancy_scores[i]:.4f}      {composite[i]:.4f}")

    # Prune neurons
    pruning_amount = 0.3  # 30%
    mask = MaskOperations.create_structured_mask(composite, amount=pruning_amount, mode='low')

    num_pruned = (~mask).sum().item()
    num_kept = mask.sum().item()

    print("\n  Pruning plan:")
    print(f"    Amount: {pruning_amount:.0%}")
    print(f"    Neurons pruned: {num_pruned}")
    print(f"    Neurons kept: {num_kept}")

    # Show which neurons are pruned
    pruned_indices = torch.where(~mask)[0]
    if len(pruned_indices) <= 20:
        print(f"    Pruned neuron indices: {pruned_indices.tolist()}")
    else:
        print(f"    Pruned neuron indices: {pruned_indices[:10].tolist()} ... (showing first 10)")

    # Apply pruning
    print("\n  Applying neuron-level mask...")
    target_proj.weight.data *= mask.unsqueeze(1).float()
    if target_proj.bias is not None:
        target_proj.bias.data *= mask.float()

    print(f"  ✓ {num_pruned} neurons zeroed out")
    print("  ✓ Can prune ANY subset of neurons (not constrained to heads)")

    return mask, composite


def head_level_pruning_demo(model, layer_idx=0):
    """
    Demonstrate HEAD-LEVEL pruning in attention.

    Aggregates neurons into heads and prunes entire heads.
    """
    print("\n" + "=" * 80)
    print("HEAD-LEVEL Attention Pruning")
    print("=" * 80)

    try:
        attn = model.model.layers[layer_idx].self_attn
        q_proj = attn.q_proj
        num_heads = 32  # LLaMA default
        head_dim = 128
    except:
        print("  Using GPT-2 (combined QKV, skipping head demo)")
        return None, None

    print("\n  Attention configuration:")
    print(f"    Number of heads: {num_heads}")
    print(f"    Dimension per head: {head_dim}")
    print(f"    Total dimensions: {num_heads * head_dim}")

    # Compute neuron-level scores first
    inputs = torch.randn(16, 4096)
    outputs = q_proj(inputs)

    redundancy = get_metric('pairwise_redundancy_gaussian', mode='output_based', num_pairs=20)
    neuron_scores = redundancy.compute(outputs=outputs)  # [4096]

    # Aggregate to head-level
    print(f"\n  Aggregating {neuron_scores.shape[0]} neurons into {num_heads} heads...")

    head_scores = []
    for head_idx in range(num_heads):
        start_idx = head_idx * head_dim
        end_idx = start_idx + head_dim

        # Average neuron scores within this head
        head_score = neuron_scores[start_idx:end_idx].mean()
        head_scores.append(head_score)

    head_scores = torch.tensor(head_scores)  # [32]

    # Show per-head scores
    print("\n  Per-head redundancy:")
    print(f"    {'Head':<6} {'Redundancy':<12} {'Assessment'}")
    print(f"    {'-'*40}")

    for h in range(min(10, num_heads)):
        assessment = "Redundant (prune)" if head_scores[h] > 0.5 else "Unique (keep)"
        print(f"    {h:<6} {head_scores[h]:.4f}      {assessment}")

    # Prune heads
    pruning_amount = 0.25  # 25% = 8 heads
    head_mask = MaskOperations.create_structured_mask(head_scores, amount=pruning_amount, mode='low')

    num_heads_pruned = (~head_mask).sum().item()

    print("\n  Pruning plan:")
    print(f"    Amount: {pruning_amount:.0%} of heads")
    print(f"    Heads pruned: {num_heads_pruned}/{num_heads}")
    print(f"    Pruned head indices: {torch.where(~head_mask)[0].tolist()}")

    # Expand to neuron-level mask
    neuron_mask = head_mask.repeat_interleave(head_dim)  # [4096]

    print("\n  Expanding to neuron-level:")
    print(f"    Neuron mask shape: {neuron_mask.shape}")
    print(f"    Neurons pruned: {(~neuron_mask).sum().item()} (= {num_heads_pruned} heads × {head_dim} dims)")

    # Apply to Q/K/V/O (all get same mask for consistency)
    print("\n  Applying to all projections (Q/K/V/O)...")
    q_proj.weight.data *= neuron_mask.unsqueeze(1).float()
    attn.k_proj.weight.data *= neuron_mask.unsqueeze(1).float()
    attn.v_proj.weight.data *= neuron_mask.unsqueeze(1).float()
    attn.o_proj.weight.data *= neuron_mask.unsqueeze(0).float()  # Input dim

    print(f"  ✓ {num_heads_pruned} heads pruned consistently across Q/K/V/O")

    return head_mask, head_scores


def compare_approaches(neuron_mask, head_mask, num_heads=32, head_dim=128):
    """Compare neuron-level vs head-level pruning."""
    print("\n" + "=" * 80)
    print("Comparison: Neuron-Level vs Head-Level")
    print("=" * 80)

    # Convert head mask to neuron mask for comparison
    if head_mask is not None:
        head_neuron_mask = head_mask.repeat_interleave(head_dim)
    else:
        print("  Head-level demo skipped (GPT-2 model)")
        return

    # Statistics
    neuron_pruned = (~neuron_mask).sum().item()
    head_equivalent_pruned = (~head_neuron_mask).sum().item()

    print("\nNeuron-Level Pruning:")
    print(f"  Neurons pruned: {neuron_pruned}")
    print("  Flexibility: Can prune ANY neurons")
    print("  Granularity: Individual neurons")

    print("\nHead-Level Pruning:")
    print(f"  Neurons pruned: {head_equivalent_pruned} (= {head_equivalent_pruned//head_dim} heads × {head_dim})")
    print(f"  Flexibility: Must prune in multiples of {head_dim}")
    print("  Granularity: Entire heads")

    print("\nKey Differences:")
    print("  ✓ Neuron-level: More flexible (any %, any neurons)")
    print("  ✓ Head-level: Cleaner (maintains head structure)")
    print("  ✓ Both supported by your framework!")


def main():
    """Main demonstration."""
    print("=" * 80)
    print("Attention Layer Pruning: Neuron vs Head Level")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Load model (GPT-2 for demo, same principles for LLaMA)
    print("\nLoading model (GPT-2 for demo)...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    model = model.to(device)
    model.eval()

    print("✓ Loaded GPT-2")
    print(f"  - Layers: {model.config.n_layer}")
    print(f"  - Hidden size: {model.config.n_embd}")
    print(f"  - Attention heads: {model.config.n_head}")

    # Analyze structure
    analyze_attention_structure(model)

    # Demonstrate neuron-level pruning
    neuron_mask, neuron_scores = neuron_level_pruning_demo(model, layer_idx=0)

    # Demonstrate head-level pruning
    # (Skip for GPT-2 since it has combined QKV, but would work for LLaMA)
    head_mask, head_scores = head_level_pruning_demo(model, layer_idx=0)

    # Compare
    if head_mask is not None:
        compare_approaches(neuron_mask, head_mask)

    # Summary
    print("\n" + "=" * 80)
    print("Summary: You Can Prune Attention at BOTH Levels!")
    print("=" * 80)
    print("""
1. NEURON-LEVEL Pruning (Fine-Grained):
   ✓ Attention projections are Linear layers with neurons
   ✓ Q projection: 4,096 neurons (LLaMA-3)
   ✓ K projection: 4,096 neurons
   ✓ V projection: 4,096 neurons
   ✓ O projection: 4,096 neurons
   ✓ Can prune ANY neurons individually
   ✓ Maximum flexibility
   ✓ Potentially better performance

2. HEAD-LEVEL Pruning (Coarse-Grained):
   ✓ Aggregate 128 neurons per head
   ✓ Prune entire heads (groups of 128)
   ✓ Maintains multi-head structure
   ✓ Cleaner, more interpretable
   ✓ Easier to implement

3. YOUR FRAMEWORK SUPPORTS BOTH:
   ✓ Neuron-level: Use scores directly on projections
   ✓ Head-level: Aggregate neurons, expand mask
   ✓ Choose based on your needs!

For LLaMA-3 Attention:
- Total neurons: 4 projections × 4,096 = 16,384 neurons per layer
- Can analyze and prune each one individually! ✓
- Or aggregate into 32 heads for head-level pruning ✓

Recommendation:
- For ANALYSIS: Use neuron-level (more detailed)
- For PRUNING: Try both, see which performs better
- For INTERPRETABILITY: Use head-level (cleaner)
    """)


if __name__ == '__main__':
    main()

