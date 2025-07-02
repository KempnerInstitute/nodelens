"""Debug script to test alignment pruning on MNIST MLP."""

import torch
import torch.nn as nn
import yaml
import logging
from src.alignment.models import create_model
from src.alignment.data import get_dataset
from src.alignment.pruning.strategies.alignment_based import AlignmentPruning
from src.alignment.pruning.base import PruningConfig

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
with open('configs/unified_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("=== Testing Alignment Pruning ===\n")

# 1. Create model
print("1. Creating model...")
model_config = config['model']
# Get MLP config and fix parameter names
mlp_config = model_config.get('mlp_config', {}).copy()
if 'activation' in mlp_config:
    mlp_config['activation_type'] = mlp_config.pop('activation')
if 'dropout' in mlp_config:
    mlp_config['dropout_rate'] = mlp_config.pop('dropout')
# Remove unsupported parameters
mlp_config.pop('use_batchnorm', None)

model = create_model(
    model_name=model_config['name'],
    dataset_name=config['dataset']['name'],
    **mlp_config
)
print(f"Model created: {model_config['name']}")
print(f"Model structure:")
for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        print(f"  {name}: {module.__class__.__name__} - weight shape: {module.weight.shape}")

# 2. Get data
print("\n2. Getting data...")
dataset_config = config['dataset']
train_loader, test_loader = get_dataset(
    dataset_name=dataset_config['name'],
    batch_size=dataset_config['batch_size'],
    num_workers=dataset_config.get('num_workers', 4),
    root=dataset_config['data_path'],
    download=dataset_config.get('download', True)
)
print(f"Dataset: {dataset_config['name']}")
print(f"Batch size: {dataset_config['batch_size']}")

# 3. Get a batch of data
print("\n3. Getting sample batch...")
inputs, labels = next(iter(train_loader))
print(f"Input shape: {inputs.shape}")
print(f"Labels shape: {labels.shape}")

# 4. Capture layer inputs using forward hooks
print("\n4. Capturing layer inputs...")
layer_inputs = {}
hooks = []

def capture_input(name):
    def hook(module, input, output):
        layer_inputs[name] = input[0].detach()
        print(f"  Captured input for {name}: shape {input[0].shape}")
    return hook

# Register hooks on linear/conv layers
for name, module in model.named_modules():
    if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
        hook = module.register_forward_hook(capture_input(name))
        hooks.append(hook)

# Forward pass
with torch.no_grad():
    outputs = model(inputs)
print(f"Forward pass output shape: {outputs.shape}")

# Remove hooks
for hook in hooks:
    hook.remove()

# 5. Test alignment pruning on each layer
print("\n5. Testing alignment pruning on each layer...")
pruning_config = PruningConfig(
    amount=0.5,
    pruning_mode='low',
    structured=True
)

strategy = AlignmentPruning(
    metric='rayleigh_quotient',
    config=pruning_config
)

for name, module in model.named_modules():
    if hasattr(module, 'weight') and len(module.weight.shape) >= 2:
        print(f"\n  Testing layer: {name}")
        print(f"    Weight shape: {module.weight.shape}")
        
        if name in layer_inputs:
            layer_input = layer_inputs[name]
            print(f"    Input shape: {layer_input.shape}")
            
            try:
                # Compute importance scores
                scores = strategy.compute_importance_scores(module, inputs=layer_input)
                print(f"    Importance scores shape: {scores.shape}")
                print(f"    Scores range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                print(f"    Mean score: {scores.mean().item():.4f}")
                
                # Apply pruning
                mask = strategy.prune(module, inputs=layer_input, amount=0.5)
                print(f"    Mask shape: {mask.shape}")
                print(f"    Sparsity: {(mask == 0).float().mean().item():.2%}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"    WARNING: No input captured for this layer")

# 6. Check if weights are actually pruned
print("\n6. Checking pruning results...")
for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        if hasattr(module, 'weight_mask'):
            mask = module.weight_mask
            actual_sparsity = (module.weight == 0).float().mean().item()
            mask_sparsity = (mask == 0).float().mean().item()
            print(f"  {name}: mask sparsity={mask_sparsity:.2%}, actual sparsity={actual_sparsity:.2%}")
        else:
            print(f"  {name}: No pruning mask found")

print("\n=== Debug Complete ===") 