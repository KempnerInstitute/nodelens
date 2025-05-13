import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import copy

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from alignment.config import ExperimentConfig
from alignment.metrics import get_metric
from alignment.models.registry import create_model
from alignment.datasets import load_dataset, DataSet
from alignment.dropout import progressive_dropout
from alignment.models.base import AlignmentNetwork # For type hinting if needed

# Helper to create a simple MLP for testing
class SimpleTestMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[32, 16], output_dim=10):
        super().__init__()
        self.network = nn.Sequential()
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            self.network.add_module(f"fc{i}", nn.Linear(current_dim, h_dim))
            self.network.add_module(f"relu{i}", nn.ReLU())
            current_dim = h_dim
        self.network.add_module("fc_out", nn.Linear(current_dim, output_dim))

    def forward(self, x):
        return self.network(x)

@pytest.fixture
def simple_mlp_model_and_data(tmp_path):
    """
    Provides a simple MLP model, a basic AlignmentNetwork wrapper, and MNIST dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a dummy base model
    base_mlp = SimpleTestMLP(input_dim=784, hidden_dims=[32,16], output_dim=10)
    
    # Create AlignmentNetwork (assuming all linear layers are alignment layers)
    # Manually identify linear layers for alignment_layer_names
    alignment_layer_info = {}
    for name, module in base_mlp.named_modules():
        if isinstance(module, nn.Linear):
            alignment_layer_info[name] = None # Using None means input to layer is its own input

    model = AlignmentNetwork(base_model=base_mlp, alignment_layer_names=alignment_layer_info)
    model.to(device)

    dataset_config_dict = {
        "dataset_name": "MNIST",
        "batch_size": 32,
        "data_path": str(tmp_path / "data"), # Use tmp_path for data
        "transform_params": {"flatten": True, "normalize": False},
        "num_workers": 0, 
        "pin_memory": False
    }
    dataset = load_dataset(dataset_config_dict, device=device)
    
    return [model], dataset, device # Return list of models for progressive_dropout

def test_progressive_dropout_output_structure(simple_mlp_model_and_data):
    """
    Tests the progressive_dropout function for basic execution and output structure.
    Based on debug_progressive_dropout.py.
    """
    networks, dataset, device = simple_mlp_model_and_data
    
    metric = get_metric("rq") # Use a real metric
    
    dropout_fractions = [0.0, 0.2, 0.5]
    pruning_mode = "layer_wise"
    dropout_mode = "scaled"
    strategy = "low_rq" # Default for test, can be parameterized later

    results_dict = progressive_dropout(
        networks=networks,
        dataset=dataset,
        dropout_fractions=dropout_fractions,
        metric=metric,
        device=device,
        pruning_mode=pruning_mode,
        dropout_mode=dropout_mode,
        strategy=strategy, # Pass the strategy
        show_progress=False, # Keep tests quiet
        debug_mode=False   # Keep tests quiet
    )

    assert isinstance(results_dict, dict), "Results should be a dictionary"
    
    # Check for expected top-level keys from progressive_dropout
    # Based on its implementation, it directly returns a dict with keys like:
    # \'accuracies\', \'losses\', \'stds\', \'dropout_fractions\', 
    # \'pruning_details\', \'pre_pruning_layer_stats\'
    # The old `debug_progressive_dropout.py` processed a potentially different structure.
    # We test the direct output of `progressive_dropout` here.

    expected_top_keys = ['accuracies', 'losses', 'stds', 'dropout_fractions', 'pruning_details', 'pre_pruning_layer_stats']
    for key in expected_top_keys:
        assert key in results_dict, f"Expected key '{key}' not found in progressive_dropout results."

    assert results_dict['dropout_fractions'] == dropout_fractions, "Dropout fractions in results should match input."

    # Check accuracies structure
    assert strategy in results_dict['accuracies'], f"Strategy '{strategy}' not found in accuracies."
    assert isinstance(results_dict['accuracies'][strategy], list), f"Accuracies for strategy '{strategy}' should be a list."
    assert len(results_dict['accuracies'][strategy]) == len(dropout_fractions), \
        f"Length of accuracies for '{strategy}' should match dropout_fractions."
    for acc in results_dict['accuracies'][strategy]:
        assert isinstance(acc, (float, np.floating)), "Accuracy values should be floats."

    # Check losses structure (similar to accuracies)
    assert strategy in results_dict['losses'], f"Strategy '{strategy}' not found in losses."
    assert isinstance(results_dict['losses'][strategy], list), f"Losses for strategy '{strategy}' should be a list."
    assert len(results_dict['losses'][strategy]) == len(dropout_fractions), \
        f"Length of losses for '{strategy}' should match dropout_fractions."
    for loss_val in results_dict['losses'][strategy]:
        assert isinstance(loss_val, (float, np.floating)), "Loss values should be floats."

    # Check stds structure (similar to accuracies)
    assert strategy in results_dict['stds'], f"Strategy '{strategy}' not found in stds."
    assert isinstance(results_dict['stds'][strategy], list), f"Stds for strategy '{strategy}' should be a list."
    assert len(results_dict['stds'][strategy]) == len(dropout_fractions), \
        f"Length of stds for '{strategy}' should match dropout_fractions."
    for std_val in results_dict['stds'][strategy]:
        assert isinstance(std_val, (float, np.floating)), "Std values should be floats."
        
    # Check pruning_details structure (more complex, basic checks here)
    assert isinstance(results_dict['pruning_details'], dict), "pruning_details should be a dict."
    if results_dict['pruning_details']: # If not empty
        assert strategy in results_dict['pruning_details'], f"Strategy '{strategy}' not in pruning_details."
        # Further checks can be added for network_idx, fraction_idx, layer_idx structure

    # Check pre_pruning_layer_stats structure
    assert isinstance(results_dict['pre_pruning_layer_stats'], dict), "pre_pruning_layer_stats should be a dict."
    if results_dict['pre_pruning_layer_stats']: # If not empty
        first_net_stats = results_dict['pre_pruning_layer_stats'][0] # Assuming stats for network 0
        assert isinstance(first_net_stats, dict), "Stats for a network should be a dict."
        # Further checks for layer_idx and metric scores

def test_pruned_neuron_activation(simple_mlp_model_and_data):
    """
    Tests if neurons marked as pruned by progressive_dropout have zero activation.
    Inspired by verify_pruning.py.
    """
    networks, dataset, device = simple_mlp_model_and_data
    original_network = networks[0] # We work with one network for this test

    metric = get_metric("rq") 
    dropout_fractions = [0.5] # Test a significant pruning fraction
    pruning_mode = "layer_wise"
    dropout_mode = "scaled" # Or "zero", scaled should also result in zero for pruned units if mask is applied before scaling
    strategy = "low_rq"

    # We need a copy of the network to be pruned by progressive_dropout
    # progressive_dropout modifies the network list in-place.
    network_to_prune = copy.deepcopy(original_network).to(device)

    results_dict = progressive_dropout(
        networks=[network_to_prune], # Pass as a list
        dataset=dataset,
        dropout_fractions=dropout_fractions,
        metric=metric,
        device=device,
        pruning_mode=pruning_mode,
        dropout_mode=dropout_mode,
        strategy=strategy,
        show_progress=False,
        debug_mode=False
    )

    pruned_network = network_to_prune # The network in the list is modified in-place
    pruned_network.eval() # Ensure model is in eval mode for activation checking

    # Get pruning details for the 50% fraction (index 0, since fractions was [0.5])
    # The structure is: results_dict[\'pruning_details\'][strategy_name][network_idx][fraction_idx][layer_idx]
    # Since we used one network (idx 0) and one fraction (idx 0 for 0.5% after baseline 0.0 which is not here)
    # Actually, progressive_dropout result for pruning_details might be simpler if only one fraction > 0 is given.
    # Let's assume `dropout_fractions` for `progressive_dropout` call was [0.0, 0.5].
    # Then details for 0.5 would be at fraction_idx=1.
    # If `dropout_fractions` was just [0.5], the interpretation of fraction_idx might be 0.
    # For safety, let's re-run progressive_dropout with [0.0, 0.5] to be clear.
    
    dropout_fractions_for_test = [0.0, 0.5]
    network_to_prune_for_activation_test = copy.deepcopy(original_network).to(device)
    results_activation_test = progressive_dropout(
        networks=[network_to_prune_for_activation_test],
        dataset=dataset,
        dropout_fractions=dropout_fractions_for_test,
        metric=metric,
        device=device,
        pruning_mode=pruning_mode,
        dropout_mode=dropout_mode, # In 'scaled' mode, pruned weights are 0, then scaled (still 0)
        strategy=strategy,
        show_progress=False,
        debug_mode=False
    )
    pruned_network_for_activation_test = network_to_prune_for_activation_test
    pruned_network_for_activation_test.eval()

    pruning_details_for_fraction = results_activation_test[\'pruning_details\'][strategy][0][1] # Net 0, Fraction 0.5 (index 1)

    # Prepare a batch of data
    data_loader = dataset.test_loader
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)

    activation_hook_handles = []
    captured_activations = {}

    def get_activation_hook(layer_name):
        def hook(module, input, output):
            captured_activations[layer_name] = output.detach()
        return hook

    # Attach hooks to alignment layers of the pruned network
    # Assuming base_model is the actual Sequential model for AlignmentNetwork here.
    # And alignment_names in AlignmentNetwork correspond to layers in base_model that were prunable.
    base_model_layers = dict(pruned_network_for_activation_test.base_model.named_modules())

    for layer_name_in_alignment_network in pruned_network_for_activation_test.alignment_names:
        if layer_name_in_alignment_network in base_model_layers:
            actual_layer_module = base_model_layers[layer_name_in_alignment_network]
            handle = actual_layer_module.register_forward_hook(get_activation_hook(layer_name_in_alignment_network))
            activation_hook_handles.append(handle)
        else:
            # This case should ideally not happen if alignment_names are correctly mapped
            print(f"Warning: Could not find layer {layer_name_in_alignment_network} in base model to attach hook.")

    # Forward pass to capture activations
    with torch.no_grad():
        _ = pruned_network_for_activation_test(inputs)

    # Remove hooks
    for handle in activation_hook_handles:
        handle.remove()

    # Verify activations
    found_pruned_neurons_to_check = False
    for layer_idx_from_details, layer_pruning_info in pruning_details_for_fraction.items():
        # map layer_idx_from_details (0, 1, 2...) to actual layer_name (e.g., 'network.fc1')
        if layer_idx_from_details < len(pruned_network_for_activation_test.alignment_names):
            layer_name = pruned_network_for_activation_test.alignment_names[layer_idx_from_details]
        else:
            print(f"Warning: layer_idx {layer_idx_from_details} out of bounds for alignment_names.")
            continue
        
        if layer_name not in captured_activations:
            # This can happen if the layer itself was not a type that produces typical activations (e.g. Flatten)
            # or if it wasn't hooked correctly. For this test MLP, all alignment layers are Linear.
            print(f"Warning: No activations captured for layer '{layer_name}'. Skipping activation check for it.")
            continue
        
        layer_activations = captured_activations[layer_name]
        pruned_indices_for_layer = layer_pruning_info.get(\"dropped_indices\", [])
        
        if not isinstance(pruned_indices_for_layer, torch.Tensor):
            pruned_indices_for_layer = torch.tensor(pruned_indices_for_layer, device=device, dtype=torch.long)

        if pruned_indices_for_layer.numel() > 0:
            found_pruned_neurons_to_check = True
            # Activations are [batch_size, num_features/neurons_in_this_layer]
            # We are checking output neurons of the *current* layer whose weights *to them* were pruned.
            # So, we check the output channels/neurons of this layer.
            for neuron_idx_tensor in pruned_indices_for_layer:
                neuron_idx = neuron_idx_tensor.item()
                if neuron_idx < layer_activations.shape[1]:
                    neuron_output_activations = layer_activations[:, neuron_idx]
                    assert torch.allclose(neuron_output_activations, torch.zeros_like(neuron_output_activations), atol=1e-6), \
                        f"Layer '{layer_name}', pruned neuron index {neuron_idx} has non-zero activations: {neuron_output_activations.abs().max().item()}"
                else:
                    print(f"Warning: Pruned index {neuron_idx} is out of bounds for layer '{layer_name}' activations shape {layer_activations.shape}")
    
    assert found_pruned_neurons_to_check, "No pruned neurons were identified to check for zero activation. Pruning might not have occurred as expected." 