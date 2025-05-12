import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

@torch.no_grad()
def collect_layer_data(
    model: nn.Module,
    dataloader: DataLoader,
    target_layers: List[str],
    num_batches: int,
    device: Union[str, torch.device],
    collect_inputs: bool = True,
    collect_outputs: bool = True,
    flatten_spatial: bool = True, # Option to flatten spatial dims for Conv layers
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Collects input and/or output activations from specified layers of a model.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader providing input data.
        target_layers: List of layer names (matching module names) to collect data from.
        num_batches: Number of batches from the dataloader to use.
        device: The device to run computations on.
        collect_inputs: Whether to collect layer inputs.
        collect_outputs: Whether to collect layer outputs.
        flatten_spatial: If True, flatten spatial dimensions (H, W) for Conv layer outputs.
                         Inputs to Conv layers are typically already flattened by previous layers or handled internally.

    Returns:
        A dictionary where keys are layer names and values are dictionaries
        containing 'input' and/or 'output' tensors concatenated across batches.
        Example: {"layer1": {"input": tensor, "output": tensor}, ...}
    """
    model.eval()
    model.to(device)

    collected_data: Dict[str, Dict[str, List[torch.Tensor]]] = {
        name: {} for name in target_layers
    }
    hooks = []

    def hook_fn(layer_name: str):
        def hook(module, input_data, output_data):
            # Ensure input_data is a tensor (handle tuple inputs)
            if isinstance(input_data, tuple):
                if len(input_data) == 0:
                     logger.warning(f"Layer {layer_name} received empty tuple as input.")
                     actual_input = None
                else:
                    actual_input = input_data[0]
            else:
                actual_input = input_data

            # Store input if requested
            if collect_inputs and actual_input is not None:
                if "input" not in collected_data[layer_name]:
                    collected_data[layer_name]["input"] = []
                # Detach and move to CPU to save GPU memory
                collected_data[layer_name]["input"].append(actual_input.detach().cpu())

            # Store output if requested
            if collect_outputs and output_data is not None:
                if "output" not in collected_data[layer_name]:
                    collected_data[layer_name]["output"] = []

                output_to_store = output_data.detach().cpu()

                # Optionally flatten spatial dimensions for Conv layer outputs
                if flatten_spatial and len(output_to_store.shape) > 2 and isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    try:
                        output_to_store = output_to_store.flatten(start_dim=1) # Flatten all but batch dim
                    except Exception as e:
                         logger.warning(f"Could not flatten output for layer {layer_name}. Shape: {output_to_store.shape}. Error: {e}")

                collected_data[layer_name]["output"].append(output_to_store)
        return hook

    # Register hooks
    module_dict = {name: mod for name, mod in model.named_modules()}
    for layer_name in target_layers:
        module = module_dict.get(layer_name)
        if module:
            hooks.append(module.register_forward_hook(hook_fn(layer_name)))
        else:
            logger.warning(f"Layer '{layer_name}' not found in model during hook registration.")


    # Process batches
    batches_processed = 0
    try:
        data_iterator = iter(dataloader)
        while batches_processed < num_batches:
            try:
                batch = next(data_iterator)
                # Assuming batch is a tuple or list where the first element is the input
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                elif isinstance(batch, torch.Tensor):
                    inputs = batch.to(device)
                else:
                    logger.warning(f"Unsupported batch type: {type(batch)}. Skipping batch.")
                    continue # Skip this batch
                
                _ = model(inputs)
                batches_processed += 1

            except StopIteration:
                logger.warning(f"DataLoader exhausted after {batches_processed} batches, requested {num_batches}.")
                break # Exit loop if dataloader is exhausted
            except Exception as e_batch:
                logger.error(f"Error processing batch {batches_processed}: {e_batch}", exc_info=True)
                # Optionally decide whether to continue to next batch or stop
                # continue 
                break # Stop processing further batches on error

    except Exception as e:
         logger.error(f"Error during forward pass for activation collection: {e}", exc_info=True)
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    if batches_processed == 0:
        logger.warning("No batches were successfully processed during activation collection.")
        return {}

    # Concatenate collected tensors
    final_data: Dict[str, Dict[str, torch.Tensor]] = {}
    for layer_name in target_layers:
        if layer_name in collected_data: # Ensure layer was processed
            final_data[layer_name] = {}
            data_dict = collected_data[layer_name]
            if "input" in data_dict and data_dict["input"]:
                try:
                    final_data[layer_name]["input"] = torch.cat(data_dict["input"], dim=0)
                except Exception as e:
                    logger.error(f"Error concatenating inputs for layer {layer_name}: {e}. Sizes: {[t.shape for t in data_dict['input']]}", exc_info=True)
            if "output" in data_dict and data_dict["output"]:
                try:
                    final_data[layer_name]["output"] = torch.cat(data_dict["output"], dim=0)
                except Exception as e:
                    logger.error(f"Error concatenating outputs for layer {layer_name}: {e}. Sizes: {[t.shape for t in data_dict['output']]}", exc_info=True)

    return final_data

# Example Usage (can be removed later)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)

#     # Simple model
#     model = nn.Sequential(
#         nn.Linear(10, 20), # layer '0'
#         nn.ReLU(),         # layer '1'
#         nn.Linear(20, 5)   # layer '2'
#     )

#     # Dummy data
#     X = torch.randn(100, 10)
#     y = torch.randint(0, 5, (100,))
#     dataset = torch.utils.data.TensorDataset(X, y)
#     loader = DataLoader(dataset, batch_size=10)

#     # Specify layers to collect from (use actual module names)
#     target_layers = ['0', '2'] # Assuming '0' is first Linear, '2' is second

#     # Collect data
#     collected_activation_data = collect_layer_data(
#         model=model,
#         dataloader=loader,
#         target_layers=target_layers,
#         num_batches=5,
#         device='cpu',
#         collect_inputs=True,
#         collect_outputs=True
#     )

#     # Print shapes
#     for layer, data in collected_activation_data.items():
#         print(f"Layer: {layer}")
#         if "input" in data:
#             print(f"  Input shape: {data['input'].shape}")
#         if "output" in data:
#             print(f"  Output shape: {data['output'].shape}") 