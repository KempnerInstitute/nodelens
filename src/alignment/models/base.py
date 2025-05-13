from warnings import warn
from typing import Optional, Union, List, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from alignment.utils.core import check_iterable, to_numpy, to_tensor, ensure_device
from alignment.utils.model_utils import (
    get_device,
    set_net_mode,
    get_maximum_strides,
    get_unfold_params,
    weighted_average,
    remove_by_idx,
    smart_pca
)
from alignment.utils.metrics_utils import AlignmentMetricsFactory as AlignmentMetrics
from alignment.utils.metrics_utils import alignment
from alignment.metrics import get_metric, AlignmentMetric
from alignment.utils.activation_utils import collect_layer_data

import logging

# Setup module logger
logger = logging.getLogger(__name__)

class AlignmentNetwork(nn.Module):
    """
    Extended to allow multiple inputs to an alignment layer.

    alignment_layer_names can be:
      - None => gather all layers with .weight
      - Dict[str, Union[str, None, List[str]]] => 
         key = alignment-layer name
         value = either:
            None => use the same layer's input
            str  => single input-layer name
            List[str] => multiple input-layer names to gather from
    """

    def __init__(
        self,
        base_model: nn.Module,
        alignment_layer_names: Optional[dict] = None,
        cnn_mode="unfold",
        **kwargs
    ):
        super().__init__()
        self.base_model = base_model
        self.alignment_layers = nn.ModuleList()
        self.alignment_names = []
        self.cnn_mode = cnn_mode
        self._is_ddp_wrapped = False
        self._initialize_layers(alignment_layer_names, **kwargs)

    def _initialize_layers(self, alignment_layer_names, **kwargs):
        if alignment_layer_names is None:
            self.layer_to_input_names = None
            for name, layer in self.base_model.named_modules():
                if not hasattr(layer, "weight"):
                    continue
                self.alignment_layers.append(layer)
                self.alignment_names.append(name)
        else:
            self.layer_to_input_names = {}
            for name, layer in self.base_model.named_modules():
                if name in alignment_layer_names.keys():
                    if not hasattr(layer, "weight"):
                        warn(
                            f"Skipping layer {name} ({layer.__class__.__name__}) - no weight attribute",
                            RuntimeWarning,
                            stacklevel=1
                        )
                        continue
                    self.alignment_layers.append(layer)
                    self.alignment_names.append(name)
                    self.layer_to_input_names[name] = alignment_layer_names[name]

    def is_classification_layer_included(self):
        classification_layer_name = [
            nm for nm, lyr in self.base_model.named_modules() if hasattr(lyr, "weight")
        ][-1]
        return (classification_layer_name in self.alignment_names)

    def num_layers(self, all=False):
        if all:
            return sum(1 for m in self.base_model.modules() if hasattr(m, "weight"))
        return len(self.alignment_layers)

    def forward(self, x, store_hidden=False):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            store_hidden: **DEPRECATED**. Kept for backward compatibility, but no longer used.
            
        Returns:
            Network output.
        """
        # Forward pass through base model
        out = self.base_model(x)
        
        return out

    def get_dropout(self):
        p = []
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                p.append(module.p)
        return p

    def set_dropout(self, p):
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def set_dropout_by_layer(self, p):
        dropout_layers = []
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)
        assert len(dropout_layers) == len(p), "p must match number of dropout layers"
        for layer, drop_prob in zip(dropout_layers, p):
            layer.p = drop_prob

    @torch.no_grad()
    def get_layer_inputs(self, x, precomputed=False):
        logger.warning("AlignmentNetwork.get_layer_inputs is deprecated due to hook refactoring. " 
                       "Use activation_utils.collect_layer_data instead.")
        return None

    @torch.no_grad()
    def get_alignment_layers(self):
        return self.alignment_layers

    @torch.no_grad()
    def get_alignment_weights(self, flatten=False):
        weights = []
        for layer in self.alignment_layers:
            w = layer.weight.data.clone()
            if flatten:
                w = w.flatten(start_dim=1)
            weights.append(w)
        return weights

    def _preprocess_inputs(self, inputs_to_layers, compress_convolutional=True):
        preprocessed = []
        for input_, layer in zip(inputs_to_layers, self.alignment_layers):
            if isinstance(layer, nn.Conv2d):
                layer_prms = get_unfold_params(layer)
                unfolded = torch.nn.functional.unfold(input_, layer.kernel_size, **layer_prms)
                if self.cnn_mode == "unfold":
                    if compress_convolutional:
                        unfolded = unfolded.transpose(1, 2).contiguous().view(-1, unfolded.size(1))
                    preprocessed.append(unfolded)
                elif self.cnn_mode == "patchwise":
                    preprocessed.append(unfolded)
                elif self.cnn_mode == "batch_patch_combined":
                    # Combines batch and patch dimensions for convolutional layers
                    # in a standardized format for alignment computations
                    unfolded = unfolded.transpose(1, 2).contiguous().view(-1, unfolded.size(1))
                    preprocessed.append(unfolded)
                else:
                    if compress_convolutional:
                        unfolded = unfolded.transpose(1, 2).contiguous()
                        B, P, F = unfolded.shape
                        unfolded = unfolded.view(B * P, F)
                    preprocessed.append(unfolded)
            else:
                if input_.dim() > 2:
                    input_ = input_.flatten(start_dim=1)
                preprocessed.append(input_)
        return preprocessed

    @torch.no_grad()
    def compare_weights(self, weights, norm=False):
        current_weights = self.get_alignment_weights()
        delta_weights = []
        for iw, cw in zip(weights, current_weights):
            if norm:
                delta_weights.append(torch.norm(cw.flatten(1) - iw.flatten(1), dim=1))
            else:
                delta_weights.append(cw - iw)
        return delta_weights

    @torch.no_grad()
    def measure_alignment_methods(
        self, 
        dataloader: DataLoader,
        methods: List[str], 
        num_batches: int = 5,
        device: Optional[Union[str, torch.device]] = None,
        scale_by_norm_for_rq: bool = False,
        metric_kwargs: Optional[Dict[str, Any]] = None
        ):
        effective_device = _normalize_device(device if device is not None else get_device(self))
        self.eval()
        metric_kwargs = metric_kwargs or {}

        target_layers_for_collection = set()
        needs_inputs = False
        needs_outputs = False
        
        source_layers_map = {}
        for align_layer_name in self.alignment_names:
            sources = [align_layer_name] 
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                input_spec = self.layer_to_input_names[align_layer_name]
                if input_spec is None: sources = [align_layer_name]
                elif isinstance(input_spec, str): sources = [input_spec]
                elif isinstance(input_spec, list): sources = input_spec
            for source_layer_name in sources:
                 target_layers_for_collection.add(source_layer_name)
                 if source_layer_name not in source_layers_map: source_layers_map[source_layer_name] = []
                 source_layers_map[source_layer_name].append(align_layer_name)

        for m_name in methods:
            m_lower = m_name.lower()
            if "rayleigh_quotient" in m_lower or "rq" in m_lower or "redundancy" in m_lower or "pid_" in m_lower:
                needs_inputs = True
            if "mi_" in m_lower or "pid_" in m_lower:
                needs_outputs = True

        layers_to_hook_for_input = set()
        layers_to_hook_for_output = set()

        for align_layer_name in self.alignment_names:
            source_layer_names = [align_layer_name]
            is_source_output = False
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                input_spec = self.layer_to_input_names[align_layer_name]
                if isinstance(input_spec, str):
                    source_layer_names = [input_spec]
                    is_source_output = True
                elif isinstance(input_spec, list):
                    source_layer_names = input_spec
                    is_source_output = True

            input_needed_for_metrics = any("rayleigh_quotient" in m.lower() or "rq" in m.lower() or "redundancy" in m.lower() for m in methods)
            
            if input_needed_for_metrics:
                if is_source_output:
                    for src in source_layer_names: layers_to_hook_for_output.add(src)
                else:
                    layers_to_hook_for_input.add(align_layer_name)

            output_needed_for_metrics = any("pid_" in m.lower() or "mi_" in m.lower() for m in methods)
            if output_needed_for_metrics:
                 layers_to_hook_for_output.add(align_layer_name)
                 
        all_target_layers_list = sorted(list(layers_to_hook_for_input.union(layers_to_hook_for_output)))
        
        collect_inputs_flag = bool(layers_to_hook_for_input)
        collect_outputs_flag = bool(layers_to_hook_for_output)

        collected_data = collect_layer_data(
            model=self.base_model if not self._is_ddp_wrapped else self.module.base_model,
            dataloader=dataloader, target_layers=all_target_layers_list, num_batches=num_batches, device=effective_device,
            collect_inputs=collect_inputs_flag, collect_outputs=collect_outputs_flag,
            flatten_spatial=(self.cnn_mode != "patchwise")
        )

        if not collected_data:
            logger.warning("Activation collection returned no data. Cannot measure alignment.")
            return [{} for _ in self.alignment_names]

        inputs_for_alignment_layers = []
        for align_layer_name in self.alignment_names:
            source_layer_names = [align_layer_name]
            use_output_as_input = False
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                input_spec = self.layer_to_input_names[align_layer_name]
                if isinstance(input_spec, str):
                    source_layer_names = [input_spec]
                    use_output_as_input = True
                elif isinstance(input_spec, list):
                    source_layer_names = input_spec
                    use_output_as_input = True

            combined_source_tensors = []
            activation_type = "output" if use_output_as_input else "input"
            
            for source_name in source_layer_names:
                if source_name in collected_data and activation_type in collected_data[source_name]:
                    combined_source_tensors.append(collected_data[source_name][activation_type])
                else:
                    logger.warning(f"Required activation '{activation_type}' not found for source layer '{source_name}' "
                                   f"needed by alignment layer '{align_layer_name}'. Skipping this source.")
            
            if not combined_source_tensors:
                 logger.warning(f"Could not gather any input activations for alignment layer '{align_layer_name}'. Appending None.")
                 inputs_for_alignment_layers.append(None)
                 continue

            if len(combined_source_tensors) == 1:
                combined_input = combined_source_tensors[0]
            else:
                dev = combined_source_tensors[0].device
                tensors_on_dev = [t.to(dev) for t in combined_source_tensors]
                try:
                    combined_input = torch.cat(tensors_on_dev, dim=1)
                except Exception as e:
                     logger.error(f"Error concatenating inputs for {align_layer_name} from sources {source_layer_names}: {e}")
                     combined_input = None

            inputs_for_alignment_layers.append(combined_input)

        preprocessed_inputs = self._preprocess_inputs(inputs_for_alignment_layers, 
                                                      compress_convolutional=(self.cnn_mode != "patchwise"))

        weights = self.get_alignment_weights(flatten=(self.cnn_mode != "patchwise"))

        all_layer_results = []
        for idx, (layer_name, layer_instance) in enumerate(zip(self.alignment_names, self.alignment_layers)):
            metrics_dict = {}
            inp = preprocessed_inputs[idx]
            w = weights[idx]

            if inp is None or w is None:
                logger.warning(f"Skipping metric calculation for layer '{layer_name}' due to missing input or weights.")
                all_layer_results.append({})
                continue
                
            inp = inp.to(effective_device)
            w = w.to(effective_device)
                
            layer_outputs = None
            if any("pid_" in m.lower() or "mi_" in m.lower() for m in methods):
                 if layer_name in collected_data and "output" in collected_data[layer_name]:
                     layer_outputs = collected_data[layer_name]["output"].to(effective_device)
                 else:
                     logger.warning(f"Output needed but not collected for layer '{layer_name}'. Metrics requiring output may fail.")

            for m_name in methods:
                try:
                    current_scale_by_norm = scale_by_norm_for_rq if m_name.upper() == "RQ" else False
                    metric_obj = get_metric(name=m_name, scale_by_norm=current_scale_by_norm)
                    
                    val = metric_obj.compute_per_node_scores(
                        layer_inputs=inp, 
                        layer_weights=w, 
                        layer_outputs=layer_outputs, 
                        device=effective_device,
                        **(metric_kwargs.get(m_name, {}))
                    )
                    metrics_dict[m_name] = val.cpu()
                except Exception as e:
                    logger.error(f"Error computing metric '{m_name}' for layer '{layer_name}': {e}", exc_info=True)
                    metrics_dict[m_name] = torch.tensor(float('nan'))

            all_layer_results.append(metrics_dict)

        return all_layer_results

    @torch.no_grad()
    def measure_alignment(self, dataloader: DataLoader, num_batches: int = 5, device: Optional[Union[str, torch.device]] = None, method="alignment", relative=True):
        metric_name = "RQ" if method == "alignment" else method
        scale_rq = relative if metric_name == "RQ" else False 
        results_per_layer = self.measure_alignment_methods(
            dataloader=dataloader,
            methods=[metric_name],
            num_batches=num_batches,
            device=device,
            scale_by_norm_for_rq=scale_rq
        )
        outputs = [layer_result.get(metric_name, torch.tensor(float('nan'))) for layer_result in results_per_layer]
        return outputs

    @torch.no_grad()
    def measure_alignment_weights(self, dataloader: DataLoader, weights_list: List[torch.Tensor], num_batches: int = 5, device: Optional[Union[str, torch.device]] = None, method="alignment", relative=True):
        effective_device = _normalize_device(device if device is not None else get_device(self))
        self.eval()
        
        target_layers_for_collection = set()
        source_layers_map = {}
        for align_layer_name in self.alignment_names:
            sources = [align_layer_name] 
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                input_spec = self.layer_to_input_names[align_layer_name]
                if input_spec is None: sources = [align_layer_name]
                elif isinstance(input_spec, str): sources = [input_spec]
                elif isinstance(input_spec, list): sources = input_spec
            for source_layer_name in sources:
                 target_layers_for_collection.add(source_layer_name)
                 if source_layer_name not in source_layers_map: source_layers_map[source_layer_name] = []
                 source_layers_map[source_layer_name].append(align_layer_name)

        layers_to_hook_for_input = set()
        layers_to_hook_for_output = set()
        for align_layer_name in self.alignment_names:
            source_layer_names = [align_layer_name]; is_source_output = False
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                 input_spec = self.layer_to_input_names[align_layer_name]
                 if isinstance(input_spec, str): source_layer_names = [input_spec]; is_source_output = True
                 elif isinstance(input_spec, list): source_layer_names = input_spec; is_source_output = True
            if is_source_output:
                for src in source_layer_names: layers_to_hook_for_output.add(src)
            else:
                layers_to_hook_for_input.add(align_layer_name)
                
        all_target_layers_list = sorted(list(layers_to_hook_for_input.union(layers_to_hook_for_output)))
        collect_inputs_flag = bool(layers_to_hook_for_input)
        collect_outputs_flag = bool(layers_to_hook_for_output)

        collected_data = collect_layer_data(
            model=self.base_model if not self._is_ddp_wrapped else self.module.base_model,
            dataloader=dataloader, target_layers=all_target_layers_list, num_batches=num_batches, device=effective_device,
            collect_inputs=collect_inputs_flag, collect_outputs=collect_outputs_flag,
            flatten_spatial=(self.cnn_mode != "patchwise")
        )

        if not collected_data: return [torch.tensor(float('nan')) for _ in self.alignment_names]

        inputs_for_alignment_layers = []
        for align_layer_name in self.alignment_names:
            source_layer_names = [align_layer_name]; use_output_as_input = False
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                 input_spec = self.layer_to_input_names[align_layer_name]
                 if isinstance(input_spec, str): source_layer_names = [input_spec]; use_output_as_input = True
                 elif isinstance(input_spec, list): source_layer_names = input_spec; use_output_as_input = True
            combined_source_tensors = []
            activation_type = "output" if use_output_as_input else "input"
            for source_name in source_layer_names:
                 if source_name in collected_data and activation_type in collected_data[source_name]: combined_source_tensors.append(collected_data[source_name][activation_type])
            if not combined_source_tensors: inputs_for_alignment_layers.append(None); continue
            if len(combined_source_tensors) == 1: combined_input = combined_source_tensors[0]
            else: 
                dev = combined_source_tensors[0].device; tensors_on_dev = [t.to(dev) for t in combined_source_tensors]
                try: combined_input = torch.cat(tensors_on_dev, dim=1)
                except Exception: combined_input = None
            inputs_for_alignment_layers.append(combined_input)

        preprocessed_inputs = self._preprocess_inputs(inputs_for_alignment_layers, compress_convolutional=(self.cnn_mode != "patchwise"))
        
        weights_flat = [w.flatten(start_dim=1).to(effective_device) for w in weights_list] 
        if len(weights_flat) != len(preprocessed_inputs):
            logger.error("Mismatch between number of provided weights and alignment layers.")
            return [torch.tensor(float('nan')) for _ in self.alignment_names]

        outputs = []
        from alignment.utils.metrics_utils import alignment as legacy_alignment_fn 
        
        for inp, w in zip(preprocessed_inputs, weights_flat):
            if inp is None or w is None:
                 outputs.append(torch.tensor(float('nan')))
                 continue
            try:
                out = legacy_alignment_fn(inp.to(effective_device), w, method=method, relative=relative)
                outputs.append(out.cpu())
            except Exception as e:
                logger.error(f"Error calling legacy alignment function: {e}")
                outputs.append(torch.tensor(float('nan')))
        return outputs

    @torch.no_grad()
    def forward_targeted_dropout(self, x, idxs, layers, dropout_mode="scaled"):
        assert check_iterable(idxs) and check_iterable(layers), "idxs & layers must be iterables with same length"
        assert len(idxs) == len(layers), "idxs and layers must be the same length"
        assert all([0 <= layer < len(self.alignment_layers) for layer in layers]), "invalid layer index"
        assert dropout_mode in ["scaled", "unscaled"], f"Invalid dropout_mode: {dropout_mode}, must be 'scaled' or 'unscaled'"
        
        total_effective_pruning = 1.0
        for idx, layer in zip(idxs, layers):
            layer_size = self.alignment_layers[layer].weight.size(0)
            fraction = idx.numel() / layer_size
            total_effective_pruning *= (1.0 - fraction)
        
        if total_effective_pruning < 0.01:
            import warnings
            warnings.warn("Combined pruning across layers is very high (>99%). This could result in near-zero accuracy.")
        
        hidden_outputs_dict = {}
        hooks = []
        
        def dropout(hook_name, dropout_idx, layer_idx):
            def dropout_hook(module, in_, out_):
                max_index = out_.shape[1]
                
                valid_idx = dropout_idx[dropout_idx < max_index]
                
                fraction_dropout = len(valid_idx) / float(max_index)
                
                out_copy = out_.clone()
                
                if valid_idx.numel() > 0:
                    out_copy[:, valid_idx] = 0
                
                if dropout_mode == "scaled":
                    scaling_factor = (1.0 - fraction_dropout)
                    out_copy = out_copy * scaling_factor
                
                hidden_outputs_dict[hook_name] = out_copy
                return out_copy
            return dropout_hook
            
        def get_output(hook_name):
            def output_hook(module, in_, out_):
                hidden_outputs_dict[hook_name] = out_
            return output_hook
            
        for idx_layer, (name, layer) in enumerate(zip(self.alignment_names, self.alignment_layers)):
            if idx_layer in layers:
                i_lyr = layers.index(idx_layer)
                d_idx = idxs[i_lyr]
                hooks.append(layer.register_forward_hook(dropout(name, d_idx, idx_layer)))
            else:
                hooks.append(layer.register_forward_hook(get_output(name)))
                
        out = self.base_model(x)
        
        for hk in hooks:
            hk.remove()
            
        assert len(hidden_outputs_dict) == len(self.alignment_names), "Missing outputs from some layers"
        hidden_outputs = [hidden_outputs_dict[nm] for nm in self.alignment_names]
        
        return out, hidden_outputs

    @torch.no_grad()
    def forward_eigenvector_dropout(self, x, eigenvalues, eigenvectors, idxs, layers):
        logger.info("Running refactored forward_eigenvector_dropout using pre-hooks.")
        device = get_device(x)
        assert check_iterable(idxs) and check_iterable(layers), "idxs/layers must be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers must be the same length"
        assert len(layers) == len(eigenvalues), "list of eigenvalues must have same length as list of layers"
        assert len(layers) == len(eigenvectors), "list of eigenvectors must have same length as list of layers"
        
        registered_hooks = []
        hidden_inputs_dict = {} # To store inputs of non-dropout layers

        # --- Define Hook Functions ---
        def make_eigen_pre_hook(evec_matrix, correction_factor):
            def pre_hook(module, input_tuple):
                original_input = input_tuple[0]
                # Ensure input is 2D for matrix multiplication [batch_size_or_flattened, features]
                # This might need adjustment based on layer type (e.g. Linear vs Conv)
                # Assuming Linear layers or inputs are already appropriately shaped for matmul with eigenvectors
                # For Conv layers, input projection is more complex (e.g., on unfolded patches)
                if not isinstance(module, nn.Linear):
                    # This simplified projection mainly works for Linear layers or pre-flattened inputs.
                    # For Conv layers, eigenvectors are typically for unfolded patches.
                    # Handling Conv layers robustly here would require unfolding, projecting, and potentially refolding,
                    # or assuming eigenvectors are compatible with flattened conv input if cnn_mode handles it.
                    logger.warning(f"Eigenvector projection in pre-hook is simplified and assumes Linear-like input for layer {module}. May not be correct for Conv layers without explicit patch handling.")
                
                projected_input = torch.matmul(original_input, evec_matrix) # Project to subspace
                reconstructed_input = torch.matmul(projected_input, evec_matrix.T) # Project back
                corrected_input = reconstructed_input * correction_factor
                return (corrected_input,) + input_tuple[1:] # Return as tuple, keeping other inputs if any
            return pre_hook

        def make_input_capture_hook(storage_dict, name_key):
            def hook(module, input_tuple, output_val):
                # Standard forward hook captures input_tuple[0]
                if input_tuple and isinstance(input_tuple[0], torch.Tensor):
                    storage_dict[name_key] = input_tuple[0].detach().cpu()
                elif input_tuple: # Store first element even if not tensor, for inspection
                    storage_dict[name_key] = input_tuple[0]
                else:
                    logger.warning(f"Input capture hook for {name_key} received empty or no input tuple.")
            return hook
        # --- End Hook Functions ---

        try:
            for layer_idx, (name, module_instance) in enumerate(zip(self.alignment_names, self.alignment_layers)):
                if layer_idx in layers: # This layer undergoes eigenvector dropout
                    eigen_dropout_params_idx = layers.index(layer_idx)
                    current_eigenvalues = eigenvalues[eigen_dropout_params_idx].to(device)
                    current_eigenvectors = eigenvectors[eigen_dropout_params_idx].to(device)
                    nodes_to_remove_indices = idxs[eigen_dropout_params_idx].to(device)
                    
                    # Keep only the eigenvectors that are NOT being dropped
                    # `remove_by_idx` removes based on indices. We need to select based on indices to *keep*.
                    # Or, if eigenvectors correspond to all nodes, select the ones to keep.
                    # Assuming eigenvectors are [num_features, num_eigenvectors]
                    # and idxs are indices of eigenvectors/values to drop.
                    
                    # Create a mask for eigenvectors/values to keep
                    num_total_eigen = current_eigenvectors.shape[1] # Assuming columns are eigenvectors
                    if current_eigenvalues.ndim == 1: num_total_eigen_val = current_eigenvalues.shape[0]
                    else: num_total_eigen_val = current_eigenvalues.shape[1] # if diag matrix
                    
                    if num_total_eigen != num_total_eigen_val:
                        logger.warning(f"Shape mismatch between eigenvectors ({current_eigenvectors.shape}) and eigenvalues ({current_eigenvalues.shape}) for layer {name}")
                        # Fallback or error needed

                    indices_to_keep = [i for i in range(num_total_eigen) if i not in nodes_to_remove_indices.tolist()]
                    if not indices_to_keep:
                        logger.warning(f"All eigenvectors are marked for removal for layer {name}. Using identity projection.")
                        # Create an identity projection or handle as error.
                        # For now, let it pass, matmul with empty tensor might error or result in zeros.
                        # A better fallback might be to not register a hook or use an identity pre-hook.
                        # This case (all eigenvectors removed) should imply output is zero or layer is fully pruned.
                        # The original forward_eigenvector_dropout was also not robust to this.
                        # Let's ensure remaining_eigenvectors is not empty for matmul.
                        remaining_eigenvectors = torch.eye(current_eigenvectors.shape[0], device=device) # Identity if all dropped
                        remaining_eigenvalues_sum = 1e-9 # Avoid div by zero, effectively no scaling

                    else:
                        remaining_eigenvectors = current_eigenvectors[:, indices_to_keep]
                        # Assuming eigenvalues is a 1D tensor corresponding to eigenvectors
                        if current_eigenvalues.ndim == 1:
                            remaining_eigenvalues_sum = torch.sum(current_eigenvalues[indices_to_keep])
                        else: # if eigenvalues is a diagonal matrix
                            remaining_eigenvalues_sum = torch.sum(torch.diag(current_eigenvalues)[indices_to_keep])

                    original_eigenvalues_sum = torch.sum(current_eigenvalues if current_eigenvalues.ndim == 1 else torch.diag(current_eigenvalues))
                    
                    if remaining_eigenvalues_sum < 1e-9:
                        correction = 1.0 # Avoid division by zero / large scaling if all remaining eigenvalues are tiny
                        logger.warning(f"Sum of remaining eigenvalues is near zero for layer {name}. Correction factor set to 1.0.")
                    else:
                        correction = torch.sqrt(original_eigenvalues_sum / remaining_eigenvalues_sum)
                    
                    hook = module_instance.register_forward_pre_hook(make_eigen_pre_hook(remaining_eigenvectors, correction))
                    registered_hooks.append(hook)
                else: # Layer not undergoing eigenvector dropout, capture its input if needed
                    hook = module_instance.register_forward_hook(make_input_capture_hook(hidden_inputs_dict, name))
                    registered_hooks.append(hook)
            
            # Perform the forward pass. Pre-hooks will modify inputs to dropout layers.
            # Standard forward hooks will capture inputs of other layers.
            final_output = self.base_model(x)

        finally:
            for hook in registered_hooks:
                hook.remove()

        # Construct the hidden_outputs list
        processed_hidden_outputs = []
        for name_key in self.alignment_names:
            # Check if this layer was a dropout layer by seeing if its name is in hidden_inputs_dict.
            # If it IS a dropout layer, its input wasn't captured by the input_capture_hook.
            if name_key in hidden_inputs_dict:
                processed_hidden_outputs.append(hidden_inputs_dict[name_key]) # Already detached and on CPU from hook
            else:
                # This was a layer that underwent eigenvector dropout, so its input wasn't stored.
                processed_hidden_outputs.append(None) 

        return final_output, processed_hidden_outputs

    def _forward_subspace(self, x, layer, metaprms, **kwargs):
        if metaprms["unfold"]:
            return self._forward_subspace_convolutional(x, layer, metaprms, **kwargs)
        else:
            return self._forward_subspace_linear(x, layer, metaprms, **kwargs)

    def _forward_subspace_linear(self, x, layer, _, subspace=None, correction=None):
        if subspace is not None:
            x = torch.matmul(torch.matmul(x, subspace), subspace.T)
            if correction is not None:
                x = x * correction
        out = layer(x)
        return out, x

    def _forward_subspace_convolutional(self, x, layer, metaprms, subspace=None, correction=None):
        def _conv_with_subspace(x, layer, subspace, correction):
            conv_layer = layer[0] if isinstance(layer, nn.Sequential) else layer
            if not isinstance(conv_layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                raise TypeError("Layer must be a convolutional layer for subspace convolution.")
            
            h_in, w_in = x.size(2), x.size(3)
            k_h, k_w = conv_layer.kernel_size
            s_h, s_w = conv_layer.stride
            p_h, p_w = conv_layer.padding
            d_h, d_w = conv_layer.dilation
            
            h_max = (h_in + 2*p_h - d_h * (k_h - 1) - 1) // s_h + 1
            w_max = (w_in + 2*p_w - d_w * (k_w - 1) - 1) // s_w + 1
            
            layer_prms = get_unfold_params(conv_layer)
            weight = conv_layer.weight.data
            weight = weight.view(weight.size(0), -1)
            x = torch.nn.functional.unfold(x, conv_layer.kernel_size, **layer_prms)
            x = torch.matmul(subspace, torch.matmul(subspace.T, x))
            if correction is not None:
                x = x * correction
            input_to_conv = x.clone()
            x = torch.matmul(weight, x).view(x.size(0), weight.size(0), h_max, w_max)
            if conv_layer.bias is not None:
                x = x + conv_layer.bias.view(-1, 1, 1)
            return x, input_to_conv
        
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return _conv_with_subspace(x, layer, subspace, correction)
        elif isinstance(layer, nn.Sequential) and isinstance(layer[0], (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return _conv_with_subspace(x, layer, subspace, correction)
        else:
            logger.warning("_forward_subspace_convolutional called with non-convolutional layer type: %s", type(layer))
            return layer(x), x

    @torch.no_grad()
    def measure_eigenfeatures(self, dataloader: DataLoader, num_batches: int = 5, device: Optional[Union[str, torch.device]] = None, with_updates=True, centered=True):
        effective_device = _normalize_device(device if device is not None else get_device(self))
        self.eval()

        target_layers_for_collection = set()
        source_layers_map = {}
        for align_layer_name in self.alignment_names:
            sources = [align_layer_name] 
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                input_spec = self.layer_to_input_names[align_layer_name]
                if input_spec is None: sources = [align_layer_name]
                elif isinstance(input_spec, str): sources = [input_spec]
                elif isinstance(input_spec, list): sources = input_spec
            for source_layer_name in sources:
                 target_layers_for_collection.add(source_layer_name)
                 if source_layer_name not in source_layers_map: source_layers_map[source_layer_name] = []
                 source_layers_map[source_layer_name].append(align_layer_name)

        layers_to_hook_for_input = set()
        layers_to_hook_for_output = set()
        for align_layer_name in self.alignment_names:
            source_layer_names = [align_layer_name]; is_source_output = False
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                 input_spec = self.layer_to_input_names[align_layer_name]
                 if isinstance(input_spec, str): source_layer_names = [input_spec]; is_source_output = True
                 elif isinstance(input_spec, list): source_layer_names = input_spec; is_source_output = True
            if is_source_output:
                for src in source_layer_names: layers_to_hook_for_output.add(src)
            else:
                layers_to_hook_for_input.add(align_layer_name)
                
        all_target_layers_list = sorted(list(layers_to_hook_for_input.union(layers_to_hook_for_output)))
        collect_inputs_flag = bool(layers_to_hook_for_input)
        collect_outputs_flag = bool(layers_to_hook_for_output)

        collected_data = collect_layer_data(
            model=self.base_model if not self._is_ddp_wrapped else self.module.base_model,
            dataloader=dataloader, target_layers=all_target_layers_list, num_batches=num_batches, device=effective_device,
            collect_inputs=collect_inputs_flag, collect_outputs=collect_outputs_flag,
            flatten_spatial=(self.cnn_mode != "patchwise")
        )

        if not collected_data: 
             logger.warning("measure_eigenfeatures: Activation collection failed.")
             return [], [], []

        inputs_for_alignment_layers = []
        for align_layer_name in self.alignment_names:
            source_layer_names = [align_layer_name]; use_output_as_input = False
            if self.layer_to_input_names and align_layer_name in self.layer_to_input_names:
                 input_spec = self.layer_to_input_names[align_layer_name]
                 if isinstance(input_spec, str): source_layer_names = [input_spec]; use_output_as_input = True
                 elif isinstance(input_spec, list): source_layer_names = input_spec; use_output_as_input = True
            combined_source_tensors = []
            activation_type = "output" if use_output_as_input else "input"
            for source_name in source_layer_names:
                 if source_name in collected_data and activation_type in collected_data[source_name]: combined_source_tensors.append(collected_data[source_name][activation_type])
            if not combined_source_tensors: inputs_for_alignment_layers.append(None); continue
            if len(combined_source_tensors) == 1: combined_input = combined_source_tensors[0]
            else: 
                dev = combined_source_tensors[0].device; tensors_on_dev = [t.to(dev) for t in combined_source_tensors]
                try: combined_input = torch.cat(tensors_on_dev, dim=1)
                except Exception: combined_input = None
            inputs_for_alignment_layers.append(combined_input)

        inp_list = self._preprocess_inputs(inputs_for_alignment_layers, compress_convolutional=(self.cnn_mode != "patchwise"))
        w_flat = self.get_alignment_weights(flatten=(self.cnn_mode != "patchwise"))
        
        valid_indices = [i for i, inp in enumerate(inp_list) if inp is not None]
        if len(valid_indices) < len(inp_list):
             logger.warning(f"measure_eigenfeatures: Could not gather inputs for {len(inp_list) - len(valid_indices)} layers. Results will be partial.")
        
        filtered_inp_list = [inp_list[i].to(effective_device) for i in valid_indices]
        filtered_w_flat = [w_flat[i].to(effective_device) for i in valid_indices]

        beta_f, eigvals_f, eigvecs_f = self._measure_layer_eigenfeatures(filtered_inp_list, filtered_w_flat, centered, with_updates)
        
        beta, eigvals, eigvecs = ([None] * len(self.alignment_names) for _ in range(3))
        for i, original_idx in enumerate(valid_indices):
             beta[original_idx] = beta_f[i]
             eigvals[original_idx] = eigvals_f[i]
             eigvecs[original_idx] = eigvecs_f[i]
             
        return beta, eigvals, eigvecs

    @torch.no_grad()
    def _measure_layer_eigenfeatures(self, inputs, weights, centered=True, with_updates=True):
        from tqdm import tqdm
        beta, eigvals, eigvecs = [], [], []
        zipped = enumerate(zip(inputs, weights))
        loop = tqdm(zipped, desc="Layer Eigenfeatures", leave=False, disable=not with_updates)
        for ii, (inp, wght) in loop:
            if inp is None or inp.numel() == 0 or inp.shape[0] < 2:
                 logger.warning(f"Skipping PCA for layer {ii} due to insufficient data (shape: {inp.shape if inp is not None else 'None'}).")
                 beta.append(None)
                 eigvals.append(None)
                 eigvecs.append(None)
                 continue
            try:
                w, v = smart_pca(inp.T, centered=centered)
                norm_wght = torch.norm(wght, dim=1, keepdim=True)
                wght_normalized = wght / (norm_wght + 1e-12)
                beta.append(wght_normalized.cpu() @ v.cpu())
                eigvals.append(w.cpu())
                eigvecs.append(v.cpu())
            except Exception as e:
                logger.error(f"Error during PCA for layer {ii}: {e}", exc_info=True)
                beta.append(None)
                eigvals.append(None)
                eigvecs.append(None)
        return beta, eigvals, eigvecs

    @torch.no_grad()
    def _process_collect_activity(self, dataset, train_set=False, with_updates=False, use_training_mode=False):
        logger.warning("_process_collect_activity seems unused and might be deprecated.")
        loader = dataset.train_loader if train_set else dataset.test_loader
        all_x, all_y = [], []
        original_mode = self.training
        if use_training_mode:
            self.train()
        else:
            self.eval()
        
        loop = tqdm(loader, desc="Collecting Activity", leave=False, disable=not with_updates)
        
        collected_count = 0
        max_samples = 10000
        current_samples = 0

        for batch in loop:
            try:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    logger.warning("Unexpected batch type in _process_collect_activity. Cannot unpack.")
                    continue
            except (IndexError, TypeError) as e:
                logger.warning(f"Error unpacking batch in _process_collect_activity: {e}")
                continue

            all_x.append(x.cpu())
            all_y.append(y.cpu())
            current_samples += x.shape[0]
            collected_count +=1
            if current_samples >= max_samples:
                 logger.info(f"Reached max samples ({max_samples}) for activity collection. Stopping.")
                 break
                
        if not all_x:
             logger.warning("No data collected in _process_collect_activity.")
             if original_mode: self.train()
             return None, None
            
        inputs = torch.cat(all_x, dim=0)
        labels = torch.cat(all_y, dim=0)

        if original_mode:
            self.train()
        else:
            self.eval()
        return inputs, labels

    @torch.no_grad()
    def measure_class_eigenfeatures(self, dataloader: DataLoader, num_batches: int = 5, device: Optional[Union[str, torch.device]] = None, eigenvectors=None, rms=False, with_updates=False):
        logger.warning("measure_class_eigenfeatures: Re-running activation collection. Ensure dataloader is not shuffled for consistency.")
        effective_device = _normalize_device(device if device is not None else get_device(self))
        self.eval()

        all_inputs = []
        all_labels = []
        batches_processed = 0
        try:
            data_iterator = iter(dataloader)
            while batches_processed < num_batches:
                try:
                    batch = next(data_iterator)
                    if isinstance(batch, (list, tuple)):
                        inputs_batch, labels_batch = batch[0].cpu(), batch[1].cpu()
                    else:
                        logger.warning(f"Unsupported batch type: {type(batch)}. Skipping.")
                        continue
                    all_inputs.append(inputs_batch)
                    all_labels.append(labels_batch)
                    batches_processed += 1
                except StopIteration:
                    logger.warning(f"DataLoader exhausted after {batches_processed} batches.")
                    break
        except Exception as e:
            logger.error(f"Error collecting inputs/labels: {e}")
            return []

        if not all_inputs:
             return []

        inputs = torch.cat(all_inputs, dim=0).to(effective_device)
        labels = torch.cat(all_labels, dim=0).to(effective_device)
        num_classes = labels.max().item() + 1

        collected_data = collect_layer_data(
            model=self.base_model if not self._is_ddp_wrapped else self.module.base_model, dataloader=dataloader, 
            target_layers=all_target_layers_list, num_batches=num_batches, device=effective_device,
            collect_inputs=collect_inputs_flag, collect_outputs=collect_outputs_flag,
            flatten_spatial=(self.cnn_mode != "patchwise")
        )

        if not collected_data: return []

        inputs_for_alignment_layers = []
        for align_layer_name in self.alignment_names:
            sources = [align_layer_name]; use_output = False; input_spec = self.layer_to_input_names.get(align_layer_name) if self.layer_to_input_names else None
            if isinstance(input_spec, str): sources = [input_spec]; use_output = True
            elif isinstance(input_spec, list): sources = input_spec; use_output = True
            tensors = []; act_type = "output" if use_output else "input"
            for src in sources: 
                if src in collected_data and act_type in collected_data[src]: tensors.append(collected_data[src][act_type])
            if not tensors: inputs_for_alignment_layers.append(None); continue
            combined = tensors[0] if len(tensors) == 1 else torch.cat([t.to(tensors[0].device) for t in tensors], dim=1)
            inputs_for_alignment_layers.append(combined)

        inp_list = self._preprocess_inputs(inputs_for_alignment_layers, compress_convolutional=True)
        
        if eigenvectors is None or len(eigenvectors) != len(inp_list):
             logger.error("Eigenvectors must be provided and match the number of alignment layers.")
             return []

        all_results = []
        loop = tqdm(enumerate(zip(inp_list, eigenvectors)), total=len(inp_list), desc="Class Eigenfeatures", disable=not with_updates)
        for i, (inp, eigvec) in loop:
            if inp is None or eigvec is None:
                all_results.append(None)
                continue
            
            inp = inp.to(effective_device)
            eigvec = eigvec.to(effective_device)

            projected = torch.matmul(inp, eigvec)
            class_means = torch.zeros(num_classes, projected.size(1), device=effective_device)
            
            for c in range(num_classes):
                class_mask = (labels == c)
                class_inputs = projected[class_mask]
                if class_inputs.size(0) > 0:
                    if rms:
                        class_means[c] = torch.sqrt(torch.mean(class_inputs**2, dim=0))
                    else:
                        class_means[c] = torch.mean(class_inputs, dim=0)
            all_results.append(class_means.cpu())
            
        return all_results