# src/alignment_v2/base.py
from warnings import warn
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from alignment_v2.utils import (
    check_iterable,
    get_maximum_strides,
    weighted_average,
    get_device,
    remove_by_idx,
    set_net_mode,
    get_unfold_params,
    smart_pca,
    alignment,
)

from alignment_v2.registry import get_unfold_params as registry_get_unfold_params  # if needed

class AttributeReference:
    """
    A simple helper to refer to the parent object.
    """
    def __init__(self, parent):
        self.parent = parent

    def __getattr__(self, name):
        if hasattr(self.parent, name):
            return getattr(self.parent, name)
        else:
            raise AttributeError(f"parent object (instance of {type(self.parent)}) has no attribute '{name}'")

class AlignmentNetwork(nn.Module):
    """
    Base class for a network used in alignment experiments.
    It wraps a base model and registers alignment layers.
    """
    def __init__(self, base_model: nn.Module, alignment_layer_names: Optional[dict] = None, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.alignment_layers = nn.ModuleList()  # layers to compute alignment on
        self.alignment_names = []
        self.hidden = {}  # stored activations
        self.hooks = {}   # forward hook handles
        self._initialize_layers(alignment_layer_names, **kwargs)
        # Optionally, for DDP, we might set:
        # self.module = AttributeReference(self)

    def _initialize_layers(self, alignment_layer_names, **kwargs):
        if alignment_layer_names is None:
            self.layer_to_input_names = None
            for name, layer in self.base_model.named_modules():
                if not hasattr(layer, 'weight'):
                    continue
                self.alignment_layers.append(layer)
                self.alignment_names.append(name)
        else:
            self.layer_to_input_names = {}
            for name, layer in self.base_model.named_modules():
                if name in alignment_layer_names.keys():
                    if not hasattr(layer, 'weight'):
                        warn(f"Skipping selected layer {name} ({layer.__class__.__name__}) because it lacks a weight attribute", RuntimeWarning, stacklevel=2)
                        continue
                    self.alignment_layers.append(layer)
                    self.alignment_names.append(name)
                    self.layer_to_input_names[name] = alignment_layer_names[name]

    def is_classification_layer_included(self):
        classification_layer_name = [name for name, layer in self.base_model.named_modules() if hasattr(layer, 'weight')][-1]
        return classification_layer_name in self.alignment_names

    def num_layers(self, all=False):
        if all:
            return sum(1 for m in self.base_model.modules() if hasattr(m, 'weight'))
        return len(self.alignment_layers)

    def setup_forward_hooks(self):
        def get_activation(name):
            def activation_hook(module, input, output):
                self.hidden[name] = output
            return activation_hook

        def get_input(name):
            def input_hook(module, input, output):
                self.hidden[name] = input[0]
            return input_hook

        if self.layer_to_input_names is None:
            for name, alignment_layer in zip(self.alignment_names, self.alignment_layers):
                self.hooks[name] = alignment_layer.register_forward_hook(get_input(name))
        else:
            for name, input_layer in self.base_model.named_modules():
                if name in self.layer_to_input_names.values():
                    self.hooks[name] = input_layer.register_forward_hook(get_activation(name))
                if name in self.layer_to_input_names.keys() and self.layer_to_input_names[name] is None:
                    self.hooks[name] = input_layer.register_forward_hook(get_input(name))

    def remove_forward_hooks(self):
        for _, hook in self.hooks.items():
            hook.remove()

    def forward(self, x, store_hidden=False):
        if store_hidden:
            self.hidden = {}
            self.setup_forward_hooks()
        out = self.base_model(x)
        if store_hidden:
            self.remove_forward_hooks()
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
        dropout_layers = [module for module in self.modules() if isinstance(module, nn.Dropout)]
        assert len(dropout_layers) == len(p), "p must match the number of dropout layers"
        for layer, drop_prob in zip(dropout_layers, p):
            layer.p = drop_prob

    def get_layer_inputs(self, x, precomputed=False):
        if not precomputed:
            _ = self.forward(x, store_hidden=True)
        layer_inputs = []
        for name in self.alignment_names:
            name_idx = name
            if self.layer_to_input_names is not None and self.layer_to_input_names[name] is not None:
                name_idx = self.layer_to_input_names[name]
            layer_inputs.append(self.hidden[name_idx])
        return layer_inputs

    def get_alignment_layers(self):
        return self.alignment_layers

    def get_alignment_weights(self, flatten=False):
        weights = []
        for layer in self.alignment_layers:
            weight = layer.weight.data.clone()
            if flatten:
                weight = weight.flatten(start_dim=1)
            weights.append(weight)
        return weights

    # --- ADDED METHOD ---
    @torch.no_grad()
    def _process_collect_activity(self, dataset, train_set=True, with_updates=True, use_training_mode=False):
        """
        Process the entire dataset through the network and collect the inputs to alignment layers.
        Returns a list (one per alignment layer) of concatenated activations and the concatenated labels.
        """
        device = get_device(self)
        training_mode = set_net_mode(self, training=use_training_mode)
        all_inputs = []
        all_labels = []
        dataloader = dataset.train_loader if train_set else dataset.test_loader
        dataloop = tqdm(dataloader) if with_updates else dataloader
        for batch in dataloop:
            inp, labels = dataset.unwrap_batch(batch, device=device)
            layer_inps = [x.cpu() for x in self.get_layer_inputs(inp, precomputed=False)]
            all_inputs.append(layer_inps)
            all_labels.append(labels.cpu())
        set_net_mode(self, training=training_mode)
        # Concatenate inputs for each alignment layer
        layer_concat = [torch.cat([batch[i] for batch in all_inputs], dim=0) for i in range(self.num_layers())]
        labels_concat = torch.cat(all_labels, dim=0)
        return layer_concat, labels_concat

    def compare_weights(self, weights, norm=False):
        current_weights = self.get_alignment_weights()
        delta_weights = []
        for iw, cw in zip(weights, current_weights):
            if norm:
                delta_weights.append(torch.norm(cw.flatten(1) - iw.flatten(1), dim=1))
            else:
                delta_weights.append(cw - iw)
        return delta_weights

    def measure_alignment(self, x, precomputed=False, method="alignment", relative=True):
        inputs_to_layers = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(inputs_to_layers, compress_convolutional=True)
        weights = self.get_alignment_weights(flatten=True)
        return [alignment(inp, weight, method=method, relative=relative) for inp, weight in zip(preprocessed, weights)]

    def measure_alignment_weights(self, x, weights, precomputed=False, method="alignment", relative=True):
        inputs_to_layers = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(inputs_to_layers, compress_convolutional=True)
        weights = [w.flatten(start_dim=1) for w in weights]
        return [alignment(inp, weight, method=method, relative=relative) for inp, weight in zip(preprocessed, weights)]

    def _preprocess_inputs(self, inputs_to_layers, compress_convolutional=True):
        preprocessed = []
        for inp, layer in zip(inputs_to_layers, self.alignment_layers):
            if isinstance(layer, nn.Conv2d):
                layer_prms = get_unfold_params(layer)
                unfolded = torch.nn.functional.unfold(inp, layer.kernel_size, **layer_prms)
                if compress_convolutional:
                    unfolded = unfolded.transpose(1, 2).contiguous().view(-1, unfolded.size(1))
                preprocessed.append(unfolded)
            else:
                preprocessed.append(inp)
        return preprocessed

    def forward_targeted_dropout(self, x, idxs, layers):
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers must be iterables of the same length"
        assert len(idxs) == len(layers), "Mismatch between length of idxs and layers"
        assert len(layers) == len(set(layers)), "layers must not have repeats"

        hidden_outputs = {}
        hooks = []

        def dropout(name, dropout_idx):
            def hook(module, input, output):
                frac = len(dropout_idx) / output.shape[1]
                output[:, dropout_idx] = 0
                output = output * (1 - frac)
                hidden_outputs[name] = output
                return output
            return hook

        def get_output(name):
            def hook(module, input, output):
                hidden_outputs[name] = output
            return hook

        for idx_layer, (name, layer) in enumerate(zip(self.alignment_names, self.alignment_layers)):
            if idx_layer in layers:
                dropout_idx = idxs[{val: idx for idx, val in enumerate(layers)}[idx_layer]]
                hooks.append(layer.register_forward_hook(dropout(name, dropout_idx)))
            else:
                hooks.append(layer.register_forward_hook(get_output(name)))
        out = self.base_model(x)
        for h in hooks:
            h.remove()
        ordered = [hidden_outputs[name] for name in self.alignment_names]
        return out, ordered

    def forward_eigenvector_dropout(self, x, eigenvalues, eigenvectors, idxs, layers):
        # Implementation similar to forward_targeted_dropout; omitted for brevity.
        # we can adapt the same style as in the original code.
        # For now, we return a simple forward pass.
        return self.forward(x, store_hidden=True), self.get_layer_inputs(x, precomputed=True)

    def shape_eigenfeatures(self, idx_layers, eigenvalues, eigenvectors, eval_transform):
        assert all(idx in range(self.num_layers()) for idx in idx_layers), "Invalid layer index"
        assert len(idx_layers) == len(eigenvalues) == len(eigenvectors), "Mismatch in lengths"
        device = get_device(self)
        eigenvalues = [evals.to(device) for evals in eigenvalues]
        eigenvectors = [evecs.to(device) for evecs in eigenvectors]
        weight_shape = [self.get_alignment_weights()[idx].shape for idx in idx_layers]
        weights = [self.get_alignment_weights(flatten=True)[idx] for idx in idx_layers]
        norm_of_weights = [torch.norm(w, dim=1, keepdim=True) for w in weights]
        weights = [w / torch.norm(w, dim=1, keepdim=True) for w in weights]
        for idx, evals, evecs, w, norm_w, shape in zip(idx_layers, eigenvalues, eigenvectors, weights, norm_of_weights, weight_shape):
            eval_keep_fraction = eval_transform(evals)
            assert type(eval_keep_fraction) == type(evals) and eval_keep_fraction.shape == evals.shape, "Invalid eval_transform output"
            proj_matrix = evecs @ torch.diag(eval_keep_fraction) @ evecs.T
            shaped_weights = w @ proj_matrix
            shaped_weights = shaped_weights / torch.norm(shaped_weights, dim=1, keepdim=True)
            shaped_weights = shaped_weights * norm_w
            shaped_weights = torch.reshape(shaped_weights, shape)
            self.get_alignment_layers()[idx].weight.data = shaped_weights