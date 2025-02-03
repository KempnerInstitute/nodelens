from warnings import warn
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from alignment_v2.utils import (check_iterable,
                             get_maximum_strides,
                             weighted_average,
                             get_device,
                             remove_by_idx,
                             set_net_mode,
                             get_unfold_params,
                             smart_pca,
                             alignment)
from alignment_v2.models.layers import (LAYER_REGISTRY, 
                                     REGISTRY_REQUIREMENTS, 
                                     check_metaparameters)

class AttributeReference:
    """
    Simple class designed to be a reference to the parent class as an attribute.
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
    This is the base class for a neural network used for alignment-related experiments.

    The point of all the wrangling of standard torch workflows in this class is to make
    it easy to perform all the alignment-related computations for networks with different
    architectures without having to rewrite similar code over and over again. In this way,
    the user only needs to pass in a base model, and specify the layers to participate in 
    alignment computation along with their disired input layer. No need for registration 
    the model and layers manually.
    """

    def __init__(self, base_model: nn.Module, alignment_layer_names: Optional[dict] = None, **kwargs):
        super().__init__()  # register it as a nn.Module
        self.base_model = base_model
        self.alignment_layers = nn.ModuleList()
        self.alignment_names = []
        self.hidden = {}
        self.hooks = {}
        self._initialize_layers(alignment_layer_names, **kwargs)

    def _initialize_layers(self, alignment_layer_names, **kwargs):
        if alignment_layer_names is None:
            self.layer_to_input_names = None
            for name, layer in self.base_model.named_modules():
                if not hasattr(layer, 'weight'): continue
                self.alignment_layers.append(layer)
                self.alignment_names.append(name)
        else:
            self.layer_to_input_names = {}
            for name, layer in self.base_model.named_modules():
                if name in alignment_layer_names.keys():
                    if not hasattr(layer, 'weight'):
                        warn_message = f"Skipping the selected layer {name} ({layer.__class__.__name__}) becuase it does not have a weight attribute"
                        warn(warn_message, RuntimeWarning, stacklevel=1)
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
        dropout_layers = []
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)

        assert len(dropout_layers) == len(p), "p must contain the same number of elements as the number of dropout layers in the network"
        for layer, drop_prob in zip(dropout_layers, p):
            layer.p = drop_prob

    @torch.no_grad()
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

    @torch.no_grad()
    def get_alignment_layers(self):
        return self.alignment_layers

    @torch.no_grad()
    def get_alignment_weights(self, flatten=False):
        weights = []
        for layer in self.alignment_layers:
            weight = layer.weight.data.clone()
            if flatten:
                weight = weight.flatten(start_dim=1)
            weights.append(weight)
        return weights

    def _preprocess_inputs(self, inputs_to_layers, compress_convolutional=True):
        preprocessed = []
        for input, layer in zip(inputs_to_layers, self.alignment_layers):
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                layer_prms = get_unfold_params(layer)
                unfolded_input = torch.nn.functional.unfold(input, layer.kernel_size, **layer_prms)
                if compress_convolutional:
                    unfolded_input = unfolded_input.transpose(1, 2).contiguous().view(-1, unfolded_input.size(1))
                preprocessed.append(unfolded_input)
            else:
                preprocessed.append(input)
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
    def measure_alignment(self, x, precomputed=False, method="alignment", relative=True):
        inputs_to_layers = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(inputs_to_layers, compress_convolutional=True)
        weights = self.get_alignment_weights(flatten=True)
        return [alignment(input, weight, method=method, relative=relative) for input, weight in zip(preprocessed, weights)]

    @torch.no_grad()
    def measure_alignment_weights(self, x, weights, precomputed=False, method="alignment", relative=True):
        inputs_to_layers = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(inputs_to_layers, compress_convolutional=True)
        weights = [w.flatten(start_dim=1) for w in weights]
        return [alignment(input, weight, method=method, relative=relative) for input, weight in zip(preprocessed, weights)]

    @torch.no_grad()
    def forward_targeted_dropout(self, x, idxs, layers):
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers) == len(set(layers)), "layers must not have any repeated elements"

        hidden_outputs_dict = {}
        hooks = []

        def dropout(name, dropout_idx):
            def dropout_hook(module, input, output):
                fraction_dropout = len(dropout_idx) / output.shape[1]
                output[:, dropout_idx] = 0
                output = output * (1 - fraction_dropout)
                hidden_outputs_dict[name] = output
                return output
            return dropout_hook
        
        def get_output(name):
            def output_hook(module, input, output):
                hidden_outputs_dict[name] = output
            return output_hook
        
        for idx_layer, (name, layer) in enumerate(zip(self.alignment_names, self.alignment_layers)):
            if idx_layer in layers:
                dropout_idx = idxs[{val: idx for idx, val in enumerate(layers)}[idx_layer]]
                hooks.append(layer.register_forward_hook(dropout(name , dropout_idx)))
            else:
                hooks.append(layer.register_forward_hook(get_output(name)))
        
        x = self.base_model(x)
        for hook in hooks:
            hook.remove()
        
        assert self.num_layers() == len(hidden_outputs_dict), "number of outputs and the number of alignment layers need to be the same"
        hidden_outputs = [hidden_outputs_dict[name] for name in self.alignment_names]
        return x, hidden_outputs

    @torch.no_grad()
    def forward_eigenvector_dropout(self, x, eigenvalues, eigenvectors, idxs, layers):
        assert check_iterable(idxs) and check_iterable(layers), "idxs and layers need to be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers) == len(set(layers)), "layers must not have any repeated elements"
        assert len(layers) == len(eigenvalues), "list of eigenvalues must have same length as list of layers"
        assert len(layers) == len(eigenvectors), "list of eigenvectors must have same length as list of layers"
        device = get_device(x)
        
        hidden_inputs_dict = {}
        hooks = []
        org_forward_methods = {}

        def get_input(name):
            def input_hook(module, input, output):
                hidden_inputs_dict[name] = input
            return input_hook

        for idx_layer, (name, layer) in enumerate(zip(self.alignment_names, self.alignment_layers)):
            if idx_layer in layers:
                idx_to_layer = {val: idx for idx, val in enumerate(layers)}[idx_layer]
                dropout_idx = idxs[idx_to_layer]
                dropout_evec = remove_by_idx(eigenvectors[idx_to_layer].to(device), dropout_idx, 1)
                dropout_eval = remove_by_idx(eigenvalues[idx_to_layer].to(device), dropout_idx, 0)
                dropout_correction = torch.sqrt(torch.sum(eigenvalues[idx_to_layer]) / torch.sum(dropout_eval))
                kwargs = dict(subspace=dropout_evec, correction=dropout_correction)
                self._forward_subspace(name, layer, hidden_inputs_dict, hooks, org_forward_methods, **kwargs)
            else:
                hooks.append(layer.register_backward_hook(get_input(name)))
        
        x = self.base_model(x)
        
        for hook in hooks:
            hook.remove()
        
        for name, layer in zip(self.alignment_names, self.alignment_layers):
            if name in org_forward_methods.keys():
                layer.forward = org_forward_methods[name]

        assert self.num_layers() == len(hidden_inputs_dict), f"number of inputs {len(hidden_inputs_dict)} and the number of alignment layers {self.num_layers()} need to be the same"
        hidden_inputs = [hidden_inputs_dict[name] for name in self.alignment_names]
        return x, hidden_inputs

    def _forward_subspace(self, name, layer, hidden_inputs_dict, hooks, org_forward_methods, subspace=None, correction=None):
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            self._forward_subspace_convolutional(name, layer, hidden_inputs_dict, org_forward_methods, subspace=subspace, correction=correction)
        else:
            self._forward_subspace_linear(name, layer, hidden_inputs_dict, hooks, subspace=subspace, correction=correction)

    def _forward_subspace_linear(self, name, layer, hidden_inputs_dict, hooks, subspace=None, correction=None):
        def subsapace_linear(name, hidden_inputs_dict, subspace, correction):
            def modify_input_hook(module, input):
                if subspace is not None:
                    input = torch.matmul(torch.matmul(input[0], subspace), subspace.T)
                    if correction is not None:
                        input = input * correction
                hidden_inputs_dict[name] = input
                return input
            return modify_input_hook

        hooks.append(layer.register_forward_pre_hook(subsapace_linear(name, hidden_inputs_dict, subspace, correction)))

    def _forward_subspace_convolutional(self, name, layer, hidden_inputs_dict, org_forward_methods, subspace=None, correction=None):
        def _conv_with_subspace(self, x, name=name, hidden_inputs_dict=hidden_inputs_dict, subspace=subspace, correction=correction):
            h_max, w_max = get_maximum_strides(x.size(2), x.size(3), self)
            layer_prms = get_unfold_params(self)
            weight = self.weight.data
            weight = weight.view(weight.size(0), -1)
            x = torch.nn.functional.unfold(x, self.kernel_size, **layer_prms)
            x = torch.matmul(subspace, torch.matmul(subspace.T, x))
            if correction is not None:
                x = x * correction
            input_to_conv = x.clone()
            hidden_inputs_dict[name] = input_to_conv
            x = torch.matmul(weight, x).view(x.size(0), weight.size(0), h_max, w_max)
            x = x + self.bias.view(-1, 1, 1)
            return x

        if subspace is not None:
            org_forward_methods[name] = layer.forward
            layer.forward = _conv_with_subspace.__get__(layer, nn.Module)

    @torch.no_grad()
    def measure_eigenfeatures(self, inputs, with_updates=True, centered=True):
        weights = self.get_alignment_weights(flatten=True)
        inputs = self._preprocess_inputs(inputs, compress_convolutional=True)
        return self._measure_layer_eigenfeatures(inputs, weights, centered=centered, with_updates=with_updates)

    def measure_class_eigenfeatures(self, inputs, labels, eigenvectors, rms=False, with_updates=True):
        classes = torch.unique(labels)
        num_classes = len(classes)
        idx_to_class = [torch.where(labels == ii)[0] for ii in classes]
        num_per_class = [len(idx) for idx in idx_to_class]
        min_per_class = min(num_per_class)
        if any([npc > min_per_class for npc in num_per_class]):
            max_per_class = max(num_per_class)
            if (max_per_class / min_per_class) > 2:
                warn_message = f"Number of elements to each class is unequal (min={min_per_class}, max={max_per_class}). Clipping examples."
                warn(warn_message, RuntimeWarning, stacklevel=1)
            idx_to_class = [idx[:min_per_class] for idx in idx_to_class]
        import torch
        idx_to_class = torch.stack(idx_to_class).unsqueeze(1)
        beta_activity = []
        inputs = self._preprocess_inputs(inputs, compress_convolutional=False)
        zipped = zip(inputs, eigenvectors, self.get_alignment_layers())
        for input, evec, layer in zipped:
            if isinstance(layer, torch.nn.modules.conv.Conv2d):
                print("measure_class_eigenfeatures has not integrated new convolutional approach")
                stride_var = torch.var(input, dim=1, keepdim=True)
                projection = torch.matmul(evec.T, input)
                projection = weighted_average(projection, stride_var, dim=2)
                beta_activity.append(projection.T.unsqueeze(0))
            else:
                beta_activity.append((input @ evec).T.unsqueeze(0))
        beta_by_class = [torch.gather(betas.expand(num_classes, -1, -1), 2, idx_to_class.expand(-1, betas.size(1), -1)) for betas in beta_activity]
        if rms:
            beta_by_class = [torch.sqrt(torch.mean(beta**2, dim=2)) for beta in beta_by_class]
        return beta_by_class

    def _measure_layer_eigenfeatures(self, inputs, weights, centered=True, with_updates=True):
        beta, eigenvalues, eigenvectors = [], [], []
        zipped = enumerate(zip(inputs, weights))
        iterate = tqdm(zipped) if with_updates else zipped
        for ii, (input, weight) in iterate:
            w, v = smart_pca(input.T, centered=centered)
            weight = weight / torch.norm(weight, dim=1, keepdim=True)
            beta.append(weight.cpu() @ v)
            eigenvalues.append(w)
            eigenvectors.append(v)
        return beta, eigenvalues, eigenvectors

    def _process_collect_activity(self, dataset, train_set=True, with_updates=True, use_training_mode=False):
        device = get_device(self)
        training_mode = set_net_mode(self, training=use_training_mode)
        allinputs = []
        alllabels = []
        dataloader = dataset.train_loader if train_set else dataset.test_loader
        from tqdm import tqdm
        dataloop = tqdm(dataloader) if with_updates else dataloader
        for batch in dataloop:
            input, labels = dataset.unwrap_batch(batch, device=device)
            layer_inputs = [input.cpu() for input in self.get_layer_inputs(input, precomputed=False)]
            allinputs.append(layer_inputs)
            alllabels.append(labels.cpu())
        set_net_mode(self, training=training_mode)
        inputs = [torch.cat([input[layer] for input in allinputs], dim=0) for layer in range(self.num_layers())]
        labels = torch.cat(alllabels, dim=0)
        return inputs, labels

    @torch.no_grad()
    def shape_eigenfeatures(self, idx_layers, eigenvalues, eigenvectors, eval_transform):
        assert all([idx in range(self.num_layers()) for idx in idx_layers]), (
            "idx_layers includes some indices not in alignment layers",
            f"(provided: {idx_layers}, alignment layer indecies: {list(range(self.num_layers()))})",
        )
        assert len(idx_layers) == len(eigenvalues), "length of idx_layers and eigenvalues doesn't match"
        assert len(idx_layers) == len(eigenvectors), "length of idx_layers and eigenvectors doesn't match"
        device = get_device(self)
        eigenvalues = [evals.to(device) for evals in eigenvalues]
        eigenvectors = [evecs.to(device) for evecs in eigenvectors]
        weight_shape = [self.get_alignment_weights()[idx].shape for idx in idx_layers]
        weights = [self.get_alignment_weights(flatten=True)[idx] for idx in idx_layers]
        norm_of_weights = [torch.norm(weight, dim=1, keepdim=True) for weight in weights]
        weights = [weight / torch.norm(weight, dim=1, keepdim=True) for weight in weights]
        from torch import diag
        zipped = zip(idx_layers, eigenvalues, eigenvectors, weights, norm_of_weights, weight_shape)
        for idx, evals, evecs, weight, norm_weight, shape in zipped:
            eval_keep_fraction = eval_transform(evals)
            assert (
                type(eval_keep_fraction) == type(evals) and eval_keep_fraction.shape == evals.shape
            ), "eval_transform returned new evals with the wrong type or shape"
            proj_matrix = evecs @ torch.diag(eval_keep_fraction) @ evecs.T
            shaped_weights = weight @ proj_matrix
            shaped_weights = shaped_weights / torch.norm(shaped_weights, dim=1, keepdim=True)
            shaped_weights = shaped_weights * norm_weight
            shaped_weights = torch.reshape(shaped_weights, shape)
            self.get_alignment_layers()[idx].weight.data = shaped_weights