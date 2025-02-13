# --------------------------------------------
# base.py
# --------------------------------------------

from warnings import warn
from typing import Optional

import torch
from torch import nn
from tqdm import tqdm

from alignment.utils import (
    check_iterable,
    get_maximum_strides,
    weighted_average,
    get_device,
    remove_by_idx,
    set_net_mode,
    get_unfold_params,
    smart_pca,
)
from alignment.models.layers import (
    LAYER_REGISTRY,
    REGISTRY_REQUIREMENTS,
    check_metaparameters,
)
from alignment.alignment_metrics import AlignmentMetrics
from alignment.alignment_metrics import alignment  

class AlignmentNetwork(nn.Module):
    """
    This is the base class for a neural network used for alignment-related experiments.

    The point of all the wrangling of standard torch workflows in this class is to make
    it easy to perform all the alignment-related computations for networks with different
    architectures without having to rewrite similar code over and over again. In this way,
    the user only needs to pass in a base model, and specify the layers to participate in 
    alignment computation along with their disired input layer. No need for registration 
    the model and layers manually.

    The forward method of **AlignmentNetwork** passes the input (*x*) through the base model
    forward method. If hidden activations are requested, then the output of each layer participated
    in the alignment computation is saved through setting up forward hooks. The alignment methods 
    are applied to the hidden activation at the output of corresponding input layer and the weight
    of layer participate in the experiment.

    Note: some shape wrangling (like that which happens between a convolutional layer and a
    linear layer are often treated as a nn.Module layer), but these don't require alignment-
    related processing.

    Note: to maintain the flexiblity of the code, user is resposible to make sure that the requested
    input layer is feasible for the requested alignment layer and has a compatible output shape for
    alignment computation on the target layer.

    An target alignment layer should have the following properties:
    1. Be a child of the nn.Module class with a forward method
    2. Have processing stage with weights for measuring alignment
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
        self.alignment_layers = nn.ModuleList() # a list of all the layers for the alignment computation
        self.alignment_names = []
        self.hidden = {} # a parallel list to maintain the output/activation of the input layers aka inputs to the alignment layers
        self.hooks = {} # a dictionary list to maintain the handle of the forward hooks on input layers
        self.cnn_mode = cnn_mode  
        self._initialize_layers(alignment_layer_names, **kwargs)

    def _initialize_layers(self, alignment_layer_names, **kwargs):
        """
        Gather modules that have a .weight. If alignment_layer_names is None,
        gather all. Otherwise gather only those in the dict.
        self.alignment_layers will hold the list of layers participating in alignment computation
        self.alignment_names will hold the names of the self.alignment_layers with the same order
        and if alignment_layer_names is not None, self.layer_to_input_names will hold a copy of this map from
        alignment layer names to their input layer names.
        """
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
        """
        convenience method to check if the last layer is included in the alignment layers?
        This check is useful since we don't dropout classification layer
        # it gets the name of the last layer in network and check if it is in alignment layer names
        """
        classification_layer_name = [
            name
            for name, layer in self.base_model.named_modules()
            if hasattr(layer, "weight")
        ][-1]
        return classification_layer_name in self.alignment_names

    def num_layers(self, all=False):
        """
        convenience method for getting the number of layers in network
        if all=False (default), will get the number of alignment layers
        if all=True, will get total number of layers in network that has
        weight attribute and can be used for alignment computation
        """
        if all:
            return sum(1 for m in self.base_model.modules() if hasattr(m, "weight"))
        return len(self.alignment_layers)

    def setup_forward_hooks(self):
        """
        For storing hidden states: either input or output, depending on config.
        """
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
                # Store the input to each alignment layer
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
        """
        standard forward pass of the base_model with option of storing 
        the activation/output of the corresponding input layers to the 
        alignment layers using forward hook
        """
        if store_hidden:
            self.hidden = {}
            self.setup_forward_hooks()
        out = self.base_model(x)
        if store_hidden:
            self.remove_forward_hooks()
        return out

    def get_dropout(self):
        """
        Return list of dropout probability for any dropout layers in network
        """
        p = []
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                p.append(module.p)
        return p

    def set_dropout(self, p):
        """
        Set dropout of all layers in a network
        Note that this will overwrite whatever was previously used
        """
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = p

    def set_dropout_by_layer(self, p):
        """
        Set dropout of each layer in a network independently

        p must be an iterable indicating the probability of dropout for each layer
        """
        dropout_layers = []
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                dropout_layers.append(module)
        assert len(dropout_layers) == len(p), "p must match dropout layers"
        for layer, drop_prob in zip(dropout_layers, p):
            layer.p = drop_prob

    @torch.no_grad()
    def get_layer_inputs(self, x, precomputed=False):
        """
        Return the raw input for each alignment layer (untransformed).
        Typically shape is 4D for conv layers, 2D for linear, etc. 
        """
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
        """
        convenience method for retrieving registered weights for alignment measurements throughout the network

        if flatten=True, will flatten weights so they have shape (nodes/channels, numel_per_weight)
        """
        weights = []
        for layer in self.alignment_layers:
            w = layer.weight.data.clone()
            if flatten:
                w = w.flatten(start_dim=1)
            weights.append(w)
        return weights

    def _preprocess_inputs(self, inputs_to_layers, compress_convolutional=True):
        """
        For each layer:
          - If it's an nn.Conv2d, do 'unfold' or 'patchwise'
          - Otherwise, fallback to flatten or keep 2D
          
        helper method for processing inputs to layers as needed for certain alignment operations

        Operations by layer type
        ------------------------
        linear layer:
            will leave inputs to layers unchanged if input to a feedforward layer
        convolutional layer:
            if compress_convolutional=True, will unfold inputs to (batch * num_strides, conv_weight_dim)
            otherwise, will unfold inputs to (batch, conv_weight_dim, num_strides)
        """
        preprocessed = []
        for input_, layer in zip(inputs_to_layers, self.alignment_layers):
            if isinstance(layer, nn.Conv2d):
                # We specifically check if it's a Conv2d
                layer_prms = get_unfold_params(layer)
                unfolded = torch.nn.functional.unfold(input_, layer.kernel_size, **layer_prms)

                if self.cnn_mode == "unfold":
                    # single-cov approach => shape => [B, F, patches]
                    # => if compress => [B*patches, F]
                    if compress_convolutional:
                        #u = unfolded
                        #unfolded = unfolded.transpose(1, 2).contiguous()
                        #print('new: ',unfolded.shape)
                        unfolded = unfolded.transpose(1, 2).contiguous().view(-1, unfolded.size(1))
                        #print('old: ',unfolded_input.shape)
                        # shape => (B, patches, F)
                        #B, P, F = unfolded.shape
                        #unfolded = unfolded.view(B * P, F)   # => (B*P, F)
                    preprocessed.append(unfolded)

                elif self.cnn_mode == "patchwise":
                    # shape => [B, F, patches], do not flatten
                    preprocessed.append(unfolded)

                else:
                    # fallback => flatten entire batch dimension
                    # e.g. if self.cnn_mode=="old" or something
                    # or just do input_.flatten(start_dim=1)
                    if compress_convolutional:
                        unfolded = unfolded.transpose(1, 2).contiguous()
                        B, P, F = unfolded.shape
                        # fallback => flatten => (B*P, F)
                        unfolded = unfolded.view(B*P, F)
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
    def measure_alignment_methods(self, x, methods, precomputed=False):
        """
        1) get_layer_inputs => raw (4D for conv)
        2) _preprocess_inputs => unfold or patchwise => 2D or 3D
        3) measure alignment
        """
        layer_inputs = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(layer_inputs, compress_convolutional=True)
        weights = self.get_alignment_weights(flatten=True)

        all_layer_results = []
        for idx, (inp, w) in enumerate(zip(preprocessed, weights)):
            metrics_dict = {}
            for m in methods:
                if self.cnn_mode == "patchwise" and inp.ndim == 3:
                    val = AlignmentMetrics.patchwise_alignment(inp, w, method=m, weigh_by_var=True)
                elif m == "delta_alignment":
                    val = AlignmentMetrics.delta_alignment(self, idx, inp)
                else:
                    val = AlignmentMetrics.measure(inp, w, method=m)
                metrics_dict[m] = val
            all_layer_results.append(metrics_dict)
        return all_layer_results

    @torch.no_grad()
    def measure_alignment(self, x, precomputed=False, method="alignment", relative=True):
        """
        Single metric approach (RQ by default).
        """
        layer_inputs = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(layer_inputs, compress_convolutional=True)
        weights = self.get_alignment_weights(flatten=True)

        outputs = []
        for inp, w in zip(preprocessed, weights):
            out = alignment(inp, w, method=method, relative=relative)
            outputs.append(out)
        return outputs

    @torch.no_grad()
    def measure_alignment_weights(self, x, weights, precomputed=False, method="alignment", relative=True):
        """
        If you want to pass in external weights, do the same unfolding or flattening.
        """
        layer_inputs = self.get_layer_inputs(x, precomputed=precomputed)
        preprocessed = self._preprocess_inputs(layer_inputs, compress_convolutional=True)
        weights_flat = [w.flatten(start_dim=1) for w in weights]
        outputs = []
        for inp, w in zip(preprocessed, weights_flat):
            out = alignment(inp, w, method=method, relative=relative)
            outputs.append(out)
        return outputs

    @torch.no_grad()
    def forward_targeted_dropout(self, x, idxs, layers):
        """
        zeroes out certain activation indices in each layer (progressive dropout).
        """
        from alignment.utils import check_iterable
        assert check_iterable(idxs) and check_iterable(layers), "idxs & layers must be iterables"
        assert len(idxs) == len(layers), "idxs/layers must match"
        hidden_outputs_dict = {}
        hooks = []

        def dropout(name, dropout_idx):
            def dropout_hook(module, input, output):
                max_index = output.shape[1]
                dropout_idx_valid = dropout_idx[dropout_idx < max_index]
                fraction_dropout = len(dropout_idx_valid) / float(max_index)
                if dropout_idx_valid.numel() > 0:
                    output[:, dropout_idx_valid] = 0
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
                i_lyr = layers.index(idx_layer)
                d_idx = idxs[i_lyr]
                hooks.append(layer.register_forward_hook(dropout(name, d_idx)))
            else:
                hooks.append(layer.register_forward_hook(get_output(name)))

        x = self.base_model(x)
        for hook in hooks:
            hook.remove()
        assert self.num_layers() == len(hidden_outputs_dict), "mismatch in alignment layers vs. outputs"
        hidden_outputs = [hidden_outputs_dict[nm] for nm in self.alignment_names]
        return x, hidden_outputs

    @torch.no_grad()
    def forward_eigenvector_dropout(self, x, eigenvalues, eigenvectors, idxs, layers):
        """
        remove entire eigenvectors in layer input subspace
        """
        from alignment.utils import check_iterable
        device = get_device(x)
        assert check_iterable(idxs) and check_iterable(layers), "idxs/layers must be iterables"
        assert len(idxs) == len(layers), "idxs/layers mismatch"
        assert len(layers) == len(eigenvalues), "eigenvalues mismatch"
        assert len(layers) == len(eigenvectors), "eigenvectors mismatch"

        hidden_inputs_dict = {}
        hooks = []
        org_forward_methods = {}

        def get_input(name):
            def input_hook(module, input, output):
                hidden_inputs_dict[name] = input
            return input_hook

        for idx_layer, (nm, lyr) in enumerate(zip(self.alignment_names, self.alignment_layers)):
            if idx_layer in layers:
                i_lyr = layers.index(idx_layer)
                d_idx = idxs[i_lyr]
                dropout_evec = remove_by_idx(eigenvectors[i_lyr].to(device), d_idx, 1)
                dropout_eval = remove_by_idx(eigenvalues[i_lyr].to(device), d_idx, 0)
                corr = torch.sqrt(torch.sum(eigenvalues[i_lyr]) / torch.sum(dropout_eval))
                kwargs = dict(subspace=dropout_evec, correction=corr)
                self._forward_subspace(nm, lyr, hidden_inputs_dict, hooks, org_forward_methods, **kwargs)
            else:
                hooks.append(lyr.register_backward_hook(get_input(nm)))

        x = self.base_model(x)
        for hk in hooks:
            hk.remove()
        for nm, lyr_ in zip(self.alignment_names, self.alignment_layers):
            if nm in org_forward_methods:
                lyr_.forward = org_forward_methods[nm]
        assert self.num_layers() == len(hidden_inputs_dict), "mismatch in alignment layers vs. outputs"
        hidden_outputs = [hidden_inputs_dict[n] for n in self.alignment_names]
        return x, hidden_outputs

    def _forward_subspace(self, name, layer, hidden_inputs_dict, hooks, org_forward_methods, subspace=None, correction=None):
        if isinstance(layer, nn.Conv2d):
            self._forward_subspace_convolutional(name, layer, hidden_inputs_dict, org_forward_methods, subspace, correction)
        else:
            self._forward_subspace_linear(name, layer, hidden_inputs_dict, hooks, subspace, correction)

    def _forward_subspace_linear(self, name, layer, hidden_inputs_dict, hooks, subspace=None, correction=None):
        """
        hooking the linear input => multiply by subspace
        """
        def subsapace_linear(_name, hidden_dict, subsp, corr):
            def modify_input_hook(module, in_):
                if subsp is not None:
                    new_input = torch.matmul(torch.matmul(in_[0], subsp), subsp.T)
                    if corr is not None:
                        new_input = new_input * corr
                    hidden_dict[_name] = new_input
                    return (new_input,)
                hidden_dict[_name] = in_[0]
                return in_
            return modify_input_hook

        hooks.append(
            layer.register_forward_pre_hook(
                subsapace_linear(name, hidden_inputs_dict, subspace, correction)
            )
        )

    def _forward_subspace_convolutional(self, name, layer, hidden_inputs_dict, org_forward_methods, subspace=None, correction=None):
        """
        Overwrite the layer's forward to do unfold -> project -> fold -> conv
        """
        def _conv_with_subspace(conv_layer, x):
            h_max, w_max = get_maximum_strides(x.size(2), x.size(3), conv_layer)
            layer_prms = get_unfold_params(conv_layer)
            weight_ = conv_layer.weight.data.view(conv_layer.weight.size(0), -1)
            x = torch.nn.functional.unfold(x, conv_layer.kernel_size, **layer_prms)
            if subspace is not None:
                x = x.transpose(1, 2)
                x = torch.matmul(x, subspace.T)
                x = torch.matmul(x, subspace)
                x = x.transpose(1, 2)
                if correction is not None:
                    x = x * correction
            hidden_inputs_dict[name] = x.clone()
            x = torch.matmul(weight_, x).view(x.size(0), weight_.size(0), h_max, w_max)
            x = x + conv_layer.bias.view(-1, 1, 1)
            return x

        if subspace is not None:
            org_forward_methods[name] = layer.forward
            layer.forward = _conv_with_subspace.__get__(layer, nn.Module)

    @torch.no_grad()
    def measure_eigenfeatures(self, inputs, with_updates=True, centered=True):
        """
        For each alignment layer, gather inputs, unfold if conv, measure PCA, etc.
        """
        w_flat = self.get_alignment_weights(flatten=True)
        inp_list = self._preprocess_inputs(self.get_layer_inputs(inputs, precomputed=False), compress_convolutional=True)
        return self._measure_layer_eigenfeatures(inp_list, w_flat, centered, with_updates)

    @torch.no_grad()
    def _measure_layer_eigenfeatures(self, inputs, weights, centered=True, with_updates=True):
        from tqdm import tqdm
        beta, eigvals, eigvecs = [], [], []
        zipped = enumerate(zip(inputs, weights))
        loop = tqdm(zipped) if with_updates else zipped
        for ii, (inp, wght) in loop:
            # shape => (N, F)
            # do PCA => shape => (F,F)
            w, v = smart_pca(inp.T, centered=centered)
            wght = wght / torch.norm(wght, dim=1, keepdim=True)
            beta.append(wght.cpu() @ v)
            eigvals.append(w)
            eigvecs.append(v)
        return beta, eigvals, eigvecs