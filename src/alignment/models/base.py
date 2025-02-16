# --------------------------------------------
# base.py
# --------------------------------------------

from warnings import warn
from typing import Optional, Union, List

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

        self.hidden = {}
        self.hooks = {}
        self.cnn_mode = cnn_mode
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
                    # Could be None, str, or list of str
                    val = alignment_layer_names[name]
                    # store as-is
                    self.layer_to_input_names[name] = val

    def is_classification_layer_included(self):
        classification_layer_name = [
            nm for nm, lyr in self.base_model.named_modules() if hasattr(lyr, "weight")
        ][-1]
        return (classification_layer_name in self.alignment_names)

    def num_layers(self, all=False):
        if all:
            return sum(1 for m in self.base_model.modules() if hasattr(m, "weight"))
        return len(self.alignment_layers)

    def setup_forward_hooks(self):
        def get_activation(hook_name):
            def activation_hook(module, in_, out_):
                self.hidden[hook_name] = out_
            return activation_hook

        def get_input(hook_name):
            def input_hook(module, in_, out_):
                self.hidden[hook_name] = in_[0]
            return input_hook

        if self.layer_to_input_names is None:
            # measure alignment for every layer with its own input
            for name, lyr in zip(self.alignment_names, self.alignment_layers):
                self.hooks[name] = lyr.register_forward_hook(get_input(name))
        else:
            # we have a dict => alignment_layer => input-layer(s)
            for name, input_layer in self.base_model.named_modules():
                # name = name of some module
                # if that name is in alignment_layer_names.values(), we set activation hook
                # but now user can specify multiple input layers as a list.
                # So we check if 'name' is among ANY of the values if they're lists or strings
                for align_nm, in_spec in self.layer_to_input_names.items():
                    if in_spec is None:
                        # alignment uses same layer's input => that layer is align_nm
                        if name == align_nm:
                            self.hooks[name] = input_layer.register_forward_hook(get_input(name))
                    elif isinstance(in_spec, str):
                        # single input layer
                        if name == in_spec:
                            self.hooks[name] = input_layer.register_forward_hook(get_activation(name))
                    elif isinstance(in_spec, list):
                        # multiple inputs => if 'name' is in that list
                        if name in in_spec:
                            # we must store them separately => key: align_nm + "@@" + name to avoid collision
                            hook_key = f"{align_nm}@@{name}"
                            self.hooks[hook_key] = input_layer.register_forward_hook(get_activation(hook_key))

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
        assert len(dropout_layers) == len(p), "p must match number of dropout layers"
        for layer, drop_prob in zip(dropout_layers, p):
            layer.p = drop_prob

    @torch.no_grad()
    def get_layer_inputs(self, x, precomputed=False):
        """
        Return the raw input for each alignment layer.
        If alignment_layer_names is None => each layer uses its own input.
        If alignment_layer_names[lyr] = "some_layer" => gather hidden["some_layer"]
        If it's a list => gather from each in the list => cat them along dim=1
        """
        if not precomputed:
            _ = self.forward(x, store_hidden=True)

        layer_inputs = []
        for align_nm in self.alignment_names:
            if self.layer_to_input_names is None:
                # user didn't specify, use self
                src_list = [align_nm]
            else:
                val = self.layer_to_input_names[align_nm]
                if val is None:
                    src_list = [align_nm]
                elif isinstance(val, str):
                    src_list = [val]
                elif isinstance(val, list):
                    src_list = val
                else:
                    raise TypeError(f"Invalid type for layer_to_input_names[{align_nm}]: {type(val)}")

            # gather the actual activation(s)
            sub_inputs = []
            for s in src_list:
                # if we used multi input hooks => "align_nm@@s"
                # only if user gave multiple input-layers for one alignment-layer
                hook_key = s
                # if we used the multi naming => check "align_nm@@s" in self.hidden
                # if it's not found, fallback to self.hidden[s]
                alt_key = f"{align_nm}@@{s}"
                if alt_key in self.hidden:
                    sub_inputs.append(self.hidden[alt_key])
                else:
                    sub_inputs.append(self.hidden[s])

            # e.g. sub_inputs = [ (B, D1), (B, D2) ], cat along dim=1 => (B, D1+D2)
            if len(sub_inputs) == 1:
                combined = sub_inputs[0]
            else:
                combined = torch.cat(sub_inputs, dim=1)
            layer_inputs.append(combined)

        return layer_inputs

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
                        # shape => (B, F, patches) => (B, patches, F) => (B*patches, F)
                        unfolded = unfolded.transpose(1, 2).contiguous().view(-1, unfolded.size(1))
                    preprocessed.append(unfolded)
                elif self.cnn_mode == "patchwise":
                    preprocessed.append(unfolded)
                else:
                    # old fallback
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
    def measure_alignment_methods(self, x, methods, precomputed=False):
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
        from alignment.utils import check_iterable
        assert check_iterable(idxs) and check_iterable(layers), "idxs & layers must be iterables"
        assert len(idxs) == len(layers), "idxs/layers length mismatch"

        hidden_outputs_dict = {}
        hooks = []

        def dropout(hook_name, dropout_idx):
            def dropout_hook(module, in_, out_):
                max_index = out_.shape[1]
                valid_idx = dropout_idx[dropout_idx < max_index]
                frac = len(valid_idx) / float(max_index)
                if valid_idx.numel() > 0:
                    out_[:, valid_idx] = 0
                out_ = out_ * (1 - frac)
                hidden_outputs_dict[hook_name] = out_
                return out_
            return dropout_hook

        def get_output(hook_name):
            def output_hook(module, in_, out_):
                hidden_outputs_dict[hook_name] = out_
            return output_hook

        for idx_layer, (name, layer) in enumerate(zip(self.alignment_names, self.alignment_layers)):
            if idx_layer in layers:
                i_lyr = layers.index(idx_layer)
                d_idx = idxs[i_lyr]
                hooks.append(layer.register_forward_hook(dropout(name, d_idx)))
            else:
                hooks.append(layer.register_forward_hook(get_output(name)))

        x = self.base_model(x)
        for hk in hooks:
            hk.remove()
        assert self.num_layers() == len(hidden_outputs_dict), f"alignment layers mismatch"
        hidden_outputs = [hidden_outputs_dict[nm] for nm in self.alignment_names]
        return x, hidden_outputs

    @torch.no_grad()
    def forward_eigenvector_dropout(self, x, eigenvalues, eigenvectors, idxs, layers):
        from alignment.utils import check_iterable
        device = get_device(x)

        assert check_iterable(idxs) and check_iterable(layers), "idxs/layers must be iterables"
        assert len(idxs) == len(layers), "length mismatch"
        assert len(layers) == len(eigenvalues), "eigenvalues mismatch"
        assert len(layers) == len(eigenvectors), "eigenvectors mismatch"

        hidden_inputs_dict = {}
        hooks = []
        org_forward_methods = {}

        def get_input(nm):
            def input_hook(module, in_, out_):
                hidden_inputs_dict[nm] = in_
            return input_hook

        for idx_layer, (nm, lyr) in enumerate(zip(self.alignment_names, self.alignment_layers)):
            if idx_layer in layers:
                i_lyr = layers.index(idx_layer)
                remove_idx = idxs[i_lyr]
                drop_evec = remove_by_idx(eigenvectors[i_lyr].to(device), remove_idx, 1)
                drop_eval = remove_by_idx(eigenvalues[i_lyr].to(device), remove_idx, 0)
                corr = torch.sqrt(torch.sum(eigenvalues[i_lyr]) / torch.sum(drop_eval))
                self._forward_subspace(nm, lyr, hidden_inputs_dict, hooks, org_forward_methods, subspace=drop_evec, correction=corr)
            else:
                hooks.append(lyr.register_backward_hook(get_input(nm)))

        x = self.base_model(x)
        for hk in hooks:
            hk.remove()
        for nm, lyr_ in zip(self.alignment_names, self.alignment_layers):
            if nm in org_forward_methods:
                lyr_.forward = org_forward_methods[nm]
        assert self.num_layers() == len(hidden_inputs_dict), "alignment layers mismatch"
        hidden_outputs = [hidden_inputs_dict[n] for n in self.alignment_names]
        return x, hidden_outputs

    def _forward_subspace(self, name, layer, hidden_inputs_dict, hooks, org_forward_methods, subspace=None, correction=None):
        if isinstance(layer, nn.Conv2d):
            self._forward_subspace_convolutional(name, layer, hidden_inputs_dict, org_forward_methods, subspace, correction)
        else:
            self._forward_subspace_linear(name, layer, hidden_inputs_dict, hooks, subspace, correction)

    def _forward_subspace_linear(self, name, layer, hidden_inputs_dict, hooks, subspace=None, correction=None):
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
        def _conv_with_subspace(conv_layer, x):
            h_max, w_max = get_maximum_strides(x.size(2), x.size(3), conv_layer)
            layer_prms = get_unfold_params(conv_layer)
            weight_ = conv_layer.weight.data.view(conv_layer.weight.size(0), -1)
            x = torch.nn.functional.unfold(x, conv_layer.kernel_size, **layer_prms)
            if subsp is not None:
                x = x.transpose(1, 2)
                x = torch.matmul(x, subsp.T)
                x = torch.matmul(x, subsp)
                x = x.transpose(1, 2)
                if correction is not None:
                    x = x * correction
            hidden_inputs_dict[name] = x.clone()
            x = torch.matmul(weight_, x).view(x.size(0), weight_.size(0), h_max, w_max)
            x = x + conv_layer.bias.view(-1, 1, 1)
            return x

        subsp = subspace  # for clarity
        if subsp is not None:
            org_forward_methods[name] = layer.forward
            layer.forward = _conv_with_subspace.__get__(layer, nn.Module)

    @torch.no_grad()
    def measure_eigenfeatures(self, inputs, with_updates=True, centered=True):
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
            w, v = smart_pca(inp.T, centered=centered)
            wght = wght / torch.norm(wght, dim=1, keepdim=True)
            beta.append(wght.cpu() @ v)
            eigvals.append(w)
            eigvecs.append(v)
        return beta, eigvals, eigvecs