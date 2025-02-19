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
            for name, lyr in zip(self.alignment_names, self.alignment_layers):
                self.hooks[name] = lyr.register_forward_hook(get_input(name))
        else:
            for name, input_layer in self.base_model.named_modules():
                for align_nm, in_spec in self.layer_to_input_names.items():
                    if in_spec is None:
                        if name == align_nm:
                            self.hooks[name] = input_layer.register_forward_hook(get_input(name))
                    elif isinstance(in_spec, str):
                        if name == in_spec:
                            self.hooks[name] = input_layer.register_forward_hook(get_activation(name))
                    elif isinstance(in_spec, list):
                        if name in in_spec:
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
        if not precomputed:
            _ = self.forward(x, store_hidden=True)
        layer_inputs = []
        for align_nm in self.alignment_names:
            if self.layer_to_input_names is None:
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
            sub_inputs = []
            for s in src_list:
                hook_key = s
                alt_key = f"{align_nm}@@{s}"
                if alt_key in self.hidden:
                    sub_inputs.append(self.hidden[alt_key])
                else:
                    sub_inputs.append(self.hidden[s])
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
                        unfolded = unfolded.transpose(1, 2).contiguous().view(-1, unfolded.size(1))
                    preprocessed.append(unfolded)
                elif self.cnn_mode == "patchwise":
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
        assert check_iterable(idxs) and check_iterable(layers), "idxs & layers must be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers) == len(set(layers)), "layers must not have any repeated elements"
        assert all([layer >= 0 and layer < len(self.layers) - 1 for layer in layers]), "dropout only works on first N-1 layers"
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
        assert check_iterable(idxs) and check_iterable(layers), "idxs/layers must be iterables with the same length"
        assert len(idxs) == len(layers), "idxs and layers need to be iterables with the same length"
        assert len(layers) == len(eigenvalues), "list of eigenvalues must have same length as list of layers"
        assert len(layers) == len(eigenvectors), "list of eigenvectors must have same length as list of layers"
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
            h_max, w_max = get_maximum_strides(x.size(2), x.size(3), layer)
            layer_prms = get_unfold_params(layer)
            weight = layer.weight.data
            weight = weight.view(weight.size(0), -1)
            x = torch.nn.functional.unfold(x, layer.kernel_size, **layer_prms)
            x = torch.matmul(subspace, torch.matmul(subspace.T, x))
            if correction is not None:
                x = x * correction
            input_to_conv = x.clone()
            x = torch.matmul(weight, x).view(x.size(0), weight.size(0), h_max, w_max)
            x = x + layer.bias.view(-1, 1, 1)
            return x, input_to_conv
        subsp = subspace
        if subsp is not None:
            org_forward_methods = {}
            org_forward_methods[name] = layer.forward
            layer.forward = _conv_with_subspace.__get__(layer, nn.Module)
        else:
            x, input_to_conv = layer(x)
        return x, input_to_conv

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

    @torch.no_grad()
    def _process_collect_activity(self, dataset, train_set=False, with_updates=False, use_training_mode=False):
        loader = dataset.train_loader if train_set else dataset.test_loader
        all_x, all_y = [], []
        original_mode = self.training
        if use_training_mode:
            self.train()
        else:
            self.eval()

        for batch in loader:
            x, y = dataset.unwrap_batch(batch)
            all_x.append(x)
            all_y.append(y)

        inputs = torch.cat(all_x, dim=0)
        labels = torch.cat(all_y, dim=0)

        if not with_updates:
            if use_training_mode:
                self.train()
            else:
                self.eval()
        set_net_mode(self, training=original_mode)
        return inputs, labels

    @torch.no_grad()
    def measure_class_eigenfeatures(self, inputs, labels, eigenvectors, rms=False, with_updates=False):
        if not with_updates:
            self.eval()
        num_classes = labels.max().item() + 1
        layer_data = []
        inp_list = self._preprocess_inputs(self.get_layer_inputs(inputs, precomputed=False), compress_convolutional=True)
        for layer_idx, (inp, vec) in enumerate(zip(inp_list, eigenvectors)):
            class_loadings = []
            for c in range(num_classes):
                mask = (labels == c)
                cdata = inp[mask]
                if cdata.size(0) == 0:
                    proj = torch.zeros(vec.size(1), device=vec.device)
                else:
                    proj = cdata @ vec
                    if rms:
                        proj = torch.sqrt(torch.mean(proj**2, dim=0))
                    else:
                        proj = torch.mean(proj, dim=0)
                class_loadings.append(proj.cpu())
            layer_data.append(torch.stack(class_loadings, dim=0))
        return layer_data