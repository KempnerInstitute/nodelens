import torch
from torch import nn
from utils import set_net_mode, default_alignment, get_unfold_params

class AlignmentNetwork(nn.Module):
    def __init__(self, base_model: nn.Module, alignment_layer_names=None, alignment_fn=None):
        super().__init__()
        self.base_model = base_model
        self.alignment_layers = nn.ModuleList()
        self.alignment_names = []
        self.hidden = {}
        self.hooks = {}
        self.layer_to_input_names = None
        self._initialize_layers(alignment_layer_names)
        self.alignment_fn = alignment_fn if alignment_fn is not None else default_alignment

    def _initialize_layers(self, aln_names):
        if aln_names is None:
            for name, module in self.base_model.named_modules():
                if hasattr(module, "weight"):
                    self.alignment_layers.append(module)
                    self.alignment_names.append(name)
        else:
            self.layer_to_input_names = {}
            for name, module in self.base_model.named_modules():
                if name in aln_names:
                    if not hasattr(module, "weight"):
                        continue
                    self.alignment_layers.append(module)
                    self.alignment_names.append(name)
                    self.layer_to_input_names[name] = aln_names[name]

    def forward(self, x, store_hidden=False):
        if store_hidden:
            self.hidden = {}
            self._setup_hooks()
        out = self.base_model(x)
        if store_hidden:
            self._remove_hooks()
        return out

    def _setup_hooks(self):
        def hook_input(name):
            def hook(module, inp, out):
                self.hidden[name] = inp[0]
            return hook
        def hook_output(name):
            def hook(module, inp, out):
                self.hidden[name] = out
            return hook
        if self.layer_to_input_names is None:
            for n, m in zip(self.alignment_names, self.alignment_layers):
                self.hooks[n] = m.register_forward_hook(hook_input(n))
        else:
            for name, module in self.base_model.named_modules():
                if name in self.layer_to_input_names.values():
                    self.hooks[name] = module.register_forward_hook(hook_output(name))
                elif name in self.layer_to_input_names and self.layer_to_input_names[name] is None:
                    self.hooks[name] = module.register_forward_hook(hook_input(name))

    def _remove_hooks(self):
        for h in self.hooks.values():
            h.remove()

    def num_layers(self, all=False):
        if all:
            return sum(1 for m in self.base_model.modules() if hasattr(m, "weight"))
        return len(self.alignment_layers)

    def is_classification_layer_included(self):
        names = [n for n, m in self.base_model.named_modules() if hasattr(m, "weight")]
        return names and names[-1] in self.alignment_names

    @torch.no_grad()
    def get_layer_inputs(self, x, precomputed=False):
        if not precomputed:
            _ = self.forward(x, store_hidden=True)
        outs = []
        for name in self.alignment_names:
            key = name
            if self.layer_to_input_names and self.layer_to_input_names[name] is not None:
                key = self.layer_to_input_names[name]
            outs.append(self.hidden[key])
        return outs

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

    @torch.no_grad()
    def measure_alignment(self, x, precomputed=False, method=None, relative=True):
        # Use the provided alignment function
        inputs = self.get_layer_inputs(x, precomputed)
        weights = self.get_alignment_weights(flatten=True)
        return [self.alignment_fn(inp, w, relative=relative) for inp, w in zip(inputs, weights)]