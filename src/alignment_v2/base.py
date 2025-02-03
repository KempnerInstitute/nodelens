import torch
from torch import nn
from warnings import warn
from utils import check_iterable, get_unfold_params, get_device, remove_by_idx, set_net_mode, alignment

class AttributeReference:
    def __init__(self,parent):
        self.parent=parent
    def __getattr__(self,name):
        if hasattr(self.parent,name):
            return getattr(self.parent,name)
        raise AttributeError(f"No attr {name} in parent")

class AlignmentNetwork(nn.Module):
    def __init__(self, base_model: nn.Module, alignment_layer_names=None, **kwargs):
        super().__init__()
        self.base_model=base_model
        self.alignment_layers=nn.ModuleList()
        self.alignment_names=[]
        self.hidden={}
        self.hooks={}
        self.layer_to_input_names=None
        self._initialize_layers(alignment_layer_names)

    def _initialize_layers(self, aln_names):
        if aln_names is None:
            for n,m in self.base_model.named_modules():
                if hasattr(m,"weight"):
                    self.alignment_layers.append(m)
                    self.alignment_names.append(n)
        else:
            self.layer_to_input_names={}
            for n,m in self.base_model.named_modules():
                if n in aln_names:
                    if hasattr(m,"weight"):
                        self.alignment_layers.append(m)
                        self.alignment_names.append(n)
                        self.layer_to_input_names[n]=aln_names[n]

    def forward(self,x,store_hidden=False):
        if store_hidden:
            self.hidden={}
            self._setup_hooks()
        out=self.base_model(x)
        if store_hidden:
            self._remove_hooks()
        return out

    def _setup_hooks(self):
        def _get_input(n):
            def hk(mod,inp,out):
                self.hidden[n]=inp[0]
            return hk
        def _get_output(n):
            def hk(mod,inp,out):
                self.hidden[n]=out
            return hk
        if self.layer_to_input_names is None:
            for n,m in zip(self.alignment_names,self.alignment_layers):
                self.hooks[n]=m.register_forward_hook(_get_input(n))
        else:
            for n,m in self.base_model.named_modules():
                if self.layer_to_input_names and n in self.layer_to_input_names.values():
                    self.hooks[n]=m.register_forward_hook(_get_output(n))
                elif self.layer_to_input_names and n in self.layer_to_input_names and self.layer_to_input_names[n] is None:
                    self.hooks[n]=m.register_forward_hook(_get_input(n))

    def _remove_hooks(self):
        for _,h in self.hooks.items():
            h.remove()

    def num_layers(self,all=False):
        if all:
            return sum(1 for n,m in self.base_model.named_modules() if hasattr(m,"weight"))
        return len(self.alignment_layers)

    def is_classification_layer_included(self):
        named_w=[n for n,m in self.base_model.named_modules() if hasattr(m,"weight")]
        if not named_w:
            return False
        return named_w[-1] in self.alignment_names

    @torch.no_grad()
    def get_layer_inputs(self, x, precomputed=False):
        if not precomputed:
            _=self.forward(x,store_hidden=True)
        r=[]
        for n in self.alignment_names:
            nm=n
            if self.layer_to_input_names and self.layer_to_input_names[n] is not None:
                nm=self.layer_to_input_names[n]
            r.append(self.hidden[nm])
        return r

    @torch.no_grad()
    def get_alignment_layers(self):
        return self.alignment_layers

    @torch.no_grad()
    def get_alignment_weights(self, flatten=False):
        ws=[]
        for layer in self.alignment_layers:
            w=layer.weight.data.clone()
            if flatten:
                w=w.flatten(start_dim=1)
            ws.append(w)
        return ws

    @torch.no_grad()
    def measure_alignment(self,x,precomputed=False,method="alignment",relative=True):
        ins=self.get_layer_inputs(x,precomputed)
        ws=self.get_alignment_weights(flatten=True)
        out=[]
        for i,w in zip(ins,ws):
            out.append(alignment(i,w,method=method,relative=relative))
        return out
    # more methods (compare_weights, measure_alignment_weights, forward_targeted_dropout, etc.) can be added as needed