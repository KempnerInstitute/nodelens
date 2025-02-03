import torch
import numpy as np
from functools import wraps
from contextlib import contextmanager
import math
from torch.linalg import matrix_rank, eigh

def get_unfold_params(layer):
    return dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation)

def set_net_mode(net, training=True):
    old = net.training
    net.train() if training else net.eval()
    return old

def test_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        modes = [set_net_mode(n, False) for n in nets]
        out = func(nets, *args, **kwargs)
        for m, n in zip(modes, nets):
            set_net_mode(n, m)
        return out
    return wrapper

def train_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        modes = [set_net_mode(n, True) for n in nets]
        out = func(nets, *args, **kwargs)
        for m, n in zip(modes, nets):
            set_net_mode(n, m)
        return out
    return wrapper

def default_alignment(x, w, relative=True):
    # Rayleigh quotient: (w^T C w) / (w^T w)
    # where C is covariance of x (assumes x is (batch, features))
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    c = torch.cov(x.T)
    rq = torch.sum((w @ c) * w, dim=1) / torch.sum(w * w, dim=1)
    if relative:
        return rq / torch.trace(c)
    return rq

def smart_pca(x, centered=True, use_rank=True, correction=True):
    # x: (samples, features)
    if centered:
        x = x - x.mean(dim=0, keepdim=True)
    if x.size(1) > x.size(0):
        # use SVD
        u, s, v = torch.linalg.svd(x, full_matrices=False)
        w = (s**2) / (x.size(0) - 1)
        # pad zeros if needed
        if w.size(0) < x.size(1):
            pad = torch.zeros(x.size(1) - w.size(0), device=w.device)
            w = torch.cat((w, pad))
        return w, v.T
    else:
        c = torch.cov(x.T)
        w, v = eigh(c)
        idx = torch.argsort(-w)
        w = w[idx]
        v = v[:, idx]
        if use_rank:
            r = matrix_rank(c)
            w[r:] = 0
        return w, v

def transpose_list(lol):
    return list(map(list, zip(*lol)))

def named_transpose(lol, reduction=None):
    if reduction is not None:
        return map(reduction, zip(*lol))
    return list(map(list, zip(*lol)))

def compute_stats_by_type(tensor, num_types, dim, method="se"):
    num = tensor.size(dim)
    per_type = num // num_types
    tensor = tensor.unsqueeze(dim)
    new_shape = list(tensor.shape)
    new_shape[dim] = num_types
    new_shape.insert(dim+1, per_type)
    tensor = tensor.view(new_shape)
    means = torch.mean(tensor, dim=dim+1)
    if method=="se":
        dev = torch.std(tensor, dim=dim+1) / np.sqrt(per_type)
    elif method=="std":
        dev = torch.std(tensor, dim=dim+1)
    elif method=="var":
        dev = torch.var(tensor, dim=dim+1)
    elif method=="range":
        dev = tensor.max(dim=dim+1).values - tensor.min(dim=dim+1).values
    else:
        raise ValueError("Unknown method")
    return means, dev

def rms(x, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(torch.mean(x**2))
    return torch.sqrt(torch.mean(x**2, dim=dim, keepdim=keepdim))

@contextmanager
def no_grad(no_grad=True):
    if no_grad:
        with torch.no_grad():
            yield
    else:
        yield

def fgsm_attack(image, epsilon, data_grad, transform, use_sign):
    if use_sign:
        data_grad = data_grad.sign()
    else:
        data_grad = data_grad.clone()
    perturbed = image + epsilon * data_grad
    perturbed = transform(perturbed)
    return perturbed

def condense_values(full):
    # full: list[network][batch][layer]
    num_layers = len(full[0][0])
    return [torch.stack([torch.cat([f[layer].unsqueeze(0) for f in net], dim=0)
                         for net in full]) for layer in range(num_layers)]

def save_checkpoint(nets, opts, results, path):
    ckpt = {f"model_state_dict_{i}": net.state_dict() for i, net in enumerate(nets)}
    ckpt.update({f"optimizer_state_dict_{i}": opt.state_dict() for i, opt in enumerate(opts)})
    ckpt.update(results)
    torch.save(ckpt, path)

def load_checkpoints(nets, opts, device, path):
    ckpt = torch.load(path, map_location=device)
    from natsort import natsorted
    model_keys = natsorted([k for k in ckpt if k.startswith("model_state_dict")])
    opt_keys = natsorted([k for k in ckpt if k.startswith("optimizer_state_dict")])
    for net, key in zip(nets, model_keys):
        net.load_state_dict(ckpt.pop(key))
    for opt, key in zip(opts, opt_keys):
        opt.load_state_dict(ckpt.pop(key))
    if device=="cuda":
        for net in nets:
            net.to(device)
    return nets, opts, ckpt