# -------------------------------------------------
# utils.py (Corrected to include get_maximum_strides)
# -------------------------------------------------

import os
import math
import zipfile
from typing import List
from warnings import warn
from contextlib import contextmanager
from functools import wraps
from natsort import natsorted
import torch
import numpy as np
from scipy.linalg import null_space
from sklearn.decomposition import IncrementalPCA
from alignment.alignment_metrics import alignment

@contextmanager
def no_grad(no_grad=True):
    if no_grad:
        with torch.no_grad():
            yield
    else:
        yield

def test_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        in_training_mode = [set_net_mode(net, training=False) for net in nets]
        func_outputs = func(nets, *args, **kwargs)
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)
        return func_outputs
    return wrapper

def train_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        in_training_mode = [set_net_mode(net, training=True) for net in nets]
        func_outputs = func(nets, *args, **kwargs)
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)
        return func_outputs
    return wrapper

def set_net_mode(net, training=True):
    """
    Helper for toggling train/eval mode of a network.
    """
    in_training_mode = net.training
    if training:
        net.train()
    else:
        net.eval()
    return in_training_mode

def get_device(obj):
    """
    Returns 'cuda' if the module or tensor is on GPU, else 'cpu'.
    """
    if isinstance(obj, torch.nn.Module):
        return next(obj.parameters()).device.type
    elif isinstance(obj, torch.Tensor):
        return "cuda" if obj.is_cuda else "cpu"
    else:
        raise ValueError("get_device: object must be nn.Module or torch.Tensor")

def check_iterable(val):
    """duck-type check if val is iterable"""
    try:
        _ = iter(val)
    except:
        return False
    else:
        return True

def get_maximum_strides(h_input, w_input, layer):
    """
    Helper for computing the number of strides h_max, w_max after
    a convolution with given kernel/stride/padding/dilation.
    """
    # layer.kernel_size, layer.stride, layer.padding, layer.dilation are tuples
    h_out = int(math.floor((h_input + 2*layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0]-1) - 1)
                           / layer.stride[0] + 1))
    w_out = int(math.floor((w_input + 2*layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1]-1) - 1)
                           / layer.stride[1] + 1))
    return h_out, w_out

def get_unfold_params(layer):
    """
    Returns the dict of stride, padding, dilation needed for unfolding
    an input for the given convolutional 'layer'.
    """
    return dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation)

def remove_by_idx(input, idx, dim):
    """
    remove part of `input` along dimension `dim` for the indices in `idx`
    """
    idx_keep = [i for i in range(input.size(dim)) if i not in idx]
    return torch.index_select(input, dim, torch.tensor(idx_keep).to(input.device))

def fractional_histogram(*args, **kwargs):
    """wrapper of np.histogram() with relative counts instead of total or density"""
    counts, bins = np.histogram(*args, **kwargs)
    counts = counts / np.sum(counts)
    return counts, bins

def edge2center(edges):
    """from a list of edges of bins (e.g. for torch.histogram()), return the centers between the edges"""
    assert edges.ndim == 1, "edges must be a 1-d array"
    return edges[:-1] + np.diff(edges) / 2

def smartcorr(input):
    """
    Wraps torch.corrcoef but zeros out rows/cols that have zero variance.
    """
    idx_zeros = torch.var(input, dim=1) == 0
    cc = torch.corrcoef(input)
    cc[idx_zeros, :] = 0
    cc[:, idx_zeros] = 0
    return cc

def batch_cov(input, centered=True, correction=True):
    """
    Batched covariance.
    If input.ndim==3, shape is (batch, dim, samples).
    If input.ndim==2, shape is (dim, samples).
    """
    assert (input.ndim == 2) or (input.ndim == 3), "input must be 2D or 3D tensor"
    assert isinstance(correction, bool), "correction must be bool"

    no_batch = (input.ndim == 2)
    if no_batch:
        input = input.unsqueeze(0)

    S = input.size(2)
    if centered:
        input = input - input.mean(dim=2, keepdim=True)
    bcov = torch.bmm(input, input.transpose(1, 2))
    bcov /= (S - 1.0 * correction)

    if no_batch:
        bcov = bcov.squeeze(0)
    return bcov

def smart_pca(input, centered=True, use_rank=True, correction=True):
    """
    Efficient PCA using SVD or eigen-decomposition depending on shape.
    Falls back on sklearn incremental PCA if it fails to converge.
    """
    # ... [the rest of your PCA code from your snippet] ...
    # Keep your original smart_pca implementation
    # for brevity, omitted here, but be sure it is present
    pass

def eigendecomposition(C, use_rank=True):
    """
    helper for eigen-decomposition
    """
    # ... same as your snippet ...
    pass

def sklearn_pca(input, use_rank=True, rank=None):
    """
    fallback using sklearn IncrementalPCA
    """
    # ... same as your snippet ...
    pass

def fast_rank(input):
    if input.size(-2) < input.size(-1):
        input = torch.transpose(input, -2, -1)
    return int(torch.linalg.matrix_rank(input))

def weighted_average(data, weights, dim, keepdim=False, ignore_nan=False):
    """
    Weighted average of 'data' along dimension 'dim',
    with weighting from 'weights'.
    """
    # ... same as your snippet ...
    pass

def fgsm_attack(image, epsilon, data_grad, transform, sign):
    """update an image with fast-gradient sign method"""
    warn("fgsm_attack is only going to be in utils temporarily!", DeprecationWarning, stacklevel=2)
    if sign:
        data_grad = data_grad.sign()
    else:
        data_grad = data_grad.clone()
    perturbed_image = image + epsilon * data_grad
    perturbed_image = transform(perturbed_image)
    return perturbed_image

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("true", "1"):
        return True
    elif s.lower() in ("false", "0"):
        return False
    else:
        raise TypeError("Boolean type expected")

def save_checkpoint(nets, optimizers, results, path):
    """
    Method for saving checkpoints for networks throughout training.
    """
    multi_model_ckpt = {f"model_state_dict_{i}": net.state_dict() for i, net in enumerate(nets)}
    multi_optimizer_ckpt = {f"optimizer_state_dict_{i}": opt.state_dict() for i, opt in enumerate(optimizers)}
    checkpoint = {**results, **multi_model_ckpt, **multi_optimizer_ckpt}
    torch.save(checkpoint, path)

def load_checkpoints(nets, optimizers, device, path):
    """
    Method for loading presaved checkpoint during training.
    """
    if device == "cpu":
        checkpoint = torch.load(path, map_location=device)
    else:
        checkpoint = torch.load(path)

    net_ids = natsorted([key for key in checkpoint if key.startswith("model_state_dict")])
    opt_ids = natsorted([key for key in checkpoint if key.startswith("optimizer_state_dict")])
    assert all(
        [oi.split("_")[-1] == ni.split("_")[-1] for oi, ni in zip(opt_ids, net_ids)]
    ), "nets and optimizers cannot be matched up from checkpoint"

    for net, net_id in zip(nets, net_ids):
        net.load_state_dict(checkpoint.pop(net_id))
    for opt, opt_id in zip(optimizers, opt_ids):
        opt.load_state_dict(checkpoint.pop(opt_id))

    if device == "cuda":
        for net in nets:
            net.to(device)
    return nets, optimizers, checkpoint

def compress_directory(output_path, directory_path=None):
    """
    Utility to compress entire directory into a zip while respecting .gitignore and ignoring .git.
    """
    pass