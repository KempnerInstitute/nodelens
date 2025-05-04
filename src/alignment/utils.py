# --------------------------------------------
# utils.py
# --------------------------------------------

import os
import math
import zipfile
from typing import List, Dict, Union, Callable, Any, Optional, Tuple, TypeVar
from warnings import warn
from contextlib import contextmanager
from functools import wraps
from natsort import natsorted
from gitignore_parser import parse_gitignore

import torch
import numpy as np
from scipy.linalg import null_space
from sklearn.decomposition import IncrementalPCA

import functools
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@contextmanager
def no_grad(no_grad=True):
    if no_grad:
        with torch.no_grad():
            yield
    else:
        yield

def test_nets(func):
    """
    Decorator to ensure input networks are in evaluation mode during function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract nets from arguments
        if len(args) > 0 and isinstance(args[0], (torch.nn.Module, list)):
            nets = args[0]
            if not isinstance(nets, list):
                nets = [nets]
            
            # Store original states
            original_states = [set_net_mode(net, training=False) for net in nets]
            
            try:
                # Run function
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore original states
                for net, state in zip(nets, original_states):
                    set_net_mode(net, training=state)
        else:
            # No networks found, just run the function
            return func(*args, **kwargs)
    return wrapper

def train_nets(func):
    @wraps(func)
    def wrapper(nets, *args, **kwargs):
        # get original training mode and set to train
        in_training_mode = [set_net_mode(net, training=True) for net in nets]
        func_outputs = func(nets, *args, **kwargs)
        # return networks to whatever mode they used to be in
        for train_mode, net in zip(in_training_mode, nets):
            set_net_mode(net, training=train_mode)
        return func_outputs
    return wrapper

def set_net_mode(net, training=True):
    """
    Helper for toggling train/eval mode of a network.
    """
    in_training_mode = net.training
    # set to training mode or evaluation mode
    if training:
        net.train()
    else:
        net.eval()
    # return original mode of network
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

def check_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable (but not a string).
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is iterable but not a string, False otherwise
    """
    if isinstance(obj, str):
        return False
    try:
        iter(obj)
        return True
    except TypeError:
        return False

def remove_by_idx(input, idx, dim):
    """
    remove part of input along dimension 'dim' for the indices in 'idx'
    """
    idx_keep = [i for i in range(input.size(dim)) if i not in idx]
    return torch.index_select(input, dim, torch.tensor(idx_keep).to(input.device))

def get_eval_transform_by_cutoff(cutoff):
    """
    Return a function that zeros out eigenvalues below a fraction 'cutoff'.
    """
    def eval_transform(evals):
        assert torch.all(evals >= 0), "found negative eigenvalues, doesn't work for 'cutoff' eval_transform"
        evals = evals / torch.sum(evals)
        return 1.0 * (evals > cutoff)
    return eval_transform

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
    assert (input.ndim == 2) or (input.ndim == 3), "input must be a 2D or 3D tensor"
    assert isinstance(correction, bool), "correction must be bool"

    no_batch = input.ndim == 2
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

def smart_pcaOLD(input, centered=True, use_rank=True, correction=True):
    """
    smart algorithm for pca optimized for speed

    input should either have shape (batch, dim, samples) or (dim, samples)
    if dim > samples, will use svd and if samples < dim will use covariance/eigh method

    will center data when centered=True

    if it fails, will fall back on performing sklearns IncrementalPCA whenever forcetry=True
    """
    assert (input.ndim == 2) or (input.ndim == 3), "input should be a matrix or batched matrices"
    assert isinstance(correction, bool), "correction should be a boolean"

    if input.ndim == 2:
        no_batch = True
        input = input.unsqueeze(0)  # create batch dimension for uniform code
    else:
        no_batch = False

    _, D, S = input.size()
    if D > S:
        # subtract mean if doing centered covariance
        if centered:
            input = input - input.mean(dim=2, keepdim=True)
        # if more dimensions than samples, it's more efficient to run svd
        v, w, _ = named_transpose([torch.linalg.svd(inp) for inp in input])
        # convert singular values to eigenvalues
        w = [ww**2 / (S - 1.0 * correction) for ww in w]
        # append zeros because svd returns w in R**k where k = min(D, S)
        w = [torch.concatenate((ww, torch.zeros(D - S))) for ww in w]

    else:
        # if more samples than dimensions, it's more efficient to run eigh
        bcov = batch_cov(input, centered=centered, correction=correction)
        w, v = named_transpose([eigendecomposition(C, use_rank=use_rank) for C in bcov])

    # return to stacked tensor across batch dimension
    w = torch.stack(w)
    v = torch.stack(v)

    # if no batch originally provided, squeeze out batch dimension
    if no_batch:
        w = w.squeeze(0)
        v = v.squeeze(0)

    # return eigenvalues and eigenvectors
    return w, v

def smart_pca(input, centered=True, use_rank=True, correction=True):
    """
    Efficient PCA using either SVD or eigen-decomposition depending on shape.
    Falls back on a known method (like sklearn) if it fails to converge.

    Args:
      input (torch.Tensor): shape could be (D, S) or (B, D, S).
        - If 2D, treat it as a single (D, S) case (no batch).
        - If 3D, treat it as (batch, D, S).
      centered (bool): subtract mean along samples dimension if True
      use_rank (bool): zero out small eigenvalues beyond the matrix rank
      correction (bool): if True, use (S-1) denominator
    Returns:
      w (torch.Tensor): eigen/singular values, shape is either (D,) or (B, D)
      v (torch.Tensor): eigen/singular vectors, shape is either (D, D) or (B, D, D)
    """
    # 1) We only handle 2D or 3D
    assert input.ndim in (2, 3), "smart_pca: input must be 2D or 3D (D,S or B,D,S)."
    assert isinstance(correction, bool), "correction must be bool (True/False)."

    # 2) Convert to batch form if 2D
    no_batch = (input.ndim == 2)
    if no_batch:
        input = input.unsqueeze(0)  # shape => (1, D, S)

    B, D, S = input.shape

    # 3) If #dims > #samples, use SVD on (D, S) for each batch
    #    else use covariance+eigendecomposition
    if D > S:
        # Possibly center the data
        if centered:
            mean_ = input.mean(dim=2, keepdim=True)  # shape (B, D, 1)
            input = input - mean_

        # We'll collect eigenvalues and eigenvectors for each batch
        w_list = []
        v_list = []
        for b in range(B):
            inp_ = input[b]  # shape (D, S)
            # Using SVD => inp_ = U * S_ * V^T
            # torch.linalg.svd => (U, S, Vh)
            U, Svals, Vh = torch.linalg.svd(inp_, full_matrices=False)
            # Convert singular values to eigenvalues
            w_ = Svals**2 / (S - (1.0 if correction else 0.0))
            v_ = Vh.transpose(0, 1)  # from shape (S, D) => (D, S)

            w_list.append(w_)
            v_list.append(v_)
        # Stack to shape => (B, D) or (B, S) … but in standard PCA we want dimension = D
        w = torch.stack(w_list, dim=0)  # shape => (B, k)
        v = torch.stack(v_list, dim=0)  # shape => (B, D, S)
        # If we truly want (B, D, D) we might need to pad or handle the mismatch if D != S

    else:
        # Use covariance => (D x D), then eigh
        bcov = batch_cov(input, centered=centered, correction=correction)
        # bcov shape => (B, D, D)
        w_list = []
        v_list = []
        for b in range(B):
            C = bcov[b]  # shape (D, D)
            w_, v_ = eigendecomposition(C, use_rank=use_rank)
            w_list.append(w_)
            v_list.append(v_)
        w = torch.stack(w_list, dim=0)  # => shape (B, D)
        v = torch.stack(v_list, dim=0)  # => shape (B, D, D)

    # 4) If we started with no batch, remove batch dim
    if no_batch:
        w = w.squeeze(0)  # => (D,)
        v = v.squeeze(0)  # => (D, D) or possibly (D, S) in the SVD branch

    return w, v

def eigendecomposition(C, use_rank=True):
    """
    helper for getting eigenvalues and eigenvectors of covariance matrix

    will measure eigenvalues and eigenvectors with torch.linalg.eigh()
    the output will be sorted from highest to lowest eigenvalue (& eigenvector)

    if use_rank=True, will measure the rank of the covariance matrix and zero
    out any eigenvalues beyond the rank (that are usually nonzero numerical errors)
    """
    try:
        w, v = torch.linalg.eigh(C)
    except torch._C._LinAlgError as error:
        # this happens if the algorithm failed to converge
        # try with sklearn's incrementalPCA algorithm
        return sklearn_pca(C, use_rank=use_rank)
    except Exception as error:
        raise error
    w_idx = torch.argsort(-w)
    w = w[w_idx]
    v = v[:, w_idx]
    # iff use_rank=True, will set eigenvalues to 0 for probable numerical errors
    if use_rank:
        crank = torch.linalg.matrix_rank(C)
        w[crank:] = 0
    return w, v

def sklearn_pca(input, use_rank=True, rank=None):
    """
    sklearn incrementalPCA algorithm serving as a replacement for eigh when it fails

    input should be a tensor with shape (num_samples, num_features) or it can be a
    covariance matrix with (num_features, num_features)

    if use_rank=True, will set num_components to the rank of input and then fill out the
    rest of the components with random orthogonal components in the null space of the true
    components and set the eigenvalues to 0

    if use_rank=False, will attempt to fit all the components
    if rank is not None, will attempt to fit #=rank components without measuring the rank directly
    (will ignore "rank" if use_rank=False)

    returns w, v where w is eigenvalues and v is eigenvectors sorted from highest to lowest
    """
    # Move tensor to CPU if it's on GPU
    if input.is_cuda:
        input = input.cpu()
        
    num_samples, num_features = input.shape
    rank = None if not use_rank else (rank if rank is not None else fast_rank(input))
    ipca = IncrementalPCA(n_components=rank).fit(input)
    v = ipca.components_
    w = ipca.singular_values_**2 / num_samples
    # if v is a subspace of input (e.g. not a full basis, fill it out)
    if v.shape[0] < num_features:
        msg = "this condition should always be true, and if not we have to find out why"
        assert w.shape[0] == v.shape[0], msg
        v_kernel = null_space(v).T
        v = np.vstack((v, v_kernel))
        w = np.concatenate((w, np.zeros(v_kernel.shape[0])))
    return torch.tensor(w, dtype=torch.float), torch.tensor(v, dtype=torch.float).T

def fast_rank(input):
    """uses transpose to speed up rank computation, otherwise normal"""
    # Move tensor to CPU if it's on GPU
    if input.is_cuda:
        input = input.cpu()
        
    if input.size(-2) < input.size(-1):
        input = torch.transpose(input, -2, -1)
    return int(torch.linalg.matrix_rank(input))

def get_maximum_strides(h_input, w_input, layer):
    """
    Helper for computing the number of strides h_max, w_max after 
    convolution with given kernel/stride/padding/dilation.
    """
    h_max = int(np.floor((h_input + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1))
    w_max = int(np.floor((w_input + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) / layer.stride[1] + 1))
    return h_max, w_max

def get_unfold_params(layer):
    return dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation)

@torch.no_grad()
def cvPCA(X1, X2):
    """X1, X2 are both (dimensions x samples)"""
    D, B = X1.shape
    assert X2.shape == (D, B), "shape mismatch"
    _, u = smart_pca(X1)
    cproj0 = X1.T @ u
    cproj1 = X2.T @ u
    ss = (cproj0 * cproj1).mean(axis=0)
    return ss

def get_num_components(nc, shape):
    return nc if nc is not None else min(shape)

@torch.no_grad()
def shuff_cvPCA(X1, X2, nshuff=5, cvmethod=cvPCA):
    D, B = X1.shape
    assert X2.shape == (D, B), "shape mismatch"
    nc = get_num_components(None, (D, B))
    ss = torch.zeros((nshuff, nc))
    X = torch.stack((X1, X2))
    for k in range(nshuff):
        iflip = 1 * (torch.rand(B) > 0.5)
        X1c = torch.gather(X, 0, iflip.view(1, 1, -1).expand(1, D, -1)).squeeze(0)
        X2c = torch.gather(X, 0, -(iflip - 1).view(1, 1, -1).expand(1, D, -1)).squeeze(0)
        ss[k] = cvmethod(X1c, X2c)
    return ss

def avg_value_by_layer(full):
    """
    Return average value per layer across a list of epochs or minibatches.
    
    **full** is a list of lists where the outer list is each snapshot through training or
    minibatch etc and each inner list is the value for each node in the network across layers
    of a particular measurement

    For example:
    num_epochs = 1000
    nodes_per_layer = [50, 40, 30, 20]
    len(full) == 1000
    len(full[i]) == 4 ... for all i
    [f.shape for f in full[i]] = [50, 40, 30, 20] ... for all i

    this method will return a tensor of size (num_layers, num_epochs) of the average value (for
    whatever value is in **full**) for each list/list of values in **full**
    """
    num_epochs = len(full)
    num_layers = len(full[0])
    avg_full = torch.zeros((num_layers, num_epochs))
    for layer in range(num_layers):
        avg_full[layer, :] = torch.tensor([torch.mean(f[layer]) for f in full])
    return avg_full.cpu()

def value_by_layer(full: List[List[torch.Tensor]], layer: int) -> torch.Tensor:
    """
    return all value measurements for a particular layer from **full**

    **full** is a list of lists where the outer list is each snapshot through training or
    minibatch etc and each inner list is the value for each node in the network across layers

    this method will return just the part of **full** corresponding to the layer indexed
    by **layer** as a tensor of shape (num_epochs, num_nodes)

    see ``avg_value_by_layer`` for a little more explanation
    """
    return torch.cat([f[layer].view(1, -1) for f in full], dim=0).cpu()

def condense_values(full: List[List[List[torch.Tensor]]]) -> List[torch.Tensor]:
    """
    condense List[List[List[Tensor]]] -> list of #=num_layers Tensors
    shape: (num_networks, num_batches, num_nodes_per_layer)
    
    returns list of #=num_layers tensors, where each tensor has shape (num_networks, num_batches, num_nodes_per_layer)

    full should be a list of list of lists
    the first list should have length = number of networks
    the second list should have length = number of batches
    the third list should have length = number of layers in the network (this has to be the same for each network!)
    the tensor should have shape = number of nodes in this layer (also must be the same for each network) (or can be anything as long as consistent across layers)
    """
    num_layers = len(full[0][0])
    return [torch.stack([value_by_layer(value, layer) for value in full]) for layer in range(num_layers)]

def transpose_list(list_of_lists):
    """helper function for transposing the order of a list of lists"""
    return list(map(list, zip(*list_of_lists)))

def named_transpose(list_of_lists, reduction=None):
    """
    helper function for transposing lists without forcing the output to be a list like transpose_list

    for example, if list_of_lists contains 10 copies of lists that each have 3 iterable elements you
    want to name "A", "B", and "C", then write:
    A, B, C = named_transpose(list_of_lists)

    if reduction is used, it will be applied to each output, otherwise will make them lists
    """
    if reduction is not None:
        return map(reduction, zip(*list_of_lists))
    return map(list, zip(*list_of_lists))

def ptp(tensor, dim=None, keepdim=False):
    """
    simple method for measuring range of tensor on requested dimension or on all data
    """
    if dim is None:
        return tensor.max() - tensor.min()
    return tensor.max(dim, keepdim).values - tensor.min(dim, keepdim).values

def rms(tensor, dim=None, keepdim=False):
    """simple method for measuring root-mean-square on requested dimension or on all data in tensor"""
    if dim is None:
        return torch.sqrt(torch.mean(tensor**2))
    return torch.sqrt(torch.mean(tensor**2, dim=dim, keepdim=keepdim))

def compute_stats_by_type(tensor, num_types, dim, method="var"):
    """
    helper method for returning the mean and variance across a certain dimension
    where multiple types are concatenated on that dimension

    for example, suppose we trained 2 networks each with 3 sets of parameters
    and concatenated the loss in a tensor like [set1-loss-net1, set1-loss-net2, set2-loss-net1, ...]
    then this would contract across the nets from each set and return the mean and variance
    """    
    num_on_dim = tensor.size(dim)
    num_per_type = int(num_on_dim / num_types)
    tensor_by_type = tensor.unsqueeze(dim)
    expand_shape = list(tensor_by_type.shape)
    expand_shape[dim + 1] = num_per_type
    expand_shape[dim] = num_types
    tensor_by_type = tensor_by_type.view(expand_shape)
    type_means = torch.mean(tensor_by_type, dim=dim + 1)
    if method == "var":
        type_dev = torch.var(tensor_by_type, dim=dim + 1)
    elif method == "std":
        type_dev = torch.std(tensor_by_type, dim=dim + 1)
    elif method == "se":
        type_dev = torch.std(tensor_by_type, dim=dim + 1) / np.sqrt(num_per_type)
    elif method == "range":
        type_dev = ptp(tensor_by_type, dim=dim + 1)
    else:
        raise ValueError(f"Method ({method}) not recognized.")
    return type_means, type_dev


def weighted_average(data, weights, dim, keepdim=False, ignore_nan=False):
    """
    take the weighted average of **data** on a certain dimension with **weights**

    weights should be a nonnegative vector that broadcasts into data
    avg = sum_i(data_i * weight_i, dim) / sum_i(weight_i, dim)

    if ignore_nan=True, (default=False), will ignore nans in weighted average
    """
    assert data.ndim == weights.ndim, "data and weights must have same number of dimensions"
    assert torch.all(weights[~torch.isnan(weights)] >= 0), "weights must be nonnegative"

    for d in dim if check_iterable(dim) else [dim]:
        assert data.size(d) == weights.size(
            d
        ), f"data and weights must have same size in averaging dimensions (data.size({d})={data.size(d)}, (weight.size({d})={weights.size(d)}))"

    # use normal sum if not ignore nan, otherwise use nansum
    sum = torch.nansum if ignore_nan else torch.sum

    # make sure nans are in the same place in weights and data for accurate division by total weight
    if ignore_nan:
        weights = weights.expand(data.size())
        weights = torch.masked_fill(weights, torch.isnan(data), torch.nan)

    # numerator & denominator of weighted average
    numerator = sum(data * weights, dim=dim, keepdim=keepdim)
    denominator = sum(weights, dim=dim, keepdim=keepdim)

    # return weighted average
    return numerator / denominator


def fgsm_attack(image, epsilon, data_grad, transform, sign):
    """update an image with fast-gradient sign method"""
    warn("fgsm_attack is only going to be in utils temporarily!", DeprecationWarning, stacklevel=2)
    # Collect the element-wise sign of the data gradient
    if sign:
        data_grad = data_grad.sign()
    else:
        data_grad = data_grad.clone()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = transform(perturbed_image)
    # Return the perturbed image
    return perturbed_image

def str2bool(str):
    if isinstance(str, bool):
        return str
    if str.lower() in ("true", "1"):
        return True
    elif str.lower() in ("false", "0"):
        return False
    else:
        raise TypeError("Boolean type expected")

def save_checkpoint(nets, optimizers, results, path):
    """
    Method for saving checkpoints for networks throughout training.
    """
    multi_model_ckpt = {f"model_state_dict_{i}": net.state_dict() for i, net in enumerate(nets)}
    multi_optimizer_ckpt = {f"optimizer_state_dict_{i}": opt.state_dict() for i, opt in enumerate(optimizers)}
    checkpoint = results | multi_model_ckpt | multi_optimizer_ckpt
    torch.save(checkpoint, path)

def load_checkpoints(nets, optimizers, device, path):
    """
    Method for loading presaved checkpoint during training.
    """
    if device == "cpu":
        checkpoint = torch.load(path, map_location=device)
    elif device == "cuda":
        checkpoint = torch.load(path)

    net_ids = natsorted([key for key in checkpoint if key.startswith("model_state_dict")])
    opt_ids = natsorted([key for key in checkpoint if key.startswith("optimizer_state_dict")])
    assert all(
        [oi.split("_")[-1] == ni.split("_")[-1] for oi, ni in zip(opt_ids, net_ids)]
    ), "nets and optimizers cannot be matched up from checkpoint"

    [net.load_state_dict(checkpoint.pop(net_id)) for net, net_id in zip(nets, net_ids)]
    [opt.load_state_dict(checkpoint.pop(opt_id)) for opt, opt_id in zip(optimizers, opt_ids)]

    if device == "cuda":
        [net.to(device) for net in nets]
    return nets, optimizers, checkpoint

def match_git(path):
    """simple method for determining if a path is a git-related file or directory"""
    if ".git" in path:
        return True
    return False

def compress_directory(output_path, directory_path=None):
    """
    Utility to compress entire directory into a zip while respecting .gitignore and ignoring .git.
    """
    if directory_path is None:
        directory_path = os.path.dirname(os.path.abspath(__file__)) + "/../.."
    gitignore_path = os.path.join(directory_path, ".gitignore")
    matches = parse_gitignore(gitignore_path)

    files_to_copy = []
    archive_names = []
    for dirpath, dirnames, files in os.walk(directory_path):
        if matches(dirpath) or match_git(dirpath):
            dirnames[:] = []
        else:
            # Filter files based on .gitignore rules (and don't save any .git files)
            keep_files = [f for f in files if not matches(f) and not match_git(f)]
            full_files = [os.path.join(dirpath, f) for f in keep_files]
            for file in full_files:
                files_to_copy.append(file)
                archive_names.append(os.path.relpath(file, directory_path))

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file, name in zip(files_to_copy, archive_names):
            zipf.write(file, arcname=name)
            
            
def condense_values(al_list):
    data_list = [item["data"] for item in al_list]
    aggregated = []
    for net_i in range(len(data_list[0])):
        net_snapshots = [d[net_i] for d in data_list]
        net_layers = []
        for layer_i in range(len(net_snapshots[0])):
            layer_values = [snap[layer_i] for snap in net_snapshots]
            net_layers.append(torch.stack(layer_values, dim=0).mean(dim=0))
        aggregated.append(net_layers)
    return aggregated

def timed(func):
    """
    Decorator to time the execution of a function.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")

def setup_logging(log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_file: Optional path to a log file
        level: Logging level (default: INFO)
    """
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        
        # Create parent directory if it doesn't exist
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")

def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert torch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor or numpy array
        
    Returns:
        Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return tensor

def to_tensor(array: Union[torch.Tensor, np.ndarray], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert numpy array to torch tensor.
    
    Args:
        array: Numpy array or PyTorch tensor
        device: Device to place tensor on
        
    Returns:
        PyTorch tensor
    """
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array
        
    if device is not None:
        tensor = tensor.to(device)
        
    return tensor

# Type variable for function decorators
F = TypeVar('F', bound=Callable[..., Any])

def timer(func: F) -> F:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
        
    Example:
        @timer
        def slow_function():
            time.sleep(1)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    return wrapper

def debug(func: F) -> F:
    """
    Decorator to add debug logging to function entry and exit.
    
    Args:
        func: Function to debug
        
    Returns:
        Wrapped function
        
    Example:
        @debug
        def sensitive_function(x, y):
            return x / y
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Entering {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__} with result={result}")
            return result
        except Exception as e:
            logger.exception(f"Exception in {func.__name__}: {e}")
            raise
    return wrapper

def ensure_device(device_arg: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Ensure a valid torch device.
    
    Args:
        device_arg: Device specification or None
        
    Returns:
        Resolved torch.device
    """
    if device_arg is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = str(device_arg)
        
    # Validate CUDA availability
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device_str = "cpu"
        
    return torch.device(device_str)

def get_model_size(model: torch.nn.Module) -> int:
    """
    Calculate the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters())

def get_memory_usage(model: torch.nn.Module) -> float:
    """
    Estimate memory usage of model in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Estimated memory usage in MB
    """
    total_bytes = 0
    for p in model.parameters():
        if p.data is not None:
            total_bytes += p.element_size() * p.numel()
        if p.grad is not None:
            total_bytes += p.element_size() * p.numel()
    
    # Convert bytes to MB
    return total_bytes / (1024 * 1024)

def get_git_revision_hash() -> str:
    """
    Get the current git revision hash.
    
    Returns:
        String with the current git hash, or "unknown" if not in a git repo
    """
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"

def initialize_logger(log_level: str = "INFO") -> None:
    """
    Initialize the logger with the specified log level.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set the logger level for this module
    logger.setLevel(numeric_level)
    
    logging.info(f"Logger initialized with level: {log_level}")

def init_seed(seed: int) -> None:
    """
    Initialize random number generator seeds for reproducibility.
    
    Args:
        seed: Integer seed for random number generators
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These settings may impact performance but improve reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed initialized to {seed}")

def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to the output file
    """
    import pickle
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.info(f"Object saved to {filepath}")