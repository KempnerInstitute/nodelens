import os
import math
import zipfile
from typing import List
from warnings import warn
from contextlib import contextmanager

import torch
import numpy as np
from scipy.linalg import null_space
from sklearn.decomposition import IncrementalPCA

@contextmanager
def no_grad(no_grad=True):
    if no_grad:
        with torch.no_grad():
            yield
    else:
        yield

def set_net_mode(net, training=True):
    in_training_mode = net.training
    if training:
        net.train()
    else:
        net.eval()
    return in_training_mode

def get_device(obj):
    if isinstance(obj, torch.nn.Module):
        return next(obj.parameters()).device.type
    elif isinstance(obj, torch.Tensor):
        return "cuda" if obj.is_cuda else "cpu"
    else:
        raise ValueError("")

def check_iterable(val):
    try:
        _ = iter(val)
    except:
        return False
    else:
        return True

def remove_by_idx(input, idx, dim):
    idx_keep = [i for i in range(input.size(dim)) if i not in idx]
    return torch.index_select(input, dim, torch.tensor(idx_keep).to(input.device))

def get_eval_transform_by_cutoff(cutoff):
    def eval_transform(evals):
        assert torch.all(evals >= 0)
        evals = evals / torch.sum(evals)
        return 1.0 * (evals > cutoff)
    return eval_transform

def fractional_histogram(*args, **kwargs):
    counts, bins = np.histogram(*args, **kwargs)
    counts = counts / np.sum(counts)
    return counts, bins

def edge2center(edges):
    return edges[:-1] + np.diff(edges) / 2

def smartcorr(input):
    idx_zeros = torch.var(input, dim=1) == 0
    cc = torch.corrcoef(input)
    cc[idx_zeros, :] = 0
    cc[:, idx_zeros] = 0
    return cc

def batch_cov(input, centered=True, correction=True):
    assert (input.ndim == 2) or (input.ndim == 3)
    assert isinstance(correction, bool)
    no_batch = input.ndim == 2
    if no_batch:
        input = input.unsqueeze(0)
    S = input.size(2)
    if centered:
        input = input - input.mean(dim=2, keepdim=True)
    bcov = torch.bmm(input, input.transpose(1, 2))
    bcov /= S - 1.0 * correction
    if no_batch:
        bcov = bcov.squeeze(0)
    return bcov

def fast_rank(input):
    if input.size(-2) < input.size(-1):
        input = torch.transpose(input, -2, -1)
    return int(torch.linalg.matrix_rank(input))

def sklearn_pca(input, use_rank=True, rank=None):
    num_samples, num_features = input.shape
    rank = None if not use_rank else (rank if rank is not None else fast_rank(input))
    ipca = IncrementalPCA(n_components=rank).fit(input)
    v = ipca.components_
    w = ipca.singular_values_**2 / num_samples
    if v.shape[0] < num_features:
        v_kernel = null_space(v).T
        v = np.vstack((v, v_kernel))
        w = np.concatenate((w, np.zeros(v_kernel.shape[0])))
    return torch.tensor(w, dtype=torch.float), torch.tensor(v, dtype=torch.float).T

def smart_pca(input, centered=True, use_rank=True, correction=True):
    assert (input.ndim == 2) or (input.ndim == 3)
    if input.ndim == 2:
        no_batch = True
        input = input.unsqueeze(0)
    else:
        no_batch = False
    _, D, S = input.size()
    if D > S:
        if centered:
            input = input - input.mean(dim=2, keepdim=True)
            
        v, w, _ = [torch.linalg.svd(inp) for inp in input]
        w = [ww**2 / (S - 1.0 * correction) for ww in w]
        w = [torch.concatenate((ww, torch.zeros(D - S))) for ww in w]
    else:
        bcov = batch_cov(input, centered=centered, correction=correction)
        out = [eigendecomposition(C, use_rank=use_rank) for C in bcov.unsqueeze(0)] if bcov.ndim == 2 else [eigendecomposition(c, use_rank=use_rank) for c in bcov]
        w, v = list(zip(*out))
    if isinstance(v, list):
        v = torch.stack(v)
        w = torch.stack(w)
    if no_batch:
        w = w.squeeze(0)
        v = v.squeeze(0)
    return w, v

def eigendecomposition(C, use_rank=True):
    try:
        w, v = torch.linalg.eigh(C)
    except torch._C._LinAlgError as error:
        return sklearn_pca(C, use_rank=use_rank)
    except Exception as error:
        raise error
    w_idx = torch.argsort(-w)
    w = w[w_idx]
    v = v[:, w_idx]
    if use_rank:
        crank = torch.linalg.matrix_rank(C)
        w[crank:] = 0
    return w, v

def alignment(input, weight, method="alignment", relative=True):
    if method == "alignment":
        cc = torch.cov(input.T)
    elif method == "similarity":
        cc = smartcorr(input.T)
    else:
        raise ValueError(f"did not recognize method ({method})")
    rq = torch.sum(torch.matmul(weight, cc) * weight, axis=1) / torch.sum(weight * weight, axis=1)
    if relative:
        return rq / torch.trace(cc)
    return rq

@torch.no_grad()
def expected_alignment_distribution(eigenvalues, relative=True, valid_rotation=True, with_rotation=True, bins=11, num_tests=100):
    N = len(eigenvalues)
    if relative:
        eigenvalues /= eigenvalues.sum()
    eigenvalues = eigenvalues.view(-1, 1).expand(-1, N * num_tests)
    if with_rotation:
        if valid_rotation:
            mixing = [torch.linalg.qr(torch.normal(0, 1 / math.sqrt(N), (N, N)))[0].T for _ in range(num_tests)]
            coefficients = torch.concatenate(mixing, axis=1) ** 2
        else:
            coefficients = torch.normal(0, 1 / math.sqrt(N), (N, N * num_tests)) ** 2
    else:
        coefficients = torch.ones((N, N * num_tests))
    coefficients = coefficients.to(get_device(eigenvalues))
    weights = eigenvalues * coefficients
    alignment = torch.sum(eigenvalues * weights, dim=0) / weights.sum(dim=0)
    counts, bins = torch.histogram(alignment.cpu(), bins=bins, density=True)
    centers = edge2center(bins)
    return counts, bins, centers

def ptp(tensor, dim=None, keepdim=False):
    if dim is None:
        return tensor.max() - tensor.min()
    return tensor.max(dim, keepdim).values - tensor.min(dim, keepdim).values

def rms(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(torch.mean(tensor**2))
    return torch.sqrt(torch.mean(tensor**2, dim=dim, keepdim=keepdim))

def compute_stats_by_type(tensor, num_types, dim, method="var"):
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
    if ignore_nan:
        w2 = weights.expand(data.size())
        w2 = torch.masked_fill(w2, torch.isnan(data), torch.nan)
        sum_op = torch.nansum
    else:
        w2 = weights
        sum_op = torch.sum
    numerator = sum_op(data * w2, dim=dim, keepdim=keepdim)
    denominator = sum_op(w2, dim=dim, keepdim=keepdim)
    return numerator / denominator

def fgsm_attack(image, epsilon, data_grad, transform, sign):
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
    multi_model_ckpt = {f"model_state_dict_{i}": net.state_dict() for i, net in enumerate(nets)}
    multi_optimizer_ckpt = {f"optimizer_state_dict_{i}": opt.state_dict() for i, opt in enumerate(optimizers)}
    checkpoint = results | multi_model_ckpt | multi_optimizer_ckpt
    torch.save(checkpoint, path)

def load_checkpoints(nets, optimizers, device, path):
    if device == "cpu":
        checkpoint = torch.load(path, map_location=device)
    elif device == "cuda":
        checkpoint = torch.load(path)
    net_ids = sorted([key for key in checkpoint if key.startswith("model_state_dict")])
    opt_ids = sorted([key for key in checkpoint if key.startswith("optimizer_state_dict")])
    for net, net_id in zip(nets, net_ids):
        net.load_state_dict(checkpoint.pop(net_id))
    for opt, opt_id in zip(optimizers, opt_ids):
        opt.load_state_dict(checkpoint.pop(opt_id))
    if device == "cuda":
        [net.to(device) for net in nets]
    return nets, optimizers, checkpoint

def compress_directory(output_path, directory_path=None):
    if directory_path is None:
        directory_path = os.path.dirname(os.path.abspath(__file__)) + "/../.."
    from gitignore_parser import parse_gitignore
    def match_git(path):
        if ".git" in path:
            return True
        return False
    gitignore_path = os.path.join(directory_path, ".gitignore")
    matches = parse_gitignore(gitignore_path)
    files_to_copy = []
    archive_names = []
    for dirpath, dirnames, files in os.walk(directory_path):
        if matches(dirpath) or match_git(dirpath):
            dirnames[:] = []
        else:
            keep_files = [f for f in files if not matches(f) and not match_git(f)]
            full_files = [os.path.join(dirpath, f) for f in keep_files]
            for file in full_files:
                files_to_copy.append(file)
                archive_names.append(os.path.relpath(file, directory_path))
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file, name in zip(files_to_copy, archive_names):
            zipf.write(file, arcname=name)