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
    """
    smart algorithm for pca optimized for speed

    input should either have shape (batch, dim, samples) or (dim, samples)
    if dim > samples, will use svd and if samples < dim will use covariance/eigh method

    will center data when centered=True

    if it fails, will fall back on performing sklearns IncrementalPCA whenever forcetry=True
    """
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
        no_batch = False

    _, D, S = input.size()
    if D > S:
        # subtract mean if doing centered pca
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


def eigendecomposition(C, use_rank=True):
    """
    helper for getting eigenvalues and eigenvectors of covariance matrix

    will measure eigenvalues and eigenvectors with torch.linalg.eigh()
    the output will be sorted from highest to lowest eigenvalue (& eigenvector)

    if use_rank=True, will measure the rank of the covariance matrix and zero
    out any eigenvalues beyond the rank (that are usually nonzero numerical errors)
    """
    try:
        # measure eigenvalues and eigenvectors
        w, v = torch.linalg.eigh(C)

    except torch._C._LinAlgError as error:
        # this happens if the algorithm failed to converge
        # try with sklearn's incrementalPCA algorithm
        return sklearn_pca(C, use_rank=use_rank)

    except Exception as error:
        # if any other exception, raise it
        raise error

    # sort by eigenvalue from highest to lowest
    w_idx = torch.argsort(-w)
    w = w[w_idx]
    v = v[:, w_idx]

    # iff use_rank=True, will set eigenvalues to 0 for probable numerical errors
    if use_rank:
        crank = torch.linalg.matrix_rank(C)  # measure rank of covariance
        w[crank:] = 0  # set eigenvalues beyond rank to 0

    # return eigenvalues and eigenvectors
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
    # dimension
    num_samples, num_features = input.shape

    # measure rank (or set to None)
    rank = None if not use_rank else (rank if rank is not None else fast_rank(input))

    # create and fit IncrementalPCA object on input data
    ipca = IncrementalPCA(n_components=rank).fit(input)

    # eigenvectors are the components
    v = ipca.components_

    # eigenvalues are the scaled singular values
    w = ipca.singular_values_**2 / num_samples

    # if v is a subspace of input (e.g. not a full basis, fill it out)
    if v.shape[0] < num_features:
        msg = "adding this because I think it should always be true, and if not I want to find out"
        assert w.shape[0] == v.shape[0], msg
        v_kernel = null_space(v).T
        v = np.vstack((v, v_kernel))
        w = np.concatenate((w, np.zeros(v_kernel.shape[0])))

    return torch.tensor(w, dtype=torch.float), torch.tensor(v, dtype=torch.float).T


def fast_rank(input):
    """uses transpose to speed up rank computation, otherwise normal"""
    if input.size(-2) < input.size(-1):
        input = torch.transpose(input, -2, -1)
    return int(torch.linalg.matrix_rank(input))


# ------------------ alignment functions ----------------------
def alignment(input, weight, method="alignment", relative=True):
    """
    measure alignment (proportion variance explained) between **input** and **weight**

    computes the rayleigh quotient between each weight vector in **weight** and the **input** fed
    into **weight**. Typically, **input** is the output in Layer L-1 and **weight** is from Layer L

    the output is normalized by the total variance in output of layer L-1 to measure the proportion
    of variance of in **input** is explained by a projection onto node's weights in **weight**

    args
    ----
        input: (batch, neurons) torch tensor
            - represents input activity being fed in to network weight layer
        weight: (num_out, num_in) torch tensor
            - represents weights multiplied by input layer
        method: string, default='alignment'
            - which method to use to measure structure in **input**
            - if 'alignment', uses covariance matrix of **input**
            - if 'similarity', uses correlation matrix of **input**
        relative: bool, default=True,
            - if True, will measure relative RQ (divide by sum of eigenvalues)

    returns
    -------
        alignment: (num_out, ) torch tensor
            - proportion of variance explained by projection of **input** onto each **weight** vector
    """
    assert method == "alignment" or method == "similarity", "method must be set to either 'alignment' or 'similarity' (or None, default is alignment)"
    if method == "alignment":
        cc = torch.cov(input.T)
    elif method == "similarity":
        cc = smartcorr(input.T)
    else:
        raise ValueError(f"did not recognize method ({method}), must be 'alignment' or 'similarity'")
    # Compute rayleigh quotient
    rq = torch.sum(torch.matmul(weight, cc) * weight, axis=1) / torch.sum(weight * weight, axis=1)
    if relative:
        # proportion of variance explained by a projection of the input onto each weight
        return rq / torch.trace(cc)
    # variance explained by a projection of the input onto each weight
    return rq


@torch.no_grad()
def expected_alignment_distribution(eigenvalues, relative=True, valid_rotation=True, with_rotation=True, bins=11, num_tests=100):
    """
    for a set of eigenvalues, measure the expected distribution given aligned weights

    relative determines whether to normalize by sum of eigenvalues
    valid_rotation determines whether we create orthonormal rotation matrices (for True)
    or just normally distributed weights with the expected variance (for False)
    bins works like histogram bins
    num_tests determines how many tests to do (it's actually num_tests*len(eigenvalues))
    """
    # otherwise, randomly sample using eigenvalue as weighted average
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

def compute_skewness_inplace(input):
    """
    Compute skewness for each input feature (dimension) using in-place operations.
    """
    mean = input.mean(dim=0)
    variance = input.var(dim=0)

    # Add epsilon to variance to avoid division by zero
    epsilon = 1e-6
    variance = variance + epsilon

    third_moment = (input - mean).pow(3).mean(dim=0)

    # Skewness formula: E[(X - μ)^3] / (E[(X - μ)^2])^(3/2)
    skewness = third_moment / variance.pow(1.5)
    
    return skewness

def compute_kurtosis_inplace(input):
    """
    Compute kurtosis for each input feature (dimension) using in-place operations.
    """
    mean = input.mean(dim=0)
    variance = input.var(dim=0)
    fourth_moment = (input - mean).pow_(4).mean(dim=0)

    # Kurtosis formula: E[(X - μ)^4] / (E[(X - μ)^2])^2 - 3
    kurtosis = fourth_moment / variance.pow(2) - 3
    return kurtosis

def compute_skewness_low_rank(input, rank_approx=50):
    """
    Compute skewness after reducing the dimensionality using PCA to the specified rank approximation.
    """
    
    pca = PCA(n_components=rank_approx)
    input_reduced = torch.Tensor(pca.fit_transform(input.cpu().numpy())).to(input.device)
    
    return compute_skewness_inplace(input_reduced)

def compute_kurtosis_low_rank(input, rank_approx=50):
    """
    Compute kurtosis after reducing the dimensionality using PCA to the specified rank approximation.
    """
    
    pca = PCA(n_components=rank_approx)
    input_reduced = torch.Tensor(pca.fit_transform(input.cpu().numpy())).to(input.device)
    
    return compute_kurtosis_inplace(input_reduced)

def compute_redundancy(weights, input_covariance):
    """
    Compute the redundancy matrix for all pairs of nodes using matrix multiplication.
    
    Args:
        weights: (n, d) torch tensor, where n is the number of nodes and d is the input dimension.
        input_covariance: (d, d) torch tensor, the covariance matrix of the input data.
    
    Returns:
        redundancy_matrix: (n, n) torch tensor, redundancy between each pair of nodes (diagonal excluded).
    """
    # Compute the redundancy matrix: R = W Σ_X W^T
    redundancy_matrix = torch.matmul(weights, torch.matmul(input_covariance, weights.T))
    
    # Zero out the diagonal elements (self-information)
    redundancy_matrix.fill_diagonal_(0)
    
    return redundancy_matrix



def alignment_expansion(input, weight, method="alignment_expansion", relative=True):
    """
    Measure alignment (proportion variance explained) between **input** and **weight**
    and compute single-node information, redundancy, and total information for the layer.
    
    args:
    ----
        input: (batch, neurons) torch tensor
            - represents input activity being fed into network weight layer
        weight: (num_out, num_in) torch tensor
            - represents weights multiplied by input layer
        method: string, default='alignment'
            - which method to use to measure structure in **input**
            - if 'alignment', uses covariance matrix of **input**
            - if 'similarity', uses correlation matrix of **input**
        relative: bool, default=True,
            - if True, will measure relative RQ (divide by sum of eigenvalues)
    
    returns:
    -------
        alignment: (num_out, ) torch tensor
            - proportion of variance explained by projection of **input** onto each **weight** vector
        single_node_info: torch tensor
            - Information carried by each node (mutual information for single nodes)
        redundancy_matrix: torch tensor
            - Pairwise redundancy (mutual information overlap between pairs of nodes)
        total_info: float
            - Total information of the layer, accounting for redundancy
    """
    
    # Step 1: Compute covariance of input
    if method == "alignment_expansion":
        cc = torch.cov(input.T)  # Covariance matrix of input
    elif method == "similarity":
        cc = torch.corrcoef(input.T)  # Correlation matrix of input
    else:
        raise ValueError(f"Method {method} not recognized. Use 'alignment' or 'similarity'.")

    # Step 2: Compute Rayleigh Quotient (RQ) for each node
    rq = torch.sum(torch.matmul(weight, cc) * weight, axis=1) / torch.sum(weight * weight, axis=1)

    # Step 3: Compute Single-Node Information for each node (with kurtosis correction)
    single_node_info = rq / torch.trace(cc)  # First term (proportional to RQ)
    
    # Compute third-order term: Skewness correction
    skewness = compute_skewness_inplace(input)
    
    # Ensure the skewness shape matches the input dimensionality
    if skewness.shape[0] == input.shape[1]:  # If skewness is (d,), reshape it
        skewness = skewness.view(1, -1)  # Reshape skewness to (1, d) for broadcasting
    
    # Apply the skewness correction term
    skewness_correction = torch.abs(torch.sum(weight * skewness, dim=1))

    # Compute kurtosis-based correction term
    kurtosis = compute_kurtosis_low_rank(input)  # Kurtosis is computed for each input feature

        # Expand kurtosis tensor to match weight shape
    # if kurtosis.dim() == 1:
    #     kurtosis = kurtosis.unsqueeze(0)  # Add a dimension to match the weight tensor shape

    kurtosis_correction = 0.5 * torch.norm(weight, p=2, dim=1) * kurtosis[:weight.shape[1]].mean()  

    #single_node_info +=  skewness_correction 
    single_node_info += kurtosis_correction 

    # Step 4: Continue with redundancy and total information as before
    redundancy_matrix = compute_redundancy(weight, cc)
    adjusted_single_node_info = adjust_information_with_redundancy(single_node_info, redundancy_matrix)#single_node_info.clone()
    
    total_info = torch.sum(adjusted_single_node_info)

    return single_node_info#, adjusted_single_node_info, redundancy_matrix, total_info

def adjust_information_with_redundancy(single_node_info, redundancy_matrix):
    """
    Adjust single-node information by subtracting redundancy for each pair of nodes.
    
    Args:
        single_node_info: (n, ) torch tensor, the single-node information for each node.
        redundancy_matrix: (n, n) torch tensor, the redundancy between each pair of nodes.
        
    Returns:
        adjusted_info: (n, ) torch tensor, adjusted single-node information.
    """
    n = single_node_info.size(0)

    # Compute pairwise differences between single-node information
    info_diffs = single_node_info.view(n, 1) - single_node_info.view(1, n)

    # Create a mask where the single-node information is smaller for each pair
    mask = (info_diffs < 0).float()

    # Subtract redundancy from the node with smaller information
    redundancy_adjustment = torch.sum(mask * redundancy_matrix, dim=1)

    # Adjust the single-node information
    adjusted_info = single_node_info - 0 * redundancy_adjustment

    return adjusted_info

def plot_information_results(single_node_info, adjusted_single_node_info, redundancy_matrix, total_info):
    """
    Plot single-node information, adjusted single-node information, redundancy matrix, and total information for the layer.

    args:
    ----
        single_node_info: torch tensor
            - Information carried by each node
        adjusted_single_node_info: torch tensor
            - Adjusted information for each node after subtracting redundancy
        redundancy_matrix: torch tensor
            - Pairwise redundancy between nodes
        total_info: float
            - Total information of the layer
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Plot single-node information
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(single_node_info)), single_node_info.cpu().detach().numpy(), alpha=0.6, label="Original Info")
    plt.bar(range(len(adjusted_single_node_info)), adjusted_single_node_info.cpu().detach().numpy(), alpha=0.6, label="Adjusted Info")
    plt.xlabel('Node Index')
    plt.ylabel('Single Node Information (MI)')
    plt.title('Single-Node Information for Each Node (Original vs Adjusted)')
    plt.legend()
    plt.show()
    
    # Plot redundancy matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(redundancy_matrix.cpu().detach().numpy(), annot=True, cmap="YlGnBu")
    plt.title('Redundancy Matrix (Pairwise Redundancy between Nodes)')
    plt.show()

    # Display total information
    print(f"Total Information for the Layer: {total_info.item():.4f}")



def get_maximum_strides(h_input, w_input, layer):
    h_max = int(np.floor((h_input + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1))
    w_max = int(np.floor((w_input + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) / layer.stride[1] + 1))
    return h_max, w_max


def get_unfold_params(layer):
    return dict(stride=layer.stride, padding=layer.padding, dilation=layer.dilation)


# ----- cvPCA methods -----
@torch.no_grad()
def cvPCA(X1, X2):
    """X1, X2 are both (dimensions x samples)"""
    D, B = X1.shape
    assert X2.shape == (D, B), "shape of X1 and X2 is not the same"
    _, u = smart_pca(X1)

    cproj0 = X1.T @ u
    cproj1 = X2.T @ u
    ss = (cproj0 * cproj1).mean(axis=0)
    return ss


def get_num_components(nc, shape):
    return nc if nc is not None else min(shape)


@torch.no_grad()
def shuff_cvPCA(X1, X2, nshuff=5, cvmethod=cvPCA):
    """X1, X2 are both (dimensions x samples)"""
    D, B = X1.shape
    assert X2.shape == (D, B), "shape of X1 and X2 is not the same"
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
    return average value per layer across training

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
    whatever value is in **full**) for each list/list
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
    condense List[List[List[Tensor]]] representing some value measured across networks, batches, and layers, for each node in the layer

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