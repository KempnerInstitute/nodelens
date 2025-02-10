# --------------------------------------------
# alignment_metrics.py
# --------------------------------------------

import torch

from alignment.utils import smart_pca, get_device

class AlignmentMetrics:
    """
    Provides static methods for various alignment metrics,
    including 'delta_alignment', but NOT 'delta_weights'.

    We also provide optional post-processing helpers if you like,
    but here we mostly keep them out to let processing code do histograms etc.
    """

    @staticmethod
    def RQ(input_, weight_):
        """
        Rayleigh Quotient alignment measure:
        proportion of variance in `input_` explained by each row of `weight_`.
        """
        return alignment(input_, weight_, method="alignment", relative=True)

    @staticmethod
    def MI_0(input_, weight_):
        """
        Placeholder for mutual information approach - version 0
        (currently reuses alignment(...) as a stand-in).
        """
        return alignment(input_, weight_, method="alignment", relative=True)

    @staticmethod
    def MI_1(input_, weight_):
        """
        Placeholder for mutual information approach - version 1
        """
        return torch.tensor(0.0)

    @staticmethod
    def delta_alignment(net, layer_idx, layer_input):
        """
        (ADDED or CHANGED) - alignment of (W_current - W_init) with the input's covariance.
        This references net._init_weights for the initial weight.
        If net._init_weights is absent, it returns zeros.
        """
        if not hasattr(net, "_init_weights"):
            weight_diff = torch.zeros_like(net.get_alignment_weights()[layer_idx])
        else:
            init_w = net._init_weights[layer_idx]
            current_w = net.get_alignment_weights()[layer_idx]
            weight_diff = current_w - init_w

        weight_diff = weight_diff.flatten(start_dim=1)
        return alignment(layer_input, weight_diff, method="alignment", relative=True)

    @staticmethod
    def measure(input_, weight_, method="RQ"):
        """
        For a single method and a single layer's (input_, weight_).
        """
        if method == "RQ":
            return AlignmentMetrics.RQ(input_, weight_)
        elif method == "MI_0":
            return AlignmentMetrics.MI_0(input_, weight_)
        elif method == "MI_1":
            return AlignmentMetrics.MI_1(input_, weight_)
        else:
            raise ValueError(f"Unknown alignment method {method}")

    @staticmethod
    def measure_methods(net, images, methods, precomputed=True):
        """
        For each layer in 'net', produce a dict {method -> tensor}.
        """
        layer_inputs = net.get_layer_inputs(images, precomputed=precomputed)
        layer_weights = net.get_alignment_weights(flatten=True)

        results_per_layer = []
        for layer_idx, (inp, wgt) in enumerate(zip(layer_inputs, layer_weights)):
            layer_dict = {}
            for m in methods:
                if m in ("RQ", "MI_0", "MI_1"):
                    val = AlignmentMetrics.measure(inp, wgt, method=m)
                elif m == "delta_alignment":
                    val = AlignmentMetrics.delta_alignment(net, layer_idx, inp)
                else:
                    raise ValueError(f"Unknown method {m}")
                layer_dict[m] = val
            results_per_layer.append(layer_dict)

        return results_per_layer

    @staticmethod
    def compute_eigenvalues(x):
        """
        Do a standard PCA with 'smart_pca' to get eigenvalues of x (batch, features).
        """
        w, v = smart_pca(x.T, centered=True)
        return w, v

    @staticmethod
    def measure_expected_distribution(method, eigenvals, bins=50, num_tests=100):
        """
        Return (counts, bin_edges) for a random alignment distribution of 'method'.
        """
        if method == "RQ":
            counts, edges, _ = expected_alignment_distribution(
                eigenvals,
                relative=True,
                valid_rotation=False,
                with_rotation=True,
                bins=bins,
                num_tests=num_tests
            )
            return counts, edges
        elif method in ["MI_0", "MI_1", "delta_alignment"]:
            # For delta_alignment we do not define a random distribution
            return None, None
        else:
            return None, None


def alignment(input, weight, method="alignment", relative=True):
    """
    measure alignment (proportion of variance explained)
    by each weight vector in 'weight' for the input's covariance.
    """
    assert method in ("alignment", "similarity"), "method must be 'alignment' or 'similarity'"
    if method == "alignment":
        cc = torch.cov(input.T)
    else:
        cc = smartcorr(input.T)

    rq = torch.sum(torch.matmul(weight, cc) * weight, axis=1) / torch.sum(weight * weight, axis=1)
    if relative:
        # proportion of variance explained
        return rq / torch.trace(cc)
    return rq


@torch.no_grad()
def expected_alignment_distribution(eigenvalues, relative=True, valid_rotation=True, with_rotation=True, bins=11, num_tests=100):
    """
    From a set of eigenvalues, simulate random weight vectors
    (optional orthonormal rotation) to measure an 'expected' alignment distribution.
    """
    import math
    import numpy as np

    N = len(eigenvalues)
    if relative:
        eigenvalues = eigenvalues / eigenvalues.sum()
    eigenvalues = eigenvalues.view(-1, 1).expand(-1, N * num_tests)

    device = eigenvalues.device
    if with_rotation:
        if valid_rotation:
            mixing = []
            for _ in range(num_tests):
                mat = torch.normal(0, 1 / math.sqrt(N), (N, N)).to(device)
                q, _ = torch.linalg.qr(mat)
                mixing.append(q.T)
            coefficients = torch.cat(mixing, dim=1) ** 2
        else:
            coefficients = torch.normal(0, 1 / math.sqrt(N), (N, N * num_tests)).to(device) ** 2
    else:
        coefficients = torch.ones((N, N * num_tests), device=device)

    weights = eigenvalues * coefficients
    align = torch.sum(eigenvalues * weights, dim=0) / weights.sum(dim=0)
    align_np = align.cpu().numpy()

    counts, bin_edges = np.histogram(align_np, bins=bins, density=True)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return torch.tensor(counts, dtype=torch.float), torch.tensor(bin_edges, dtype=torch.float), torch.tensor(centers, dtype=torch.float)


def smartcorr(input):
    """
    Wraps torch.corrcoef but zeros out rows/cols that have zero variance.
    """
    idx_zeros = torch.var(input, dim=1) == 0
    cc = torch.corrcoef(input)
    cc[idx_zeros, :] = 0
    cc[:, idx_zeros] = 0
    return cc