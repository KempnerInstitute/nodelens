# --------------------------------------------
# alignment_metrics.py
# --------------------------------------------

import torch
from alignment.utils import smart_pca, get_device

class AlignmentMetrics:
    """
    Provides static methods for various alignment metrics,
    including 'delta_alignment'.
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
        alignment of (W_current - W_init) with the input's covariance.
        This references net._init_weights for the initial weight.
        If net._init_weights is absent, returns zeros.
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
    def patchwise_alignment(inp, w, method="RQ", weigh_by_var=True):
        """
        'Patchwise' alignment for CNN. 
         - inp shape: [B, F, patches], e.g. B=mini-batch, F=inC*kH*kW, patches=outH*outW
         - w shape: [outC, F], the flattened filter weights.
         - We compute alignment for each patch’s (B,F) input. 
         - Weighted by patch-level variance across the batch dimension.
        """
        B, F, P = inp.shape
        var_patches = torch.var(inp, dim=0, keepdim=False)  # => (F, P)
        patchwise_var = var_patches.sum(dim=0)              # => shape (P,)

        all_patch_vals = []
        all_patch_vars = []

        for p in range(P):
            patch_data = inp[:, :, p]     # shape => [B, F]
            cc = torch.cov(patch_data.T)  # shape (F, F)
            num_ = torch.sum(torch.matmul(w, cc) * w, dim=1)
            denom_ = torch.sum(w*w, dim=1)

            patch_rq = num_ / denom_
            if method == "RQ":
                patch_rq = patch_rq / torch.trace(cc)
            patch_weight = patchwise_var[p] if weigh_by_var else 1.0

            all_patch_vals.append(patch_rq * patch_weight)
            all_patch_vars.append(patch_weight)

        total_weight = torch.stack(all_patch_vars).sum()
        sum_rq = torch.stack(all_patch_vals, dim=0).sum(dim=0)  # shape => (outC,)

        if total_weight > 0:
            final_rq = sum_rq / total_weight
        else:
            final_rq = sum_rq * 0
        return final_rq  # shape => (outC,)

    @staticmethod
    def measure_methods(net, images, methods, precomputed=True):
        """
        For each layer in 'net', produce a dict {method -> tensor}.
        1) get_layer_inputs
        2) preprocess => unfold or patchwise
        3) flatten weights
        4) measure alignment or patchwise alignment
        """
        # 1) Get raw inputs
        raw_inputs = net.get_layer_inputs(images, precomputed=precomputed)
        # 2) Unfold or flatten them
        preprocessed = net._preprocess_inputs(raw_inputs, compress_convolutional=True)
        # 3) Flatten weights
        layer_weights = net.get_alignment_weights(flatten=True)

        results_per_layer = []
        for layer_idx, (inp, wgt) in enumerate(zip(preprocessed, layer_weights)):
            layer_dict = {}
            for m in methods:
                if m == "delta_alignment":
                    val = AlignmentMetrics.delta_alignment(net, layer_idx, inp)
                else:
                    # if net says "patchwise" & input is 3D => do patchwise
                    if net.cnn_mode == "patchwise" and inp.ndim == 3:
                        val = AlignmentMetrics.patchwise_alignment(inp, wgt, method=m, weigh_by_var=True)
                    else:
                        # single-cov
                        val = AlignmentMetrics.measure(inp, wgt, method=m)
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
            return None, None
        else:
            return None, None

def alignment(input, weight, method="alignment", relative=True):
    """
    measure alignment by computing RQ = (w^T C w) / (w^T w),
    optionally normalized by trace(C).

    - input shape can be (N, F) for single-cov approach,
      or (B, F, patches) if you do patchwise in patchwise_alignment.
    - weight shape (outC, F).
    """
    if input.ndim == 2:
        # standard single-cov approach
        cc = torch.cov(input.T)
        numerator = torch.sum(torch.matmul(weight, cc) * weight, dim=1)
        denom = torch.sum(weight * weight, dim=1)
        rq = numerator / denom
        if method == "alignment" and relative:
            rq = rq / torch.trace(cc)
        return rq
    else:
        # user should call patchwise_alignment or do proper flatten
        raise ValueError(
            f"alignment() got {input.ndim}-D data. For patchwise CNN, call patchwise_alignment(...). "
            "For single-cov, ensure input is 2D via net._preprocess_inputs(...)."
        )


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
    return (torch.tensor(counts, dtype=torch.float),
            torch.tensor(bin_edges, dtype=torch.float),
            torch.tensor(centers, dtype=torch.float))