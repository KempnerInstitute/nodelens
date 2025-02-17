# --------------------------------------------
# plotting.py
# --------------------------------------------

import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl

from alignment.utils import compute_stats_by_type, named_transpose, transpose_list, rms




def plot_train_results(exp, train_results, test_results, prms):
    """
    Plot training curves for loss and accuracy, plus test performance.
    Optionally plots alignment across training if present in `train_results`.

    If 'loss' or 'accuracy' are missing in train_results (e.g. do_train=false),
    we skip the training curve plot. We can still show test performance if present.
    """

    # If do_train was false, we might not have train_results["loss"] or train_results["accuracy"].
    # We'll do a check:
    has_train_loss = "loss" in train_results and train_results["loss"].dim() == 2
    has_train_acc  = "accuracy" in train_results and train_results["accuracy"].dim() == 2

    # In normal training, shape is (num_replicates, num_epochs)
    # If missing, we can skip them. We'll define num_train_epochs = 0 if missing
    if has_train_loss:
        num_train_epochs = train_results["loss"].size(1)  ### This is the number of epochs
    else:
        num_train_epochs = 0

    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]

    print("getting statistics on run data...")
    plot_alignment = "alignment" in train_results
    alignment = None
    if plot_alignment:
        # Build a dictionary keyed by method
        alignment_by_method = {}
        for record in train_results["alignment"]:
            # same logic as before for building alignment_by_method
            # unchanged except for skipping code if  do_train was false
            if isinstance(record, torch.Tensor):
                method_key = 'RQ'
                if record.dim() == 3:
                    net_avg = record.mean(dim=0)
                elif record.dim() == 4:
                    net_avg = record.squeeze(0).mean(dim=0)
                else:
                    raise ValueError(f"Unexpected tensor shape: {record.shape}")
                for layer_idx in range(net_avg.size(0)):
                    if method_key not in alignment_by_method:
                        alignment_by_method[method_key] = {}
                    if layer_idx not in alignment_by_method[method_key]:
                        alignment_by_method[method_key][layer_idx] = []
                    alignment_by_method[method_key][layer_idx].append(torch.mean(net_avg[layer_idx]).item())

            elif isinstance(record, dict):
                data_field = record["data"]
                if isinstance(data_field, list):
                    for net_data in data_field:
                        for layer_idx, layer_dict in enumerate(net_data):
                            for method_key, tensor in layer_dict.items():
                                if method_key not in alignment_by_method:
                                    alignment_by_method[method_key] = {}
                                if layer_idx not in alignment_by_method[method_key]:
                                    alignment_by_method[method_key][layer_idx] = []
                                alignment_by_method[method_key][layer_idx].append(torch.mean(tensor).item())
                elif isinstance(data_field, torch.Tensor):
                    method_key = 'RQ'
                    if data_field.dim() == 3:
                        net_avg = data_field.mean(dim=0)
                    elif data_field.dim() == 4:
                        net_avg = data_field.squeeze(0).mean(dim=0)
                    else:
                        raise ValueError(f"Unexpected record['data'] shape: {data_field.shape}")
                    for layer_idx in range(net_avg.size(0)):
                        if method_key not in alignment_by_method:
                            alignment_by_method[method_key] = {}
                        if layer_idx not in alignment_by_method[method_key]:
                            alignment_by_method[method_key][layer_idx] = []
                        alignment_by_method[method_key][layer_idx].append(torch.mean(net_avg[layer_idx]).item())
                else:
                    raise TypeError(f"record['data'] has unexpected type {type(data_field)}")
            else:
                raise TypeError(f"train_results['alignment'] has an element of type {type(record)}")

        # Now average across records for each method and each layer.
        alignment_avg = {}
        for method_key, layers_dict in alignment_by_method.items():
            layer_avgs = []
            for layer_idx in sorted(layers_dict.keys()):
                vals = torch.tensor(layers_dict[layer_idx])
                layer_avgs.append(torch.mean(vals))
            alignment_avg[method_key] = torch.stack(layer_avgs, dim=0)
        alignment = alignment_avg

    cmap = mpl.colormaps["tab10"]

    # If we do have training loss/accuracy, let's do the normal stats
    if has_train_loss:
        # shape is (replicates, epochs)
        # We'll transpose to (epochs, replicates) => shape (E,R)
        train_loss_mean, train_loss_se = compute_stats_by_type(
            train_results["loss"].transpose(0,1),  
            num_types=num_types,
            dim=1,
            method="se"
        )
    else:
        train_loss_mean, train_loss_se = None, None

    if has_train_acc:
        train_acc_mean, train_acc_se = compute_stats_by_type(
            train_results["accuracy"].transpose(0,1),
            num_types=num_types,
            dim=1,
            method="se"
        )
    else:
        train_acc_mean, train_acc_se = None, None

    # test_results["loss"] and ["accuracy"] should be shape (replicates,). 
    # If only one pass of test, that's fine. We'll produce a single point for each replicate
    test_loss_mean, test_loss_se = None, None
    test_acc_mean, test_acc_se = None, None
    if ("loss" in test_results) and (test_results["loss"].dim() == 1):
        test_loss_mean, test_loss_se = compute_stats_by_type(
            test_results["loss"],
            num_types=num_types,
            dim=0,
            method="se"
        )
    if ("accuracy" in test_results) and (test_results["accuracy"].dim() == 1):
        test_acc_mean, test_acc_se = compute_stats_by_type(
            test_results["accuracy"],
            num_types=num_types,
            dim=0,
            method="se"
        )

    print("plotting run data...")
    xOffset = [-0.2, 0.2]
    get_x = lambda idx: [xOffset[0] + idx, xOffset[1] + idx]

    alpha = 0.3
    figdim = 3
    figratio = 2
    width_ratios = [figdim, figdim / figratio, figdim, figdim / figratio]

    fig, ax = plt.subplots(1, 4, figsize=(sum(width_ratios), figdim), width_ratios=width_ratios, layout="constrained")

    # -----------
    # Left two panels: training loss & test loss
    # -----------
    for idx, label in enumerate(labels):
        if train_loss_mean is not None:
            cmn = train_loss_mean[:, idx]  # shape => (num_train_epochs,)
            cse = train_loss_se[:, idx]
            ax[0].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
            ax[0].fill_between(range(num_train_epochs), cmn + cse, cmn - cse, color=(cmap(idx), alpha))

        if test_loss_mean is not None:
            tmn = test_loss_mean[idx]
            tse = test_loss_se[idx]
            ax[1].plot(get_x(idx), [tmn] * 2, color=cmap(idx), label=label, lw=4)
            ax[1].plot([idx, idx], [tmn - tse, tmn + tse], color=cmap(idx), lw=1.5)

    ax[0].set_xlabel("Training Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Training Loss")
    ax[1].set_title("Testing")

    if train_loss_mean is not None:
        ax[0].set_ylim(0, None)
        ylims = ax[0].get_ylim()
        ax[1].set_ylim(ylims)
    ax[1].set_ylabel("Loss")
    ax[1].set_xlim(-0.5, len(labels) - 0.5)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    # -----------
    # Right two panels: training acc & test acc
    # -----------
    for idx, label in enumerate(labels):
        if train_acc_mean is not None:
            cmn = train_acc_mean[:, idx]
            cse = train_acc_se[:, idx]
            ax[2].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
            ax[2].fill_between(range(num_train_epochs), cmn + cse, cmn - cse, color=(cmap(idx), alpha))

        if test_acc_mean is not None:
            tmn = test_acc_mean[idx]
            tse = test_acc_se[idx]
            ax[3].plot(get_x(idx), [tmn]*2, color=cmap(idx), label=label, lw=4)
            ax[3].plot([idx, idx], [tmn - tse, tmn + tse], color=cmap(idx), lw=1.5)

    ax[2].set_xlabel("Training Epoch")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title("Training Accuracy")
    ax[3].set_title("Testing")
    if train_acc_mean is not None:
        ax[2].set_ylim(0, 100)
        ax[3].set_ylim(0, 100)
    ax[3].set_xlim(-0.5, len(labels) - 0.5)
    ax[3].set_xticks(range(len(labels)))
    ax[3].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    
    exp.plot_ready("train_test_performance")

    # -----------
    # If alignment data is present, plot it
    # -----------
    if plot_alignment and alignment is not None:
        for method_key, align_tensor in alignment.items():
            num_layers = align_tensor.size(0)
            fig2, ax2 = plt.subplots(1, num_layers, figsize=(num_layers * figdim, figdim), layout="constrained", sharex=True, squeeze=False)
            for layer in range(num_layers):
                val = align_tensor[layer].item() * 100
                ax2[0, layer].axhline(val, color=cmap(0), label=method_key)
                ax2[0, layer].set_xlabel("Snapshots")
                ax2[0, layer].set_ylabel("Alignment (%)")
                ax2[0, layer].set_title(f"Layer {layer} - {method_key}")
            exp.plot_ready(f"train_alignment_{method_key}")



def plot_dropout_results(exp, dropout_results, dropout_parameters, prms, dropout_type="nodes"):
    """
    Visualize progressive dropout experiment results (loss & accuracy),
    comparing dropout from highest alignment, lowest alignment, or random nodes.
    """
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]
    cmap = mpl.colormaps["Set1"]
    alpha = 0.3
    msize = 10
    figdim = 3

    num_layers = dropout_results["progdrop_loss_high"].size(2)
    names = ["From high", "From low", "Random"]
    dropout_fraction = dropout_results["dropout_fraction"]
    by_layer = dropout_results["by_layer"]
    extra_name = "by_layer" if by_layer else "all_layers"
    extra_name += dropout_type

    print("measuring statistics on dropout analysis...")
    loss_mean_high, loss_se_high = compute_stats_by_type(dropout_results["progdrop_loss_high"], num_types=num_types, dim=0, method="se")
    loss_mean_low,  loss_se_low  = compute_stats_by_type(dropout_results["progdrop_loss_low"],  num_types=num_types, dim=0, method="se")
    loss_mean_rand, loss_se_rand = compute_stats_by_type(dropout_results["progdrop_loss_rand"], num_types=num_types, dim=0, method="se")

    acc_mean_high,  acc_se_high  = compute_stats_by_type(dropout_results["progdrop_acc_high"], num_types=num_types, dim=0, method="se")
    acc_mean_low,   acc_se_low   = compute_stats_by_type(dropout_results["progdrop_acc_low"],  num_types=num_types, dim=0, method="se")
    acc_mean_rand,  acc_se_rand  = compute_stats_by_type(dropout_results["progdrop_acc_rand"], num_types=num_types, dim=0, method="se")

    loss_mean = [loss_mean_high, loss_mean_low, loss_mean_rand]
    loss_se   = [loss_se_high,  loss_se_low,   loss_se_rand]
    acc_mean  = [acc_mean_high, acc_mean_low,  acc_mean_rand]
    acc_se    = [acc_se_high,   acc_se_low,    acc_se_rand]

    print("plotting dropout results...")
    fig, ax = plt.subplots(
        num_layers,
        num_types,
        figsize=(num_types * figdim, num_layers * figdim),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False
    )
    ax = np.reshape(ax, (num_layers, num_types))

    names = ["From high", "From low", "Random"]
    num_exp = len(names)

    # --- ADDED ---
    # If we stored alignment_names in 'results', we can retrieve them
    alignment_names = None
    if "alignment_names" in dropout_results:
        alignment_names = dropout_results["alignment_names"]
    elif "alignment_names" in prms:
        alignment_names = prms["alignment_names"]
    elif "alignment_names" in exp.__dict__.get("args", {}):
        alignment_names = exp.args.alignment_layers  # fallback, if relevant

    # If not in dropout_results, check results in case we saved it there
    # We can do:
    if alignment_names is None and hasattr(exp, "results"):
        alignment_names = exp.results.get("alignment_names", None)
    # or skip if not found

    for idx, label in enumerate(labels):
        for layer in range(num_layers):
            for iexp, name in enumerate(names):
                cmn = loss_mean[iexp][idx, :, layer]
                cse = loss_se[iexp][idx, :, layer]
                ax[layer, idx].plot(
                    dropout_fraction,
                    cmn,
                    color=cmap(iexp),
                    marker=".",
                    markersize=msize,
                    label=name,
                )
                ax[layer, idx].fill_between(dropout_fraction, cmn + cse, cmn - cse, color=(cmap(iexp), alpha))

            if layer == 0:
                ax[layer, idx].set_title(label)
            if layer == num_layers - 1:
                ax[layer, idx].set_xlabel("Dropout Fraction")
                ax[layer, idx].set_xlim(0, 1)
            if idx == 0:
                # Label left y-axis
                # Attempt to label layer name if available
                lab = "Loss w/ Dropout"
                if alignment_names is not None and layer < len(alignment_names):
                    lab = f"{alignment_names[layer]}: Loss"
                ax[layer, idx].set_ylabel(lab)

            if iexp == num_exp - 1:
                ax[layer, idx].legend(loc="best")

    exp.plot_ready("prog_dropout_" + extra_name + "_loss")


    fig, ax = plt.subplots(
        num_layers,
        num_types,
        figsize=(num_types * figdim, num_layers * figdim),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False
    )
    ax = np.reshape(ax, (num_layers, num_types))

    for idx, label in enumerate(labels):
        for layer in range(num_layers):
            for iexp, name in enumerate(names):
                cmn = acc_mean[iexp][idx, :, layer]
                cse = acc_se[iexp][idx, :, layer]
                ax[layer, idx].plot(
                    dropout_fraction,
                    cmn,
                    color=cmap(iexp),
                    marker=".",
                    markersize=msize,
                    label=name,
                )
                ax[layer, idx].fill_between(dropout_fraction, cmn + cse, cmn - cse, color=(cmap(iexp), alpha))

            ax[layer, idx].set_ylim(0, 100)
            if layer == 0:
                ax[layer, idx].set_title(label)
            if layer == num_layers - 1:
                ax[layer, idx].set_xlabel("Dropout Fraction")
                ax[layer, idx].set_xlim(0, 1)
            if idx == 0:
                # Label left y-axis
                lab = "Accuracy w/ Dropout"
                if alignment_names is not None and layer < len(alignment_names):
                    lab = f"{alignment_names[layer]}: Accuracy"
                ax[layer, idx].set_ylabel(lab)

            if iexp == num_exp - 1:
                ax[layer, idx].legend(loc="best")

    exp.plot_ready("prog_dropout_" + extra_name + "_accuracy")


def plot_eigenfeatures(exp, results, prms):
    """
    method for plotting results related to eigen-analysis (beta, eigenvalues, class_betas).
    """
    beta, eigvals, class_betas, class_names = (
        results["beta"],
        results["eigvals"],
        results["class_betas"],
        results["class_names"],
    )
    beta = [[torch.abs(b) for b in net_beta] for net_beta in beta]
    class_betas = [[rms(cb, dim=2) for cb in net_class_beta] for net_class_beta in class_betas]

    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]
    cmap = mpl.colormaps["tab10"]
    class_cmap = mpl.colormaps["viridis"].resampled(len(class_names))

    print("measuring statistics of eigenfeature analyses...")

    beta = [torch.stack(b) for b in transpose_list(beta)]
    eigvals = [torch.stack(ev) for ev in transpose_list(eigvals)]
    class_betas = [torch.stack(cb) for cb in transpose_list(class_betas)]

    beta = [b / b.sum(dim=2, keepdim=True) for b in beta]
    eigvals = [ev / ev.sum(dim=1, keepdim=True) for ev in eigvals]
    class_betas = [cb / cb.sum(dim=2, keepdim=True) for cb in class_betas]

    statprms = lambda method: dict(num_types=num_types, dim=0, method=method)
    mean_evals, var_evals = named_transpose([compute_stats_by_type(ev, **statprms("var")) for ev in eigvals])
    sorted_beta = [torch.sort(b, descending=True, dim=2).values for b in beta]
    mean_beta, se_beta = named_transpose([compute_stats_by_type(b, **statprms("var")) for b in beta])
    mean_sorted, se_sorted = named_transpose([compute_stats_by_type(b, **statprms("var")) for b in sorted_beta])
    mean_class_beta, se_class_beta = named_transpose([compute_stats_by_type(cb, **statprms("var")) for cb in class_betas])

    print("plotting eigenfeature results...")
    figdim = 3
    alpha = 0.3
    num_layers = len(mean_beta)
    fig, ax = plt.subplots(2, num_layers, figsize=(num_layers * figdim, figdim * 2), layout="constrained", squeeze=False)

    for layer in range(num_layers):
        num_input = mean_evals[layer].size(1)
        num_nodes = mean_beta[layer].size(1)
        for idx, label in enumerate(labels):
            mn_ev = mean_evals[layer][idx]
            se_ev = var_evals[layer][idx]
            mn_beta_ = torch.mean(mean_beta[layer][idx], dim=0)
            se_beta_ = torch.std(mean_beta[layer][idx], dim=0) / np.sqrt(num_nodes)
            mn_sort = torch.mean(mean_sorted[layer][idx], dim=0)
            se_sort = torch.std(mean_sorted[layer][idx], dim=0) / np.sqrt(num_nodes)

            ax[0, layer].plot(
                range(num_input),
                mn_ev,
                color=cmap(idx),
                linestyle="--",
                label="eigvals" if idx == 0 else None,
            )
            ax[0, layer].plot(range(num_input), mn_beta_, color=cmap(idx), label=label)
            ax[0, layer].fill_between(range(num_input), mn_beta_ + se_beta_, mn_beta_ - se_beta_, color=(cmap(idx), alpha))

            ax[1, layer].plot(range(num_input), mn_sort, color=cmap(idx), label=label)
            ax[1, layer].fill_between(range(num_input), mn_sort + se_sort, mn_sort - se_sort, color=(cmap(idx), alpha))

            ax[0, layer].set_xscale("log")
            ax[1, layer].set_xscale("log")
            ax[0, layer].set_xlabel("Input Dimension")
            ax[1, layer].set_xlabel("Sorted Input Dim")
            ax[0, layer].set_ylabel("Relative Eigval & Beta")
            ax[1, layer].set_ylabel("Relative Beta (Sorted)")
            ax[0, layer].set_title(f"Layer {layer}")
            ax[1, layer].set_title(f"Layer {layer}")

            if layer == num_layers - 1:
                ax[0, layer].legend(loc="best")
                ax[1, layer].legend(loc="best")

    exp.plot_ready("eigenfeatures")

    fig, ax = plt.subplots(1, num_layers, figsize=(num_layers * figdim, figdim), layout="constrained", squeeze=False)
    for layer in range(num_layers):
        num_input = mean_evals[layer].size(1)
        num_nodes = mean_beta[layer].size(1)
        for idx, label in enumerate(labels):
            mn_ev = mean_evals[layer][idx]
            se_ev = var_evals[layer][idx]
            mn_beta_ = torch.mean(mean_beta[layer][idx], dim=0)
            se_beta_ = torch.std(mean_beta[layer][idx], dim=0) / np.sqrt(num_nodes)
            ax[0, layer].plot(
                range(num_input),
                mn_ev,
                color=cmap(idx),
                linestyle="--",
                label="eigvals" if idx == 0 else None,
            )
            ax[0, layer].plot(range(num_input), mn_beta_, color=cmap(idx), label=label)
            ax[0, layer].fill_between(range(num_input), mn_beta_ + se_beta_, mn_beta_ - se_beta_, color=(cmap(idx), alpha))

            ax[0, layer].set_xscale("log")
            ax[0, layer].set_yscale("log")
            ax[0, layer].set_xlabel("Input Dimension")
            ax[0, layer].set_ylabel("Relative Eigval & Beta")
            ax[0, layer].set_title(f"Layer {layer}")

            if layer == num_layers - 1:
                ax[0, layer].legend(loc="best")

    exp.plot_ready("eigenfeatures_loglog")

    fig, ax = plt.subplots(
        num_types,
        num_layers,
        figsize=(num_layers * figdim, figdim * num_types),
        layout="constrained",
        squeeze=False,
    )
    ax = np.reshape(ax, (num_types, num_layers))
    for layer in range(num_layers):
        num_input = mean_evals[layer].size(1)
        for idx, label in enumerate(labels):
            for idx_class, class_name in enumerate(class_names):
                mn_data = mean_class_beta[layer][idx][idx_class]
                se_data = se_class_beta[layer][idx][idx_class]
                ax[idx, layer].plot(range(num_input), mn_data, color=class_cmap(idx_class), label=class_name)
                ax[idx, layer].fill_between(
                    range(num_input),
                    mn_data + se_data,
                    mn_data - se_data,
                    color=(class_cmap(idx_class), alpha),
                )

            ax[idx, layer].set_xscale("log")
            ax[idx, layer].set_yscale("linear")
            ax[idx, layer].set_xlabel("Input Dimension")
            if layer == 0:
                ax[idx, layer].set_ylabel(f"{label}\nClass Loading (RMS)")
            if idx == 0:
                ax[idx, layer].set_title(f"Layer {layer}")
            if layer == num_layers - 1:
                ax[idx, layer].legend(loc="upper right", fontsize=6)

    exp.plot_ready("class_eigenfeatures")

def plot_adversarial_results(exp, eigen_results, adversarial_results, prms):
    """
    Visualize adversarial attack results (accuracy vs epsilon) and structure 
    of adversarial perturbations in the eigenvector basis.
    """
    accuracy, beta, eigvals = (
        adversarial_results["accuracy"],
        adversarial_results["betas"],
        eigen_results["eigvals"],
    )
    epsilons, use_sign = adversarial_results["use_sign"], adversarial_results["use_sign"]

    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]
    cmap = mpl.colormaps["tab10"]

    print("measuring statistics of adversarial analyses...")

    accuracy = torch.stack([torch.stack(acc) for acc in transpose_list(accuracy)])  # shape (num_epsilon, num_nets)
    eigvals = [torch.stack(ev) for ev in transpose_list(eigvals)]
    beta = [torch.stack(b) for b in beta]

    beta = [b / b.sum(dim=2, keepdim=True) for b in beta]
    eigvals = [ev / ev.sum(dim=1, keepdim=True) for ev in eigvals]

    statprms = lambda dim, method: dict(num_types=num_types, dim=dim, method=method)
    mean_acc, se_acc = compute_stats_by_type(accuracy, **statprms(1, "var"))

    from alignment.core.utils import named_transpose
    mean_beta, se_beta = named_transpose([compute_stats_by_type(b, **statprms(1, "var")) for b in beta])
    mean_evals, var_evals = named_transpose([compute_stats_by_type(ev, **statprms(0, "var")) for ev in eigvals])

    print("plotting adversarial success results...")
    figdim = 3
    alpha = 0.3
    num_layers = len(mean_beta)
    fig, ax = plt.subplots(1, 1, figsize=(figdim, figdim), layout="constrained")
    for idx, label in enumerate(labels):
        ax.plot(epsilons, mean_acc[:, idx], color=cmap(idx), label=label)
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Accuracy")
        ax.set_title("Adversarial Attack Success")
        ax.legend(loc="best")

    exp.plot_ready("adversarial_success")

    print("plotting adversarial structure...")
    fig, ax = plt.subplots(1, num_layers, figsize=(num_layers * figdim, figdim), layout="constrained", squeeze=False)
    for layer in range(num_layers):
        num_input = mean_beta[layer].size(2)
        for idx, label in enumerate(labels):
            ax[0, layer].plot(
                range(num_input),
                torch.nanmean(mean_beta[layer][:, idx], dim=0).detach(),
                color=cmap(idx),
                label=label,
            )
        ax[0, layer].set_xscale("log")
        ax[0, layer].set_xlabel("Input Dimension")
        ax[0, layer].set_ylabel("Average Component of Pertubation")
        ax[0, layer].set_title(f"Layer {layer}")
        if layer == num_layers - 1:
            ax[0, layer].legend(loc="best", fontsize=6)

    exp.plot_ready("adversarial_structure")

def plot_rf(rf, width, alignment=None, alignBounds=None, showRFs=None, figSize=5):
    """
    Helper for visualizing receptive fields in a grid. 
    Optionally color-coded by alignment.
    """
    if showRFs is not None:
        rf = rf.reshape(rf.shape[0], -1)
        idxRandom = np.random.choice(range(rf.shape[0]), showRFs, replace=False)
        rf = rf[idxRandom, :]
    else:
        showRFs = rf.shape[0]
    rf = rf.T / np.abs(rf).max(axis=1)
    rf = rf.T
    rf = rf.reshape(showRFs, width, width)
    if alignment is not None:
        cmap = mpl.cm.get_cmap("rainbow", rf.shape[0])
        cmapPeak = lambda x: cmap(x)
        if alignBounds is not None:
            alignment = alignment - alignBounds[0]
            alignment = alignment / (alignBounds[1] - alignBounds[0])
        else:
            alignment = alignment - alignment.min()
            alignment = alignment / alignment.max()

    n = int(np.ceil(np.sqrt(rf.shape[0])))
    fig, axes = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, squeeze=False)
    fig.set_size_inches(figSize, figSize)

    N = 1000
    for i in tqdm(range(rf.shape[0])):
        ax = axes[i // n][i % n]
        if alignment is not None:
            vals = np.ones((N, 4))
            cAlignment = alignment[i].numpy()
            cPeak = cmapPeak(alignment[i].numpy())
            vals[:, 0] = np.linspace(0, cPeak[0], N)
            vals[:, 1] = np.linspace(0, cPeak[1], N)
            vals[:, 2] = np.linspace(0, cPeak[2], N)
            usecmap = mpl.colors.ListedColormap(vals)
            ax.imshow(rf[i], cmap=usecmap, vmin=-1, vmax=1)
        else:
            ax.imshow(rf[i], cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    for j in range(rf.shape[0], n * n):
        ax = axes[j // n][j % n]
        ax.imshow(np.ones_like(rf[0]) * -1, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    return fig