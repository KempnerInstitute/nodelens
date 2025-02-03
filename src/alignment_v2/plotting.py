# src/alignment_v2/plotting.py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib as mpl
import torch
from alignment_v2.utils import compute_stats_by_type, named_transpose, transpose_list, rms

def plot_train_results(exp, train_results, test_results, prms):
    """
    Plot training and testing performance.
    Shows training loss and accuracy curves over epochs and test loss/accuracy as summary points.
    """
    num_train_epochs = train_results["loss"].size(0)
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]

    print("Computing training statistics...")
    plot_alignment = "alignment" in train_results
    if plot_alignment:
        # For each snapshot, average the alignment (across dimensions)
        alignment = torch.stack([torch.mean(align, dim=2) for align in train_results["alignment"]])
    
    cmap = mpl.colormaps["tab10"]

    train_loss_mean, train_loss_se = compute_stats_by_type(train_results["loss"],
                                                           num_types=num_types, dim=1, method="se")
    train_acc_mean, train_acc_se = compute_stats_by_type(train_results["accuracy"],
                                                         num_types=num_types, dim=1, method="se")

    if plot_alignment:
        align_mean, align_se = compute_stats_by_type(alignment,
                                                      num_types=num_types, dim=1, method="se")

    test_loss_mean, test_loss_se = compute_stats_by_type(torch.tensor(test_results["loss"]),
                                                         num_types=num_types, dim=0, method="se")
    test_acc_mean, test_acc_se = compute_stats_by_type(torch.tensor(test_results["accuracy"]),
                                                       num_types=num_types, dim=0, method="se")

    print("Plotting training and testing performance...")
    x_offset = [-0.2, 0.2]
    get_x = lambda idx: [x_offset[0] + idx, x_offset[1] + idx]

    alpha = 0.3
    figdim = 3
    figratio = 2
    width_ratios = [figdim, figdim / figratio, figdim, figdim / figratio]
    fig, ax = plt.subplots(1, 4, figsize=(sum(width_ratios), figdim),
                           width_ratios=width_ratios, constrained_layout=True)

    # Training loss plot
    for idx, label in enumerate(labels):
        cmn = train_loss_mean[:, idx]
        cse = train_loss_se[:, idx]
        ax[0].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
        ax[0].fill_between(range(num_train_epochs), cmn + cse, cmn - cse, color=(cmap(idx), alpha))
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Training Loss")
    ax[0].set_title("Training Loss")
    ax[0].set_ylim(0, None)

    # Testing loss plot
    for idx, label in enumerate(labels):
        tmn = test_loss_mean[idx]
        tse = test_loss_se[idx]
        ax[1].plot(get_x(idx), [tmn, tmn], color=cmap(idx), lw=4, label=label)
        ax[1].plot([idx, idx], [tmn - tse, tmn + tse], color=cmap(idx), lw=1.5)
    ax[1].set_xticks(range(num_types))
    ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax[1].set_ylabel("Testing Loss")
    ax[1].set_title("Testing Loss")

    # Training accuracy plot
    for idx, label in enumerate(labels):
        cmn = train_acc_mean[:, idx]
        cse = train_acc_se[:, idx]
        ax[2].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
        ax[2].fill_between(range(num_train_epochs), cmn + cse, cmn - cse, color=(cmap(idx), alpha))
    ax[2].set_xlabel("Epoch")
    ax[2].set_ylabel("Training Accuracy (%)")
    ax[2].set_title("Training Accuracy")
    ax[2].set_ylim(0, 100)

    # Testing accuracy plot
    for idx, label in enumerate(labels):
        tmn = test_acc_mean[idx]
        tse = test_acc_se[idx]
        ax[3].plot(get_x(idx), [tmn, tmn], color=cmap(idx), lw=4, label=label)
        ax[3].plot([idx, idx], [tmn - tse, tmn + tse], color=cmap(idx), lw=1.5)
    ax[3].set_xticks(range(num_types))
    ax[3].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax[3].set_ylabel("Testing Accuracy (%)")
    ax[3].set_title("Testing Accuracy")
    ax[3].set_ylim(0, 100)

    plt.show()

def plot_dropout_results(exp, dropout_results, dropout_parameters, prms, dropout_type="nodes"):
    """
    Plot dropout results.
    Plots the loss and accuracy versus dropout fraction for three pruning strategies:
    dropout from high alignment, low alignment, and random.
    """
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val} - dropout {dropout_type}" for val in prms["vals"]]
    cmap = mpl.colormaps["Set1"]
    alpha = 0.3
    msize = 10
    figdim = 3

    num_layers = dropout_results["progdrop_loss_high"].size(2)
    curve_names = ["From high", "From low", "Random"]
    num_curves = len(curve_names)
    dropout_fraction = dropout_results["dropout_fraction"].numpy()
    by_layer = dropout_results["by_layer"]
    extra_name = ("by_layer" if by_layer else "all_layers") + dropout_type

    # Compute loss statistics
    loss_mean_high, loss_se_high = compute_stats_by_type(dropout_results["progdrop_loss_high"],
                                                         num_types=num_types, dim=0, method="se")
    loss_mean_low, loss_se_low = compute_stats_by_type(dropout_results["progdrop_loss_low"],
                                                       num_types=num_types, dim=0, method="se")
    loss_mean_rand, loss_se_rand = compute_stats_by_type(dropout_results["progdrop_loss_rand"],
                                                         num_types=num_types, dim=0, method="se")
    loss_means = [loss_mean_high, loss_mean_low, loss_mean_rand]
    loss_ses = [loss_se_high, loss_se_low, loss_se_rand]

    # Plot loss curves for each network type (averaging over layers)
    fig_loss, axes_loss = plt.subplots(1, num_types, figsize=(num_types*figdim, figdim), constrained_layout=True)
    if num_types == 1:
        axes_loss = [axes_loss]
    for idx in range(num_types):
        for c in range(num_curves):
            # Here we average over layers for simplicity
            mean_curve = loss_means[c][idx, :, :].mean(dim=1).numpy()
            se_curve = loss_ses[c][idx, :, :].mean(dim=1).numpy()
            axes_loss[idx].plot(dropout_fraction, mean_curve, marker=".", markersize=msize,
                                color=cmap(c), label=curve_names[c])
            axes_loss[idx].fill_between(dropout_fraction, mean_curve + se_curve, mean_curve - se_curve,
                                        color=(cmap(c), alpha))
        axes_loss[idx].set_title(labels[idx])
        axes_loss[idx].set_xlabel("Dropout Fraction")
        axes_loss[idx].set_ylabel("Loss with Dropout")
        axes_loss[idx].set_xlim(0, 1)
    plt.suptitle("Progressive Dropout Loss")
    plt.show()

    # Compute accuracy statistics
    acc_mean_high, acc_se_high = compute_stats_by_type(dropout_results["progdrop_acc_high"],
                                                       num_types=num_types, dim=0, method="se")
    acc_mean_low, acc_se_low = compute_stats_by_type(dropout_results["progdrop_acc_low"],
                                                     num_types=num_types, dim=0, method="se")
    acc_mean_rand, acc_se_rand = compute_stats_by_type(dropout_results["progdrop_acc_rand"],
                                                       num_types=num_types, dim=0, method="se")
    acc_means = [acc_mean_high, acc_mean_low, acc_mean_rand]
    acc_ses = [acc_se_high, acc_se_low, acc_se_rand]

    # Plot accuracy curves for each network type
    fig_acc, axes_acc = plt.subplots(1, num_types, figsize=(num_types*figdim, figdim), constrained_layout=True)
    if num_types == 1:
        axes_acc = [axes_acc]
    for idx in range(num_types):
        for c in range(num_curves):
            mean_curve = acc_means[c][idx, :, :].mean(dim=1).numpy()
            se_curve = acc_ses[c][idx, :, :].mean(dim=1).numpy()
            axes_acc[idx].plot(dropout_fraction, mean_curve, marker=".", markersize=msize,
                               color=cmap(c), label=curve_names[c])
            axes_acc[idx].fill_between(dropout_fraction, mean_curve + se_curve, mean_curve - se_curve,
                                       color=(cmap(c), alpha))
        axes_acc[idx].set_title(labels[idx])
        axes_acc[idx].set_xlabel("Dropout Fraction")
        axes_acc[idx].set_ylabel("Accuracy with Dropout (%)")
        axes_acc[idx].set_xlim(0, 1)
        axes_acc[idx].set_ylim(0, 100)
    plt.suptitle("Progressive Dropout Accuracy")
    plt.show()

def plot_eigenfeatures(exp, results, prms):
    """
    Plot eigenfeature analysis results.
    For simplicity, this function plots the mean eigenvalue and the mean beta (projected weight)
    for each alignment layer.
    """
    beta = results["eigen_results"]["beta"]
    eigvals = results["eigen_results"]["eigvals"]
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]
    cmap = mpl.colormaps["tab10"]

    print("Plotting eigenfeatures...")
    num_layers = len(beta)
    fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*3, 3), constrained_layout=True)
    if num_layers == 1:
        ax = [ax]
    for layer in range(num_layers):
        # For each layer, compute the average (over nodes) of beta and eigenvalues for each network type.
        for idx in range(num_types):
            # Assume beta[layer] is a tensor of shape (num_networks, dim)
            b = beta[layer][idx] if beta[layer].ndim == 2 else beta[layer]
            mean_b = b.mean().item()
            # Plot as a bar for simplicity
            ax[layer].bar(idx, mean_b, color=cmap(idx), label=labels[idx] if layer == 0 else "")
        ax[layer].set_title(f"Layer {layer}")
    if num_types > 0:
        ax[0].legend()
    plt.suptitle("Eigenfeatures (Mean Beta)")
    plt.show()