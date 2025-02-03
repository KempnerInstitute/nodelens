import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from alignment_v2.utils import compute_stats_by_type, named_transpose, transpose_list, rms

def plot_train_results(exp, train_results, test_results, prms):
    """
    Plot training and testing loss and accuracy curves.
    Plots training curves as lines with shaded error and testing values as error bars.
    """
    num_train_epochs = train_results["loss"].size(0)
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]

    print("Calculating training statistics...")
    # Calculate mean and standard error for training loss and accuracy.
    train_loss_mean, train_loss_se = compute_stats_by_type(train_results["loss"], num_types=num_types, dim=1, method="se")
    train_acc_mean, train_acc_se   = compute_stats_by_type(train_results["accuracy"], num_types=num_types, dim=1, method="se")
    test_loss_mean, test_loss_se   = compute_stats_by_type(torch.tensor(test_results["loss"]), num_types=num_types, dim=0, method="se")
    test_acc_mean, test_acc_se     = compute_stats_by_type(torch.tensor(test_results["accuracy"]), num_types=num_types, dim=0, method="se")

    print("Plotting training and testing curves...")
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    epochs = np.arange(num_train_epochs)
    # Training loss
    for idx, label in enumerate(labels):
        axs[0].plot(epochs, train_loss_mean[:, idx].numpy(), color=cmap(idx), label=label)
        axs[0].fill_between(epochs,
                            (train_loss_mean[:, idx] + train_loss_se[:, idx]).numpy(),
                            (train_loss_mean[:, idx] - train_loss_se[:, idx]).numpy(),
                            color=cmap(idx), alpha=0.3)
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Training Loss")
    axs[0].legend()

    # Testing loss (using error bars)
    for idx, label in enumerate(labels):
        axs[1].errorbar([idx], [test_loss_mean[idx].item()], yerr=test_loss_se[idx].item(),
                        fmt='o', color=cmap(idx), label=label)
    axs[1].set_xticks(range(num_types))
    axs[1].set_xticklabels(labels, rotation=45)
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Testing Loss")

    # Training accuracy
    for idx, label in enumerate(labels):
        axs[2].plot(epochs, train_acc_mean[:, idx].numpy(), color=cmap(idx), label=label)
        axs[2].fill_between(epochs,
                            (train_acc_mean[:, idx] + train_acc_se[:, idx]).numpy(),
                            (train_acc_mean[:, idx] - train_acc_se[:, idx]).numpy(),
                            color=cmap(idx), alpha=0.3)
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy (%)")
    axs[2].set_title("Training Accuracy")
    axs[2].legend()

    # Testing accuracy
    for idx, label in enumerate(labels):
        axs[3].errorbar([idx], [test_acc_mean[idx].item()], yerr=test_acc_se[idx].item(),
                        fmt='o', color=cmap(idx), label=label)
    axs[3].set_xticks(range(num_types))
    axs[3].set_xticklabels(labels, rotation=45)
    axs[3].set_ylabel("Accuracy (%)")
    axs[3].set_title("Testing Accuracy")
    axs[3].legend()

    plt.tight_layout()
    plt.show()
    # Uncomment if using a custom experiment method:
    # exp.plot_ready("train_test_performance")

def plot_dropout_results(exp, dropout_results, dropout_parameters, prms, dropout_type="nodes"):
    """
    Plot dropout experiment results.
    For each network type, plots the loss and accuracy versus dropout fraction for three methods:
      - Dropout from high alignment nodes
      - Dropout from low alignment nodes
      - Random dropout
    """
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val} - dropout {dropout_type}" for val in prms["vals"]]
    cmap = mpl.colormaps["Set1"]
    alpha = 0.3
    msize = 10
    figdim = 3

    # Number of alignment layers (assumed consistent across networks)
    num_layers = dropout_results["progdrop_loss_high"].size(2)
    method_names = ["from high", "from low", "random"]
    num_methods = len(method_names)
    dropout_fraction = dropout_results["dropout_fraction"].numpy()
    by_layer = dropout_results["by_layer"]
    extra_name = ("by_layer" if by_layer else "all_layers") + dropout_type

    print("Computing dropout statistics...")
    loss_mean_high, loss_se_high = compute_stats_by_type(dropout_results["progdrop_loss_high"], num_types=num_types, dim=0, method="se")
    loss_mean_low, loss_se_low   = compute_stats_by_type(dropout_results["progdrop_loss_low"], num_types=num_types, dim=0, method="se")
    loss_mean_rand, loss_se_rand = compute_stats_by_type(dropout_results["progdrop_loss_rand"], num_types=num_types, dim=0, method="se")

    acc_mean_high, acc_se_high   = compute_stats_by_type(dropout_results["progdrop_acc_high"], num_types=num_types, dim=0, method="se")
    acc_mean_low, acc_se_low     = compute_stats_by_type(dropout_results["progdrop_acc_low"], num_types=num_types, dim=0, method="se")
    acc_mean_rand, acc_se_rand   = compute_stats_by_type(dropout_results["progdrop_acc_rand"], num_types=num_types, dim=0, method="se")

    loss_means = [loss_mean_high, loss_mean_low, loss_mean_rand]
    loss_ses   = [loss_se_high, loss_se_low, loss_se_rand]
    acc_means  = [acc_mean_high, acc_mean_low, acc_mean_rand]
    acc_ses    = [acc_se_high, acc_se_low, acc_se_rand]

    print("Plotting dropout loss curves...")
    fig, axs = plt.subplots(num_layers, num_types, figsize=(num_types*figdim, num_layers*figdim),
                            sharex=True, sharey=True)
    axs = np.reshape(axs, (num_layers, num_types))
    for type_idx, label in enumerate(labels):
        for layer in range(num_layers):
            for m in range(num_methods):
                cur_loss = loss_means[m][type_idx, :, layer].numpy()
                cur_se = loss_ses[m][type_idx, :, layer].numpy()
                axs[layer, type_idx].plot(dropout_fraction, cur_loss, marker=".", markersize=msize,
                                           color=cmap(m), label=method_names[m])
                axs[layer, type_idx].fill_between(dropout_fraction, cur_loss+cur_se, cur_loss-cur_se,
                                                  color=cmap(m), alpha=alpha)
            if layer == 0:
                axs[layer, type_idx].set_title(label)
            if layer == num_layers-1:
                axs[layer, type_idx].set_xlabel("Dropout Fraction")
            if type_idx == 0:
                axs[layer, type_idx].set_ylabel("Loss with Dropout")
            axs[layer, type_idx].set_xlim(0, 1)
            axs[layer, type_idx].legend(loc="best")
    plt.tight_layout()
    plt.show()
    # exp.plot_ready("prog_dropout_" + extra_name + "_loss")

    print("Plotting dropout accuracy curves...")
    fig, axs = plt.subplots(num_layers, num_types, figsize=(num_types*figdim, num_layers*figdim),
                            sharex=True, sharey=True)
    axs = np.reshape(axs, (num_layers, num_types))
    for type_idx, label in enumerate(labels):
        for layer in range(num_layers):
            for m in range(num_methods):
                cur_acc = acc_means[m][type_idx, :, layer].numpy()
                cur_se = acc_ses[m][type_idx, :, layer].numpy()
                axs[layer, type_idx].plot(dropout_fraction, cur_acc, marker=".", markersize=msize,
                                           color=cmap(m), label=method_names[m])
                axs[layer, type_idx].fill_between(dropout_fraction, cur_acc+cur_se, cur_acc-cur_se,
                                                  color=cmap(m), alpha=alpha)
            axs[layer, type_idx].set_ylim(0, 100)
            if layer == 0:
                axs[layer, type_idx].set_title(label)
            if layer == num_layers-1:
                axs[layer, type_idx].set_xlabel("Dropout Fraction")
            if type_idx == 0:
                axs[layer, type_idx].set_ylabel("Accuracy with Dropout")
            axs[layer, type_idx].set_xlim(0, 1)
            axs[layer, type_idx].legend(loc="best")
    plt.tight_layout()
    plt.show()
    # exp.plot_ready("prog_dropout_" + extra_name + "_accuracy")

def plot_eigenfeatures(exp, results, prms):
    """
    Plot eigenfeatures extracted from each alignment layer.
    Here we simply plot the mean eigenvalue per node for each layer.
    """
    beta = results["beta"]
    eigvals = results["eigvals"]
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]
    cmap = mpl.colormaps["tab10"]
    print("Plotting eigenfeatures...")
    num_layers = len(eigvals)
    fig, axs = plt.subplots(1, num_layers, figsize=(num_layers*4, 4))
    for layer in range(num_layers):
        for idx in range(num_types):
            mean_eig = torch.mean(eigvals[layer][idx], dim=0).numpy()
            axs[layer].plot(mean_eig, color=cmap(idx), label=labels[idx])
        axs[layer].set_xscale("log")
        axs[layer].set_title(f"Layer {layer}")
        axs[layer].legend(loc="best")
    plt.tight_layout()
    plt.show()
    # exp.plot_ready("eigenfeatures")

def plot_adversarial_results(exp, eigen_results, adversarial_results, prms):
    """
    Plot adversarial attack success.
    This function plots accuracy vs. epsilon for adversarial attacks.
    """
    accuracy = adversarial_results["accuracy"]
    epsilons = adversarial_results["epsilons"]
    num_types = len(prms["vals"])
    labels = [f"{prms['name']}={val}" for val in prms["vals"]]
    cmap = mpl.colormaps["tab10"]
    print("Plotting adversarial success...")
    plt.figure(figsize=(4,4))
    for idx in range(num_types):
        plt.plot(epsilons, accuracy[:, idx].numpy(), color=cmap(idx), label=labels[idx])
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title("Adversarial Attack Success")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    # exp.plot_ready("adversarial_success")
