#!/usr/bin/env python
# coding: utf-8

# In[1]:


################################################################################
# Cell 1: Imports & Global Settings
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.spatial.distance as sdist

import copy
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


# In[2]:


################################################################################
# Cell 2: Define the MLP Architecture
#
# We'll make a simple 4-layer MLP with hidden dims [1024, 512, 256] 
# final out_features=10 for CIFAR-10 classification
# input_dim=3*32*32=3072
################################################################################

class SimpleMLP(nn.Module):
    """4-layer MLP for CIFAR-10 classification:
       input -> [linear(1024) -> ReLU] -> [linear(512) -> ReLU] 
          -> [linear(256) -> ReLU] -> [linear(10)] -> output
    """
    def __init__(self, input_dim=3*32*32, num_classes=10, hidden_dims=[1024,512,256]):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for hdim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hdim))
            prev_dim = hdim
        self.final_layer = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        # x shape: (B, 3, 32, 32)
        # Flatten
        x = x.view(x.size(0), -1)
        for linear_layer in self.layers:
            x = F.relu(linear_layer(x))
        x = self.final_layer(x)
        return x

    def num_layers(self):
        # ignoring final classification layer for "hidden" layers
        return len(self.layers)

    def layer_weights(self, layer_idx):
        return self.layers[layer_idx].weight  # shape (out_features, in_features)

    def layer_biases(self, layer_idx):
        return self.layers[layer_idx].bias

    def final_layer_weights(self):
        return self.final_layer.weight
    
    def final_layer_bias(self):
        return self.final_layer.bias


# In[3]:


################################################################################
# Cell 3: CIFAR-10 Data Loading
# We'll just do a train/test split with normalization. 
# For demonstration, we keep batch_size moderate (e.g., 128 or 256).
################################################################################

def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# In[4]:


################################################################################
# Cell 4: Rayleigh Quotient (RQ), approximate MI, redundancy, and helper functions
#
# For layer 0, we computed the covariance of the input images (3072-dim).
# For subsequent layers, we need the covariance of the activations.
################################################################################

def estimate_data_cov(model, loader, subset_size=512, device='cpu'):
    """
    (Retained for layer 0, if needed)
    Estimate a covariance matrix of the flattened input images.
    """
    X_samples = []
    collected = 0
    for images, _ in loader:
        images = images.to(device)
        bsz = images.size(0)
        if collected + bsz > subset_size:
            needed = subset_size - collected
            images = images[:needed]
        X_samples.append(images)
        collected += images.size(0)
        if collected >= subset_size:
            break
    X_samples = torch.cat(X_samples, dim=0)  # shape (subset_size, 3,32,32)
    X_flat = X_samples.view(X_samples.size(0), -1)  # shape (subset_size, 3072)
    mean_ = X_flat.mean(dim=0, keepdim=True)
    X_centered = X_flat - mean_
    cov = (X_centered.t() @ X_centered) / (X_centered.size(0) - 1)
    return cov, mean_

def get_activation(model, x, layer_idx):
    """
    Returns the activation after the ReLU of the specified hidden layer.
    """
    x = x.view(x.size(0), -1)
    for i, linear_layer in enumerate(model.layers):
        x = F.relu(linear_layer(x))
        if i == layer_idx:
            return x
    return x

def estimate_layer_cov(model, loader, layer_idx, subset_size=512, device='cpu'):
    """
    Estimate the covariance matrix of the activations at the given hidden layer.
    For layer 0, compute covariance of raw input (flattened).
    For other layers, compute covariance of activations from the previous layer.
    """
    if int(layer_idx) == 0:
        # Get raw input covariance
        X_samples = []
        collected = 0
        for images, _ in loader:
            images = images.to(device)
            bsz = images.size(0)
            if collected + bsz > subset_size:
                needed = subset_size - collected
                images = images[:needed]
            X_samples.append(images)
            collected += images.size(0)
            if collected >= subset_size:
                break
        X_samples = torch.cat(X_samples, dim=0)  # shape: (subset_size, 3,32,32)
        X_flat = X_samples.view(X_samples.size(0), -1)  # shape: (subset_size, 3072)
        mean_ = X_flat.mean(dim=0, keepdim=True)
        X_centered = X_flat - mean_
        cov = (X_centered.t() @ X_centered) / (X_centered.size(0) - 1)
        return cov.to(device), mean_.to(device)
    else:
        activations = []
        collected = 0
        for images, _ in loader:
            images = images.to(device)
            # Get activation from the previous layer
            act = get_activation(model, images, int(layer_idx) - 1)
            activations.append(act)
            collected += act.size(0)
            if collected >= subset_size:
                break
        activations = torch.cat(activations, dim=0)[:subset_size]  # shape: (subset_size, d)
        mean_act = activations.mean(dim=0, keepdim=True)
        act_centered = activations - mean_act
        cov = (act_centered.t() @ act_centered) / (activations.size(0) - 1)
        return cov.to(device), mean_act.to(device)

def compute_per_node_RQ_MI(model, layer_idx, cov_input, device='cpu'):
    """
    RQ(node) ~ (w_node^T cov_input w_node) / (w_node^T w_node)
    MI(node) ~ log(1 + w_node^T cov_input w_node) (approximation)
    """
    W = model.layer_weights(layer_idx).detach().to(device)  # shape (out_features, in_features)
    w_norm = torch.sum(W * W, dim=1) + 1e-9  # (out_features,)
    temp = torch.matmul(W, cov_input)           # (out_features, in_features)
    quad = torch.sum(temp * W, dim=1)            # (out_features,)
    RQ = quad / w_norm
    MI = torch.log1p(quad)
    return RQ.detach().cpu().numpy(), MI.detach().cpu().numpy()

def compute_pairwise_redundancy(model, layer_idx, cov_input, device='cpu'):
    """
    redundancy(i,j) ~ (w_i^T cov_input w_j) / sqrt((w_i^T w_i)*(w_j^T w_j))
    """
    W = model.layer_weights(layer_idx).detach().to(device)
    M = torch.matmul(torch.matmul(W, cov_input), W.t())
    w_norm = torch.sqrt(torch.sum(W * W, dim=1, keepdim=True))
    denom = w_norm @ w_norm.t()
    R = M / (denom + 1e-9)
    return R.detach().cpu().numpy()


# In[5]:


################################################################################
# Cell 5: Training Loop with Metric Collection (Updated)
################################################################################

def train_and_collect_metrics(model, train_loader, test_loader, num_epochs=3, device='cpu'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    all_metrics = {}  # dict epoch -> layer_idx -> { 'RQ':..., 'MI':..., 'redundancy_mat':...}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100.0
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                test_correct += torch.sum(preds == labels).item()
                test_total += labels.size(0)
        test_acc = test_correct / test_total * 100.0
        print(f"           Test Accuracy: {test_acc:.2f}%")

        # For each layer, compute the covariance of its input activations
        layer_dict = {}
        for layer_idx in range(model.num_layers()):
            # For layer 0, you may choose to use the original input covariance via estimate_data_cov,
            # but here we uniformly compute the covariance from the activation at each layer.
            cov_input, _ = estimate_layer_cov(model, train_loader, layer_idx, subset_size=256, device=device)
            RQ_vals, MI_vals = compute_per_node_RQ_MI(model, layer_idx, cov_input, device=device)
            redund_mat = compute_pairwise_redundancy(model, layer_idx, cov_input, device=device)
            layer_dict[layer_idx] = {
                'RQ': RQ_vals,
                'MI': MI_vals,
                'redundancy_mat': redund_mat
            }
        all_metrics[epoch] = layer_dict
        
    return all_metrics


def compute_per_node_dRQ(avg_grad, cov_input, device='cpu'):
    """
    Compute dRQ using the average gradient matrix (delta weights).
    dRQ(node) = (g_node^T cov_input g_node) / (g_node^T g_node),
    where g_node is the gradient vector for that neuron.
    """
    g = avg_grad.to(device)  # shape (out_features, in_features)
    g_norm = torch.sum(g * g, dim=1) + 1e-9
    temp = torch.matmul(g, cov_input)
    quad = torch.sum(temp * g, dim=1)
    dRQ = quad / g_norm
    return dRQ.detach().cpu().numpy()

def train_with_gradient_tracking(model, train_loader, criterion, optimizer, device='cuda', num_epochs=10):
    """
    Train the model while tracking gradient information and computing per-epoch metrics.
    Returns:
        model: trained model,
        gradient_norms: dict mapping epoch -> layer_idx -> per-neuron average gradient norm,
        avg_gradients_epoch: dict mapping epoch -> layer_idx -> average gradient matrix,
        all_metrics: dict mapping epoch -> layer_idx -> {
                        'RQ': ..., 'MI': ..., 'redundancy_mat': ...,
                        'grad': gradient norm per neuron, 'dRQ': dRQ computed using avg grad }.
    """
    gradient_norms = {}
    avg_gradients_epoch = {}
    all_metrics = {}
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        # Initialize accumulators for each layer
        sum_grads = [None] * model.num_layers()
        count_grads = [0] * model.num_layers()
        # Also store per-neuron gradient norms for each batch in this epoch
        epoch_gradients = {i: [] for i in range(model.num_layers())}
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Accumulate gradients for each layer
            for i, layer in enumerate(model.layers):
                if layer.weight.grad is not None:
                    if sum_grads[i] is None:
                        sum_grads[i] = layer.weight.grad.detach().clone()
                    else:
                        sum_grads[i] += layer.weight.grad.detach().clone()
                    count_grads[i] += 1
                    # Record gradient norm per neuron for this batch
                    for j in range(layer.weight.size(0)):
                        neuron_grad = layer.weight.grad[j, :].detach().norm().item()
                        if len(epoch_gradients[i]) <= j:
                            epoch_gradients[i].append([neuron_grad])
                        else:
                            epoch_gradients[i][j].append(neuron_grad)
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        
        # Compute average gradient matrix and average per-neuron gradient norm for each layer
        avg_grads = {}
        epoch_grad_norms = {}
        for i in range(model.num_layers()):
            if count_grads[i] > 0:
                avg_grads[i] = sum_grads[i] / count_grads[i]
            else:
                avg_grads[i] = torch.zeros_like(model.layers[i].weight)
            epoch_grad_norms[i] = np.array([np.mean(neuron_grads) for neuron_grads in epoch_gradients[i]])
        gradient_norms[epoch] = epoch_grad_norms
        avg_gradients_epoch[epoch] = avg_grads
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / batch_count:.4f}")
        
        # Now compute metrics for each layer for this epoch.
        model.eval()
        epoch_metrics = {}
        for layer_idx in range(model.num_layers()):
            # Compute covariance from the activation input to the layer.
            cov_input, _ = estimate_layer_cov(model, train_loader, layer_idx, subset_size=256, device=device)
            RQ_vals, MI_vals = compute_per_node_RQ_MI(model, layer_idx, cov_input, device=device)
            redund_mat = compute_pairwise_redundancy(model, layer_idx, cov_input, device=device)
            grad_metric = gradient_norms[epoch][layer_idx]  # per-neuron average gradient norm
            dRQ_vals = compute_per_node_dRQ(avg_grads[layer_idx], cov_input, device=device)
            epoch_metrics[layer_idx] = {
                'RQ': RQ_vals,
                'MI': MI_vals,
                'redundancy_mat': redund_mat,
                'grad': grad_metric,
                'dRQ': dRQ_vals
            }
        all_metrics[epoch] = epoch_metrics
        
    return model, gradient_norms, avg_gradients_epoch, all_metrics

# def train_with_gradient_tracking(model, train_loader, criterion, optimizer, device='cuda', num_epochs=10):
#     """
#     Train the model while tracking:
#       - Gradient norms per neuron per layer (averaged over batches)
#       - Average gradient vectors per neuron per layer (delta weights)
#       - And, at the end of each epoch, compute RQ, MI, redundancy, and store the covariance matrix
#          for later use in computing dRQ.
    
#     Returns:
#       model, gradient_norms, avg_gradients_epoch, all_metrics
#     """
#     gradient_norms = {}       # {epoch: {layer_idx: np.array of shape (num_neurons,)}}
#     avg_gradients_epoch = {}  # {epoch: {layer_idx: np.array of shape (num_neurons, in_features)}}
#     all_metrics = {}          # {epoch: {layer_idx: {'RQ':..., 'MI':..., 'redundancy_mat':..., 'cov':...}}}
#     model = model.to(device)
 
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         # For each layer, accumulate gradients over batches.
#         epoch_gradients = {i: [] for i in range(len(model.layers))}
#         sum_gradients = {}
#         for i, layer in enumerate(model.layers):
#             sum_gradients[i] = torch.zeros_like(layer.weight, device=device)
        
#         num_batches = 0
#         for inputs, labels in train_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             # Accumulate gradients for each layer.
#             for i, layer in enumerate(model.layers):
#                 # layer.weight.grad has shape (out_features, in_features)
#                 sum_gradients[i] += layer.weight.grad.detach()
#                 # Also store per-neuron gradient norm (over the weight row)
#                 grad_norms = layer.weight.grad.detach().norm(dim=1).cpu().numpy()  # shape (out_features,)
#                 epoch_gradients[i].append(grad_norms)
#             optimizer.step()
#             running_loss += loss.item() * inputs.size(0)
#             num_batches += 1
        
#         # Average per-neuron gradient norms and gradient vectors over batches.
#         gradient_norms[epoch] = {}
#         avg_gradients_epoch[epoch] = {}
#         for i in range(len(model.layers)):
#             # Average gradient norm: stack along batch dimension and take mean
#             grads_array = np.stack(epoch_gradients[i], axis=0)  # shape (num_batches, out_features)
#             gradient_norms[epoch][i] = np.mean(grads_array, axis=0)  # mean per neuron
#             # Average gradient vector for each neuron:
#             avg_grad = (sum_gradients[i] / num_batches).detach().cpu().numpy()  # shape (out_features, in_features)
#             avg_gradients_epoch[epoch][i] = avg_grad
        
#         avg_loss = running_loss / len(train_loader.dataset)
#         print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
#         # Compute per-layer metrics (RQ, MI, redundancy) using covariance of activations.
#         layer_metrics = {}
#         for layer_idx in range(model.num_layers()):
#             # For layer 0, compute covariance of raw inputs; for others, use activation of previous layer.
#             cov_input, _ = estimate_layer_cov(model, train_loader, layer_idx, subset_size=256, device=device)
#             RQ_vals, MI_vals = compute_per_node_RQ_MI(model, layer_idx, cov_input, device=device)
#             redund_mat = compute_pairwise_redundancy(model, layer_idx, cov_input, device=device)
#             layer_metrics[layer_idx] = {
#                 'RQ': RQ_vals,
#                 'MI': MI_vals,
#                 'redundancy_mat': redund_mat,
#                 'cov': cov_input.cpu().detach().numpy()  # store covariance for later dRQ computation
#             }
#         all_metrics[epoch] = layer_metrics
        
#     return model, gradient_norms, avg_gradients_epoch, all_metrics


# In[6]:


################################################################################
# Cell 6: Plotting utilities
################################################################################

def plot_metric_distribution(all_metrics, metric_name='RQ'):
    """
    For the 1st and last epochs, plot histogram distribution of the chosen metric
    for each layer. 
    """
    epochs_list = sorted(all_metrics.keys())
    if len(epochs_list) < 1:
        print("No epochs stored!")
        return

    first_epoch = epochs_list[0]
    last_epoch  = epochs_list[-1]

    # how many layers
    num_layers = len(all_metrics[first_epoch].keys())

    fig, axes = plt.subplots(2, num_layers, figsize=(5*num_layers, 8))
    if num_layers==1:
        # ensure axes is 2D
        axes = np.expand_dims(axes, axis=1)

    for idx,layer_idx in enumerate(sorted(all_metrics[first_epoch].keys())):
        data_first = all_metrics[first_epoch][layer_idx][metric_name]
        data_last  = all_metrics[last_epoch][layer_idx][metric_name]

        axes[0,idx].hist(data_first, bins=20, color='blue', alpha=0.7)
        axes[0,idx].set_title(f"{metric_name} Dist (Layer {layer_idx}) - Ep {first_epoch}")
        axes[0,idx].set_ylabel("Count")

        axes[1,idx].hist(data_last, bins=20, color='green', alpha=0.7)
        axes[1,idx].set_title(f"{metric_name} Dist (Layer {layer_idx}) - Ep {last_epoch}")
        axes[1,idx].set_ylabel("Count")

    plt.tight_layout()
    plt.show()

def compute_node_level_redundancy(redund_matrix):
    """
    average row i for node i => 1D array
    ignoring diagonal
    """
    N = redund_matrix.shape[0]
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        rowvals = np.delete(redund_matrix[i], i)
        out[i] = np.mean(rowvals) if len(rowvals)>0 else 0
    return out

def correlation_vs_steps(all_metrics, metric_pairs=[('RQ','MI'), ('RQ','redund'), ('MI','redund'), ('grad','dRQ')]):
    """
    For each pair of metrics specified in metric_pairs, compute and plot the Pearson correlation 
    across all neurons in each layer over epochs.
    
    Valid keys in each layer's metrics: 
      'RQ', 'MI', 'redundancy_mat', 'grad', and 'dRQ'.
    For 'redund', we compute node-level redundancy as the mean of each row of the redundancy matrix 
    (ignoring the diagonal).
    """
    epochs_list = sorted(all_metrics.keys())
    num_epochs = len(epochs_list)
    if num_epochs < 1:
        print("No data in all_metrics!")
        return

    layer_list = sorted(all_metrics[epochs_list[0]].keys())
    # Initialize dictionary for storing correlations.
    corr_results = {layer_idx: {} for layer_idx in layer_list}
    for layer_idx in layer_list:
        for pair in metric_pairs:
            corr_results[layer_idx][pair] = []

    for epoch in epochs_list:
        for layer_idx in layer_list:
            metrics = all_metrics[epoch][layer_idx]
            RQ_vals = metrics['RQ']
            MI_vals = metrics['MI']
            redund_vals = compute_node_level_redundancy(metrics['redundancy_mat'])
            grad_vals = metrics['grad']
            dRQ_vals = metrics['dRQ']
            for (m1, m2) in metric_pairs:
                if m1 == 'RQ':
                    arr1 = RQ_vals
                elif m1 == 'MI':
                    arr1 = MI_vals
                elif m1 == 'redund':
                    arr1 = redund_vals
                elif m1 == 'grad':
                    arr1 = grad_vals
                elif m1 == 'dRQ':
                    arr1 = dRQ_vals
                else:
                    arr1 = None

                if m2 == 'RQ':
                    arr2 = RQ_vals
                elif m2 == 'MI':
                    arr2 = MI_vals
                elif m2 == 'redund':
                    arr2 = redund_vals
                elif m2 == 'grad':
                    arr2 = grad_vals
                elif m2 == 'dRQ':
                    arr2 = dRQ_vals
                else:
                    arr2 = None

                if arr1 is None or arr2 is None or len(arr1) == 0 or len(arr2) == 0:
                    corr_ = 0
                else:
                    corr_, _ = pearsonr(arr1, arr2)
                corr_results[layer_idx][(m1, m2)].append(corr_)

    # Plot correlations vs. epochs for each metric pair and each layer.
    for pair in metric_pairs:
        fig, axes = plt.subplots(1, len(layer_list), figsize=(5*len(layer_list), 4), sharey=True)
        if len(layer_list) == 1:
            axes = [axes]
        for i, layer_idx in enumerate(layer_list):
            cvals = corr_results[layer_idx][pair]
            axes[i].plot(range(num_epochs), cvals, marker='o')
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel("Correlation")
            axes[i].set_ylim([-1, 1])
            axes[i].set_title(f"Layer {layer_idx}: {pair[0]} vs {pair[1]}")
            axes[i].axhline(0, color='k', ls='--', alpha=0.5)
        plt.suptitle(f"Correlation vs Epochs for {pair} across layers")
        plt.tight_layout()
        plt.show()        
        
def plot_hierarchical_clustering(all_metrics, epoch, layer_idx):
    """
    Hierarchical clustering of redundancy matrix (epoch, layer)
    Shows both the dendrogram and the clustered matrix
    """
    redund_mat = all_metrics[epoch][layer_idx]['redundancy_mat']
    dist = sdist.pdist(redund_mat, metric='euclidean')  # or 'cosine'
    Z = linkage(dist, method='ward')
    
    # Get the order of samples based on the hierarchical clustering
    dendro_idx = dendrogram(Z, no_plot=True)['leaves']
    
    # Reorder the redundancy matrix based on clustering
    redund_mat_clustered = redund_mat[dendro_idx, :][:, dendro_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot dendrogram
    dendrogram(Z, ax=ax1)
    ax1.set_title(f"Dendrogram (Layer {layer_idx}, Epoch {epoch})")
    ax1.set_xlabel("Node Index")
    ax1.set_ylabel("Distance")
    
    # Plot clustered matrix as heatmap
    im = ax2.imshow(redund_mat_clustered, cmap='viridis', aspect='auto')
    ax2.set_title(f"Clustered Redundancy Matrix (Layer {layer_idx}, Epoch {epoch})")
    ax2.set_xlabel("Node Index (Clustered)")
    ax2.set_ylabel("Node Index (Clustered)")
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label="Redundancy")
    
    plt.tight_layout()
    plt.show()
    
    
def plot_clustered_redundancy(all_metrics, layer_idx=0, epochs_to_plot=None):
    """
    Plot clustered redundancy matrix for specified epochs.
    
    Args:
        all_metrics: Dictionary of metrics across epochs
        layer_idx: Which layer's redundancy to visualize (default: 0)
        epochs_to_plot: List of epochs to plot (if None, uses epoch 1 and last epoch)
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import SpectralClustering
    import numpy as np
    
    # If epochs not specified, use epoch 1 and last epoch
    if epochs_to_plot is None:
        epochs = sorted(list(all_metrics.keys()))
        epochs_to_plot = [1, epochs[-1]]
    
    fig, axs = plt.subplots(1, len(epochs_to_plot), figsize=(15, 6))
    if len(epochs_to_plot) == 1:
        axs = [axs]
    
    for i, epoch in enumerate(epochs_to_plot):
        if epoch not in all_metrics:
            print(f"Epoch {epoch} not found in metrics. Skipping.")
            continue
        
        # Get redundancy matrix
        redund_matrix = all_metrics[epoch][layer_idx]['redundancy_mat']
        
        # Apply spectral clustering to get cluster assignments
        n_clusters = min(10, redund_matrix.shape[0] // 10)  # Reasonable number of clusters
        clustering = SpectralClustering(n_clusters=n_clusters, 
                                       affinity='precomputed',
                                       assign_labels='discretize',
                                       random_state=0)
        
        # Convert redundancy to affinity (higher redundancy = higher affinity)
        affinity = np.abs(redund_matrix)
        np.fill_diagonal(affinity, 0)  # Zero out diagonal for better clustering
        
        # Get cluster assignments
        cluster_labels = clustering.fit_predict(affinity)
        
        # Sort nodes by cluster
        idx = np.argsort(cluster_labels)
        sorted_matrix = redund_matrix[idx][:, idx]
        
        # Plot
        im = axs[i].imshow(sorted_matrix, cmap='viridis', vmin=-1, vmax=1)
        axs[i].set_title(f'Epoch {epoch} - Layer {layer_idx}')
        axs[i].set_xlabel('Neurons (clustered)')
        axs[i].set_ylabel('Neurons (clustered)')
    
    fig.colorbar(im, ax=axs, label='Redundancy')
    plt.tight_layout()
    plt.show()
    
    
def plot_gradient_metric_correlations(gradient_norms, all_metrics, metric_name='RQ'):
    """Plot correlations between gradient norms and other metrics."""
    num_epochs = len(gradient_norms)
    num_layers = len(gradient_norms[0]) if 0 in gradient_norms else 0
    
    fig, axes = plt.subplots(num_layers, 3, figsize=(18, 5*num_layers))
    if num_layers == 1:
        axes = np.array([axes])
    
    correlation_over_time = {layer_idx: [] for layer_idx in range(num_layers)}
    
    for epoch in range(num_epochs):
        for layer_idx in range(num_layers):
            if layer_idx in gradient_norms[epoch] and layer_idx in all_metrics[epoch]:
                grad_norms = gradient_norms[epoch][layer_idx]
                metric_vals = all_metrics[epoch][layer_idx][metric_name]
                
                if len(axes.shape) == 1:
                    ax = axes[0] if layer_idx == 0 else axes[layer_idx]
                else:
                    ax = axes[layer_idx, 0]
                
                # Plot latest epoch correlation
                if epoch == num_epochs - 1:
                    ax.scatter(grad_norms, metric_vals, alpha=0.7)
                    ax.set_xlabel('Average Gradient Norm')
                    ax.set_ylabel(f'{metric_name}')
                    ax.set_title(f'Layer {layer_idx}: Gradient vs {metric_name}')
                
                # Calculate correlation
                corr = np.corrcoef(grad_norms, metric_vals)[0, 1]
                correlation_over_time[layer_idx].append(corr)
    
    # Plot correlation over time
    for layer_idx in range(num_layers):
        if len(axes.shape) == 1:
            ax = axes[1] if layer_idx == 0 else axes[layer_idx+1]
        else:
            ax = axes[layer_idx, 1]
        
        ax.plot(range(num_epochs), correlation_over_time[layer_idx])
        ax.set_xlabel('Epoch')
        ax.set_ylabel(f'Correlation with {metric_name}')
        ax.set_title(f'Layer {layer_idx}: Correlation over Time')
    
    # Add heatmap showing correlations across all metrics at final epoch
    for layer_idx in range(num_layers):
        if len(axes.shape) == 1:
            ax = axes[2] if layer_idx == 0 else axes[layer_idx+2]
        else:
            ax = axes[layer_idx, 2]
        
        if layer_idx in gradient_norms[num_epochs-1]:
            grad_norms = gradient_norms[num_epochs-1][layer_idx]
            metrics_dict = all_metrics[num_epochs-1][layer_idx]
            
            corr_values = []
            metric_names = []
            
            for m_name, m_vals in metrics_dict.items():
                corr = np.corrcoef(grad_norms, m_vals)[0, 1]
                corr_values.append(corr)
                metric_names.append(m_name)
            
            # Create correlation bar chart
            ax.bar(metric_names, corr_values)
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Correlation with Gradient Norm')
            ax.set_title(f'Layer {layer_idx}: Gradient Correlations')
            plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_mean_metrics_over_epochs(all_metrics, gradient_norms, avg_gradients_epoch):
    """
    For each epoch and for each hidden layer, compute the mean (over nodes) for:
      - RQ, MI (from all_metrics)
      - dRQ computed from the average delta weights (avg_gradients_epoch) using the stored covariance
      - grad (average gradient norm from gradient_norms)
      - For redundancy, compute the average over all off-diagonal entries.
    
    Then, plot one curve per layer for each metric over epochs.
    """
    epochs = sorted(all_metrics.keys())
    num_layers = len(all_metrics[epochs[0]])
    
    # Initialize dictionaries to store mean values per epoch for each layer.
    mean_RQ = {layer: [] for layer in range(num_layers)}
    mean_MI = {layer: [] for layer in range(num_layers)}
    mean_redund = {layer: [] for layer in range(num_layers)}
    mean_grad = {layer: [] for layer in range(num_layers)}
    mean_dRQ = {layer: [] for layer in range(num_layers)}
    
    for epoch in epochs:
        for layer in range(num_layers):
            # Retrieve stored metrics.
            RQ_vals = all_metrics[epoch][layer]['RQ']
            MI_vals = all_metrics[epoch][layer]['MI']           
            dRQ_vals = all_metrics[epoch][layer]['dRQ']
            redund_mat = all_metrics[epoch][layer]['redundancy_mat']
            # For redundancy, average over off-diagonals.
            n = redund_mat.shape[0]
            avg_redund = (np.sum(redund_mat) - np.sum(np.diag(redund_mat))) / (n*(n-1)) if n > 1 else 0
            mean_RQ[layer].append(np.mean(RQ_vals))
            mean_MI[layer].append(np.mean(MI_vals))
            mean_redund[layer].append(avg_redund)
            
            # Gradient: average gradient norm (from gradient_norms)
            grad_vals = gradient_norms[epoch][layer]
            mean_grad[layer].append(np.mean(grad_vals))
            
            # dRQ: compute from avg_gradients_epoch and stored covariance.
            #cov_epoch = torch.from_numpy(all_metrics[epoch][layer]['cov']).to(device)
            #delta_W = torch.from_numpy(avg_gradients_epoch[epoch][layer]).to(device)
            #dRQ_vals = compute_per_node_dRQ(delta_W, cov_epoch)
            mean_dRQ[layer].append(np.mean(dRQ_vals))
    
    # Create a separate plot for each metric.
    metrics = {
        'RQ': mean_RQ,
        'MI': mean_MI,
        'dRQ': mean_dRQ,
        'grad': mean_grad,
        'redundancy': mean_redund
    }
    
    for metric_name, data_dict in metrics.items():
        plt.figure(figsize=(8,6))
        for layer in range(num_layers):
            plt.plot(epochs, data_dict[layer], marker='o', label=f'Layer {layer}')
        plt.xlabel('Epoch')
        plt.ylabel(f'Mean {metric_name}')
        plt.title(f'Mean {metric_name} Over Epochs (averaged over nodes)')
        plt.legend()
        plt.tight_layout()
        plt.show()


# In[7]:


################################################################################
# Cell 7: Pruning & Distillation (Updated)
################################################################################
def prune_nodes(model, metric_vals, fraction=0.2, mode="lowest", device="cuda"):
    """
    metric_vals: list of np arrays, one per layer, each shape = (num_nodes,),
                 specifying the "score" for each node in that layer.
    fraction: fraction of nodes to prune per layer.
    mode: "lowest", "highest", or "random"
    device: device to put the model on.
    
    Returns a new model with fewer out_features in each hidden layer.
    
    IMPORTANT: This version updates the input dimension for each pruned layer using the kept nodes
    from the previous layer.
    """
    new_layers = []
    layer_keep_indices = []  # List to store the kept output indices for each layer.
    
    # For the first layer, the input indices remain all (i.e. full input dimension).
    input_keep_indices = np.arange(model.layers[0].in_features)
    model = model.cpu()  
    for layer_idx, linear_layer in enumerate(model.layers):
        # Get the original weight matrix and select only the columns corresponding to the current input.
        old_w = linear_layer.weight.data.cpu().numpy()  # shape (orig_out, orig_in)
        old_w = old_w[:, input_keep_indices]  # effective weight: (orig_out, current_in)
        old_b = linear_layer.bias.data.cpu().numpy()   # shape (orig_out,)
        
        # Get the metric for this layer (vector of scores per node)
        score_arr = metric_vals[layer_idx]
        out_feats = score_arr.shape[0]
        
        # Determine number of nodes to prune, ensuring at least one node remains.
        max_prune = out_feats - 1
        num_prune = min(max_prune, int(np.round(fraction * out_feats)))
        if num_prune < 1:
            num_prune = 0

        # Determine which nodes to keep based on the chosen mode.
        if mode == "lowest":
            idx_sorted = np.argsort(score_arr)  # ascending order
            keep = idx_sorted[num_prune:]
        elif mode == "highest":
            idx_sorted = np.argsort(score_arr)
            keep = idx_sorted[:-num_prune]
        else:  # random
            all_idx = np.arange(out_feats)
            np.random.shuffle(all_idx)
            keep = all_idx[num_prune:]
        
        # Make sure at least one node is kept.
        if len(keep) == 0:
            print(f"Warning: Layer {layer_idx} would have 0 nodes. Keeping at least one node.")
            if mode == "lowest":
                keep = np.array([idx_sorted[num_prune - 1]])
            elif mode == "highest":
                keep = np.array([idx_sorted[0]])
            else:
                keep = np.array([all_idx[num_prune]])
        
        keep = np.sort(keep)
        layer_keep_indices.append(keep)
        
        # Prune the weights and biases according to the kept indices.
        W_new = old_w[keep, :]  # shape (new_out, current_in)
        b_new = old_b[keep]     # shape (new_out,)
        
        current_in = W_new.shape[1]
        new_out = W_new.shape[0]
        new_linear = nn.Linear(current_in, new_out)
        new_linear.weight.data = torch.from_numpy(W_new).to(device)
        new_linear.bias.data = torch.from_numpy(b_new).to(device)
        new_layers.append(new_linear)
        
        # Update the input indices for the next layer:
        # The output of the current layer (i.e., new_linear) will be the input for the next layer.
        input_keep_indices = keep.copy()  # These indices now become the effective "input" for the next layer.
    
    # For the final classification layer, adjust its input dimensions using the last pruned layer's kept indices.
    old_final_weight = model.final_layer.weight.data.cpu().numpy()  # shape (num_classes, orig_last_out)
    old_final_bias = model.final_layer.bias.data.cpu().numpy()        # shape (num_classes,)
    Wf_new = old_final_weight[:, input_keep_indices]  # shape (num_classes, new_last_out)
    bf_new = old_final_bias

    final_layer_new = nn.Linear(Wf_new.shape[1], Wf_new.shape[0])
    final_layer_new.weight.data = torch.from_numpy(Wf_new).to(device)
    final_layer_new.bias.data = torch.from_numpy(bf_new).to(device)

    # Build the new pruned MLP model.
    pruned_model = SimpleMLP(input_dim=model.layers[0].in_features, 
                             num_classes=model.final_layer.out_features,
                             hidden_dims=[])
    pruned_model.layers = nn.ModuleList(new_layers)
    pruned_model.final_layer = final_layer_new
    pruned_model = pruned_model.to(device)
    return pruned_model

def evaluate_accuracy(model, data_loader, device='cpu'):
    model.eval()
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _,preds = torch.max(outputs,1)
            correct += (preds==labels).sum().item()
            total+=labels.size(0)
    return (correct/total)*100

def distill_student(teacher, student, train_loader, device='cpu', alpha=0.5, T=1.0, epochs=1):
    """
    L = cross_entropy(student, label) + alpha * KL_div(student_logits, teacher_logits)
    teacher is fixed. We'll do 1 epoch for demonstration.
    """
    teacher.eval()
    student=student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=5e-4)
    ce_loss_func = nn.CrossEntropyLoss()

    for e in range(epochs):
        student.train()
        running_loss=0
        total_examples=0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                tlogits = teacher(images)/T
            slogits = student(images)/T

            ce_loss = ce_loss_func(slogits, labels)
            logp_s = F.log_softmax(slogits, dim=1)
            p_t = F.softmax(tlogits, dim=1)
            kl_div = F.kl_div(logp_s, p_t, reduction='batchmean')

            loss = ce_loss + alpha*kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()*images.size(0)
            total_examples+=images.size(0)

        epoch_loss = running_loss/ total_examples
        print(f"Distill Epoch {e+1}, Loss={epoch_loss:.4f}")

    return student


# In[10]:


def main_experiment():
    # 1) Data Loading
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=256)
    
    # 2) Create and train model with gradient tracking
    model = SimpleMLP(input_dim=3*32*32, hidden_dims=[1024,512,256], num_classes=10)
    model, gradient_norms, avg_gradients_epoch, all_metrics = train_with_gradient_tracking(
        model, train_loader, nn.CrossEntropyLoss(), 
        optim.Adam(model.parameters(), lr=0.001),
        device=device, num_epochs=50
    )

    model = model.to(device)
        
    # 3) Plot distributions of RQ and MI for each layer (first & last epochs)
    plot_metric_distribution(all_metrics, metric_name='RQ')
    plot_metric_distribution(all_metrics, metric_name='MI')
    #plot_metric_distribution(all_metrics, metric_name='redund')
    plot_metric_distribution(all_metrics, metric_name='dRQ')
    
    
    plot_mean_metrics_over_epochs(all_metrics, gradient_norms, avg_gradients_epoch)


    # 4) Plot correlation-vs-steps for various metric pairs, including the new 'grad' and 'dRQ'
    correlation_vs_steps(all_metrics, metric_pairs=[
        ('RQ', 'MI'), ('RQ', 'redund'), ('MI', 'redund'), ('dRQ', 'RQ'), ('dRQ', 'MI'), ('grad', 'dRQ'), ('grad', 'RQ'), ('grad', 'MI'), ('grad', 'redund')
    ])
    
    # 5) Hierarchical clustering on redundancy matrices:
    first_epoch = sorted(all_metrics.keys())[0]
    last_epoch = sorted(all_metrics.keys())[-1]
    # For example, visualize for layer 0 and layer 1 at first and last epoch:
    #plot_hierarchical_clustering(all_metrics, epoch=first_epoch, layer_idx=0)
    #plot_hierarchical_clustering(all_metrics, epoch=last_epoch,  layer_idx=0)
    plot_hierarchical_clustering(all_metrics, epoch=first_epoch, layer_idx=1)
    plot_hierarchical_clustering(all_metrics, epoch=last_epoch,  layer_idx=1)
    
    # 6) Pruning & Distillation analysis:
    # Make a copy of the trained model as teacher
    teacher_model = copy.deepcopy(model).to(device)
    base_acc = evaluate_accuracy(teacher_model, test_loader, device=device)
    print(f"\nTeacher (unpruned) accuracy = {base_acc:.2f}%\n")
    
    # Use RQ metric from the last epoch for pruning (you can change this to MI or dRQ as desired)
    def get_layer_metric_vals(all_metrics, epoch, metric_name='RQ'):
        layer_vals = []
        for layer_idx in sorted(all_metrics[epoch].keys()):
            layer_vals.append(all_metrics[epoch][layer_idx][metric_name])
        return layer_vals
    
    prune_metric_vals = get_layer_metric_vals(all_metrics, last_epoch, 'RQ')
    # Define a range of pruning fractions and modes
    fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    modes = ['highest']
    
    results_dict = {}
    for mode_ in modes:
        accs_before = []
        accs_after = []
        for frac in fractions:
            pruned_model = prune_nodes(model, prune_metric_vals, fraction=frac, mode=mode_, device=device)
            acc_before = evaluate_accuracy(pruned_model, test_loader, device=device)
            
            # Distill the pruned model using the teacher model (for one epoch)
            distilled = distill_student(teacher_model, pruned_model, train_loader, device=device, alpha=0.5, T=1.0, epochs=1)
            acc_after = evaluate_accuracy(distilled, test_loader, device=device)
            
            accs_before.append(acc_before)
            accs_after.append(acc_after)
        results_dict[mode_] = (accs_before, accs_after)
    
    # Plot the test accuracy vs pruning fraction (before and after distillation)
    plt.figure(figsize=(10,6))
    for mode_ in modes:
        acc_b, acc_a = results_dict[mode_]
        plt.plot(fractions, acc_b, '-o', label=f"{mode_} (before)")
        plt.plot(fractions, acc_a, '--o', label=f"{mode_} (after)")
    plt.axhline(base_acc, color='k', ls=':', label="Unpruned Baseline")
    plt.xlabel("Pruning Fraction")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Pruning Fraction (RQ-based) Before/After Distillation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    prune_metric_vals = get_layer_metric_vals(all_metrics, last_epoch, 'MI')
    # Define a range of pruning fractions and modes
    fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    modes = ['highest']
    
    results_dict = {}
    for mode_ in modes:
        accs_before = []
        accs_after = []
        for frac in fractions:
            pruned_model = prune_nodes(model, prune_metric_vals, fraction=frac, mode=mode_, device=device)
            acc_before = evaluate_accuracy(pruned_model, test_loader, device=device)
            
            # Distill the pruned model using the teacher model (for one epoch)
            distilled = distill_student(teacher_model, pruned_model, train_loader, device=device, alpha=0.5, T=1.0, epochs=1)
            acc_after = evaluate_accuracy(distilled, test_loader, device=device)
            
            accs_before.append(acc_before)
            accs_after.append(acc_after)
        results_dict[mode_] = (accs_before, accs_after)
    
    # Plot the test accuracy vs pruning fraction (before and after distillation)
    plt.figure(figsize=(10,6))
    for mode_ in modes:
        acc_b, acc_a = results_dict[mode_]
        plt.plot(fractions, acc_b, '-o', label=f"{mode_} (before)")
        plt.plot(fractions, acc_a, '--o', label=f"{mode_} (after)")
    plt.axhline(base_acc, color='k', ls=':', label="Unpruned Baseline")
    plt.xlabel("Pruning Fraction")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Pruning Fraction (RQ-based) Before/After Distillation")
    plt.legend()
    plt.tight_layout()
    plt.show()
    prune_metric_vals = get_layer_metric_vals(all_metrics, last_epoch, 'dRQ')
    # Define a range of pruning fractions and modes
    fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    modes = ['highest']
    
    results_dict = {}
    for mode_ in modes:
        accs_before = []
        accs_after = []
        for frac in fractions:
            pruned_model = prune_nodes(model, prune_metric_vals, fraction=frac, mode=mode_, device=device)
            acc_before = evaluate_accuracy(pruned_model, test_loader, device=device)
            
            # Distill the pruned model using the teacher model (for one epoch)
            distilled = distill_student(teacher_model, pruned_model, train_loader, device=device, alpha=0.5, T=1.0, epochs=1)
            acc_after = evaluate_accuracy(distilled, test_loader, device=device)
            
            accs_before.append(acc_before)
            accs_after.append(acc_after)
        results_dict[mode_] = (accs_before, accs_after)
    
    # Plot the test accuracy vs pruning fraction (before and after distillation)
    plt.figure(figsize=(10,6))
    for mode_ in modes:
        acc_b, acc_a = results_dict[mode_]
        plt.plot(fractions, acc_b, '-o', label=f"{mode_} (before)")
        plt.plot(fractions, acc_a, '--o', label=f"{mode_} (after)")
    plt.axhline(base_acc, color='k', ls=':', label="Unpruned Baseline")
    plt.xlabel("Pruning Fraction")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Pruning Fraction (RQ-based) Before/After Distillation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    prune_metric_vals = get_layer_metric_vals(all_metrics, last_epoch, 'redund')
    # Define a range of pruning fractions and modes
    fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    modes = ['highest']
    
    results_dict = {}
    for mode_ in modes:
        accs_before = []
        accs_after = []
        for frac in fractions:
            pruned_model = prune_nodes(model, prune_metric_vals, fraction=frac, mode=mode_, device=device)
            acc_before = evaluate_accuracy(pruned_model, test_loader, device=device)
            
            # Distill the pruned model using the teacher model (for one epoch)
            distilled = distill_student(teacher_model, pruned_model, train_loader, device=device, alpha=0.5, T=1.0, epochs=1)
            acc_after = evaluate_accuracy(distilled, test_loader, device=device)
            
            accs_before.append(acc_before)
            accs_after.append(acc_after)
        results_dict[mode_] = (accs_before, accs_after)
    
    # Plot the test accuracy vs pruning fraction (before and after distillation)
    plt.figure(figsize=(10,6))
    for mode_ in modes:
        acc_b, acc_a = results_dict[mode_]
        plt.plot(fractions, acc_b, '-o', label=f"{mode_} (before)")
        plt.plot(fractions, acc_a, '--o', label=f"{mode_} (after)")
    plt.axhline(base_acc, color='k', ls=':', label="Unpruned Baseline")
    plt.xlabel("Pruning Fraction")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Pruning Fraction (RQ-based) Before/After Distillation")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run the experiment
if __name__=="__main__":
    main_experiment()

