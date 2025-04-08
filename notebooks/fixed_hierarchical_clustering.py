def plot_hierarchical_clustering(all_metrics, epoch, layer_idx):
    """
    Hierarchical clustering of redundancy matrix (epoch, layer)
    Shows both the dendrogram and the clustered matrix
    """
    redund_mat = all_metrics[epoch][layer_idx][\"redundancy_mat\"]
    dist = sdist.pdist(redund_mat, metric=\"euclidean\")  # or \"cosine\"
    Z = linkage(dist, method=\"ward\")
    
    # Get the order of samples based on the hierarchical clustering
    dendro_idx = dendrogram(Z, no_plot=True)[\"leaves\"]
    
    # Reorder the redundancy matrix based on clustering
    redund_mat_clustered = redund_mat[dendro_idx, :][:, dendro_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot dendrogram
    dendrogram(Z, ax=ax1)
    ax1.set_title(f\"Dendrogram (Layer {layer_idx}, Epoch {epoch})\")
    ax1.set_xlabel(\"Node Index\")
    ax1.set_ylabel(\"Distance\")
    
    # Plot clustered matrix as heatmap
    im = ax2.imshow(redund_mat_clustered, cmap=\"viridis\", aspect=\"auto\")
    ax2.set_title(f\"Clustered Redundancy Matrix (Layer {layer_idx}, Epoch {epoch})\")
    ax2.set_xlabel(\"Node Index (Clustered)\")
    ax2.set_ylabel(\"Node Index (Clustered)\")
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label=\"Redundancy\")
    
    plt.tight_layout()
    plt.show()
