def prune_nodes(model, metric_vals, fraction=0.2, mode="lowest", device="cuda"):
    """
    metric_vals: list of np arrays, one per layer, each shape = (num_nodes,),
                 specifying the "score" for each node in that layer. 
    fraction: fraction of nodes to prune
    mode: "lowest", "highest", "random"
    device: device to put the model on
    Returns a new model with fewer out_features in each hidden layer.
    """
    new_model = SimpleMLP()  # just to replicate structure, but we will override
    new_layers = []
    layer_indices_map = []

    for layer_idx, linear_layer in enumerate(model.layers):
        score_arr = metric_vals[layer_idx]
        out_feats = score_arr.shape[0]
        
        # Ensure we keep at least one node per layer
        max_prune = out_feats - 1
        num_prune = min(max_prune, int(np.round(fraction*out_feats)))
        if num_prune < 1:
            num_prune = 0

        if mode=="lowest":
            idx_sorted = np.argsort(score_arr)  # ascending
            pruned = idx_sorted[:num_prune]
            keep = idx_sorted[num_prune:]
        elif mode=="highest":
            idx_sorted = np.argsort(score_arr)
            pruned = idx_sorted[-num_prune:]
            keep = idx_sorted[:-num_prune]
        else: # random
            all_idx = np.arange(out_feats)
            np.random.shuffle(all_idx)
            pruned = all_idx[:num_prune]
            keep = all_idx[num_prune:]
        
        # Double check we're keeping at least one node
        if len(keep) == 0:
            print(f"Warning: Layer {layer_idx} would have 0 nodes. Keeping at least one node.")
            if mode == "lowest":
                keep = np.array([idx_sorted[num_prune-1]])
            elif mode == "highest":
                keep = np.array([idx_sorted[0]])
            else:
                keep = np.array([all_idx[num_prune]])
        
        keep = np.sort(keep)
        old_w = linear_layer.weight.data.cpu().numpy()  # shape (out_feats, in_feats)
        old_b = linear_layer.bias.data.cpu().numpy()

        W_new = old_w[keep,:]
        b_new = old_b[keep]

        out_feats_new = len(keep)
        in_feats_old = W_new.shape[1]
        new_linear = nn.Linear(in_feats_old, out_feats_new)
        # Move to correct device
        new_linear.weight.data = torch.from_numpy(W_new).to(device)
        new_linear.bias.data = torch.from_numpy(b_new).to(device)
        new_layers.append(new_linear)
        layer_indices_map.append(keep)

    # final classification
    old_final_weight = model.final_layer.weight.data.cpu().numpy() # shape(num_classes, last_out)
    old_final_bias = model.final_layer.bias.data.cpu().numpy()   # shape(num_classes,)
    last_keep = layer_indices_map[-1]
    Wf_new = old_final_weight[:, last_keep]  # shape(num_classes, len(keep))
    bf_new = old_final_bias

    final_layer_new = nn.Linear(Wf_new.shape[1], Wf_new.shape[0])
    # Move to correct device
    final_layer_new.weight.data = torch.from_numpy(Wf_new).to(device)
    final_layer_new.bias.data = torch.from_numpy(bf_new).to(device)

    # Build new MLP
    pruned_model = SimpleMLP(input_dim=model.layers[0].in_features, 
                             num_classes=model.final_layer.out_features,
                             hidden_dims=[])
    pruned_model.layers = nn.ModuleList(new_layers)
    pruned_model.final_layer = final_layer_new
    # Move the entire model to the device
    pruned_model = pruned_model.to(device)
    return pruned_model
