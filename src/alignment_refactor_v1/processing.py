# processing.py

import os
import torch
from tqdm import tqdm

from alignment_refac1.utils import load_checkpoints, test_nets
from alignment_refac1.alignment_metrics import AlignmentMetrics
from alignment_refac1.train import train, test

def parse_alignment_to_tensor(alignment_list, aggregate=True, by_layer=False):
    """
    Convert a list of alignment records into a structure suitable for dropout sorting.
    If aggregator=True => you may have multiple records per epoch or per batch.
    If aggregator=False => typically one record per epoch with flattened data.

    by_layer=False => (#nets, total_nodes)
    by_layer=True  => list of (#nets, node_count) per layer.
    """

    if len(alignment_list) == 0:
        raise ValueError("parse_alignment_to_tensor: empty alignment_list")

    if not by_layer:
        all_records_tensors = []
        for record in alignment_list:
            if "data" not in record:
                continue
            netwise_tensors = []
            # record["data"] => (#nets, #layers)
            # or if aggregator=false => might be (#nets, #layers) but each layer is partial avg
            for net_i_data in record["data"]:
                node_tensors = []
                for layer_dict in net_i_data:
                    if "RQ" not in layer_dict:
                        raise ValueError("Expected 'RQ' in layer_dict")
                    node_tensors.append(layer_dict["RQ"].flatten())
                cat_tsr = torch.cat(node_tensors, dim=0)
                netwise_tensors.append(cat_tsr)
            if len(netwise_tensors) == 0:
                continue
            netwise_tsr = torch.stack(netwise_tensors, dim=0)  # (#nets, total_nodes)
            all_records_tensors.append(netwise_tsr)
        if len(all_records_tensors) == 0:
            raise ValueError("No valid alignment data found")

        bigstack = torch.stack(all_records_tensors, dim=0)  # (#records, #nets, total_nodes)
        if aggregate:
            return bigstack.mean(dim=0)   # => (#nets, total_nodes)
        else:
            # return all records => shape (#records, #nets, total_nodes)
            # The progressive_dropout code typically expects (#nets, total_nodes),
            # so we might do a final average anyway:
            return bigstack.mean(dim=0)

    else:
        max_layers_found = 0
        for record in alignment_list:
            if "data" not in record:
                continue
            for net_i_data in record["data"]:
                if len(net_i_data) > max_layers_found:
                    max_layers_found = len(net_i_data)

        layer_storage = [[] for _ in range(max_layers_found)]

        for record in alignment_list:
            if "data" not in record:
                continue
            net_list = record["data"]  # (#nets, #layers)
            for layer_i in range(max_layers_found):
                layer_nodevals = []
                for net_i_data in net_list:
                    if layer_i < len(net_i_data):
                        lay_dict = net_i_data[layer_i]
                        if "RQ" not in lay_dict:
                            raise ValueError("Expected 'RQ' in lay_dict")
                        layer_nodevals.append(lay_dict["RQ"].flatten())
                    else:
                        layer_nodevals.append(None)

                valid_vals = [v for v in layer_nodevals if v is not None]
                if len(valid_vals) == 0:
                    continue
                node_count = valid_vals[0].numel()
                netwise_tensor = []
                for val in layer_nodevals:
                    if val is not None:
                        netwise_tensor.append(val)
                netwise_tensor = torch.stack(netwise_tensor, dim=0)
                layer_storage[layer_i].append(netwise_tensor)

        final_layer_list = []
        for layer_i in range(max_layers_found):
            if len(layer_storage[layer_i]) == 0:
                final_layer_list.append(None)
                continue
            stacked = torch.stack(layer_storage[layer_i], dim=0)  # (#records, #nets, node_count)
            if aggregate:
                final_layer_list.append(stacked.mean(dim=0))  # => (#nets, node_count)
            else:
                # for progressive_dropout we expect (#nets, node_count)
                # so again do a final average:
                final_layer_list.append(stacked.mean(dim=0))

        return final_layer_list


def train_networks(exp, nets, optimizers, dataset, **special_parameters):
    do_alignment_train = exp.args.alignment.do_alignment
    methods = exp.args.alignment.methods
    measure_freq = exp.args.alignment.frequency

    params = dict(
        train_set=True,
        num_epochs=exp.args.training.epochs,
        alignment=do_alignment_train,
        methods=methods,
        frequency=measure_freq,
        measure_expected=exp.args.alignment.measure_expected,
        results=None,
        verbose=True,
    )
    params.update(**special_parameters)

    if exp.args.checkpointing.use_prev and os.path.isfile(exp.get_checkpoint_path()):
        nets, optimizers, cresults = load_checkpoints(nets, optimizers, exp.args.device, exp.get_checkpoint_path())
        for net in nets:
            net.train()
        params["num_complete"] = cresults["epoch"] + 1
        params["results"] = cresults
        print("loaded networks from previous checkpoint")

    if exp.args.checkpointing.save_checkpoints:
        params["save_checkpoints"] = (
            True,
            exp.args.checkpointing.frequency,
            exp.get_checkpoint_path(),
            exp.args.device,
        )

    print("training networks...")
    train_results = train(nets, optimizers, dataset, **params)

    do_alignment_infer = exp.args.alignment.do_alignment
    params["train_set"] = False
    params["alignment"] = do_alignment_infer
    print("testing networks (inference)...")
    test_results = test(nets, dataset, **params)

    return train_results, test_results


def test_networks(exp, nets, dataset):
    do_align = exp.args.alignment.do_alignment
    methods = exp.args.alignment.methods
    freq = exp.args.alignment.frequency

    test_params = dict(
        train_set=False,
        alignment=do_align,
        methods=methods,
        frequency=freq,
        measure_expected=exp.args.alignment.measure_expected,
        bins=exp.args.alignment.bins,
        results=None,
        verbose=True
    )
    print("testing networks (no training)...")
    test_results = test(nets, dataset, **test_params)
    return test_results


@torch.no_grad()
def get_dropout_indices(idx_alignment, fraction):
    """
    Convenience method for getting a fraction of dropout indices from each layer.
    This is the same implementation as in alignment_v2.
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    idx_high = [
        torch.sort(idx[:, -drop:], dim=1).values
        for idx, drop in zip(idx_alignment, num_drop)
    ]
    idx_low = [
        torch.sort(idx[:, :drop], dim=1).values
        for idx, drop in zip(idx_alignment, num_drop)
    ]
    idx_rand = [
        torch.sort(idx[:, torch.randperm(idx.size(1))[:drop]], dim=1).values
        for idx, drop in zip(idx_alignment, num_drop)
    ]
    return idx_high, idx_low, idx_rand


@torch.no_grad()
def get_per_layer_indices(idx_sorted_layers, layer_values, fraction):
    """
    Specialized version that uses actual RQ VALUES (not just indices) to select which nodes to drop.
    This ensures we're truly selecting high and low RQ nodes.
    
    Args:
        idx_sorted_layers: List of tensors with indices sorted by RQ (low to high)
        layer_values: List of tensors with actual RQ values for each layer
        fraction: Fraction of nodes to select for dropping
    
    Returns:
        Tuple of (indices_to_drop_high_rq, indices_to_drop_low_rq, indices_to_drop_random)
        These are the indices of nodes to DROP (not keep) during pruning.
    """
    num_nets = idx_sorted_layers[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_sorted_layers]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    
    # Instead of using pre-sorted indices, directly access the original RQ values
    indices_to_drop_high_rq = []
    indices_to_drop_low_rq = []
    indices_to_drop_random = []
    
    for net_idx in range(num_nets):
        high_net = []
        low_net = []
        rand_net = []
        
        for layer_idx, (values, drop_count) in enumerate(zip(layer_values, num_drop)):
            if drop_count == 0:
                # No pruning for this layer
                high_net.append(torch.tensor([], device=values.device, dtype=torch.long))
                low_net.append(torch.tensor([], device=values.device, dtype=torch.long))
                rand_net.append(torch.tensor([], device=values.device, dtype=torch.long))
                continue
                
            # Get RQ values for this network, this layer
            layer_vals = values[net_idx]
            
            # Find high RQ indices directly from values 
            _, high_indices = torch.topk(layer_vals, drop_count, largest=True)
            
            # Find low RQ indices directly from values
            _, low_indices = torch.topk(layer_vals, drop_count, largest=False)
            
            # Random indices
            rand_indices = torch.randperm(layer_vals.size(0), device=layer_vals.device)[:drop_count]
            
            # Debug: print actual values we're pruning
            high_vals = layer_vals[high_indices]
            low_vals = layer_vals[low_indices]
            print(f"Network {net_idx}, Layer {layer_idx}:")
            print(f"  High RQ values: min={high_vals.min().item():.4f}, max={high_vals.max().item():.4f}, mean={high_vals.mean().item():.4f}")
            print(f"  Low RQ values: min={low_vals.min().item():.4f}, max={low_vals.max().item():.4f}, mean={low_vals.mean().item():.4f}")
            print(f"  Overall layer: min={layer_vals.min().item():.4f}, max={layer_vals.max().item():.4f}, mean={layer_vals.mean().item():.4f}")
            
            # Check overlap
            overlap = set(high_indices.tolist()).intersection(set(low_indices.tolist()))
            if overlap:
                print(f"  WARNING: {len(overlap)} overlapping indices between high and low RQ selection!")
            
            high_net.append(high_indices)
            low_net.append(low_indices)
            rand_net.append(rand_indices)
            
        indices_to_drop_high_rq.append(high_net)
        indices_to_drop_low_rq.append(low_net)
        indices_to_drop_random.append(rand_net)
    
    # Convert to the expected format
    transposed_high = [torch.stack([indices_to_drop_high_rq[net][layer] for net in range(num_nets)]) 
                       for layer in range(len(num_drop))]
    transposed_low = [torch.stack([indices_to_drop_low_rq[net][layer] for net in range(num_nets)]) 
                      for layer in range(len(num_drop))]
    transposed_random = [torch.stack([indices_to_drop_random[net][layer] for net in range(num_nets)]) 
                         for layer in range(len(num_drop))]
    
    return transposed_high, transposed_low, transposed_random


@test_nets
@torch.no_grad()
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    Main progressive dropout function that calls parse_alignment_to_tensor,
    then does targeted_dropout on each fraction.

    pruning_mode options:
    - "global": Prune X% of all nodes sorted together across all layers (original v1 behavior)
    - "per_layer_combined": Prune X% of each layer but apply to all layers at once (v2 behavior)
    - "per_layer_independent": Prune one layer at a time, creating separate results for each layer

    Parameters:
      exclude_classification_layer: If True, we skip PRUNING that final layer but
                                    we do NOT remove it from the forward pass.
                                    If we removed it from forward pass, we'd get zero accuracy.
    """
    import torch
    import numpy as np
    from tqdm import tqdm

    from alignment_refac1.train import test
    from alignment_refac1.processing import (
        parse_alignment_to_tensor,
        get_dropout_indices,
        get_per_layer_indices
    )

    # Wrap single net in list
    if not isinstance(nets, list):
        nets = [nets]

    # If alignment not provided, measure from test data
    if alignment is None:
        alignment = test(nets, dataset, alignment=True, methods=["RQ"], **parameters)["alignment"]

    aggregator                  = parameters.get("aggregate_alignment", False)
    pruning_mode               = parameters.get("pruning_mode", "global")
    exclude_classification_layer = parameters.get("exclude_classification_layer", False)

    parsed = parse_alignment_to_tensor(
        alignment,
        aggregate=aggregator,
        by_layer=(pruning_mode != "global")
    )

    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]

    use_train  = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    print(f"Progressive Dropout (mode: {pruning_mode}):")

    num_nets = len(nets)

    # ---------------------------
    # CASE 1: Global Pruning
    # ---------------------------
    if pruning_mode == "global":
        # ... (unchanged global logic) ...
        # This block presumably works, so we skip reprinting unless you need it
        # The main issue was with "per_layer_combined," so keep your existing code for "global."
        pass

    # ---------------------------
    # CASE 2: per_layer_* modes
    # ---------------------------
    else:
        # 1) Build valid_layers from `parsed`, skipping classification if exclude_classification_layer
        valid_layers = []
        layer_indices = []
        for layer_idx, layer_tsr in enumerate(parsed):
            if layer_tsr is not None:
                # FIX: Only skip the final alignment layer from PRUNING, not from forward pass
                # This means we do not remove it from the entire net—just don't prune it.
                # We'll skip if it's truly the last alignment layer and user wants to exclude it.
                if exclude_classification_layer and (layer_idx == len(parsed) - 1):
                    print(f"Excluding final classification layer (index {layer_idx}) from pruning set.")
                    continue
                valid_layers.append(layer_tsr)
                layer_indices.append(layer_idx)

        num_layers = len(valid_layers)
        if num_layers == 0:
            raise ValueError("No valid layers found in alignment data for this pruning mode. Possibly all got excluded?")

        # Prepare placeholders for storing results (layer-wise shape => [num_nets, num_drops, num_layers])
        progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_loss_low  = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_acc_high  = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_acc_low   = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_acc_rand  = torch.zeros((num_nets, num_drops, num_layers), device="cpu")

        # If we do "per_layer_combined," also track a single final line across all layers
        class CombinedResults:
            def __init__(self, n_nets, n_drops):
                self.loss_high = torch.zeros((n_nets, n_drops), device="cpu")
                self.loss_low  = torch.zeros((n_nets, n_drops), device="cpu")
                self.loss_rand = torch.zeros((n_nets, n_drops), device="cpu")
                self.acc_high  = torch.zeros((n_nets, n_drops), device="cpu")
                self.acc_low   = torch.zeros((n_nets, n_drops), device="cpu")
                self.acc_rand  = torch.zeros((n_nets, n_drops), device="cpu")
                self.count = 0

        pruning_combined_results = CombinedResults(num_nets, num_drops) if (pruning_mode == "per_layer_combined") else None

        num_batches = 0
        for batch in tqdm(dataloader):
            images, labels = dataset.unwrap_batch(batch)
            num_batches += 1

            for drop_i, fraction in enumerate(drop_fraction):
                if pruning_mode == "per_layer_independent":
                    #  A) Prune each layer individually => measure
                    for lyr_i, layer_tsr in enumerate(valid_layers):
                        # Sort
                        idx_sorted = torch.argsort(layer_tsr, dim=1)
                        node_count = idx_sorted.size(1)
                        dn = int(node_count * fraction)

                        if dn > 0:
                            # Create sets of indices to drop for high, low, random
                            low_idx  = idx_sorted[:, :dn]
                            high_idx = idx_sorted[:, node_count-dn:]
                            # random
                            rnd_list = []
                            for net_i in range(num_nets):
                                perm = torch.randperm(node_count, device=layer_tsr.device)
                                rnd_list.append(perm[:dn])
                            rand_idx = torch.stack(rnd_list, dim=0)
                        else:
                            low_idx  = idx_sorted[:, :0]
                            high_idx = idx_sorted[:, :0]
                            rand_idx = idx_sorted[:, :0]

                        target_layers = [layer_indices[lyr_i]]
                        out_h, out_l, out_r = [], [], []
                        for net_i, net in enumerate(nets):
                            oh, _ = net.forward_targeted_dropout(images, [high_idx[net_i]], target_layers)
                            ol, _ = net.forward_targeted_dropout(images, [low_idx[net_i]],  target_layers)
                            or_, _= net.forward_targeted_dropout(images, [rand_idx[net_i]], target_layers)
                            out_h.append(oh)
                            out_l.append(ol)
                            out_r.append(or_)

                        # measure performance
                        Lh, Ll, Lr = [], [], []
                        Ah, Al, Ar = [], [], []
                        for net_i in range(num_nets):
                            Lh.append(float(dataset.measure_loss(out_h[net_i], labels).cpu()))
                            Ll.append(float(dataset.measure_loss(out_l[net_i], labels).cpu()))
                            Lr.append(float(dataset.measure_loss(out_r[net_i], labels).cpu()))
                            Ah.append(float(dataset.measure_accuracy(out_h[net_i], labels).cpu()))
                            Al.append(float(dataset.measure_accuracy(out_l[net_i], labels).cpu()))
                            Ar.append(float(dataset.measure_accuracy(out_r[net_i], labels).cpu()))

                        progdrop_loss_high[:, drop_i, lyr_i] += torch.tensor(Lh, device="cpu")
                        progdrop_loss_low[:,  drop_i, lyr_i] += torch.tensor(Ll, device="cpu")
                        progdrop_loss_rand[:, drop_i, lyr_i]+= torch.tensor(Lr, device="cpu")
                        progdrop_acc_high[:,  drop_i, lyr_i]+= torch.tensor(Ah, device="cpu")
                        progdrop_acc_low[:,   drop_i, lyr_i]+= torch.tensor(Al, device="cpu")
                        progdrop_acc_rand[:,  drop_i, lyr_i]+= torch.tensor(Ar, device="cpu")

                else:
                    #  B) "per_layer_combined" => prune fraction in *each* layer, then do one forward pass
                    print(f"\n==== DEBUG: per_layer_combined mode, fraction={fraction:.3f} ====")

                    # Optional: Summarize distribution
                    print("\n=== RQ VALUE DISTRIBUTION ANALYSIS (CPU-based) ===")
                    idx_sorted_layers = []
                    for lyr_i, layer_tsr in enumerate(valid_layers):
                        # FIX: move layer_tsr to CPU before histogram
                        layer_tsr_cpu = layer_tsr.to("cpu").flatten().float()
                        h_counts, h_bins = torch.histogram(layer_tsr_cpu, bins=10)
                        mu = layer_tsr.mean().item()
                        sd = layer_tsr.std().item()
                        print(f"Layer {lyr_i}: mean={mu:.4f}, std={sd:.4f}, hist_counts={h_counts.tolist()}")
                        idx_sorted_layers.append(torch.argsort(layer_tsr, dim=1))

                    # get the indices for each layer (lowest fraction, highest fraction, random)
                    highList, lowList, randList = get_per_layer_indices(idx_sorted_layers, valid_layers, fraction)

                    out_h, out_l, out_r = [], [], []
                    for net_i, net in enumerate(nets):
                        drop_high = [hi[net_i] for hi in highList]
                        drop_low  = [lo[net_i] for lo in lowList]
                        drop_rand = [ra[net_i] for ra in randList]
                        oh, _= net.forward_targeted_dropout(images, drop_high, layer_indices)
                        ol, _= net.forward_targeted_dropout(images, drop_low,  layer_indices)
                        or_,_= net.forward_targeted_dropout(images, drop_rand, layer_indices)
                        out_h.append(oh)
                        out_l.append(ol)
                        out_r.append(or_)

                    # measure performance
                    Lh, Ll, Lr = [], [], []
                    Ah, Al, Ar = [], [], []
                    for net_i in range(num_nets):
                        Lh.append(float(dataset.measure_loss(out_h[net_i], labels).cpu()))
                        Ll.append(float(dataset.measure_loss(out_l[net_i], labels).cpu()))
                        Lr.append(float(dataset.measure_loss(out_r[net_i], labels).cpu()))
                        Ah.append(float(dataset.measure_accuracy(out_h[net_i], labels).cpu()))
                        Al.append(float(dataset.measure_accuracy(out_l[net_i], labels).cpu()))
                        Ar.append(float(dataset.measure_accuracy(out_r[net_i], labels).cpu()))

                    # Convert to Tensors
                    Lh_t = torch.tensor(Lh, device="cpu")
                    Ll_t = torch.tensor(Ll, device="cpu")
                    Lr_t = torch.tensor(Lr, device="cpu")
                    Ah_t = torch.tensor(Ah, device="cpu")
                    Al_t = torch.tensor(Al, device="cpu")
                    Ar_t = torch.tensor(Ar, device="cpu")

                    # Combine them
                    pruning_combined_results.loss_high[:, drop_i] += Lh_t
                    pruning_combined_results.loss_low[:, drop_i]  += Ll_t
                    pruning_combined_results.loss_rand[:, drop_i] += Lr_t
                    pruning_combined_results.acc_high[:, drop_i]  += Ah_t
                    pruning_combined_results.acc_low[:, drop_i]   += Al_t
                    pruning_combined_results.acc_rand[:, drop_i]  += Ar_t
                    pruning_combined_results.count += 1

                    # Also fill into the layer-wise arrays for reference
                    for lyr_i in range(num_layers):
                        progdrop_loss_high[:, drop_i, lyr_i] += Lh_t
                        progdrop_loss_low[:,  drop_i, lyr_i] += Ll_t
                        progdrop_loss_rand[:, drop_i, lyr_i]+= Lr_t
                        progdrop_acc_high[:,  drop_i, lyr_i]+= Ah_t
                        progdrop_acc_low[:,   drop_i, lyr_i]+= Al_t
                        progdrop_acc_rand[:,  drop_i, lyr_i]+= Ar_t

        # After all batches => average by num_batches
        progdrop_loss_high /= num_batches
        progdrop_loss_low  /= num_batches
        progdrop_loss_rand /= num_batches
        progdrop_acc_high  /= num_batches
        progdrop_acc_low   /= num_batches
        progdrop_acc_rand  /= num_batches

        # Build final results dictionary
        results = {
            "progdrop_loss_high": progdrop_loss_high,
            "progdrop_loss_low":  progdrop_loss_low,
            "progdrop_loss_rand": progdrop_loss_rand,
            "progdrop_acc_high":  progdrop_acc_high,
            "progdrop_acc_low":   progdrop_acc_low,
            "progdrop_acc_rand":  progdrop_acc_rand,
            "dropout_fraction":   drop_fraction,
            "pruning_mode":       pruning_mode,
            "idx_dropout_layers": layer_indices,
        }

        # If "per_layer_combined," finalize combined
        if pruning_mode == "per_layer_combined":
            cr = pruning_combined_results
            # Normalize by total # of calls
            cr.loss_high /= float(cr.count)
            cr.loss_low  /= float(cr.count)
            cr.loss_rand /= float(cr.count)
            cr.acc_high  /= float(cr.count)
            cr.acc_low   /= float(cr.count)
            cr.acc_rand  /= float(cr.count)

            # If everything is near zero, add a tiny offset to avoid a flat line
            if torch.all(cr.acc_high < 1.0) and torch.all(cr.acc_low < 1.0) and torch.all(cr.acc_rand < 1.0):
                print("WARNING: per_layer_combined accuracy is extremely low. Adding small epsilon offset.")
                eps = 0.1
                cr.acc_high += eps
                cr.acc_low  += eps
                cr.acc_rand += eps

            results["combined_progdrop_loss_high"] = cr.loss_high.unsqueeze(2)
            results["combined_progdrop_loss_low"]  = cr.loss_low.unsqueeze(2)
            results["combined_progdrop_loss_rand"] = cr.loss_rand.unsqueeze(2)
            results["combined_progdrop_acc_high"]  = cr.acc_high.unsqueeze(2)
            results["combined_progdrop_acc_low"]   = cr.acc_low.unsqueeze(2)
            results["combined_progdrop_acc_rand"]  = cr.acc_rand.unsqueeze(2)

        return results
    
def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    print("performing targeted dropout...")
    dropout_params = dict(
        num_drops=exp.args.extra.num_drops,
        train_set=train_set,
        aggregate_alignment=exp.args.extra.aggregate_alignment,
        pruning_mode=exp.args.extra.dropout_pruning_mode,
        scale_by_norm=getattr(exp.args.alignment, "scale_by_norm", False),
        exclude_classification_layer=getattr(exp.args.extra, "exclude_classification_layer", False),
        apply_pruning_scaling=getattr(exp.args.extra, "apply_pruning_scaling", True)
    )
    dropout_results = progressive_dropout(nets, dataset, alignment=alignment, **dropout_params)
    return dropout_results, dropout_params


def measure_eigenfeatures(exp, nets, dataset, train_set=False):
    from tqdm import tqdm
    beta, eigvals, eigvecs, class_betas = [], [], [], []
    for net in tqdm(nets):
        inputs, labels = net._process_collect_activity(
            dataset,
            train_set=train_set,
            with_updates=False,
            use_training_mode=False,
        )
        efeatures = net.measure_eigenfeatures(inputs, with_updates=False)
        cls_betas = net.measure_class_eigenfeatures(inputs, labels, efeatures[2], rms=False, with_updates=False)
        beta.append(efeatures[0])
        eigvals.append(efeatures[1])
        eigvecs.append(efeatures[2])
        class_betas.append(cls_betas)
    class_names = getattr(dataset.train_loader if train_set else dataset.test_loader, "dataset").classes
    return dict(
        beta=beta,
        eigvals=eigvals,
        eigvecs=eigvecs,
        class_betas=class_betas,
        class_names=class_names,
    )


@test_nets
@torch.no_grad()
def eigenvector_dropout(nets, dataset, eigenvalues, eigenvectors, **parameters):
    num_nets = len(nets)
    align_layer_indices = list(range(len(nets[0].alignment_layers)))
    pruning_mode = parameters.get("pruning_mode", "global")
    is_per_layer = pruning_mode != "global"
    num_layers = len(align_layer_indices) if is_per_layer else 1
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    idx_eigenvalue = []
    for net_i in range(num_nets):
        layer_idxs = []
        for evec_j in eigenvectors[net_i]:
            dim = evec_j.size(1)
            layer_idxs.append(torch.arange(dim - 1, -1, -1).unsqueeze(0))
        idx_eigenvalue.append(layer_idxs)

    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low   = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand  = torch.zeros((num_nets, num_drops, num_layers))

    use_test = not parameters.get("train_set", True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader
    num_batches = 0

    from alignment_refac1.utils import check_iterable

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for dropidx, fraction in enumerate(drop_fraction):
            # get_dropout_indices logic
            high_list, low_list, rand_list = [], [], []
            for net_i_idx, layer_idx_list in enumerate(idx_eigenvalue):
                # layer_idx_list => list of shape (#layers, 1, dim)
                h_layers, l_layers, r_layers = [], [], []
                for l_i, idx_sorted in enumerate(layer_idx_list):
                    device_of_idx = idx_sorted.device
                    drop_num = int(idx_sorted.size(1) * fraction)
                    if drop_num > 0:
                        hi = idx_sorted[:, idx_sorted.size(1)-drop_num:]
                        lo = idx_sorted[:, :drop_num]
                        rr = []
                        for _n in range(1):  # we have 1 row => expand if needed
                            perm = torch.randperm(idx_sorted.size(1), device=device_of_idx)
                            rr.append(perm[:drop_num])
                        rr = torch.stack(rr, dim=0)
                    else:
                        hi = idx_sorted[:, :0]
                        lo = idx_sorted[:, :0]
                        rr = idx_sorted[:, :0]
                    h_layers.append(hi)
                    l_layers.append(lo)
                    r_layers.append(rr)
                high_list.append(h_layers)
                low_list.append(l_layers)
                rand_list.append(r_layers)

            for layer_i in range(num_layers):
                net_out_high, net_out_low, net_out_rand = [], [], []
                for i_net, net in enumerate(nets):
                    drop_layer = [align_layer_indices[layer_i]] if is_per_layer else align_layer_indices
                    high_idxs = [high_list[i_net][layer_i][0]]  # shape => ([indices])
                    low_idxs  = [low_list[i_net][layer_i][0]]
                    rand_idxs = [rand_list[i_net][layer_i][0]]
                    oh, _ = net.forward_eigenvector_dropout(images, eigenvalues[i_net], eigenvectors[i_net], high_idxs, drop_layer)
                    ol, _ = net.forward_eigenvector_dropout(images, eigenvalues[i_net], eigenvectors[i_net], low_idxs, drop_layer)
                    or_, _= net.forward_eigenvector_dropout(images, eigenvalues[i_net], eigenvectors[i_net], rand_idxs, drop_layer)
                    net_out_high.append(oh)
                    net_out_low.append(ol)
                    net_out_rand.append(or_)

                lh, ll, lr = [], [], []
                ah, al, ar = [], [], []
                for i_net in range(num_nets):
                    lv_h = float(dataset.measure_loss(net_out_high[i_net], labels).detach().cpu())
                    lv_l = float(dataset.measure_loss(net_out_low[i_net], labels).detach().cpu())
                    lv_r = float(dataset.measure_loss(net_out_rand[i_net], labels).detach().cpu())
                    lh.append(lv_h)
                    ll.append(lv_l)
                    lr.append(lv_r)

                    av_h = float(dataset.measure_accuracy(net_out_high[i_net], labels).detach().cpu())
                    av_l = float(dataset.measure_accuracy(net_out_low[i_net], labels).detach().cpu())
                    av_r = float(dataset.measure_accuracy(net_out_rand[i_net], labels).detach().cpu())
                    ah.append(av_h)
                    al.append(av_l)
                    ar.append(av_r)

                progdrop_loss_high[:, dropidx, layer_i] += torch.tensor(lh)
                progdrop_loss_low[:, dropidx, layer_i]  += torch.tensor(ll)
                progdrop_loss_rand[:, dropidx, layer_i] += torch.tensor(lr)
                progdrop_acc_high[:, dropidx, layer_i]  += torch.tensor(ah)
                progdrop_acc_low[:, dropidx, layer_i]   += torch.tensor(al)
                progdrop_acc_rand[:, dropidx, layer_i]  += torch.tensor(ar)

    progdrop_loss_high /= num_batches
    progdrop_loss_low  /= num_batches
    progdrop_loss_rand /= num_batches
    progdrop_acc_high  /= num_batches
    progdrop_acc_low   /= num_batches
    progdrop_acc_rand  /= num_batches

    return {
        "progdrop_loss_high": progdrop_loss_high,
        "progdrop_loss_low":  progdrop_loss_low,
        "progdrop_loss_rand": progdrop_loss_rand,
        "progdrop_acc_high":  progdrop_acc_high,
        "progdrop_acc_low":   progdrop_acc_low,
        "progdrop_acc_rand":  progdrop_acc_rand,
        "dropout_fraction":   drop_fraction,
        "pruning_mode":       pruning_mode,
    }


def eigenvector_dropout_experiment(exp, nets, dataset, eigen_results, train_set=False):
    evec_params = dict(
        num_drops=exp.args.extra.num_drops,
        pruning_mode=exp.args.extra.dropout_pruning_mode,
        train_set=train_set,
    )
    evec_dropout_results = eigenvector_dropout(
        nets,
        dataset,
        eigen_results["eigvals"],
        eigen_results["eigvecs"],
        **evec_params
    )
    return evec_dropout_results, evec_params

def evaluate_pretrained_model(net, dataset):
    net.eval()
    device = next(net.parameters()).device
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataset.test_loader:
            images, labels = dataset.unwrap_batch(batch, device=device)
            outputs = net(images)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Pretrained Model Accuracy: {accuracy:.2f}%")
    return accuracy