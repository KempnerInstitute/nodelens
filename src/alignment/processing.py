# processing.py

import os
import torch
from tqdm import tqdm

from alignment.utils import load_checkpoints, test_nets
from alignment.alignment_metrics import AlignmentMetrics
from alignment.train import train, test

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
    """
    if not isinstance(nets, list):
        nets = [nets]

    if alignment is None:
        alignment = test(nets, dataset, alignment=True, methods=["RQ"], **parameters)["alignment"]

    aggregator = parameters.get("aggregate_alignment", False)
    pruning_mode = parameters.get("pruning_mode", "global")

    parsed = parse_alignment_to_tensor(alignment, aggregate=aggregator, by_layer=(pruning_mode != "global"))

    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    print(f"Progressive Dropout (mode: {pruning_mode}):")

    num_nets = len(nets)

    if pruning_mode == "global":
        # Original v1 behavior - prune X% of all nodes together
        if not isinstance(parsed, torch.Tensor) or parsed.dim() != 2:
            raise ValueError("Expected shape (#nets, total_nodes) for pruning_mode='global'")

        idx_sorted = torch.argsort(parsed, dim=1)  # (#nets, total_nodes)
        total_nodes = idx_sorted.size(1)

        progdrop_loss_high = torch.zeros((num_nets, num_drops, 1), device="cpu")
        progdrop_loss_low  = torch.zeros((num_nets, num_drops, 1), device="cpu")
        progdrop_loss_rand = torch.zeros((num_nets, num_drops, 1), device="cpu")
        progdrop_acc_high  = torch.zeros((num_nets, num_drops, 1), device="cpu")
        progdrop_acc_low   = torch.zeros((num_nets, num_drops, 1), device="cpu")
        progdrop_acc_rand  = torch.zeros((num_nets, num_drops, 1), device="cpu")

        num_batches = 0
        for batch in tqdm(dataloader):
            images, labels = dataset.unwrap_batch(batch)
            num_batches += 1

            for dropidx, fraction in enumerate(drop_fraction):
                dn = int(total_nodes * fraction)
                if dn > 0:
                    hi = idx_sorted[:, total_nodes - dn : total_nodes]
                    lo = idx_sorted[:, :dn]
                    rr = []
                    for i_net in range(num_nets):
                        perm = torch.randperm(total_nodes, device=idx_sorted.device)
                        rr.append(perm[:dn])
                    rr = torch.stack(rr, dim=0)
                else:
                    hi = idx_sorted[:, :0]
                    lo = idx_sorted[:, :0]
                    rr = idx_sorted[:, :0]

                out_high, out_low, out_rand = [], [], []
                for i_net, net in enumerate(nets):
                    oh, _ = net.forward_targeted_dropout(images, [hi[i_net]], [0])
                    ol, _ = net.forward_targeted_dropout(images, [lo[i_net]], [0])
                    or_, _= net.forward_targeted_dropout(images, [rr[i_net]], [0])
                    out_high.append(oh)
                    out_low.append(ol)
                    out_rand.append(or_)

                lh, ll, lr = [], [], []
                ah, al, ar = [], [], []
                for i_net in range(num_nets):
                    lv_h = float(dataset.measure_loss(out_high[i_net], labels).cpu())
                    lv_l = float(dataset.measure_loss(out_low[i_net], labels).cpu())
                    lv_r = float(dataset.measure_loss(out_rand[i_net], labels).cpu())
                    lh.append(lv_h)
                    ll.append(lv_l)
                    lr.append(lv_r)

                    av_h = float(dataset.measure_accuracy(out_high[i_net], labels).cpu())
                    av_l = float(dataset.measure_accuracy(out_low[i_net], labels).cpu())
                    av_r = float(dataset.measure_accuracy(out_rand[i_net], labels).cpu())
                    ah.append(av_h)
                    al.append(av_l)
                    ar.append(av_r)

                progdrop_loss_high[:, dropidx, 0] += torch.tensor(lh, device="cpu")
                progdrop_loss_low[:, dropidx, 0]  += torch.tensor(ll, device="cpu")
                progdrop_loss_rand[:, dropidx, 0] += torch.tensor(lr, device="cpu")
                progdrop_acc_high[:, dropidx, 0]  += torch.tensor(ah, device="cpu")
                progdrop_acc_low[:, dropidx, 0]   += torch.tensor(al, device="cpu")
                progdrop_acc_rand[:, dropidx, 0]  += torch.tensor(ar, device="cpu")

        progdrop_loss_high /= num_batches
        progdrop_loss_low  /= num_batches
        progdrop_loss_rand /= num_batches
        progdrop_acc_high  /= num_batches
        progdrop_acc_low   /= num_batches
        progdrop_acc_rand  /= num_batches

        results = {
            "progdrop_loss_high": progdrop_loss_high,
            "progdrop_loss_low":  progdrop_loss_low,
            "progdrop_loss_rand": progdrop_loss_rand,
            "progdrop_acc_high":  progdrop_acc_high,
            "progdrop_acc_low":   progdrop_acc_low,
            "progdrop_acc_rand":  progdrop_acc_rand,
            "dropout_fraction":   drop_fraction,
            "pruning_mode":       pruning_mode,
            "idx_dropout_layers": [0],
        }
        return results
    else:
        # Per-layer parsing for both "per_layer_combined" and "per_layer_independent" modes
        valid_layers = []
        layer_indices = []
        for i, layer_tsr in enumerate(parsed):
            if layer_tsr is not None:
                valid_layers.append(layer_tsr)
                layer_indices.append(i)

        num_layers = len(valid_layers)
        if num_layers == 0:
            raise ValueError("No valid layers found in alignment data when using per-layer pruning")

        progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_loss_low  = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_acc_high  = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_acc_low   = torch.zeros((num_nets, num_drops, num_layers), device="cpu")
        progdrop_acc_rand  = torch.zeros((num_nets, num_drops, num_layers), device="cpu")

        num_batches = 0
        for batch in tqdm(dataloader):
            images, labels = dataset.unwrap_batch(batch)
            num_batches += 1

            for dropidx, fraction in enumerate(drop_fraction):
                for lyr_idx, layer_tsr in enumerate(valid_layers):
                    idx_sorted = torch.argsort(layer_tsr, dim=1)
                    node_count = idx_sorted.size(1)
                    dn = int(node_count * fraction)
                    if dn > 0:
                        hi = idx_sorted[:, node_count - dn : node_count]
                        lo = idx_sorted[:, :dn]
                        rr = []
                        for i_net in range(num_nets):
                            perm = torch.randperm(node_count, device=idx_sorted.device)
                            rr.append(perm[:dn])
                        rr = torch.stack(rr, dim=0)
                    else:
                        hi = idx_sorted[:, :0]
                        lo = idx_sorted[:, :0]
                        rr = idx_sorted[:, :0]

                    # Determine which layers to apply pruning to
                    if pruning_mode == "per_layer_independent":
                        # Just prune the current layer
                        target_layers = [layer_indices[lyr_idx]]
                        
                        # Process outputs for current layer only
                        out_high, out_low, out_rand = [], [], []
                        for i_net, net in enumerate(nets):
                            oh, _ = net.forward_targeted_dropout(images, [hi[i_net]], target_layers)
                            ol, _ = net.forward_targeted_dropout(images, [lo[i_net]], target_layers)
                            or_, _= net.forward_targeted_dropout(images, [rr[i_net]], target_layers)
                            out_high.append(oh)
                            out_low.append(ol)
                            out_rand.append(or_)
                            
                        # Record metrics for this layer
                        lh, ll, lr = [], [], []
                        ah, al, ar = [], [], []
                        for i_net in range(num_nets):
                            lv_h = float(dataset.measure_loss(out_high[i_net], labels).cpu())
                            lv_l = float(dataset.measure_loss(out_low[i_net], labels).cpu())
                            lv_r = float(dataset.measure_loss(out_rand[i_net], labels).cpu())
                            lh.append(lv_h)
                            ll.append(lv_l)
                            lr.append(lv_r)

                            av_h = float(dataset.measure_accuracy(out_high[i_net], labels).cpu())
                            av_l = float(dataset.measure_accuracy(out_low[i_net], labels).cpu())
                            av_r = float(dataset.measure_accuracy(out_rand[i_net], labels).cpu())
                            ah.append(av_h)
                            al.append(av_l)
                            ar.append(av_r)

                        progdrop_loss_high[:, dropidx, lyr_idx] += torch.tensor(lh, device="cpu")
                        progdrop_loss_low[:, dropidx, lyr_idx]  += torch.tensor(ll, device="cpu")
                        progdrop_loss_rand[:, dropidx, lyr_idx] += torch.tensor(lr, device="cpu")
                        progdrop_acc_high[:, dropidx, lyr_idx]  += torch.tensor(ah, device="cpu")
                        progdrop_acc_low[:, dropidx, lyr_idx]   += torch.tensor(al, device="cpu")
                        progdrop_acc_rand[:, dropidx, lyr_idx]  += torch.tensor(ar, device="cpu")
                    
                    else:  # "per_layer_combined"
                        # For per_layer_combined, we still run the loop for each layer
                        # but collect pruning results for all layers together
                        
                        # Apply pruning to each layer independently but record results in each layer's slot
                        # This is primarily for visualization purposes to show effect on each layer
                        out_high, out_low, out_rand = [], [], []
                        for i_net, net in enumerate(nets):
                            # Just focus on current layer for this loop iteration
                            oh, _ = net.forward_targeted_dropout(images, [hi[i_net]], [layer_indices[lyr_idx]])
                            ol, _ = net.forward_targeted_dropout(images, [lo[i_net]], [layer_indices[lyr_idx]])
                            or_, _= net.forward_targeted_dropout(images, [rr[i_net]], [layer_indices[lyr_idx]])
                            
                            out_high.append(oh)
                            out_low.append(ol)
                            out_rand.append(or_)
                        
                        # Record metrics for this layer
                        lh, ll, lr = [], [], []
                        ah, al, ar = [], [], []
                        for i_net in range(num_nets):
                            lv_h = float(dataset.measure_loss(out_high[i_net], labels).cpu())
                            lv_l = float(dataset.measure_loss(out_low[i_net], labels).cpu())
                            lv_r = float(dataset.measure_loss(out_rand[i_net], labels).cpu())
                            lh.append(lv_h)
                            ll.append(lv_l)
                            lr.append(lv_r)

                            av_h = float(dataset.measure_accuracy(out_high[i_net], labels).cpu())
                            av_l = float(dataset.measure_accuracy(out_low[i_net], labels).cpu())
                            av_r = float(dataset.measure_accuracy(out_rand[i_net], labels).cpu())
                            ah.append(av_h)
                            al.append(av_l)
                            ar.append(av_r)

                        progdrop_loss_high[:, dropidx, lyr_idx] += torch.tensor(lh, device="cpu")
                        progdrop_loss_low[:, dropidx, lyr_idx]  += torch.tensor(ll, device="cpu")
                        progdrop_loss_rand[:, dropidx, lyr_idx] += torch.tensor(lr, device="cpu")
                        progdrop_acc_high[:, dropidx, lyr_idx]  += torch.tensor(ah, device="cpu")
                        progdrop_acc_low[:, dropidx, lyr_idx]   += torch.tensor(al, device="cpu")
                        progdrop_acc_rand[:, dropidx, lyr_idx]  += torch.tensor(ar, device="cpu")

        progdrop_loss_high /= num_batches
        progdrop_loss_low  /= num_batches
        progdrop_loss_rand /= num_batches
        progdrop_acc_high  /= num_batches
        progdrop_acc_low   /= num_batches
        progdrop_acc_rand  /= num_batches

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
        
        # For per_layer_combined mode, we need to aggregate the results across layers
        # to produce a single figure showing the combined effect of pruning
        if pruning_mode == "per_layer_combined":
            # Average the results across all layers to get one result per dropout fraction
            combined_loss_high = progdrop_loss_high.mean(dim=2)
            combined_loss_low = progdrop_loss_low.mean(dim=2)
            combined_loss_rand = progdrop_loss_rand.mean(dim=2)
            combined_acc_high = progdrop_acc_high.mean(dim=2)
            combined_acc_low = progdrop_acc_low.mean(dim=2)
            combined_acc_rand = progdrop_acc_rand.mean(dim=2)
            
            # Add a new dimension to match expected shape for plotting code
            results["combined_progdrop_loss_high"] = combined_loss_high.unsqueeze(2)
            results["combined_progdrop_loss_low"] = combined_loss_low.unsqueeze(2)
            results["combined_progdrop_loss_rand"] = combined_loss_rand.unsqueeze(2)
            results["combined_progdrop_acc_high"] = combined_acc_high.unsqueeze(2)
            results["combined_progdrop_acc_low"] = combined_acc_low.unsqueeze(2)
            results["combined_progdrop_acc_rand"] = combined_acc_rand.unsqueeze(2)
            
        return results


def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    print("performing targeted dropout...")
    dropout_params = dict(
        num_drops=exp.args.extra.num_drops,
        train_set=train_set,
        aggregate_alignment=exp.args.extra.aggregate_alignment,
        pruning_mode=exp.args.extra.dropout_pruning_mode
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

    from alignment.utils import check_iterable

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