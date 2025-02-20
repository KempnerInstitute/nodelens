import os
import torch
from tqdm import tqdm

from alignment.utils import load_checkpoints, test_nets
from alignment.alignment_metrics import AlignmentMetrics
from alignment.train import train, test

def parse_alignment_to_tensor(alignment_list, aggregate=True, by_layer=False):
    """
    Convert a list of alignment records into a structure suitable for dropout sorting.
    Each record in alignment_list is typically:
      {
        "epoch": int or str,
        "batch": int or str,
        "data": [
           # for net i in 0..N-1
           [
             # list of layer_dict
             { "RQ": tensor(out_features,) }, ...
           ],
           ...
        ]
      }

    If by_layer=False => we concatenate all layers for each net into a single vector,
      shape => (#nets, total_nodes) after averaging or stacking.

    If by_layer=True => we gather each layer separately, so that for layer_i we do NOT zero-pad.
      We'll build a list of length = max # layers across all records, but for each layer_i we
      only store the actual node values for that layer in each net. We then average across the records
      if aggregate=True.

      The final return for by_layer=True is a list of shape (#layers,) where each element
      is a tensor with shape (num_nets, node_count_for_this_layer) representing the average
      alignment for that layer across all records.
    """
    if len(alignment_list) == 0:
        raise ValueError("parse_alignment_to_tensor: empty alignment_list")

    if not by_layer:
        all_records_tensors = []
        for record in alignment_list:
            netwise_tensors = []
            for net_i_data in record["data"]:
                node_tensors = []
                for layer_dict in net_i_data:
                    if "RQ" not in layer_dict:
                        raise ValueError("Expected 'RQ' in layer_dict")
                    node_tensors.append(layer_dict["RQ"].flatten())
                cat_tsr = torch.cat(node_tensors, dim=0)
                netwise_tensors.append(cat_tsr)
            netwise_tsr = torch.stack(netwise_tensors, dim=0)
            all_records_tensors.append(netwise_tsr)

        bigstack = torch.stack(all_records_tensors, dim=0)  # (#records, #nets, total_nodes)
        if aggregate:
            return bigstack.mean(dim=0)                     # => (#nets, total_nodes)
        else:
            return bigstack                                 # => (#records, #nets, total_nodes)

    else:
        max_layers_found = 0
        for record in alignment_list:
            for net_i_data in record["data"]:
                if len(net_i_data) > max_layers_found:
                    max_layers_found = len(net_i_data)

        # layer_storage[layer_i] => list of Tensors, each (#nets_that_have_layer, node_count_for_that_layer)
        layer_storage = [[] for _ in range(max_layers_found)]

        for record in alignment_list:
            net_list = record["data"]  # shape => (#nets,)
            for layer_i in range(max_layers_found):
                layer_nodevals = []
                for net_i_data in net_list:
                    if layer_i < len(net_i_data):
                        lay_dict = net_i_data[layer_i]
                        if "RQ" not in lay_dict:
                            raise ValueError("Expected 'RQ' in lay_dict")
                        node_vals = lay_dict["RQ"].flatten()
                        layer_nodevals.append(node_vals)
                    else:
                        layer_nodevals.append(None)

                valid_vals = [v for v in layer_nodevals if v is not None]
                if len(valid_vals) == 0:
                    continue

                node_counts = set(v.numel() for v in valid_vals)
                if len(node_counts) > 1:
                    raise ValueError(f"Mismatch in node counts among nets for layer {layer_i}")

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
            stacked = torch.stack(layer_storage[layer_i], dim=0)  # (#records, #nets_that_have_layer, node_count)
            if aggregate:
                final_layer_list.append(stacked.mean(dim=0))  
            else:
                final_layer_list.append(stacked)
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
        nets, optimizers, results = load_checkpoints(nets, optimizers, exp.args.device, exp.get_checkpoint_path())
        for net in nets:
            net.train()
        params["num_complete"] = results["epoch"] + 1
        params["results"] = results
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
    If by_layer=False => parse_alignment_to_tensor returns (#nets, total_nodes).
      We do single-lump node removal across that dimension.

    If by_layer=True => parse_alignment_to_tensor returns a list of Tensors,
      each shaped (#nets, node_count). We'll handle each layer separately.
    """
    if not isinstance(nets, list):
        nets = [nets]

    if alignment is None:
        alignment = test(nets, dataset, alignment=True, methods=["RQ"], **parameters)["alignment"]

    aggregator = parameters.get("aggregate_alignment", False)
    by_layer = parameters.get("by_layer", False)
    parsed = parse_alignment_to_tensor(alignment, aggregate=aggregator, by_layer=by_layer)

    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    print("Progressive Dropout:")

    num_nets = len(nets)

    if not by_layer:
        if not isinstance(parsed, torch.Tensor) or parsed.dim() != 2:
            raise ValueError("Expected parsed to be shape (#nets, total_nodes) for by_layer=False")

        idx_sorted = torch.argsort(parsed, dim=1)  # shape => (#nets, total_nodes)
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
                    # Protect from out-of-bounds if i_net >= hi.size(0)
                    if i_net >= hi.size(0):
                        continue
                    oh, _ = net.forward_targeted_dropout(images, [hi[i_net]], [0])
                    ol, _ = net.forward_targeted_dropout(images, [lo[i_net]], [0])
                    or_, _= net.forward_targeted_dropout(images, [rr[i_net]], [0])
                    out_high.append(oh)
                    out_low.append(ol)
                    out_rand.append(or_)

                lh, ll, lr = [], [], []
                ah, al, ar = [], [], []
                for i_net in range(num_nets):
                    if i_net >= len(out_high):
                        # means we didn't append an output for that net 
                        lh.append(0.0)
                        ll.append(0.0)
                        lr.append(0.0)
                        ah.append(0.0)
                        al.append(0.0)
                        ar.append(0.0)
                        continue
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
            "by_layer":           False,
            "idx_dropout_layers": [0],
        }
        return results

    else:
        # by_layer=True => parsed is a list of Tensors or None
        # filter out layers that are None
        valid_layers = []
        layer_indices = []
        for i, layer_tsr in enumerate(parsed):
            if layer_tsr is not None:
                valid_layers.append(layer_tsr)
                layer_indices.append(i)

        num_layers = len(valid_layers)
        if num_layers == 0:
            raise ValueError("No valid layers found in alignment data when by_layer=True")

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
                    idx_sorted = torch.argsort(layer_tsr, dim=1)  # (#nets, node_count)
                    node_count = layer_tsr.size(1)

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

                    out_high, out_low, out_rand = [], [], []
                    for i_net, net in enumerate(nets):
                        if i_net >= hi.size(0):
                            out_high.append(None)
                            out_low.append(None)
                            out_rand.append(None)
                            continue
                        oh, _ = net.forward_targeted_dropout(images, [hi[i_net]], [layer_indices[lyr_idx]])
                        ol, _ = net.forward_targeted_dropout(images, [lo[i_net]], [layer_indices[lyr_idx]])
                        or_, _= net.forward_targeted_dropout(images, [rr[i_net]], [layer_indices[lyr_idx]])
                        out_high.append(oh)
                        out_low.append(ol)
                        out_rand.append(or_)

                    lh, ll, lr = [], [], []
                    ah, al, ar = [], [], []
                    for i_net in range(num_nets):
                        if out_high[i_net] is None:
                            lh.append(0.0)
                            ll.append(0.0)
                            lr.append(0.0)
                            ah.append(0.0)
                            al.append(0.0)
                            ar.append(0.0)
                            continue
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
            "by_layer":           True,
            "idx_dropout_layers": layer_indices,
        }
        return results


def get_dropout_indices(idx_alignment, fraction):
    """
    Utility for eigenvector dropout, etc. but not typically used 
    for the by_layer approach now that we do direct sorting.

    Returns lists of indices for highest, lowest, and random sets.
    """
    idx_high_list = []
    idx_low_list = []
    idx_rand_list = []
    for idx_sorted in idx_alignment:
        device_of_idx = idx_sorted.device
        num_nets, num_nodes = idx_sorted.shape
        drop_num = int(num_nodes * fraction)
        if drop_num > 0:
            hi = torch.index_select(
                idx_sorted,
                dim=1,
                index=torch.arange(num_nodes - drop_num, num_nodes, device=device_of_idx)
            )
            lo = torch.index_select(
                idx_sorted,
                dim=1,
                index=torch.arange(drop_num, device=device_of_idx)
            )
            rand = []
            for i_net in range(num_nets):
                perm = torch.randperm(num_nodes, device=device_of_idx)
                rand.append(perm[:drop_num])
            rand = torch.stack(rand, dim=0)
        else:
            hi = idx_sorted[:, :0]
            lo = idx_sorted[:, :0]
            rand = idx_sorted[:, :0]
        idx_high_list.append(hi)
        idx_low_list.append(lo)
        idx_rand_list.append(rand)
    return idx_high_list, idx_low_list, idx_rand_list

def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    print("performing targeted dropout...")
    dropout_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
        aggregate_alignment=exp.args.extra.aggregate_alignment
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
    """
    Applies progressive dropout in eigenvector space. 
    """
    num_nets = len(nets) if isinstance(nets, list) else 1
    if not isinstance(nets, list):
        nets = [nets]

    align_layer_indices = list(range(len(nets[0].alignment_layers)))
    by_layer = parameters.get("by_layer", False)
    num_layers = len(align_layer_indices) if by_layer else 1

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

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_eigenvalue, fraction)
            for layer_i in range(num_layers):
                if layer_i >= len(idx_high):
                    break
                if by_layer:
                    drop_layer = [align_layer_indices[layer_i]]
                    drop_high  = [idx_high[layer_i]]
                    drop_low   = [idx_low[layer_i]]
                    drop_rand  = [idx_rand[layer_i]]
                else:
                    drop_layer = align_layer_indices
                    drop_high  = idx_high
                    drop_low   = idx_low
                    drop_rand  = idx_rand

                out_high, out_low, out_rand_ = [], [], []
                for i_net, net in enumerate(nets):
                    if layer_i >= len(eigenvectors[i_net]):
                        out_high.append(None)
                        out_low.append(None)
                        out_rand_.append(None)
                        continue
                    high_idxs = [dr[i_net, :] for dr in drop_high]
                    low_idxs  = [dr[i_net, :] for dr in drop_low]
                    rand_idxs = [dr[i_net, :] for dr in drop_rand]

                    oh, _ = net.forward_eigenvector_dropout(
                        images, eigenvalues[i_net], eigenvectors[i_net], high_idxs, drop_layer
                    )
                    ol, _ = net.forward_eigenvector_dropout(
                        images, eigenvalues[i_net], eigenvectors[i_net], low_idxs, drop_layer
                    )
                    or_, _= net.forward_eigenvector_dropout(
                        images, eigenvalues[i_net], eigenvectors[i_net], rand_idxs, drop_layer
                    )
                    out_high.append(oh)
                    out_low.append(ol)
                    out_rand_.append(or_)

                lh, ll, lr = [], [], []
                ah, al, ar = [], [], []
                for idxn in range(num_nets):
                    if out_high[idxn] is None:
                        lh.append(0.0)
                        ll.append(0.0)
                        lr.append(0.0)
                        ah.append(0.0)
                        al.append(0.0)
                        ar.append(0.0)
                        continue
                    lv_h = float(dataset.measure_loss(out_high[idxn], labels).cpu())
                    lv_l = float(dataset.measure_loss(out_low[idxn], labels).cpu())
                    lv_r = float(dataset.measure_loss(out_rand_[idxn], labels).cpu())
                    lh.append(lv_h)
                    ll.append(lv_l)
                    lr.append(lv_r)

                    av_h = float(dataset.measure_accuracy(out_high[idxn], labels).cpu())
                    av_l = float(dataset.measure_accuracy(out_low[idxn], labels).cpu())
                    av_r = float(dataset.measure_accuracy(out_rand_[idxn], labels).cpu())
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

    results = {
        "progdrop_loss_high": progdrop_loss_high,
        "progdrop_loss_low":  progdrop_loss_low,
        "progdrop_loss_rand": progdrop_loss_rand,
        "progdrop_acc_high":  progdrop_acc_high,
        "progdrop_acc_low":   progdrop_acc_low,
        "progdrop_acc_rand":  progdrop_acc_rand,
        "dropout_fraction":   drop_fraction,
        "by_layer":           by_layer,
    }
    return results


def eigenvector_dropout_experiment(exp, nets, dataset, eigen_results, train_set=False):
    evec_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
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