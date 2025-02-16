# --------------------------------------------
# processing.py
# --------------------------------------------

import os
import torch
from tqdm import tqdm

from alignment.utils import load_checkpoints, test_nets
import alignment.train as train
from alignment.alignment_metrics import AlignmentMetrics


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
    train_results = train.train(nets, optimizers, dataset, **params)

    do_alignment_infer = exp.args.alignment.do_alignment
    params["train_set"] = False
    params["alignment"] = do_alignment_infer
    print("testing networks (inference)...")
    test_results = train.test(nets, dataset, **params)

    return train_results, test_results


def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    print("performing targeted dropout...")
    # get user param from config
    single_layer_mode = getattr(exp.args.extra, "single_layer_mode", False)  # --- CHANGED ---
    # e.g. single_layer_mode = True => do single-layer dropout instead of all-layer or by_layer

    # the existing progressive_dropout call
    dropout_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
        single_layer_mode=single_layer_mode  # --- CHANGED ---
    )

    # call a new function or pass param
    dropout_results = progressive_dropout(nets, dataset, alignment=alignment, **dropout_params)
    return dropout_results, dropout_params


@test_nets
@torch.no_grad()
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    The same function as before, but we'll add logic for single_layer_mode.

    If single_layer_mode=True, we do a separate measurement for each alignment layer individually.

    For normal usage, single_layer_mode=False => unchanged behavior.
    """
    if not isinstance(nets, list):
        nets = [nets]

    if alignment is None:
        alignment = train.test(nets, dataset, alignment=True, methods=["RQ"], **parameters)["alignment"]

    alignment_layers = []
    for layerdata in alignment:
        all_nets_rq = []
        for net_i_data in layerdata["data"]:
            net_nodes = []
            for layer_dict in net_i_data:
                net_nodes.append(layer_dict["RQ"].flatten())
            flattened = torch.cat(net_nodes, dim=0)
            all_nets_rq.append(flattened)
        stacked = torch.stack(all_nets_rq, dim=0)
        alignment_layers.append(stacked)

    idx_alignment = [torch.argsort(al, dim=1) for al in alignment_layers]

    num_snapshots = len(idx_alignment)
    by_layer = parameters.get("by_layer", False)
    num_layers = num_snapshots if by_layer else 1

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]

    single_layer_mode = parameters.get("single_layer_mode", False)  # --- CHANGED ---

    # preallocate
    # shape => (num_nets, num_drops, num_layers)
    # if single_layer_mode => we might want shape => (num_nets, num_drops, #alignment_layers?)
    # We'll keep existing shape for demonstration, or we can store (num_layers * some dimension)...

    import torch
    from tqdm import tqdm
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low   = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand  = torch.zeros((num_nets, num_drops, num_layers))

    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    num_batches = 0
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for drop_idx, frac in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_alignment, frac)

            if single_layer_mode:
                # We want to measure dropout of each alignment layer *separately*,
                # i.e. do a forward pass with only layerX dropped, measure performance, then layerY, etc.
                # We'll store them in the same arrays for now. We'll do layer_i in [0..num_layers).
                # But note if num_snapshots != # of alignment layers, might mismatch. We'll assume they match.
                # For each snapshot, we do a separate forward pass.

                for layer_i in range(num_layers):
                    # drop only the alignment for that layer
                    # i.e. drop_layer = [layer_i], drop_idxs = [ idx_high[layer_i][net_i, :] ]
                    # if layer_i >= len(idx_high): break or skip
                    if layer_i >= len(idx_high):
                        break

                    drop_high_use = [ idx_high[layer_i] ]
                    drop_low_use  = [ idx_low[layer_i]  ]
                    drop_rand_use = [ idx_rand[layer_i] ]

                    out_high, out_low, out_rand = [], [], []
                    for i_net, net in enumerate(nets):
                        # gather the actual node indices
                        high_idxs = [ drop_high_use[0][i_net, :] ]
                        low_idxs  = [ drop_low_use[0][i_net, :] ]
                        rand_idxs = [ drop_rand_use[0][i_net, :] ]

                        # forward_targeted_dropout => pass layers=[layer_i], idxs=[ high_idxs ],
                        # but be sure the "layer_i" is the alignment-layer index in the net
                        # if net uses alignment_names in order, we do layers=[layer_i].
                        # We'll assume that the net's alignment_names indices match snapshot i.
                        # In practice, you'd do a mapping. We'll do the naive approach:

                        oh, _ = net.forward_targeted_dropout(images, high_idxs, [layer_i])
                        ol, _ = net.forward_targeted_dropout(images, low_idxs,  [layer_i])
                        or_, _= net.forward_targeted_dropout(images, rand_idxs, [layer_i])

                        out_high.append(oh)
                        out_low.append(ol)
                        out_rand.append(or_)

                    # measure
                    loss_high = [dataset.measure_loss(oh, labels).item() for oh in out_high]
                    loss_low  = [dataset.measure_loss(ol, labels).item() for ol in out_low]
                    loss_rand = [dataset.measure_loss(or_, labels).item() for or_ in out_rand]

                    acc_high = [dataset.measure_accuracy(oh, labels) for oh in out_high]
                    acc_low  = [dataset.measure_accuracy(ol, labels) for ol in out_low]
                    acc_rand = [dataset.measure_accuracy(or_, labels) for or_ in out_rand]

                    # store => for demonstration, store in [drop_idx, layer_i] position
                    progdrop_loss_high[:, drop_idx, layer_i] += torch.tensor(loss_high)
                    progdrop_loss_low[:, drop_idx, layer_i]  += torch.tensor(loss_low)
                    progdrop_loss_rand[:, drop_idx, layer_i] += torch.tensor(loss_rand)
                    progdrop_acc_high[:, drop_idx, layer_i]  += torch.tensor(acc_high)
                    progdrop_acc_low[:, drop_idx, layer_i]   += torch.tensor(acc_low)
                    progdrop_acc_rand[:, drop_idx, layer_i]  += torch.tensor(acc_rand)

            else:
                # the original logic => either drop them all or something
                for layer_i in range(num_layers):
                    if layer_i >= len(idx_high):
                        break
                    if by_layer:
                        drop_high_use = [ idx_high[layer_i] ]
                        drop_low_use  = [ idx_low[layer_i] ]
                        drop_rand_use = [ idx_rand[layer_i] ]
                        drop_layer = [layer_i]
                    else:
                        drop_high_use = idx_high
                        drop_low_use  = idx_low
                        drop_rand_use = idx_rand
                        drop_layer = list(range(num_snapshots))

                    out_high = []
                    out_low  = []
                    out_rand = []
                    for i_net, net in enumerate(nets):
                        high_idxs = [dr[i_net, :] for dr in drop_high_use]
                        low_idxs  = [dr[i_net, :] for dr in drop_low_use]
                        rand_idxs = [dr[i_net, :] for dr in drop_rand_use]

                        oh, _ = net.forward_targeted_dropout(images, high_idxs, drop_layer)
                        ol, _ = net.forward_targeted_dropout(images, low_idxs, drop_layer)
                        or_, _= net.forward_targeted_dropout(images, rand_idxs, drop_layer)

                        out_high.append(oh)
                        out_low.append(ol)
                        out_rand.append(or_)

                    loss_high = [dataset.measure_loss(oh, labels).item() for oh in out_high]
                    loss_low  = [dataset.measure_loss(ol, labels).item() for ol in out_low]
                    loss_rand = [dataset.measure_loss(or_, labels).item() for or_ in out_rand]

                    acc_high = [dataset.measure_accuracy(oh, labels) for oh in out_high]
                    acc_low  = [dataset.measure_accuracy(ol, labels) for ol in out_low]
                    acc_rand = [dataset.measure_accuracy(or_, labels) for or_ in out_rand]

                    progdrop_loss_high[:, drop_idx, layer_i] += torch.tensor(loss_high)
                    progdrop_loss_low[:, drop_idx, layer_i]  += torch.tensor(loss_low)
                    progdrop_loss_rand[:, drop_idx, layer_i] += torch.tensor(loss_rand)
                    progdrop_acc_high[:, drop_idx, layer_i]  += torch.tensor(acc_high)
                    progdrop_acc_low[:, drop_idx, layer_i]   += torch.tensor(acc_low)
                    progdrop_acc_rand[:, drop_idx, layer_i]  += torch.tensor(acc_rand)

    # average
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
        "single_layer_mode":  single_layer_mode,  # --- CHANGED ---
    }
    return results


def get_dropout_indices(idx_alignment, fraction):
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    idx_high = [idx[:, -drop:] for idx, drop in zip(idx_alignment, num_drop)]
    idx_low  = [idx[:, :drop]  for idx, drop in zip(idx_alignment, num_drop)]
    idx_rand = [
        torch.stack([torch.randperm(nodes)[:drop] for _ in range(num_nets)], dim=0)
        for nodes, drop in zip(num_nodes, num_drop)
    ]
    return idx_high, idx_low, idx_rand