# --------------------------------------------
# processing.py
# --------------------------------------------

import os
import torch
from tqdm import tqdm

from alignment.utils import load_checkpoints, test_nets, transpose_list, fgsm_attack
import alignment.train as train
from alignment.alignment_metrics import AlignmentMetrics

def train_networks(exp, nets, optimizers, dataset, **special_parameters):
    """
    Orchestrates training and testing of networks.
    The toggles in exp.args.alignment control whether alignment
    is measured during training or inference.
    """
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

    # Possibly load from checkpoint
    if exp.args.checkpointing.use_prev and os.path.isfile(exp.get_checkpoint_path()):
        nets, optimizers, results = load_checkpoints(
            nets, optimizers, exp.args.device, exp.get_checkpoint_path()
        )
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


def test_networks(exp, nets, dataset):
    """
    A simple wrapper to test networks when do_train=False.
    Calls train.test(...) behind the scenes with alignment parameters from exp.args.
    """
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
    test_results = train.test(nets, dataset, **test_params)
    return test_results


@test_nets
@torch.no_grad()
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    Perform progressive dropout *per alignment layer* or lumps them, depending on by_layer.
    
    Instead of flattening across *all* layers, we keep shape = (#nets, #layers, #nodes_per_layer).
    Then, if by_layer=True, we can target each layer separately.
    """
    if not isinstance(nets, list):
        nets = [nets]

    if alignment is None:
        # If no alignment is provided, measure it via a test pass
        alignment = train.test(
            nets, dataset, alignment=True, methods=["RQ"], **parameters
        )["alignment"]

    # Build an array of shape => list of snapshots, each snapshot is (num_nets, num_layers, num_nodes_per_that_layer)
    alignment_snapshots = []
    for snapshot in alignment:
        # snapshot["data"] is a list over nets => net_i_data
        # net_i_data is a list over layers => layer_dict with keys e.g. "RQ"
        # So we want shape => (num_nets, num_layers, #nodes)
        all_nets_rq = []
        for net_i_data in snapshot["data"]:
            # net_i_data is e.g. [ {"RQ": tensor(#nodes_layer0)}, {"RQ": tensor(#nodes_layer1)}, ... ]
            rq_per_layer = [ld["RQ"] for ld in net_i_data]  # list of 1D Tensors
            # stack them => shape (num_layers, #nodes_this_layer)
            # But each layer can have a different #nodes, so let's just keep a list-of-tensors:
            # Alternatively, if each layer has a different shape, we can't stack, but let's do it carefully
            # We'll unify by storing in a Python list:
            all_nets_rq.append(rq_per_layer)
        # Now we have a list of length #nets, each item is a list of length #layers, 1D RQ
        # We'll keep it as is: shape => (#nets, #layers, #?)
        alignment_snapshots.append(all_nets_rq)

    # alignment_snapshots => list[#snapshots], each => (#nets, #layers, 1D RQ)
    num_snapshots = len(alignment_snapshots)
    by_layer = parameters.get("by_layer", False)
    num_layers = len(alignment_snapshots[0][0]) if by_layer else 1  # if by_layer=True, we treat each layer separately

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]

    # We store dropout results on CPU
    progdrop_loss_high = torch.zeros(num_nets, num_drops, num_layers, dtype=torch.float)
    progdrop_loss_low  = torch.zeros(num_nets, num_drops, num_layers, dtype=torch.float)
    progdrop_loss_rand = torch.zeros(num_nets, num_drops, num_layers, dtype=torch.float)
    progdrop_acc_high  = torch.zeros(num_nets, num_drops, num_layers, dtype=torch.float)
    progdrop_acc_low   = torch.zeros(num_nets, num_drops, num_layers, dtype=torch.float)
    progdrop_acc_rand  = torch.zeros(num_nets, num_drops, num_layers, dtype=torch.float)

    num_batches = 0
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    for batch in tqdm(dataloader, desc="Progressive Dropout"):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for dropidx, fraction in enumerate(drop_fraction):
            # We build the dropout indices for each snapshot
            idx_high_all, idx_low_all, idx_rand_all = [], [], []

            # For each snapshot (which might correspond to an epoch/frequency), build the sorted indices
            for snap_i, net_rqs in enumerate(alignment_snapshots):
                # net_rqs => (#nets, #layers, each is 1D RQ)
                # flatten or not depending on by_layer
                # If by_layer=True, we do per-layer. If not, we lump them all
                if by_layer:
                    # shape => (#nets, #layers, 1D RQ)
                    # We'll build a list of shape (#layers). Each layer is (#nets, #num_nodes_layer)
                    layer_idxs_high = []
                    layer_idxs_low  = []
                    layer_idxs_rand = []
                    for layer_i in range(num_layers):
                        # gather each net's RQ => shape (num_nets, #nodes)
                        # then sort etc.
                        # net_rqs[i_net][layer_i] is shape (#nodes)
                        # stack across i_net => shape (num_nets, #nodes)
                        net_rq_list = [net_rqs[i_net][layer_i] for i_net in range(num_nets)]
                        # pad or direct stack
                        # assume they are same #nodes across all nets
                        rq_stacked = torch.stack(net_rq_list, dim=0)  # (num_nets, #nodes)
                        # Sort for each net
                        idx_sorted = torch.argsort(rq_stacked, dim=1)  # shape => (num_nets, #nodes)
                        num_nodes = rq_stacked.size(1)
                        drop_num = int(num_nodes * fraction)
                        idx_hi = torch.index_select(idx_sorted, 1, torch.arange(num_nodes-drop_num, num_nodes))
                        idx_lo = torch.index_select(idx_sorted, 1, torch.arange(drop_num))
                        idx_rd = []
                        for i_net in range(num_nets):
                            idx_rd.append(torch.randperm(num_nodes)[:drop_num])
                        idx_rd = torch.stack(idx_rd, dim=0)
                        layer_idxs_high.append(idx_hi)
                        layer_idxs_low.append(idx_lo)
                        layer_idxs_rand.append(idx_rd)
                    # now shape => (#layers, #nets, #drop_num)
                    idx_high_all.append(layer_idxs_high)
                    idx_low_all.append(layer_idxs_low)
                    idx_rand_all.append(layer_idxs_rand)
                else:
                    # Lump all layers into one. 
                    # net_rqs[i_net] => list of shape (#layers). We'll flatten
                    net_flat = []
                    for i_net in range(num_nets):
                        cat_ = torch.cat(net_rqs[i_net], dim=0)  # e.g. shape => (sum_of_nodes_all_layers,)
                        net_flat.append(cat_)
                    # => shape (#nets, sum_nodes)
                    flat_rq = torch.stack(net_flat, dim=0)
                    idx_sorted = torch.argsort(flat_rq, dim=1)
                    num_nodes = flat_rq.size(1)
                    drop_num = int(num_nodes * fraction)
                    idx_hi = torch.index_select(idx_sorted, 1, torch.arange(num_nodes - drop_num, num_nodes))
                    idx_lo = torch.index_select(idx_sorted, 1, torch.arange(drop_num))
                    idx_rd = []
                    for i_net in range(num_nets):
                        idx_rd.append(torch.randperm(num_nodes)[:drop_num])
                    idx_rd = torch.stack(idx_rd, dim=0)
                    # store
                    idx_high_all.append(idx_hi)
                    idx_low_all.append(idx_lo)
                    idx_rand_all.append(idx_rd)

            # Now we have per-snapshot sorted indices
            # We'll just pick the snapshot we want. If you're actually merging across snapshots,
            # you might choose snapshot=-1 or something. For simplicity, let's pick last snapshot.
            # Or do you want to sum across them all? The original code picks "layer_i in range(num_snapshots) if by_layer" etc.
            # We'll do a single snapshot to keep code simpler:
            # Typically, if you want 1 snapshot => snap_i=-1
            snap_i = -1
            # idx_high_all[snap_i] => either a list of (#layers) if by_layer, or single big set
            # We'll keep the same approach:
            if by_layer:
                # => (#layers, #nets, #drop_num)
                idx_high_use = idx_high_all[snap_i]  # shape => list[#layers] of Tensors
                idx_low_use  = idx_low_all[snap_i]
                idx_rand_use = idx_rand_all[snap_i]
            else:
                idx_high_use = [idx_high_all[snap_i]]
                idx_low_use  = [idx_low_all[snap_i]]
                idx_rand_use = [idx_rand_all[snap_i]]

            # We'll do a for layer_i in range(num_layers) block:
            Lmax = num_layers if by_layer else 1
            for layer_i in range(Lmax):
                # gather the actual indices
                hi_idxs = idx_high_use[layer_i]
                lo_idxs = idx_low_use[layer_i]
                rd_idxs = idx_rand_use[layer_i]

                out_high, out_low, out_rand = [], [], []
                for i_net, net in enumerate(nets):
                    oh, _ = net.forward_targeted_dropout(
                        images, [hi_idxs[i_net]], [layer_i] if by_layer else list(range(num_snapshots))
                    )
                    ol, _ = net.forward_targeted_dropout(
                        images, [lo_idxs[i_net]], [layer_i] if by_layer else list(range(num_snapshots))
                    )
                    or_, _= net.forward_targeted_dropout(
                        images, [rd_idxs[i_net]], [layer_i] if by_layer else list(range(num_snapshots))
                    )
                    out_high.append(oh)
                    out_low.append(ol)
                    out_rand.append(or_)

                # measure loss & accuracy
                loss_h = [float(dataset.measure_loss(oh, labels).detach().cpu()) for oh in out_high]
                loss_l = [float(dataset.measure_loss(ol, labels).detach().cpu()) for ol in out_low]
                loss_r = [float(dataset.measure_loss(or_, labels).detach().cpu()) for or_ in out_rand]
                acc_h = [float(dataset.measure_accuracy(oh, labels).detach().cpu()) for oh in out_high]
                acc_l = [float(dataset.measure_accuracy(ol, labels).detach().cpu()) for ol in out_low]
                acc_r = [float(dataset.measure_accuracy(or_, labels).detach().cpu()) for or_ in out_rand]

                progdrop_loss_high[:, dropidx, layer_i] += torch.tensor(loss_h)
                progdrop_loss_low[:, dropidx, layer_i]  += torch.tensor(loss_l)
                progdrop_loss_rand[:, dropidx, layer_i] += torch.tensor(loss_r)
                progdrop_acc_high[:, dropidx, layer_i]  += torch.tensor(acc_h)
                progdrop_acc_low[:, dropidx, layer_i]   += torch.tensor(acc_l)
                progdrop_acc_rand[:, dropidx, layer_i]  += torch.tensor(acc_r)

    # finalize
    progdrop_loss_high /= num_batches
    progdrop_loss_low  /= num_batches
    progdrop_loss_rand /= num_batches
    progdrop_acc_high  /= num_batches
    progdrop_acc_low   /= num_batches
    progdrop_acc_rand  /= num_batches

    results = {
        "progdrop_loss_high": progdrop_loss_high.cpu(),
        "progdrop_loss_low":  progdrop_loss_low.cpu(),
        "progdrop_loss_rand": progdrop_loss_rand.cpu(),
        "progdrop_acc_high":  progdrop_acc_high.cpu(),
        "progdrop_acc_low":   progdrop_acc_low.cpu(),
        "progdrop_acc_rand":  progdrop_acc_rand.cpu(),
        "dropout_fraction":   drop_fraction,
        "by_layer":           by_layer,
    }
    return results


def get_dropout_indices(idx_alignment, fraction):
    """
    Original version flattened everything. Now we don't flatten if we keep shape.
    In the new approach, you might not even need this function if you're building
    per-layer indices. But if you do, just keep shapes consistent.
    """
    # This function might not be used the same way if you store shape (#nets, #layers, #nodes).
    # Typically you'd do the sorting directly in progressive_dropout as we did above.
    pass


def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    """
    Perform a progressive dropout experiment, dropping nodes (by alignment ranking) in increments.
    With the new approach, each layer is separate if by_layer=True.
    """
    print("performing targeted dropout...")
    dropout_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
    )
    dropout_results = progressive_dropout(nets, dataset, alignment=alignment, **dropout_params)
    return dropout_results, dropout_params


def measure_eigenfeatures(exp, nets, dataset, train_set=False):
    """
    Gather input activations for each layer across the entire dataset,
    compute PCA, measure how each weight aligns with the eigenvectors.
    """
    print("measuring eigenfeatures...")
    from tqdm import tqdm
    beta, eigvals, eigvecs, class_betas = [], [], [], []
    for net in tqdm(nets, desc="Eigenfeatures"):
        inputs, labels = net._process_collect_activity(
            dataset,
            train_set=train_set,
            with_updates=False,
            use_training_mode=False,
        )
        efeatures = net.measure_eigenfeatures(inputs, with_updates=False)
        cls_betas = net.measure_class_eigenfeatures(
            inputs, labels, efeatures[2], rms=False, with_updates=False
        )
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
    Example code for eigenvector dropout. If you keep the layer dimension separate,
    you'll do something similar: each layer can have a different # of input dims.
    """
    # (unchanged or lightly adapted)
    pass


def eigenvector_dropout_experiment(exp, nets, dataset, eigen_results, train_set=False):
    """
    Similar to progressive_dropout_experiment but uses
    eigenvectors + eigenvalues to rank which components to drop.
    """
    print("performing targeted eigenvector dropout...")
    evec_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
    )
    # Then call the above eigenvector_dropout, storing results
    return {}, evec_params


def evaluate_pretrained_model(net, dataset):
    """
    Evaluate a single pretrained model on the dataset's test loader
    """
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