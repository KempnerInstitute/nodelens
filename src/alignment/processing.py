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
    method for testing network on supervised learning problem with progressive dropout.
    Based on the average alignment, we sort the nodes from low-to-high alignment
    for each alignment layer. Then we systematically drop top X% or bottom X% or 
    random X% of nodes, measuring performance at each fraction.

    NOTE: We only drop *alignment* layers, not all layers of the net. 
    This is crucial if we only want to drop out specific layers, e.g. those in alignment_layers.
    """
    if not isinstance(nets, list):
        nets = [nets]

    # If alignment is None, compute alignment from test(...) 
    if alignment is None:
        alignment = train.test(
            nets, dataset, alignment=True, methods=["RQ"], **parameters
        )["alignment"]

    # Let's get the number of alignment layers from the first net
    # net.alignment_layers is a list, so we do:
    alignment_layer_count = len(nets[0].alignment_layers)
    
    # 'alignment' is typically a list of snapshots, one per alignment layer
    # shape => #=alignment_layer_count or snapshots
    # We'll build alignment_layers as in your code:
    alignment_layers = []
    for layerdata in alignment:
        # layerdata["data"] is a list with one entry per net, each is a list of dicts
        # one dict per alignment layer => { "RQ": <some tensor> } or similar
        all_nets_rq = []
        for net_i_data in layerdata["data"]:
            # net_i_data => [ { "RQ": ... }, { "RQ": ... }, ... ] (one per alignment layer)
            net_nodes = []
            for layer_dict in net_i_data:
                # e.g. layer_dict["RQ"] => shape (# of nodes in that layer)
                net_nodes.append(layer_dict["RQ"].flatten())
            flattened = torch.cat(net_nodes, dim=0)
            all_nets_rq.append(flattened)
        stacked = torch.stack(all_nets_rq, dim=0)  # shape => (num_nets, total_nodes)
        alignment_layers.append(stacked)

    # idx_alignment => a list of shape (#=alignment_layer_count) or snapshots
    # each item => idx = torch.argsort(al, dim=1), sorting from low to high
    idx_alignment = [torch.argsort(al, dim=1) for al in alignment_layers]

    # 'by_layer' means we handle each alignment layer separately,
    # else we treat them all at once (like your code).
    by_layer = parameters.get("by_layer", False)
    # If by_layer=True => we do alignment_layer_count loops
    # else => 1 loop for "all"
    num_layers = alignment_layer_count if by_layer else 1

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]

    # Preallocate
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low   = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand  = torch.zeros((num_nets, num_drops, num_layers))

    num_batches = 0
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for dropidx, fraction in enumerate(drop_fraction):
            # get the indices to drop (top, bottom, random) for each alignment layer
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_alignment, fraction)

            # Now if by_layer=True, we handle each layer separately
            # else we handle them all in one pass
            for layer_i in range(num_layers):
                if by_layer:
                    # We only drop the nodes from the single alignment layer 'layer_i'
                    # => so pick out idx_*[layer_i] => shape (num_nets, #_drop)
                    # We'll pass [layer_i] to forward_targeted_dropout
                    drop_high_use = [idx_high[layer_i]]
                    drop_low_use  = [idx_low[layer_i]]
                    drop_rand_use = [idx_rand[layer_i]]
                    drop_layer = [layer_i]  # alignment-layer index
                else:
                    # We handle all alignment layers at once
                    drop_high_use = idx_high
                    drop_low_use  = idx_low
                    drop_rand_use = idx_rand
                    # The entire set [0,1,..., alignment_layer_count-1]
                    drop_layer = list(range(alignment_layer_count))

                # forward pass with top X% dropped
                out_high = [
                    net.forward_targeted_dropout(
                        images, [dr[i_net, :] for dr in drop_high_use], drop_layer
                    )[0]
                    for i_net, net in enumerate(nets)
                ]
                # forward pass with bottom X% dropped
                out_low = [
                    net.forward_targeted_dropout(
                        images, [dr[i_net, :] for dr in drop_low_use], drop_layer
                    )[0]
                    for i_net, net in enumerate(nets)
                ]
                # forward pass with random X% dropped
                out_rand = [
                    net.forward_targeted_dropout(
                        images, [dr[i_net, :] for dr in drop_rand_use], drop_layer
                    )[0]
                    for i_net, net in enumerate(nets)
                ]

                loss_high = [dataset.measure_loss(oh, labels).item() for oh in out_high]
                loss_low  = [dataset.measure_loss(ol, labels).item() for ol in out_low]
                loss_rand = [dataset.measure_loss(or_, labels).item() for or_ in out_rand]

                acc_high = [dataset.measure_accuracy(oh, labels) for oh in out_high]
                acc_low  = [dataset.measure_accuracy(ol, labels) for ol in out_low]
                acc_rand = [dataset.measure_accuracy(or_, labels) for or_ in out_rand]

                progdrop_loss_high[:, dropidx, layer_i] += torch.tensor(loss_high)
                progdrop_loss_low[:, dropidx, layer_i]  += torch.tensor(loss_low)
                progdrop_loss_rand[:, dropidx, layer_i] += torch.tensor(loss_rand)

                progdrop_acc_high[:, dropidx, layer_i]  += torch.tensor(acc_high)
                progdrop_acc_low[:, dropidx, layer_i]   += torch.tensor(acc_low)
                progdrop_acc_rand[:, dropidx, layer_i]  += torch.tensor(acc_rand)

    # divide by num_batches to get mean
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
        "dropout_fraction":    drop_fraction,
        "by_layer":            by_layer,
        # We store the alignment-layer indices if helpful, e.g. range(alignment_layer_count)
        "idx_dropout_layers":  list(range(alignment_layer_count)),
    }
    return results


def get_dropout_indices(idx_alignment, fraction):
    """
    convenience method for getting a fraction of dropout indices from each layer

    idx_alignment is a list of length (#=alignment_layer_count).
    Each item is shape => (num_nets, total_nodes_in_that_layer).
    For fraction X, we pick top X% or bottom X% or random.

    Returns three lists of the same shape as idx_alignment:
      idx_high[layer_i] => shape (num_nets, #_drop)
      idx_low[layer_i]  => shape (num_nets, #_drop)
      idx_rand[layer_i] => shape (num_nets, #_drop)
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes_per_layer = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes_per_layer]

    idx_high = []
    idx_low  = []
    idx_rand = []
    for layer_i, drop_count in enumerate(num_drop):
        idx = idx_alignment[layer_i]  # shape => (num_nets, total_nodes)
        # top X% => last "drop_count" columns
        # because idx is sorted from low->high? 
        # Actually 'torch.argsort' is ascending, so top X% means idx[:, -drop_count:]
        # bottom X% => idx[:, :drop_count]
        if drop_count == 0:
            # no dropout
            idx_high.append(torch.empty((num_nets, 0), dtype=torch.long))
            idx_low.append(torch.empty((num_nets, 0), dtype=torch.long))
            idx_rand.append(torch.empty((num_nets, 0), dtype=torch.long))
        else:
            top_ = idx[:, -drop_count:]
            bottom_ = idx[:, :drop_count]
            # random => pick 'drop_count' random indices for each net
            layer_rand = []
            total_nodes = idx.size(1)
            for _ in range(num_nets):
                random_perm = torch.randperm(total_nodes)
                layer_rand.append(random_perm[:drop_count])
            layer_rand = torch.stack(layer_rand, dim=0)  # shape => (num_nets, drop_count)

            idx_high.append(top_)
            idx_low.append(bottom_)
            idx_rand.append(layer_rand)

    return idx_high, idx_low, idx_rand


def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    """
    Perform a progressive dropout experiment, dropping nodes 
    (by alignment ranking) in increments. If alignment is None,
    we re-compute alignment (RQ) on the test set by default.

    This function is a simple wrapper around progressive_dropout(...).
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
    beta, eigvals, eigvecs, class_betas = [], [], [], []
    for net in tqdm(nets):
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
    method for testing network on supervised learning problem with eigenvector dropout
    Instead of dropping entire nodes, we drop entire eigenvectors
    based on their relative alignment or eigenvalues, for each alignment layer.

    Like progressive_dropout, but we rank eigenvalues and remove top X% or bottom X% or random X% 
    of eigenvectors. 
    If by_layer=True, do each alignment layer separately; else do them all at once.

    Returns a dict with average loss/acc across dropout fractions in 
    e.g. 'progdrop_loss_high', 'progdrop_acc_high', etc.
    """
    if not isinstance(nets, list):
        nets = [nets]

    align_layer_count = len(nets[0].alignment_layers)
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = align_layer_count if by_layer else 1

    # Build an index in ascending order for each net/layer if you want 
    # or descending => we can do the same approach as your original code
    # We'll just do ascending first, then picking top X% => we pick from the end
    idx_eigenvalue = []
    for net_i in range(num_nets):
        layer_idxs = []
        for evec_j in eigenvectors[net_i]:
            dim = evec_j.size(1)  # evec_j => shape (dim, dim)
            # We'll create an ascending index: 0..dim-1
            # If you want descending, do range(dim-1,...,0)
            # but let's do ascending, so top X% => last slice
            layer_idxs.append(torch.arange(dim).unsqueeze(0))  # shape => (1, dim)
        idx_eigenvalue.append(layer_idxs)

    # Preallocate
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low   = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand  = torch.zeros((num_nets, num_drops, num_layers))

    num_batches = 0
    use_test = not parameters.get("train_set", True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        # for each fraction => drop fraction of eigenvectors
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_eigenvalue, fraction)

            for layer_i in range(num_layers):
                if by_layer:
                    drop_high = [idx_high[layer_i]]
                    drop_low  = [idx_low[layer_i]]
                    drop_rand = [idx_rand[layer_i]]
                    drop_layer = [layer_i]
                    # gather the relevant subset of eigenvals/evecs
                    evals = []
                    evecs = []
                    for net_i in range(num_nets):
                        evals.append([eigenvalues[net_i][layer_i]])
                        evecs.append([eigenvectors[net_i][layer_i]])
                else:
                    drop_high = idx_high
                    drop_low  = idx_low
                    drop_rand = idx_rand
                    drop_layer = list(range(align_layer_count))
                    evals = eigenvalues
                    evecs = eigenvectors

                out_high = []
                for net_i, net in enumerate(nets):
                    if by_layer:
                        idxs_to_drop = [drop_high[0][net_i]]
                        evec_list = evecs[net_i]
                        eval_list = evals[net_i]
                    else:
                        idxs_to_drop = [drop_high[L][net_i] for L in drop_layer]
                        evec_list = evecs[net_i]
                        eval_list = evals[net_i]

                    out, _ = net.forward_eigenvector_dropout(images, eval_list, evec_list, idxs_to_drop, drop_layer)
                    out_high.append(out)

                out_low = []
                for net_i, net in enumerate(nets):
                    if by_layer:
                        idxs_to_drop = [drop_low[0][net_i]]
                        evec_list = evecs[net_i]
                        eval_list = evals[net_i]
                    else:
                        idxs_to_drop = [drop_low[L][net_i] for L in drop_layer]
                        evec_list = evecs[net_i]
                        eval_list = evals[net_i]
                    out, _ = net.forward_eigenvector_dropout(images, eval_list, evec_list, idxs_to_drop, drop_layer)
                    out_low.append(out)

                out_rand_ = []
                for net_i, net in enumerate(nets):
                    if by_layer:
                        idxs_to_drop = [drop_rand[0][net_i]]
                        evec_list = evecs[net_i]
                        eval_list = evals[net_i]
                    else:
                        idxs_to_drop = [drop_rand[L][net_i] for L in drop_layer]
                        evec_list = evecs[net_i]
                        eval_list = evals[net_i]
                    out, _ = net.forward_eigenvector_dropout(images, eval_list, evec_list, idxs_to_drop, drop_layer)
                    out_rand_.append(out)

                # measure loss, acc
                loss_high = [dataset.measure_loss(oh, labels).item() for oh in out_high]
                loss_low  = [dataset.measure_loss(ol, labels).item() for ol in out_low]
                loss_rand = [dataset.measure_loss(or_, labels).item() for or_ in out_rand_]

                acc_high = [dataset.measure_accuracy(oh, labels) for oh in out_high]
                acc_low  = [dataset.measure_accuracy(ol, labels) for ol in out_low]
                acc_rand = [dataset.measure_accuracy(or_, labels) for or_ in out_rand_]

                progdrop_loss_high[:, dropidx, layer_i] += torch.tensor(loss_high)
                progdrop_loss_low[:, dropidx, layer_i]  += torch.tensor(loss_low)
                progdrop_loss_rand[:, dropidx, layer_i] += torch.tensor(loss_rand)

                progdrop_acc_high[:, dropidx, layer_i]  += torch.tensor(acc_high)
                progdrop_acc_low[:, dropidx, layer_i]   += torch.tensor(acc_low)
                progdrop_acc_rand[:, dropidx, layer_i]  += torch.tensor(acc_rand)

    # average over batches
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
        "dropout_fraction":    drop_fraction,
        "by_layer":            by_layer,
    }
    return results


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
    evec_dropout_results = eigenvector_dropout(
        nets,
        dataset,
        eigen_results["eigvals"],
        eigen_results["eigvecs"],
        **evec_params
    )
    return evec_dropout_results, evec_params


def evaluate_pretrained_model(net, dataset):
    """
    Measure test accuracy of a loaded pretrained model on 'dataset.test_loader'.
    'net' should already be on the correct device and in eval mode.
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