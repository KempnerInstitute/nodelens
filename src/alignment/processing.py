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


@test_nets
@torch.no_grad()
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    method for testing network on supervised learning problem with progressive dropout.
    Based on the average alignment, we sort the nodes from low-to-high alignment.
    Then we systematically drop top X% or bottom X% or random X% of nodes,
    measuring performance at each fraction.
    """
    if not (isinstance(nets, list)):
        nets = [nets]

    if alignment is None:
        # Use the test() function from train.py to gather alignment
        alignment = train.test(nets, dataset, alignment=True, methods=["RQ"], **parameters)["alignment"]

    # We'll build alignment_layers (list of Tensors), each shape = (1, total_nodes)
    alignment_layers = []
    for layerdata in alignment:
        net0_data = layerdata["data"][0]
        all_nodes = []
        for layer_dict in net0_data:
            rqs_ = layer_dict["RQ"].flatten()
            all_nodes.append(rqs_)
        flattened = torch.cat(all_nodes, dim=0)
        alignment_layers.append(flattened.unsqueeze(0))

    # idx_alignment: each entry is shape (1, total_nodes)
    idx_alignment = [torch.argsort(al, dim=1) for al in alignment_layers]

    # number of alignment snapshots
    num_snapshots = len(idx_alignment)

    by_layer = parameters.get("by_layer", False)
    # if by_layer=True, we treat each snapshot as a separate "layer" in dropout
    # else we treat all snapshots as one big layer
    num_layers = num_snapshots if by_layer else 1

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]

    # accumulators
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand = torch.zeros((num_nets, num_drops, num_layers))

    num_batches = 0
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        # get high/low/random idx for each snapshot
        # idx_alignment is a list of length num_snapshots
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_alignment, fraction)

            for layer in range(num_layers):
                if by_layer:
                    # we drop only the single snapshot
                    drop_high = [idx_high[layer]]
                    drop_low = [idx_low[layer]]
                    drop_rand = [idx_rand[layer]]
                    drop_layer = [layer]
                else:
                    # we drop all snapshots
                    drop_high = idx_high
                    drop_low = idx_low
                    drop_rand = idx_rand
                    drop_layer = list(range(num_snapshots))

                out_high = [
                    net.forward_targeted_dropout(
                        images, [dr[idx, :] for dr in drop_high], drop_layer
                    )[0]
                    for idx, net in enumerate(nets)
                ]
                out_low = [
                    net.forward_targeted_dropout(
                        images, [dr[idx, :] for dr in drop_low], drop_layer
                    )[0]
                    for idx, net in enumerate(nets)
                ]
                out_rand = [
                    net.forward_targeted_dropout(
                        images, [dr[idx, :] for dr in drop_rand], drop_layer
                    )[0]
                    for idx, net in enumerate(nets)
                ]

                loss_high = [dataset.measure_loss(oh, labels).item() for oh in out_high]
                loss_low = [dataset.measure_loss(ol, labels).item() for ol in out_low]
                loss_rand = [dataset.measure_loss(or_, labels).item() for or_ in out_rand]

                acc_high = [dataset.measure_accuracy(oh, labels) for oh in out_high]
                acc_low = [dataset.measure_accuracy(ol, labels) for ol in out_low]
                acc_rand = [dataset.measure_accuracy(or_, labels) for or_ in out_rand]

                progdrop_loss_high[:, dropidx, layer] += torch.tensor(loss_high)
                progdrop_loss_low[:, dropidx, layer] += torch.tensor(loss_low)
                progdrop_loss_rand[:, dropidx, layer] += torch.tensor(loss_rand)
                progdrop_acc_high[:, dropidx, layer] += torch.tensor(acc_high)
                progdrop_acc_low[:, dropidx, layer] += torch.tensor(acc_low)
                progdrop_acc_rand[:, dropidx, layer] += torch.tensor(acc_rand)

    results = {
        "progdrop_loss_high": progdrop_loss_high / num_batches,
        "progdrop_loss_low": progdrop_loss_low / num_batches,
        "progdrop_loss_rand": progdrop_loss_rand / num_batches,
        "progdrop_acc_high": progdrop_acc_high / num_batches,
        "progdrop_acc_low": progdrop_acc_low / num_batches,
        "progdrop_acc_rand": progdrop_acc_rand / num_batches,
        "dropout_fraction": drop_fraction,
        "by_layer": by_layer,
        "idx_dropout_layers": [ix for ix in range(num_snapshots)],
    }
    return results


def get_dropout_indices(idx_alignment, fraction):
    """
    convenience method for getting a fraction of dropout indices from each layer

    idx_alignment should be a list of the indices of alignment (sorted from lowest to highest)
    where len(idx_alignment)=num_layers_per_network and each element is a tensor such that
    idx_alignment[0].shape=(num_nets, num_nodes_per_layer)

    returns a fraction of indices to drop of highest, lowest, and random alignment
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    idx_high = [idx[:, -drop:] for idx, drop in zip(idx_alignment, num_drop)]
    idx_low = [idx[:, :drop] for idx, drop in zip(idx_alignment, num_drop)]
    idx_rand = [
        torch.stack([torch.randperm(nodes)[:drop] for _ in range(num_nets)], dim=0)
        for nodes, drop in zip(num_nodes, num_drop)
    ]
    return idx_high, idx_low, idx_rand




def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    """
    (unchanged) Perform a progressive dropout experiment,
    dropping nodes (by alignment ranking) in increments.
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
    method for testing network on supervised learning problem with eigenvector dropout
    Instead of dropping nodes, we drop entire eigenvectors
    based on their relative alignment or eigenvalues.
    """
    if not (isinstance(nets, list)):
        nets = [nets]

    n_alignment_idx = nets[0].num_layers()
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = n_alignment_idx if by_layer else 1

    # Example code to build "indices" from eigenvalues, etc.:
    # (This is just an outline consistent with your code style)
    # We'll pretend we want to sort eigenvalues from largest to smallest
    idx_eigenvalue = []
    for netidx in range(num_nets):
        # we assume eigenvectors[netidx] is a list of length n_alignment_idx
        # each entry is a matrix
        # to replicate your "argsort" logic, we do:
        layer_idxs = []
        for layer_evecs in eigenvectors[netidx]:
            dim = layer_evecs.size(1)
            # just an example: reversed order
            layer_idxs.append(torch.arange(dim - 1, -1, -1).unsqueeze(0))
        idx_eigenvalue.append(layer_idxs)
    # you will want to adapt the above logic to your actual data structure.

    # accumulators
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand = torch.zeros((num_nets, num_drops, num_layers))

    num_batches = 0
    use_test = not parameters.get("train_set", True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for dropidx, fraction in enumerate(drop_fraction):
            # Suppose you define a helper for eigen‐drop indices
            # or adapt your existing "get_dropout_indices"
            # ...
            # Then run forward with net.forward_eigenvector_dropout(...)

            # The rest is the same pattern as progressive_dropout:
            pass

    # This function would similarly fill in the results dict
    # just like progressive_dropout does.
    # For brevity, we leave the full logic out here.
    # Return a results dict:
    results = {}
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