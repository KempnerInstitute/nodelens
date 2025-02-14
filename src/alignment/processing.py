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


def test_networks(exp, nets, dataset):
    """
    A simple wrapper to test networks when do_train=False.

    Calls train.test(...) behind the scenes with alignment parameters from exp.args.
    """
    do_align = exp.args.alignment.do_alignment
    methods = exp.args.alignment.methods
    freq = exp.args.alignment.frequency

    # build param dict for train.test
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
    Based on the average alignment, we sort the nodes from low-to-high alignment.
    Then we systematically drop top X% or bottom X% or random X% of nodes,
    measuring performance at each fraction.
    """
    if not (isinstance(nets, list)):
        nets = [nets]

    if alignment is None:
        # Use the test() function from train.py to gather alignment
        alignment = train.test(nets, dataset, alignment=True, methods=["RQ"], **parameters)["alignment"]

    # Build alignment_layers so that for each alignment snapshot we have a tensor of shape (num_nets, total_nodes)
    alignment_layers = []
    for layerdata in alignment:
        # layerdata["data"] is a list with one entry per net.
        all_nets_rq = []
        for net_i_data in layerdata["data"]:
            # net_i_data is a list of dicts (one per layer); we flatten all "RQ" values.
            net_nodes = []
            for layer_dict in net_i_data:
                net_nodes.append(layer_dict["RQ"].flatten())
            flattened = torch.cat(net_nodes, dim=0)  # shape: (total_nodes_for_this_snapshot,)
            all_nets_rq.append(flattened)
        stacked = torch.stack(all_nets_rq, dim=0)  # shape: (num_nets, total_nodes)
        alignment_layers.append(stacked)

    idx_alignment = [torch.argsort(al, dim=1) for al in alignment_layers]

    # Number of snapshots determines the number of dropout "layers"
    num_snapshots = len(idx_alignment)
    by_layer = parameters.get("by_layer", False)
    num_layers = num_snapshots if by_layer else 1

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]

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

        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_alignment, fraction)

            for layer in range(num_layers):
                if by_layer:
                    drop_high_use = [idx_high[layer]]
                    drop_low_use = [idx_low[layer]]
                    drop_rand_use = [idx_rand[layer]]
                    drop_layer = [layer]
                else:
                    drop_high_use = idx_high
                    drop_low_use = idx_low
                    drop_rand_use = idx_rand
                    drop_layer = list(range(num_snapshots))

                out_high = [
                    net.forward_targeted_dropout(
                        images, [dr[i_net, :] for dr in drop_high_use], drop_layer
                    )[0]
                    for i_net, net in enumerate(nets)
                ]
                out_low = [
                    net.forward_targeted_dropout(
                        images, [dr[i_net, :] for dr in drop_low_use], drop_layer
                    )[0]
                    for i_net, net in enumerate(nets)
                ]
                out_rand = [
                    net.forward_targeted_dropout(
                        images, [dr[i_net, :] for dr in drop_rand_use], drop_layer
                    )[0]
                    for i_net, net in enumerate(nets)
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
    Perform a progressive dropout experiment,
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

    idx_eigenvalue = []
    for netidx in range(num_nets):
        layer_idxs = []
        for layer_evecs in eigenvectors[netidx]:
            dim = layer_evecs.size(1)
            layer_idxs.append(torch.arange(dim - 1, -1, -1).unsqueeze(0))
        idx_eigenvalue.append(layer_idxs)

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
            pass  # user can implement the logic for dropping eigenvectors

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



def evaluate_pretrained_model(net, dataset):
    """
    Measure test accuracy of a loaded pretrained model on 'dataset.test_loader'.
    'net' should already be on the correct device and in eval mode.
    """
    net.eval()  # ensure we're in eval mode (no dropout, etc.)
    device = next(net.parameters()).device

    total_correct = 0
    total_samples = 0

    # Turn off gradient calculations
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