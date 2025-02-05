import os
import torch
from tqdm import tqdm

from alignment_v2 import train
from alignment_v2.utils import load_checkpoints, test_nets, transpose_list, fgsm_attack

def train_networks(exp, nets, optimizers, dataset, **special_parameters):
    """train and test networks with new approach"""
    do_alignment_train = exp.args.alignment.compute_during_training
    methods = exp.args.alignment.methods
    measure_freq = exp.args.alignment.frequency

    params = dict(
        train_set=True,
        num_epochs=exp.args.training.epochs,
        alignment=do_alignment_train,
        methods=methods,
        frequency=measure_freq,
        measure_weight_deltas=exp.args.alignment.measure_weight_deltas,
        run=exp.wandb_run,
    )
    params.update(**special_parameters)

    # handle checkpoints
    if exp.args.checkpointing.use_prev and os.path.isfile(exp.get_checkpoint_path()):
        nets, optimizers, results = load_checkpoints(nets, optimizers, exp.args.device, exp.get_checkpoint_path())
        for net in nets:
            net.train()
        params["num_complete"] = results["epoch"] + 1
        params["results"] = results
        print("loaded networks from previous checkpoint")

    if exp.args.checkpointing.save_checkpoints:
        params["save_checkpoints"] = (True, exp.args.checkpointing.frequency, exp.get_checkpoint_path(), exp.args.device)

    print("training networks...")
    train_results = train.train(nets, optimizers, dataset, **params)

    # do testing loop if user wants alignment during inference or just final test
    do_alignment_infer = exp.args.alignment.compute_during_inference
    params["train_set"] = False
    params["alignment"] = do_alignment_infer

    print("testing networks (inference)...")
    test_results = train.test(nets, dataset, **params)

    return train_results, test_results

def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    print("performing targeted dropout...")
    dropout_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
    )
    dropout_results = progressive_dropout(nets, dataset, alignment=alignment, **dropout_params)
    return dropout_results, dropout_params

def measure_eigenfeatures(exp, nets, dataset, train_set=False):
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

def eigenvector_dropout(exp, nets, dataset, eigen_results, train_set=False):
    print("performing targeted eigenvector dropout...")
    evec_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
    )
    evec_dropout_results = train.eigenvector_dropout(
        nets,
        dataset,
        eigen_results["eigvals"],
        eigen_results["eigvecs"],
        **evec_params
    )
    return evec_dropout_results, evec_params

@torch.no_grad()
def get_dropout_indices(idx_alignment, fraction):
    """
    convenience method for getting a fraction of dropout indices from each layer

    idx_alignment should be a list of the indices of alignment (sorted from lowest to highest)
    where len(idx_alignment)=num_layers_per_network and each element is a tensor such that
    idx_alignment[0].shape=(num_nodes_per_layer, num_networks)

    returns a fraction of indices to drop of highest, lowest, and random alignment
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    idx_high = [idx[:, -drop:] for idx, drop in zip(idx_alignment, num_drop)]
    idx_low = [idx[:, :drop] for idx, drop in zip(idx_alignment, num_drop)]
    idx_rand = [torch.stack([torch.randperm(nodes)[:drop] for _ in range(num_nets)], dim=0) for nodes, drop in zip(num_nodes, num_drop)]
    return idx_high, idx_low, idx_rand

@torch.no_grad()
@test_nets
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    method for testing network on supervised learning problem with progressive dropout

    takes as input a list of networks (usually trained) and a dataset, along with some
    experiment parameters (although there are defaults coded into the method)

    note that this only works when each network in the list has the same architecture!
    to analyze a group of networks with different architectures, run this function multiple
    times and concatenate the results.

    alignment, when provided is a list of the alignment measurement for each layer of
    each network. It is expected that each alignment list has the structure:
    len(alignment)=num_alignment_layers
    alignment[i].shape = (num_networks, num_batches, num_dimensions_per_layer)
    : see how the outputs of ``measure_alignment`` is handled in the ``train`` and ``test``
    functions of this module to understand how to structure it in that format.

    will measure the loss and accuracy on the dataset using targeted dropout, where
    the method will progressively dropout more and more nodes based on the highest, lowest,
    or random alignment. Can either do it for each layer separately or all togehter using
    the parameters['by_layer'] kwarg.
    """

    # input argument check
    if not (isinstance(nets, list)):
        nets = [nets]

    # get the number of alignment layers
    n_alignment_idx = nets[0].num_layers()

    # get alignment of networks if not provided
    if alignment is None:
        alignment = test(nets, dataset, **parameters)["alignment"]

    # check if alignment has the right length (ie number of layers) (otherwise can't make assumptions about where the classification layer is)
    assert len(alignment) == n_alignment_idx, "the number of layers in **alignment** doesn't correspond to the number of alignment layers"

    # don't dropout classification layer if included as an alignment layer
    if nets[0].is_classification_layer_included():
        n_alignment_idx -= 1
        alignment.pop(-1)

    # get average alignment (across batches) and index of average alignment by node
    alignment = [torch.mean(align, dim=1) for align in alignment]
    idx_alignment = [torch.argsort(align, dim=1) for align in alignment]

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = n_alignment_idx if by_layer else 1

    # preallocate tracker tensors
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand = torch.zeros((num_nets, num_drops, num_layers))

    # to keep track of how many values have been added
    num_batches = 0

    # retrieve requested dataloader from dataset
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    # let dataloader be outer loop to minimize extract / load / transform time
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        # get dropout indices for this fraction of dropouts
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_alignment, fraction)

            # do drop out for each layer (or across all depending on parameters)
            for layer in range(num_layers):
                if by_layer:
                    drop_high, drop_low, drop_rand = (
                        [idx_high[layer]],
                        [idx_low[layer]],
                        [idx_rand[layer]],
                    )
                    drop_layer = [layer]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = [ix for ix in range(n_alignment_idx)]

                # get output with targeted dropout
                out_high = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_high], drop_layer)[0] for idx, net in enumerate(nets)]
                out_low = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_low], drop_layer)[0] for idx, net in enumerate(nets)]
                out_rand = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_rand], drop_layer)[0] for idx, net in enumerate(nets)]

                # get loss with targeted dropout
                loss_high = [dataset.measure_loss(out, labels).item() for out in out_high]
                loss_low = [dataset.measure_loss(out, labels).item() for out in out_low]
                loss_rand = [dataset.measure_loss(out, labels).item() for out in out_rand]

                # get accuracy with targeted dropout
                acc_high = [dataset.measure_accuracy(out, labels) for out in out_high]
                acc_low = [dataset.measure_accuracy(out, labels) for out in out_low]
                acc_rand = [dataset.measure_accuracy(out, labels) for out in out_rand]

                # add to storage tensors
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
        "idx_dropout_layers": [ix for ix in range(n_alignment_idx)],
    }

    return results


@torch.no_grad()
@test_nets
def eigenvector_dropout(nets, dataset, eigenvalues, eigenvectors, **parameters):
    """
    method for testing network on supervised learning problem with eigenvector dropout

    takes as input a list of networks (usually trained) and a dataset, along with the
    eigenvectors of the input to each alignment layer along with some experiment
    parameters (although there are defaults coded into the method)

    note that this only works when each network in the list has the same architecture!
    to analyze a group of networks with different architectures, run this function multiple
    times and concatenate the results.

    eigenvectors must have the following structure:
    a list of lists of eigenvectors to each layer for each network such that:
    len(eigenvectors) = num_networks
    len(eigenvectors[i]) = num_alignment_layers for all i
    eigenvectors[i][j].shape = (num_dim_input_to_j, num_dim_input_to_j)

    eigenvalues should have same structure except have vectors instead of square matrices

    will measure the loss and accuracy on the dataset using targeted dropout, where
    the method will progressively dropout more and more eigenvectors based on the highest,
    lowest, or random eigenvalues. Can either do it for each layer separately or all
    together using the parameters['by_layer'] kwarg.
    """

    # input argument check
    if not (isinstance(nets, list)):
        nets = [nets]

    # get index to each alignment layer
    n_alignment_idx = nets[0].num_layers()

    # check if alignment has the right length (ie number of layers) (otherwise can't make assumptions about where the classification layer is)
    assert all(
        [len(ev) == n_alignment_idx for ev in eigenvectors]
    ), "the number of layers in **eigenvectors** doesn't correspond to the number of alignment layers"
    assert all(
        [len(ev) == n_alignment_idx for ev in eigenvalues]
    ), "the number of layers in **eigenvalues** doesn't correspond to the number of alignment layers"

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = n_alignment_idx if by_layer else 1

    # create index of eigenvalue for compatibility with get_dropout_indices
    idx_eigenvalue = [torch.fliplr(torch.tensor(range(0, ev.size(1))).expand(num_nets, -1)) for ev in eigenvectors[0]]

    # preallocate tracker tensors
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand = torch.zeros((num_nets, num_drops, num_layers))

    # to keep track of how many values have been added
    num_batches = 0

    # retrieve requested dataloader from dataset
    use_test = not parameters.get("train_set", True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # let dataloader be outer loop to minimize extract / load / transform time
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        # get dropout indices for this fraction of dropouts
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_eigenvalue, fraction)

            # do drop out for each layer (or across all depending on parameters)
            for layer in range(num_layers):
                if by_layer:
                    drop_high, drop_low, drop_rand = (
                        [idx_high[layer]],
                        [idx_low[layer]],
                        [idx_rand[layer]],
                    )
                    drop_layer = [layer]
                    drop_evals = [[evals[layer]] for evals in eigenvalues]
                    drop_evecs = [[evecs[layer]] for evecs in eigenvectors]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = [ix for ix in range(n_alignment_idx)]
                    drop_evals = deepcopy(eigenvalues)
                    drop_evecs = deepcopy(eigenvectors)

                # get output with targeted dropout
                out_high = [
                    net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_high], drop_layer)[0]
                    for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs))
                ]

                out_low = [
                    net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_low], drop_layer)[0]
                    for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs))
                ]

                out_rand = [
                    net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_rand], drop_layer)[0]
                    for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs))
                ]

                # get loss with targeted dropout
                loss_high = [dataset.measure_loss(out, labels).item() for out in out_high]
                loss_low = [dataset.measure_loss(out, labels).item() for out in out_low]
                loss_rand = [dataset.measure_loss(out, labels).item() for out in out_rand]

                # get accuracy with targeted dropout
                acc_high = [dataset.measure_accuracy(out, labels) for out in out_high]
                acc_low = [dataset.measure_accuracy(out, labels) for out in out_low]
                acc_rand = [dataset.measure_accuracy(out, labels) for out in out_rand]

                # add to storage tensors
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
        "idx_dropout_layers": [ix for ix in range(n_alignment_idx)],
    }

    return results

