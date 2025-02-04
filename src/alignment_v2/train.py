from copy import deepcopy
import torch
from tqdm import tqdm

from alignment_v2.utils import (
    transpose_list,
    condense_values,
    test_nets,
    train_nets,
    save_checkpoint,
    smart_pca,
    expected_alignment_distribution,
)

@train_nets
def train(nets, optimizers, dataset, **parameters):
    """
    method for training network on supervised learning problem
    """

    if not isinstance(nets, list):
        nets = [nets]
    if not isinstance(optimizers, list):
        optimizers = [optimizers]
    assert len(nets) == len(optimizers), "nets and optimizers need to be equal length lists"

    verbose = parameters.get("verbose", True)
    num_nets = len(nets)
    use_train = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    num_steps = len(dataset.train_loader) * parameters["num_epochs"]

    wandb_run = parameters.get("run", None)

    measure_alignment = parameters.get("alignment", True)
    measure_delta_weights = parameters.get("delta_weights", False)
    measure_frequency = parameters.get("frequency", 1)
    compare_expected = parameters.get("compare_expected", False)

    manual_shape = parameters.get("manual_shape", False)
    manual_frequency = parameters.get("manual_frequency", -1)
    manual_transforms = parameters.get("manual_transforms", None)
    manual_layers = parameters.get("manual_layers", None)

    results = parameters.get("results", False)
    num_complete = parameters.get("num_complete", 0)
    save_ckpt, freq_ckpt, path_ckpt, dev = parameters.get("save_checkpoints", (False, 1, "", ""))

    if not results:
        results = {
            "loss": torch.zeros((num_steps, num_nets)),
            "accuracy": torch.zeros((num_steps, num_nets)),
        }
        if measure_alignment:
            results["alignment"] = []
        if measure_delta_weights:
            results["delta_weights"] = []
            results["init_weights"] = [net.get_alignment_weights() for net in nets]

    if num_complete > 0:
        print("resuming training from checkpoint on epoch", num_complete)

    epoch_loop = range(num_complete, parameters["num_epochs"])
    if verbose:
        epoch_loop = tqdm(epoch_loop, desc="training epoch")

    for epoch in epoch_loop:
        batch_loop = dataloader
        if verbose:
            batch_loop = tqdm(batch_loop, desc="minibatch", leave=False)

        for idx, batch in enumerate(batch_loop):
            cidx = epoch * len(dataloader) + idx
            images, labels = dataset.unwrap_batch(batch)

            for opt in optimizers:
                opt.zero_grad()

            outputs = [net(images, store_hidden=True) for net in nets]
            loss = [dataset.measure_loss(output, labels) for output in outputs]
            for l, opt in zip(loss, optimizers):
                l.backward()
                opt.step()

            results["loss"][cidx] = torch.tensor([l.item() for l in loss])
            results["accuracy"][cidx] = torch.tensor([dataset.measure_accuracy(output, labels).cpu() for output in outputs])

            if idx % measure_frequency == 0:
                if measure_alignment:
                    results["alignment"].append([
                        net.measure_alignment(images, precomputed=True, method="alignment")
                        for net in nets
                    ])
                if measure_delta_weights:
                    c_delta_weights = [
                        net.compare_weights(init_weight)
                        for net, init_weight in zip(nets, results["init_weights"])
                    ]
                    results["delta_weights"].append(c_delta_weights)

            if wandb_run is not None:
                wandb_run.log(
                    {f"losses/loss-{ii}": l.item() for ii, l in enumerate(loss)}
                    | {f"accuracies/accuracy-{ii}": dataset.measure_accuracy(output, labels) for ii, output in enumerate(outputs)}
                    | {"batch": cidx}
                )

        if manual_shape:
            if ((epoch + 1) % manual_frequency == 0) and (epoch < parameters["num_epochs"] - 1):
                for net, transform in zip(nets, manual_transforms):
                    inputs, _ = net._process_collect_activity(dataset, train_set=False, with_updates=False, use_training_mode=False)
                    _, eigenvalues, eigenvectors = net.measure_eigenfeatures(inputs, with_updates=False)
                    eigenvalues = [eigenvalues[ml] for ml in manual_layers]
                    eigenvectors = [eigenvectors[ml] for ml in manual_layers]
                    net.shape_eigenfeatures(manual_layers, eigenvalues, eigenvectors, transform)

        if save_ckpt and (epoch % freq_ckpt == 0):
            save_checkpoint(
                nets,
                optimizers,
                results | {"prms": parameters, "epoch": epoch, "device": dev},
                path_ckpt,
            )

    return results

@torch.no_grad()
@test_nets
def test(nets, dataset, **parameters):
    """
    method for testing network on supervised learning problem
    """

    wandb_run = parameters.get("run", None)
    verbose = parameters.get("verbose", True)
    num_nets = len(nets)
    use_test = not parameters.get("train_set", False)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    total_loss = [0 for _ in range(num_nets)]
    total_correct = [0 for _ in range(num_nets)]
    num_batches = 0

    measure_alignment = parameters.get("alignment", True)
    alignment_log = []

    batch_loop = tqdm(dataloader) if verbose else dataloader
    for batch in batch_loop:
        images, labels = dataset.unwrap_batch(batch)

        outputs = [net(images, store_hidden=True) for net in nets]
        for idx, output in enumerate(outputs):
            total_loss[idx] += dataset.measure_loss(output, labels).item()
            total_correct[idx] += dataset.measure_accuracy(output, labels).item()
        num_batches += 1

        if measure_alignment:
            alignment_log.append([
                net.measure_alignment(images, precomputed=True, method="alignment")
                for net in nets
            ])

    results = {
        "loss": [x / num_batches for x in total_loss],
        "accuracy": [c / num_batches for c in total_correct],
    }

    if measure_alignment:
        results["alignment"] = condense_values(transpose_list(alignment_log))

    if wandb_run is not None:
        # ensure it's a wandb run
        if hasattr(wandb_run, "summary"):
            wandb_run.summary["test_loss"] = torch.mean(torch.tensor(results["loss"]))
            wandb_run.summary["test_accuracy"] = torch.mean(torch.tensor(results["accuracy"]))

    return results

@torch.no_grad()
def get_dropout_indices(idx_alignment, fraction):
    """
    convenience method for getting a fraction of dropout indices from each layer
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
    """
    if not (isinstance(nets, list)):
        nets = [nets]

    n_alignment_idx = nets[0].num_layers()

    if alignment is None:
        alignment = test(nets, dataset, **parameters)["alignment"]

    assert len(alignment) == n_alignment_idx, "the number of layers in **alignment** doesn't correspond to the number of alignment layers"

    if nets[0].is_classification_layer_included():
        n_alignment_idx -= 1
        alignment.pop(-1)

    alignment = [torch.mean(align, dim=1) for align in alignment]
    idx_alignment = [torch.argsort(align, dim=1) for align in alignment]

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = n_alignment_idx if by_layer else 1

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
                    drop_high, drop_low, drop_rand = (
                        [idx_high[layer]],
                        [idx_low[layer]],
                        [idx_rand[layer]],
                    )
                    drop_layer = [layer]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = [ix for ix in range(n_alignment_idx)]

                out_high = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_high], drop_layer)[0] for idx, net in enumerate(nets)]
                out_low = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_low], drop_layer)[0] for idx, net in enumerate(nets)]
                out_rand = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_rand], drop_layer)[0] for idx, net in enumerate(nets)]

                loss_high = [dataset.measure_loss(out, labels).item() for out in out_high]
                loss_low = [dataset.measure_loss(out, labels).item() for out in out_low]
                loss_rand = [dataset.measure_loss(out, labels).item() for out in out_rand]

                acc_high = [dataset.measure_accuracy(out, labels) for out in out_high]
                acc_low = [dataset.measure_accuracy(out, labels) for out in out_low]
                acc_rand = [dataset.measure_accuracy(out, labels) for out in out_rand]

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
    """
    if not (isinstance(nets, list)):
        nets = [nets]

    n_alignment_idx = nets[0].num_layers()

    from copy import deepcopy
    assert all(
        [len(ev) == n_alignment_idx for ev in eigenvectors]
    ), "the number of layers in **eigenvectors** doesn't correspond to the number of alignment layers"
    assert all(
        [len(ev) == n_alignment_idx for ev in eigenvalues]
    ), "the number of layers in **eigenvalues** doesn't correspond to the number of alignment layers"

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = n_alignment_idx if by_layer else 1

    idx_eigenvalue = [torch.fliplr(torch.tensor(range(0, ev.size(1))).expand(num_nets, -1)) for ev in eigenvectors[0]]

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
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_eigenvalue, fraction)

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

                loss_high = [dataset.measure_loss(out, labels).item() for out in out_high]
                loss_low = [dataset.measure_loss(out, labels).item() for out in out_low]
                loss_rand = [dataset.measure_loss(out, labels).item() for out in out_rand]

                acc_high = [dataset.measure_accuracy(out, labels) for out in out_high]
                acc_low = [dataset.measure_accuracy(out, labels) for out in out_low]
                acc_rand = [dataset.measure_accuracy(out, labels) for out in out_rand]

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