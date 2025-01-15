from copy import copy, deepcopy

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
    alignment,
    alignment_expansion,
)


@train_nets
def train(nets, optimizers, dataset, **parameters):
    """method for training network on supervised learning problem"""

    # input argument checks
    if not (isinstance(nets, list)):
        nets = [nets]
    if not (isinstance(optimizers, list)):
        optimizers = [optimizers]
    assert len(nets) == len(optimizers), "nets and optimizers need to be equal length lists"

    # check if we should print progress bars
    verbose = parameters.get("verbose", True)

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    use_train = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    num_steps = len(dataset.train_loader) * parameters["num_epochs"]

    # --- optional W&B logging ---
    run = parameters.get("run")

    # --- optional analyses ---
    measure_alignment = parameters.get("alignment", True)
    measure_alignment_expansion = parameters.get("alignment_expansion", False)
    
    measure_delta_weights = parameters.get("delta_weights", False)
    measure_delta_alignment = parameters.get("delta_alignment", False)
    measure_frequency = parameters.get("frequency", 1)
    compare_expected = parameters.get("compare_expected", False)

    # --- optional training method: manual shaping with eigenvectors ---
    manual_shape = parameters.get("manual_shape", False)  # true or False, whether to do this
    manual_frequency = parameters.get("manual_frequency", -1)
    manual_transforms = parameters.get("manual_transforms", None)  # len()==len(nets) callable methods
    manual_layers = parameters.get("manual_layers", None)  # index to which layers

    # --- create results dictionary if not provided and handle checkpoint info ---
    results = parameters.get("results", False)
    num_complete = parameters.get("num_complete", 0)
    save_ckpt, freq_ckpt, path_ckpt, dev = parameters.get("save_checkpoints", (False, 1, "", ""))
    if not results:
        # initialize dictionary for storing performance across epochs
        results = {
            "loss": torch.zeros((num_steps, num_nets)),
            "accuracy": torch.zeros((num_steps, num_nets)),
        }

        # measure alignment throughout training
        if measure_alignment:
            results["alignment"] = []

        if measure_alignment_expansion:
            results["alignment_0"] = []
            results["alignment_1"] = []
            results["alignment_2"] = []
            results["alignment_red"] = []

        # measure weight norm throughout training
        if measure_delta_weights:
            results["delta_weights"] = []
            results["init_weights"] = [net.module.get_alignment_weights() if hasattr(net, "module") else net.get_alignment_weights() for net in nets]

        # measure alignment of weight updates throughout training
        if measure_delta_alignment:
            if not "init_weights" in results:
                results["init_weights"] = [net.module.get_alignment_weights() if hasattr(net, "module") else net.get_alignment_weights() for net in nets]
            results["delta_alignment"] = []

        # compare true alignment distribution to expected distribution (according to Fiete alignment definition)
        if compare_expected:
            calign_bins = torch.linspace(0, 1, 301)
            results["compare_alignment_bins"] = calign_bins
            results["compare_alignment_expected"] = []
            results["compare_alignment_observed"] = []
            if measure_delta_alignment:
                results["compare_delta_alignment_observed"] = []

    elif results["loss"].shape[0] < num_steps:
        add_steps = num_steps - results["loss"].shape[0]
        assert (add_steps / (parameters["num_epochs"] - num_complete)) == len(
            dataset.train_loader
        ), "Number of new steps needs to multiple of epochs and num minibatches"
        results["loss"] = torch.vstack((results["loss"], torch.zeros((add_steps, num_nets))))
        results["accuracy"] = torch.vstack((results["accuracy"], torch.zeros((add_steps, num_nets))))

    if num_complete > 0:
        print("resuming training from checkpoint on epoch", num_complete)

    # --- training loop ---
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
                    alignment_vals = []
                    for net in nets:
                        real_net = net.module if hasattr(net, "module") else net
                        alignment_val = real_net.measure_alignment(images, precomputed=True, method="alignment")
                        alignment_vals.append(alignment_val)
                    results["alignment"].append(alignment_vals)
                    
                if measure_alignment_expansion:
                    # Old lines commented, replaced with the new approach
                    # results["alignment_0"].append([net.measure_alignment_expansion(images, precomputed=True, method="alignment_0") for net in nets])
                    # ...
                    alignment0_vals, alignment1_vals, alignment2_vals, alignmentred_vals = [], [], [], []
                    for net in nets:
                        real_net = net.module if hasattr(net, "module") else net
                        alignment0_vals.append(real_net.measure_alignment_expansion(images, precomputed=True, method="alignment_0"))
                        alignment1_vals.append(real_net.measure_alignment_expansion(images, precomputed=True, method="alignment_1"))
                        alignment2_vals.append(real_net.measure_alignment_expansion(images, precomputed=True, method="alignment_2"))
                        alignmentred_vals.append(real_net.measure_alignment_expansion(images, precomputed=True, method="alignment_red"))
                    results["alignment_0"].append(alignment0_vals)
                    results["alignment_1"].append(alignment1_vals)
                    results["alignment_2"].append(alignment2_vals)
                    results["alignment_red"].append(alignmentred_vals)

                if measure_delta_weights or measure_delta_alignment:
                    c_delta_weights = []
                    for net, init_weight in zip(nets, results["init_weights"]):
                        real_net = net.module if hasattr(net, "module") else net
                        c_delta_weights.append(real_net.compare_weights(init_weight))
                    if measure_delta_weights:
                        results["delta_weights"].append(c_delta_weights)
                    if measure_delta_alignment:
                        c_delta_alignment = []
                        for net, weights_ in zip(nets, c_delta_weights):
                            real_net = net.module if hasattr(net, "module") else net
                            c_delta_alignment.append(
                                real_net.measure_alignment_weights(images, weights_, precomputed=True, method="alignment")
                            )
                        results["delta_alignment"].append(c_delta_alignment)

                if compare_expected:
                    if measure_alignment:
                        c_alignment = results["alignment"][-1]
                    else:
                        c_alignment = []
                        for net in nets:
                            real_net = net.module if hasattr(net, "module") else net
                            c_alignment.append(real_net.measure_alignment(images, precomputed=True, method="alignment"))
                    c_inputs = []
                    for net in nets:
                        real_net = net.module if hasattr(net, "module") else net
                        c_inputs.append(real_net.get_layer_inputs(images, precomputed=True))
                    for i in range(len(nets)):
                        net_ = nets[i]
                        real_net_ = net_.module if hasattr(net_, "module") else net_
                        c_inputs[i] = real_net_._preprocess_inputs(c_inputs[i])
                    c_evals = []
                    for cin in c_inputs:
                        c_eval_sub = []
                        for c in cin:
                            w,_ = smart_pca(c.T)
                            c_eval_sub.append(w)
                        c_evals.append(c_eval_sub)
                    calign_bins = results["compare_alignment_bins"]
                    c_dist = []
                    for c_eval in c_evals:
                        subdist = []
                        for ev in c_eval:
                            subdist.append(expected_alignment_distribution(ev, valid_rotation=False, bins=calign_bins)[0])
                        c_dist.append(subdist)
                    t_dist = []
                    for c_align in c_alignment:
                        align_sub = []
                        for align in c_align:
                            align_sub.append(torch.histogram(align.cpu(), bins=calign_bins, density=True)[0])
                        t_dist.append(align_sub)
                    results["compare_alignment_expected"].append(c_dist)
                    results["compare_alignment_observed"].append(t_dist)
                    if measure_delta_alignment:
                        d_alignment = results["delta_alignment"][-1]
                        d_dist = []
                        for d_align in d_alignment:
                            align_sub = []
                            for dalign in d_align:
                                align_sub.append(torch.histogram(dalign.cpu(), bins=calign_bins, density=True)[0])
                            d_dist.append(align_sub)
                        results["compare_delta_alignment_observed"].append(d_dist)

            if run is not None:
                run.log(
                    {f"losses/loss-{ii}": l.item() for ii, l in enumerate(loss)}
                    | {f"accuracies/accuracy-{ii}": dataset.measure_accuracy(output, labels) for ii, output in enumerate(outputs)}
                    | {"batch": cidx}
                )

        if manual_shape:
            if ((epoch + 1) % manual_frequency == 0) and (epoch < parameters["num_epochs"] - 1):
                for net, transform in tqdm(zip(nets, manual_transforms), desc="manual shaping", leave=False):
                    real_net = net.module if hasattr(net, "module") else net
                    inputs, _ = real_net._process_collect_activity(dataset, train_set=False, with_updates=False, use_training_mode=False)
                    _, eigenvalues, eigenvectors = real_net.measure_eigenfeatures(inputs, with_updates=False)
                    idx_to_layer_lookup = {layer: idx for idx, layer in enumerate(real_net.get_alignment_layer_indices())}
                    eigenvalues = [eigenvalues[idx_to_layer_lookup[ml]] for ml in manual_layers]
                    eigenvectors = [eigenvectors[idx_to_layer_lookup[ml]] for ml in manual_layers]
                    real_net.shape_eigenfeatures(manual_layers, eigenvalues, eigenvectors, transform)

        if save_ckpt & (epoch % freq_ckpt == 0):
            save_checkpoint(
                nets,
                optimizers,
                results | {"prms": parameters, "epoch": epoch, "device": dev},
                path_ckpt,
            )

    for k in [
        "alignment",
        "alignment_0",
        "alignment_1",
        "alignment_2",
        "alignment_red",
        "delta_weights",
        "delta_alignment",
        "avgcorr",
        "fullcorr",
        "compare_alignment_expected",
        "compare_alignment_observed",
        "compare_delta_alignment_observed",
    ]:
        if k not in results.keys():
            continue
        results[k] = condense_values(transpose_list(results[k]))

    return results


@torch.no_grad()
@test_nets
def test(nets, dataset, **parameters):
    """method for testing network on supervised learning problem"""

    run = parameters.get("run")

    if not (isinstance(nets, list)):
        nets = [nets]

    verbose = parameters.get("verbose", True)
    num_nets = len(nets)

    use_test = not parameters.get("train_set", False)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    total_loss = [0 for _ in range(num_nets)]
    num_correct = [0 for _ in range(num_nets)]
    num_batches = 0

    measure_alignment = parameters.get("alignment", True)
    measure_alignment_expansion = parameters.get("alignment_expansion", True)

    if measure_alignment:
        alignment = []
    if measure_alignment_expansion:
        alignment_0 = []
        alignment_1 = []
        alignment_2 = []
        alignment_red = []

    batch_loop = tqdm(dataloader) if verbose else dataloader
    for batch in batch_loop:
        images, labels = dataset.unwrap_batch(batch)

        outputs = [net(images, store_hidden=True) for net in nets]

        for idx, output in enumerate(outputs):
            total_loss[idx] += dataset.measure_loss(output, labels).item()
            num_correct[idx] += dataset.measure_accuracy(output, labels).item()

        num_batches += 1

        if measure_alignment:
            a_vals = []
            for net in nets:
                real_net = net.module if hasattr(net, "module") else net
                a_vals.append(real_net.measure_alignment(images, precomputed=True, method="alignment"))
            alignment.append(a_vals)

        if measure_alignment_expansion:
            a0_vals, a1_vals, a2_vals, ared_vals = [], [], [], []
            for net in nets:
                real_net = net.module if hasattr(net, "module") else net
                a0_vals.append(real_net.measure_alignment_expansion(images, precomputed=True, method="alignment_0"))
                a1_vals.append(real_net.measure_alignment_expansion(images, precomputed=True, method="alignment_1"))
                a2_vals.append(real_net.measure_alignment_expansion(images, precomputed=True, method="alignment_2"))
                ared_vals.append(real_net.measure_alignment_expansion(images, precomputed=True, method="alignment_red"))
            alignment_0.append(a0_vals)
            alignment_1.append(a1_vals)
            alignment_2.append(a2_vals)
            alignment_red.append(ared_vals)
    
    results = {
        "loss": [loss / num_batches for loss in total_loss],
        "accuracy": [correct / num_batches for correct in num_correct],
    }

    if measure_alignment:
        results["alignment"] = condense_values(transpose_list(alignment))
    if measure_alignment_expansion:
        results["alignment_0"] = condense_values(transpose_list(alignment_0))
        results["alignment_1"] = condense_values(transpose_list(alignment_1))
        results["alignment_2"] = condense_values(transpose_list(alignment_2))
        results["alignment_red"] = condense_values(transpose_list(alignment_red))

    if run is not None:
        run.summary["test_loss"] = torch.mean(torch.tensor(results["loss"]))
        run.summary["test_accuracy"] = torch.mean(torch.tensor(results["accuracy"]))

    return results

@torch.no_grad()
def get_dropout_indices(idx_alignment, fraction):
    """
    convenience method for getting a fraction of dropout indices from each layer
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    idx_high = [torch.sort(idx[:, -drop:], dim=1).values for idx, drop in zip(idx_alignment, num_drop)]
    idx_low = [torch.sort(idx[:, :drop], dim=1).values for idx, drop in zip(idx_alignment, num_drop)]
    idx_rand = [torch.sort(idx[:, torch.randperm(idx.size(1))[:drop]],dim=1).values for idx, drop in zip(idx_alignment, num_drop)]
    return idx_high, idx_low, idx_rand

@torch.no_grad()
def get_dropout_indices000(idx_alignment, fraction):
    """
    Returns dropout indices for high, low, and random alignment for each layer and each network.
    """
    num_layers = len(idx_alignment)
    num_nets = idx_alignment[0].size(0)

    idx_high = []
    idx_low = []
    idx_rand = []

    for layer_idx in range(num_layers):
        layer_alignment = idx_alignment[layer_idx]
        layer_dimension = layer_alignment.size(1)
        num_drop = int(layer_dimension * fraction)
        sorted_indices = torch.argsort(layer_alignment, dim=1)
        high_indices = sorted_indices[:, -num_drop:]
        high_indices = torch.sort(high_indices, dim=1).values
        idx_high.append(high_indices)
        low_indices = sorted_indices[:, :num_drop]
        low_indices = torch.sort(low_indices, dim=1).values
        idx_low.append(low_indices)
        rand_indices = torch.randperm(layer_dimension, device=layer_alignment.device)[:num_drop]
        rand_indices = rand_indices.repeat(num_nets, 1)
        idx_rand.append(rand_indices)

    return idx_high, idx_low, idx_rand

@torch.no_grad()
@test_nets
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    method for testing network on supervised learning problem with progressive dropout
    """

    if not (isinstance(nets, list)):
        nets = [nets]

    real_net = nets[0].module if hasattr(nets[0], 'module') else nets[0]
    idx_dropout_layers = real_net.get_alignment_layer_indices()

    if alignment is None:
        alignment = test(nets, dataset, **parameters)["alignment"]

    alignment = [alignment[i] for i in idx_dropout_layers]
    assert len(alignment) == len(idx_dropout_layers), "the number of layers in **alignment** doesn't correspond to the number of alignment layers"

    classification_layer = nets[0].module.num_layers(all=True) - 1 if hasattr(nets[0], 'module') else nets[0].num_layers(all=True) - 1
    if classification_layer in idx_dropout_layers:
        idx_dropout_layers.pop(-1)
        alignment.pop(-1)

    alignment = [torch.mean(align, dim=1) for align in alignment]
    idx_alignment = [torch.argsort(align, dim=1) for align in alignment]

    print(len(idx_alignment))
    print(idx_alignment[0].shape)
    
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = len(idx_dropout_layers) if by_layer else 1

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
                    drop_layer = [idx_dropout_layers[layer]]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = copy(idx_dropout_layers)

                out_high = []
                for idx, net_ in enumerate(nets):
                    real_net_ = net_.module if hasattr(net_, "module") else net_
                    out_ = real_net_.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_high], drop_layer)[0]
                    out_high.append(out_)
                out_low = []
                for idx, net_ in enumerate(nets):
                    real_net_ = net_.module if hasattr(net_, "module") else net_
                    out_ = real_net_.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_low], drop_layer)[0]
                    out_low.append(out_)
                out_rand = []
                for idx, net_ in enumerate(nets):
                    real_net_ = net_.module if hasattr(net_, "module") else net_
                    out_ = real_net_.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_rand], drop_layer)[0]
                    out_rand.append(out_)

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
        "idx_dropout_layers": idx_dropout_layers,
    }

    return results


@torch.no_grad()
def prune_network_by_alignment(nets, idx_alignment, fraction, method="high"):
    """
    Prune the network by alignment method ('high', 'low', or 'random') 
    based on the alignment values and fraction provided.
    """
    pruned_nets = []
    num_nets = len(nets)

    for net_idx, net in enumerate(nets):
        pruned_net = type(net)()
        pruned_net.load_state_dict(net.state_dict())

        for layer_idx, idx in enumerate(idx_alignment):
            num_nodes = idx.size(1)
            num_drop = min(int(num_nodes * fraction), num_nodes)
            if method == "high":
                drop_idx = idx[:, -num_drop:]
            elif method == "low":
                drop_idx = idx[:, :num_drop]
            elif method == "random":
                drop_idx = torch.stack([torch.randperm(num_nodes)[:num_drop] for _ in range(idx.size(0))], dim=0)
            else:
                raise ValueError("Unknown pruning method: Choose 'high', 'low', or 'random'.")
            drop_idx = drop_idx.clamp(0, num_nodes - 1)
            for layer in pruned_net.modules():
                if hasattr(layer, 'weight'):
                    drop_idx = drop_idx[drop_idx < layer.weight.data.size(1)]
                    layer.weight.data[:, drop_idx] = 0
                    pruned_nets.append(pruned_net)

    return pruned_nets


def train_network(net, dataset, epochs=5, learning_rate=0.001):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    
    for epoch in range(epochs):
        for batch in dataset.train_loader:
            images, labels = dataset.unwrap_batch(batch)
            optimizer.zero_grad()
            outputs = net(images)
            loss = dataset.measure_loss(outputs, labels)
            loss.backward()
            optimizer.step()
    
    return net


def progressive_dropout_train(nets, dataset, alignment=None, **parameters):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(nets, list):
        nets = [nets]
    nets = [net.to(device) for net in nets]

    real_net_0 = nets[0].module if hasattr(nets[0], 'module') else nets[0]
    idx_dropout_layers = real_net_0.get_alignment_layer_indices()
    
    if alignment is None:
        alignment = test(nets, dataset, **parameters)["alignment"]

    assert len(alignment) == len(idx_dropout_layers), \
        f"Mismatch in alignment layer count: alignment has {len(alignment)} layers, expected {len(idx_dropout_layers)} layers."

    classification_layer = real_net_0.num_layers(all=True) - 1
    if classification_layer in idx_dropout_layers:
        idx_dropout_layers.pop(-1)
        alignment.pop(-1)

    alignment = [torch.mean(align, dim=1) for align in alignment]
    idx_alignment = [torch.argsort(align, dim=1) for align in alignment]

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = len(idx_dropout_layers) if by_layer else 1

    progdrop_loss_high_before, progdrop_loss_low_before, progdrop_loss_rand_before = [], [], []
    progdrop_acc_high_before, progdrop_acc_low_before, progdrop_acc_rand_before = [], [], []

    progdrop_loss_high_after, progdrop_loss_low_after, progdrop_loss_rand_after = [], [], []
    progdrop_acc_high_after, progdrop_acc_low_after, progdrop_acc_rand_after = [], [], []

    num_batches = 0
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        images = images.to(device)
        labels = labels.to(device)

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
                    drop_layer = [idx_dropout_layers[layer]]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = deepcopy(idx_dropout_layers)

                pruned_nets_high = [train_network(prune_network_by_alignment([net], idx_alignment, fraction, method="high")[0].to(device), dataset) for net in nets]
                pruned_nets_low = [train_network(prune_network_by_alignment([net], idx_alignment, fraction, method="low")[0].to(device), dataset) for net in nets]
                pruned_nets_rand = [train_network(prune_network_by_alignment([net], idx_alignment, fraction, method="random")[0].to(device), dataset) for net in nets]

                out_high_before = [net_(images) for net_ in pruned_nets_high]
                out_low_before = [net_(images) for net_ in pruned_nets_low]
                out_rand_before = [net_(images) for net_ in pruned_nets_rand]

                loss_high_before = [dataset.measure_loss(out, labels).item() for out in out_high_before]
                loss_low_before = [dataset.measure_loss(out, labels).item() for out in out_low_before]
                loss_rand_before = [dataset.measure_loss(out, labels).item() for out in out_rand_before]

                acc_high_before = [dataset.measure_accuracy(out, labels) for out in out_high_before]
                acc_low_before = [dataset.measure_accuracy(out, labels) for out in out_low_before]
                acc_rand_before = [dataset.measure_accuracy(out, labels) for out in out_rand_before]

                progdrop_loss_high_before.append(loss_high_before)
                progdrop_loss_low_before.append(loss_low_before)
                progdrop_loss_rand_before.append(loss_rand_before)
                progdrop_acc_high_before.append(acc_high_before)
                progdrop_acc_low_before.append(acc_low_before)
                progdrop_acc_rand_before.append(acc_rand_before)

                out_high_after = [net_(images) for net_ in pruned_nets_high]
                out_low_after = [net_(images) for net_ in pruned_nets_low]
                out_rand_after = [net_(images) for net_ in pruned_nets_rand]

                loss_high_after = [dataset.measure_loss(out, labels).item() for out in out_high_after]
                loss_low_after = [dataset.measure_loss(out, labels).item() for out in out_low_after]
                loss_rand_after = [dataset.measure_loss(out, labels).item() for out in out_rand_after]

                acc_high_after = [dataset.measure_accuracy(out, labels) for out in out_high_after]
                acc_low_after = [dataset.measure_accuracy(out, labels) for out in out_low_after]
                acc_rand_after = [dataset.measure_accuracy(out, labels) for out in out_rand_after]

                progdrop_loss_high_after.append(loss_high_after)
                progdrop_loss_low_after.append(loss_low_after)
                progdrop_loss_rand_after.append(loss_rand_after)
                progdrop_acc_high_after.append(acc_high_after)
                progdrop_acc_low_after.append(acc_low_after)
                progdrop_acc_rand_after.append(acc_rand_after)

    results = {
        "progdrop_loss_high_before": torch.tensor(progdrop_loss_high_before) / num_batches,
        "progdrop_loss_low_before": torch.tensor(progdrop_loss_low_before) / num_batches,
        "progdrop_loss_rand_before": torch.tensor(progdrop_loss_rand_before) / num_batches,
        "progdrop_acc_high_before": torch.tensor(progdrop_acc_high_before) / num_batches,
        "progdrop_acc_low_before": torch.tensor(progdrop_acc_low_before) / num_batches,
        "progdrop_acc_rand_before": torch.tensor(progdrop_acc_rand_before) / num_batches,
        "progdrop_loss_high_after": torch.tensor(progdrop_loss_high_after) / num_batches,
        "progdrop_loss_low_after": torch.tensor(progdrop_loss_low_after) / num_batches,
        "progdrop_loss_rand_after": torch.tensor(progdrop_loss_rand_after) / num_batches,
        "progdrop_acc_high_after": torch.tensor(progdrop_acc_high_after) / num_batches,
        "progdrop_acc_low_after": torch.tensor(progdrop_acc_low_after) / num_batches,
        "progdrop_acc_rand_after": torch.tensor(progdrop_acc_rand_after) / num_batches,
        "dropout_fraction": drop_fraction,
        "by_layer": by_layer,
        "idx_dropout_layers": idx_dropout_layers,
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

    idx_dropout_layers = nets[0].module.get_alignment_layer_indices() if hasattr(nets[0], 'module') else nets[0].get_alignment_layer_indices()

    assert all([len(ev) == len(idx_dropout_layers) for ev in eigenvectors]), "the number of layers in **eigenvectors** doesn't correspond to the number of alignment layers"
    assert all([len(ev) == len(idx_dropout_layers) for ev in eigenvalues]), "the number of layers in **eigenvalues** doesn't correspond to the number of alignment layers"

    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = len(idx_dropout_layers) if by_layer else 1

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
                    drop_layer = [idx_dropout_layers[layer]]
                    drop_evals = [[ev[layer]] for ev in eigenvalues]
                    drop_evecs = [[evc[layer]] for evc in eigenvectors]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = copy(idx_dropout_layers)
                    drop_evals = deepcopy(eigenvalues)
                    drop_evecs = deepcopy(eigenvectors)

                out_high = []
                for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs)):
                    real_net = net.module if hasattr(net, 'module') else net
                    out_ = real_net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_high], drop_layer)[0]
                    out_high.append(out_)
                out_low = []
                for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs)):
                    real_net = net.module if hasattr(net, 'module') else net
                    out_ = real_net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_low], drop_layer)[0]
                    out_low.append(out_)
                out_rand = []
                for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs)):
                    real_net = net.module if hasattr(net, 'module') else net
                    out_ = real_net.forward_eigenvector_dropout(images, evals, evecs, [drop[idx, :] for drop in drop_rand], drop_layer)[0]
                    out_rand.append(out_)

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
        "idx_dropout_layers": idx_dropout_layers,
    }

    return results