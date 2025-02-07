# --------------------------------------------
# train.py
# --------------------------------------------

import torch
from tqdm import tqdm
from copy import deepcopy

from alignment.utils import train_nets, test_nets, save_checkpoint
from alignment.alignment_metrics import AlignmentMetrics

@train_nets
def train(nets, optimizers, dataset, **parameters):
    """
    A single function for supervised training:

      - If 'alignment' in parameters is True, measure alignment 
        for each method, including 'delta_alignment'.
      - Build histograms from observed alignment
      - Build random distribution from PCA for each method

    This returns the updated 'results' dictionary.
    """
    do_align = parameters.get("alignment", False)
    methods = parameters.get("methods", ["RQ"])
    freq = parameters.get("frequency", 1)
    measure_expected = parameters.get("measure_expected", True)
    bins = parameters.get("bins", 50)

    num_epochs = parameters["num_epochs"]
    results = parameters.get("results", {})
    if "alignment" not in results:
        results["alignment"] = []
    if "alignment_distribution" not in results:
        results["alignment_distribution"] = []
    if "expected_distribution" not in results:
        results["expected_distribution"] = []

    save_ckpt_info = parameters.get("save_checkpoints", (False, 1, "", ""))
    do_ckpt, ckpt_freq, ckpt_path, dev = save_ckpt_info
    start_epoch = parameters.get("num_complete", 0)

    if not results:
        # if no dictionary was provided, create one
        results = dict(
            alignment=[],
            alignment_distribution=[],
            expected_distribution=[],
        )

    # retrieve train loader
    use_train = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    verbose = parameters.get("verbose", True)
    if verbose:
        print(f"Starting training loop with epochs={num_epochs}, do_align={do_align}, methods={methods}")

    for epoch in range(start_epoch, num_epochs):
        loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False) if verbose else dataloader
        for batch_idx, batch in enumerate(loop):
            images, labels = dataset.unwrap_batch(batch)
            # zero grads / step
            for net, opt in zip(nets, optimizers):
                opt.zero_grad()
                out = net(images, store_hidden=True)
                loss_val = dataset.measure_loss(out, labels)
                loss_val.backward()
                opt.step()

            if do_align and (batch_idx % freq == 0):
                # measure raw alignment for each net, each layer, each method
                align_data = []
                for net in nets:
                    layer_metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
                    align_data.append(layer_metrics)

                # store raw alignment
                results["alignment"].append({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "data": align_data
                })

                # build histogram from raw alignment
                dist_data = []
                for net_layer_list in align_data:
                    layer_dists = []
                    for layer_dict in net_layer_list:
                        method_dists = {}
                        for m, val_tensor in layer_dict.items():
                            val_cpu = val_tensor.detach().cpu()
                            c, e = torch.histogram(val_cpu, bins=bins, density=True)
                            method_dists[m] = (c, e)
                        layer_dists.append(method_dists)
                    dist_data.append(layer_dists)
                results["alignment_distribution"].append({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "data": dist_data
                })

                # measure random distribution if measure_expected
                if measure_expected:
                    exp_data = []
                    for net in nets:
                        layer_inps = net.get_layer_inputs(images, precomputed=True)
                        layer_exp_list = []
                        for inp in layer_inps:
                            w, _ = AlignmentMetrics.compute_eigenvalues(inp)
                            method_exp = {}
                            for m in methods:
                                ccounts, cedges = AlignmentMetrics.measure_expected_distribution(m, w, bins=bins)
                                method_exp[m] = (ccounts, cedges)
                            layer_exp_list.append(method_exp)
                        exp_data.append(layer_exp_list)
                    results["expected_distribution"].append({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "data": exp_data
                    })

        # checkpoint
        if do_ckpt and (epoch % ckpt_freq == 0):
            cpy_res = deepcopy(results)
            cpy_res["epoch"] = epoch
            cpy_res["device"] = dev
            cpy_res["prms"] = parameters
            save_checkpoint(nets, optimizers, cpy_res, ckpt_path)

    return results

@test_nets
@torch.no_grad()
def test(nets, dataset, **parameters):
    """
    A single function for a test pass / evaluation:

      - If 'alignment' in parameters is True, measure alignment 
        for each method, including 'delta_alignment'.
      - Build histograms for observed alignment
      - Build random distribution from PCA for each method

    Returns updated 'results' dictionary.
    """
    do_align = parameters.get("alignment", False)
    methods = parameters.get("methods", ["RQ"])
    freq = parameters.get("frequency", 1)
    measure_expected = parameters.get("measure_expected", True)
    bins = parameters.get("bins", 50)

    results = parameters.get("results", {})
    if "alignment" not in results:
        results["alignment"] = []
    if "alignment_distribution" not in results:
        results["alignment_distribution"] = []
    if "expected_distribution" not in results:
        results["expected_distribution"] = []

    verbose = parameters.get("verbose", True)

    # use test loader by default (or train_loader if train_set param is set)
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    if verbose:
        dataloader = tqdm(dataloader, desc="Testing")

    batch_idx = 0
    for batch in dataloader:
        images, labels = dataset.unwrap_batch(batch)

        # forward pass
        for net in nets:
            out = net(images, store_hidden=True)

        if do_align and (batch_idx % freq == 0):
            align_data = []
            for net in nets:
                layer_metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
                align_data.append(layer_metrics)

            results["alignment"].append({
                "test_batch": batch_idx,
                "data": align_data
            })

            # build observed distribution
            dist_data = []
            for net_layer_list in align_data:
                layer_dists = []
                for layer_dict in net_layer_list:
                    method_dists = {}
                    for m, val_tensor in layer_dict.items():
                        val_cpu = val_tensor.detach().cpu()
                        c, e = torch.histogram(val_cpu, bins=bins, density=True)
                        method_dists[m] = (c, e)
                    layer_dists.append(method_dists)
                dist_data.append(layer_dists)

            results["alignment_distribution"].append({
                "test_batch": batch_idx,
                "data": dist_data
            })

            # measure random distribution if measure_expected
            if measure_expected:
                exp_data = []
                for net in nets:
                    layer_inps = net.get_layer_inputs(images, precomputed=True)
                    layer_exp_list = []
                    for inp in layer_inps:
                        w, _ = AlignmentMetrics.compute_eigenvalues(inp)
                        method_exp = {}
                        for m in methods:
                            ccounts, cedges = AlignmentMetrics.measure_expected_distribution(m, w, bins=bins)
                            method_exp[m] = (ccounts, cedges)
                        layer_exp_list.append(method_exp)
                    exp_data.append(layer_exp_list)

                results["expected_distribution"].append({
                    "test_batch": batch_idx,
                    "data": exp_data
                })

        batch_idx += 1

    return results


# ------------------------
# [UNCHANGED from older code]
# progressive_dropout_experiment, measure_eigenfeatures, eigenvector_dropout
# plus the get_dropout_indices function
# ------------------------

def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    """
    (unchanged)
    Perform a progressive dropout experiment, dropping nodes (by alignment ranking) progressively
    in increments, then measuring accuracy/loss after each fraction of dropout.
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
    (unchanged)
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
    """
    (unchanged)
    Similar to progressive_dropout_experiment but uses 
    eigenvectors + eigenvalues to rank which components to drop.
    """
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

# [UNCHANGED, important function for progressive_dropout: get_dropout_indices]
@torch.no_grad()
def get_dropout_indices(idx_alignment, fraction):
    """
    (unchanged)
    convenience method for getting a fraction of dropout indices from each layer

    idx_alignment should be a list of the indices of alignment (sorted from lowest to highest)
    where len(idx_alignment)=num_layers_per_network and each element is a tensor such that
    idx_alignment[0].shape=(num_nets, num_nodes_per_layer)

    returns a fraction of indices to drop of highest, lowest, and random alignment

    This is used by progressive_dropout(nets, dataset, alignment=..., ...)
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    idx_high = [idx[:, -drop:] for idx, drop in zip(idx_alignment, num_drop)]
    idx_low = [idx[:, :drop] for idx, drop in zip(idx_alignment, num_drop)]
    idx_rand = [torch.stack([torch.randperm(nodes)[:drop] for _ in range(num_nets)], dim=0)
                for nodes, drop in zip(num_nodes, num_drop)]
    return idx_high, idx_low, idx_rand

@test_nets
@torch.no_grad()
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    (unchanged)
    method for testing network on supervised learning problem with progressive dropout.
    """
    if not (isinstance(nets, list)):
        nets = [nets]

    n_alignment_idx = nets[0].num_layers()
    if alignment is None:
        alignment = test(nets, dataset, **parameters)["alignment"]

    assert len(alignment) == n_alignment_idx, "the number of layers in **alignment** doesn't match"
    if nets[0].is_classification_layer_included():
        n_alignment_idx -= 1
        alignment.pop(-1)

    # average across batches
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
                    drop_high = [idx_high[layer]]
                    drop_low = [idx_low[layer]]
                    drop_rand = [idx_rand[layer]]
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
