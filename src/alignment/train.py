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
    
    # If user passed "results": None, we fix it to a dict
    results = parameters.get("results", None)
    if not isinstance(results, dict):
        results = {}
    
    # Ensure sub-keys exist
    if "alignment" not in results:
        results["alignment"] = []
    if "alignment_distribution" not in results:
        results["alignment_distribution"] = []
    if "expected_distribution" not in results:
        results["expected_distribution"] = []

    # NEW CODE: ensure 'loss' key exists, to store train losses
    if "loss" not in results:
        results["loss"] = []

    # Extract checkpoint information
    save_ckpt_info = parameters.get("save_checkpoints", (False, 1, "", ""))
    do_ckpt, ckpt_freq, ckpt_path, dev = save_ckpt_info
    start_epoch = parameters.get("num_complete", 0)

    # retrieve train loader
    use_train = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    verbose = parameters.get("verbose", True)
    if verbose:
        print(f"Starting training loop with epochs={num_epochs}, do_align={do_align}, methods={methods}")

    for epoch in range(start_epoch, num_epochs):
        loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False) if verbose else dataloader

        # NEW CODE: accumulate batch losses each epoch
        epoch_loss_sum = 0.0
        epoch_loss_count = 0

        for batch_idx, batch in enumerate(loop):
            images, labels = dataset.unwrap_batch(batch)
            # zero grads / step
            for net, opt in zip(nets, optimizers):
                opt.zero_grad()
                out = net(images, store_hidden=True)
                loss_val = dataset.measure_loss(out, labels)
                loss_val.backward()
                opt.step()

            # record the total batch loss
            epoch_loss_sum += loss_val.item()
            epoch_loss_count += 1

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

        # NEW CODE: after finishing the epoch, compute average epoch loss
        avg_epoch_loss = epoch_loss_sum / max(epoch_loss_count, 1)
        results["loss"].append(avg_epoch_loss)

        # checkpoint
        if do_ckpt and (epoch % ckpt_freq == 0):
            cpy_res = deepcopy(results)
            cpy_res["epoch"] = epoch
            cpy_res["device"] = dev
            cpy_res["prms"] = parameters
            save_checkpoint(nets, optimizers, cpy_res, ckpt_path)

    # NEW CODE: convert the recorded list of loss floats to a tensor
    results["loss"] = torch.tensor(results["loss"])

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

    # If user passed "results": None, fix it:
    results = parameters.get("results", None)
    if not isinstance(results, dict):
        results = {}

    # Ensure sub-keys exist
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