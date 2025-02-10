# --------------------------------------------
# train.py
# --------------------------------------------

import numpy as np
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
    
    results = parameters.get("results", None)
    if not isinstance(results, dict):
        results = {}
    
    if "alignment" not in results:
        results["alignment"] = []
    if "alignment_distribution" not in results:
        results["alignment_distribution"] = []
    if "expected_distribution" not in results:
        results["expected_distribution"] = []

    # NEW CODE: We need a 2D shape for 'loss', i.e. (#replicates, #epochs).
    num_replicates = len(nets)
    if "loss" not in results:
        results["loss"] = torch.zeros(num_replicates, num_epochs, dtype=torch.float)

    # Extract checkpoint information
    save_ckpt_info = parameters.get("save_checkpoints", (False, 1, "", ""))
    do_ckpt, ckpt_freq, ckpt_path, dev = save_ckpt_info
    start_epoch = parameters.get("num_complete", 0)

    use_train = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    verbose = parameters.get("verbose", True)
    if verbose:
        print(f"Starting training loop with epochs={num_epochs}, do_align={do_align}, methods={methods}")

    for epoch in range(start_epoch, num_epochs):
        replicate_loss_sums = [0.0] * num_replicates
        replicate_loss_counts = [0] * num_replicates

        loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False) if verbose else dataloader
        for batch_idx, batch in enumerate(loop):
            images, labels = dataset.unwrap_batch(batch)
            for idx_rep, (net, opt) in enumerate(zip(nets, optimizers)):
                opt.zero_grad()
                out = net(images, store_hidden=True)
                loss_val = dataset.measure_loss(out, labels)
                loss_val.backward()
                opt.step()

                replicate_loss_sums[idx_rep] += loss_val.item()
                replicate_loss_counts[idx_rep] += 1

            if do_align and (batch_idx % freq == 0):
                align_data = []
                for net in nets:
                    layer_metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
                    align_data.append(layer_metrics)
                results["alignment"].append({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "data": align_data
                })

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

        for idx_rep in range(num_replicates):
            if replicate_loss_counts[idx_rep] > 0:
                avg_loss = replicate_loss_sums[idx_rep] / replicate_loss_counts[idx_rep]
            else:
                avg_loss = 0.0
            results["loss"][idx_rep, epoch] = avg_loss

        if do_ckpt and (epoch % ckpt_freq == 0):
            cpy_res = deepcopy(results)
            cpy_res["epoch"] = epoch
            cpy_res["device"] = dev
            cpy_res["prms"] = parameters
            save_checkpoint(nets, optimizers, cpy_res, ckpt_path)

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

    results = parameters.get("results", None)
    if not isinstance(results, dict):
        results = {}

    if "alignment" not in results:
        results["alignment"] = []
    if "alignment_distribution" not in results:
        results["alignment_distribution"] = []
    if "expected_distribution" not in results:
        results["expected_distribution"] = []

    num_nets = len(nets)
    test_losses = [[] for _ in range(num_nets)]
    test_accs = [[] for _ in range(num_nets)]

    verbose = parameters.get("verbose", True)
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    if verbose:
        dataloader = tqdm(dataloader, desc="Testing")

    batch_idx = 0
    for batch in dataloader:
        images, labels = dataset.unwrap_batch(batch)
        for i, net in enumerate(nets):
            out = net(images, store_hidden=True)
            loss_val = dataset.measure_loss(out, labels).item()
            acc_val = dataset.measure_accuracy(out, labels)
            test_losses[i].append(loss_val)
            test_accs[i].append(acc_val)
        if do_align and (batch_idx % freq == 0):
            align_data = []
            for net in nets:
                layer_metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
                align_data.append(layer_metrics)
            results["alignment"].append({
                "test_batch": batch_idx,
                "data": align_data
            })

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

    avg_test_losses = torch.tensor([np.mean(loss_list) if len(loss_list) > 0 else 0.0 for loss_list in test_losses], dtype=torch.float)
    avg_test_accs = torch.tensor([np.mean(acc_list) if len(acc_list) > 0 else 0.0 for acc_list in test_accs], dtype=torch.float)
    results["loss"] = avg_test_losses
    results["accuracy"] = avg_test_accs

    return results