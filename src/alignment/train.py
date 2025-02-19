# --------------------------------------------
# train.py
# --------------------------------------------
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
import wandb

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
      - Optionally log via wandb

    This returns the updated 'results' dictionary.
    """
    do_align         = parameters.get("alignment", False)
    methods          = parameters.get("methods", ["RQ"])
    freq             = parameters.get("frequency", 1)
    measure_expected = parameters.get("measure_expected", True)
    bins             = parameters.get("bins", 50)

    num_epochs  = parameters["num_epochs"]
    results     = parameters.get("results", None)
    use_wandb   = parameters.get("use_wandb", False)  # <-- check if user wants wandb
    wandb_inited= wandb.run is not None               # <-- check if wandb.init() was actually called

    if not isinstance(results, dict):
        results = {}

    # We keep track of alignment, distribution, and expected_distribution
    if "alignment" not in results:
        results["alignment"] = []
    if "alignment_distribution" not in results:
        results["alignment_distribution"] = []
    if "expected_distribution" not in results:
        results["expected_distribution"] = []

    # We want a 2D shape for both 'loss' and 'accuracy' => (#replicates, #epochs)
    num_replicates = len(nets)
    if "loss" not in results:
        results["loss"] = torch.zeros(num_replicates, num_epochs, dtype=torch.float)
    if "accuracy" not in results:
        results["accuracy"] = torch.zeros(num_replicates, num_epochs, dtype=torch.float)

    # Extract checkpoint info
    save_ckpt_info = parameters.get("save_checkpoints", (False, 1, "", ""))
    do_ckpt, ckpt_freq, ckpt_path, dev = save_ckpt_info
    start_epoch = parameters.get("num_complete", 0)

    use_train  = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    verbose = parameters.get("verbose", True)
    if verbose:
        print(f"Starting training loop with epochs={num_epochs}, do_align={do_align}, methods={methods}")

    for epoch in range(start_epoch, num_epochs):
        replicate_loss_sums   = [0.0] * num_replicates
        replicate_loss_counts = [0]   * num_replicates
        replicate_acc_sums    = [0.0] * num_replicates
        replicate_acc_counts  = [0]   * num_replicates

        # list for collecting alignment (RQ) across batches if we want to do wandb logs
        epoch_rq_values = []

        loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False) if verbose else dataloader

        for batch_idx, batch in enumerate(loop):
            images, labels = dataset.unwrap_batch(batch)
            # -------------
            # update each replicate's net
            # -------------
            for idx_rep, (net, opt) in enumerate(zip(nets, optimizers)):
                opt.zero_grad()
                out = net(images, store_hidden=True)
                loss_val = dataset.measure_loss(out, labels)
                loss_val.backward()
                opt.step()

                replicate_loss_sums[idx_rep]   += loss_val.item()
                replicate_loss_counts[idx_rep] += 1

                # measure accuracy
                acc_val = dataset.measure_accuracy(out, labels)
                replicate_acc_sums[idx_rep]    += float(acc_val)
                replicate_acc_counts[idx_rep]  += 1

            # -------------
            # measure alignment if do_align is True (and optional freq check)
            # -------------
            if do_align and (epoch % freq == 0):
                align_data = []
                for net in nets:
                    layer_metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
                    align_data.append(layer_metrics)
                    print(epoch)
                # store alignment snapshot
                results["alignment"].append({
                    "epoch": epoch,
                    "batch": batch_idx,
                    "data": align_data
                })

                # measure alignment distribution
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
                            if inp.ndim == 4:
                                inp = inp.flatten(start_dim=1)
                            w_vals, _ = AlignmentMetrics.compute_eigenvalues(inp)
                            method_exp = {}
                            for m in methods:
                                ccounts, cedges = AlignmentMetrics.measure_expected_distribution(m, w_vals, bins=bins)
                                method_exp[m] = (ccounts, cedges)
                            layer_exp_list.append(method_exp)
                        exp_data.append(layer_exp_list)

                    results["expected_distribution"].append({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "data": exp_data
                    })

                # store RQ for possible logging
                if len(align_data) > 0:
                    first_net_first_layer = align_data[0][0]
                    if "RQ" in first_net_first_layer:
                        rq_values = first_net_first_layer["RQ"]
                        epoch_rq_values.append(rq_values.mean().item())

        # End of one epoch => average replicate losses & accuracies
        for idx_rep in range(num_replicates):
            if replicate_loss_counts[idx_rep] > 0:
                avg_loss = replicate_loss_sums[idx_rep] / replicate_loss_counts[idx_rep]
            else:
                avg_loss = 0.0
            results["loss"][idx_rep, epoch] = avg_loss

            if replicate_acc_counts[idx_rep] > 0:
                avg_acc = replicate_acc_sums[idx_rep] / replicate_acc_counts[idx_rep]
            else:
                avg_acc = 0.0
            results["accuracy"][idx_rep, epoch] = avg_acc

        # Optionally log to wandb if the user has wandb and it is properly initialized
        avg_loss_across_nets = float(torch.mean(results["loss"][:, epoch]))
        avg_acc_across_nets  = float(torch.mean(results["accuracy"][:, epoch]))
        if len(epoch_rq_values) > 0:
            mean_rq_epoch = float(np.mean(epoch_rq_values))
        else:
            mean_rq_epoch = 0.0

        if use_wandb and wandb_inited:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss_across_nets,
                "train_acc": avg_acc_across_nets,
                "train_alignment_RQ_epoch": mean_rq_epoch
            })

        if do_ckpt and (epoch % ckpt_freq == 0):
            cpy_res = deepcopy(results)
            cpy_res["epoch"]  = epoch
            cpy_res["device"] = dev
            cpy_res["prms"]   = parameters
            save_checkpoint(nets, optimizers, cpy_res, ckpt_path)

    results["loss"]     = results["loss"].clone().detach()
    results["accuracy"] = results["accuracy"].clone().detach()

    return results