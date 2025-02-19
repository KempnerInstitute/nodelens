# train.py

import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from alignment.utils import train_nets, test_nets, load_checkpoints, save_checkpoint
from alignment.alignment_metrics import AlignmentMetrics


@train_nets
def train(nets, optimizers, dataset, **parameters):
    """
    A single function for supervised training with batch/epoch aggregator logic
    and alignment distribution computations.

    Args:
      nets (List[nn.Module]): list of replicate networks.
      optimizers (List[torch.optim.Optimizer]): matching optimizers, length=#nets.
      dataset: a dataset wrapper with train_loader/test_loader.
      alignment (bool): if True, measure alignment each epoch (or freq) in training.
      aggregate_alignment (bool): if True => store alignment each batch;
                                  if False => store once per epoch.
      methods (List[str]): alignment methods (["RQ","delta_alignment",...]).
      frequency (int): how often (epochs) to measure alignment.
      measure_expected (bool): if True, measure expected distributions via PCA.
      bins (int): # bins for alignment histogram.
      num_epochs (int): total epochs to train.
      results (dict or None): existing results dict or None => create new.
      save_checkpoints (tuple): (bool do_ckpt, ckpt_freq, ckpt_path, device).
      train_set (bool): if True => use dataset.train_loader, else dataset.test_loader.
      verbose (bool): whether to print progress info.
      use_wandb (bool): if True => log stats in wandb (wandb must be initialized).
      ...
    Returns:
      A dict of results, with keys:
        "loss", "accuracy" => shape (#nets, #epochs)
        "alignment", "alignment_distribution", "expected_distribution" => lists of data
    """

    do_align         = parameters.get("alignment", False)
    methods          = parameters.get("methods", ["RQ"])
    freq             = parameters.get("frequency", 1)
    measure_expected = parameters.get("measure_expected", True)
    bins             = parameters.get("bins", 50)
    aggregate        = parameters.get("aggregate_alignment", False)

    num_epochs = parameters["num_epochs"]
    results    = parameters.get("results", None)
    use_wandb  = parameters.get("use_wandb", False)

    # Attempt to detect wandb initialization
    wandb_run  = None
    try:
        import wandb
        wandb_run = wandb.run
    except ImportError:
        pass
    wandb_inited = (wandb_run is not None)

    if not isinstance(results, dict):
        results = {}

    # Create keys to store alignment, distribution, etc.
    if "alignment" not in results:
        results["alignment"] = []
    if "alignment_distribution" not in results:
        results["alignment_distribution"] = []
    if "expected_distribution" not in results:
        results["expected_distribution"] = []

    num_replicates = len(nets)
    if "loss" not in results:
        results["loss"] = torch.zeros(num_replicates, num_epochs, dtype=torch.float)
    if "accuracy" not in results:
        results["accuracy"] = torch.zeros(num_replicates, num_epochs, dtype=torch.float)

    save_ckpt_info = parameters.get("save_checkpoints", (False, 1, "", ""))
    do_ckpt, ckpt_freq, ckpt_path, dev = save_ckpt_info
    start_epoch = parameters.get("num_complete", 0)

    use_train  = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    verbose    = parameters.get("verbose", True)

    if verbose:
        print(f"Starting training loop with epochs={num_epochs}, alignment={do_align}, "
              f"aggregate_alignment={aggregate}, methods={methods}")

    for epoch in range(start_epoch, num_epochs):
        replicate_loss_sums   = [0.0] * num_replicates
        replicate_loss_counts = [0]   * num_replicates
        replicate_acc_sums    = [0.0] * num_replicates
        replicate_acc_counts  = [0]   * num_replicates

        # If aggregator=False, we'll collect alignment data, etc. in local lists
        epoch_align_data = []
        epoch_dist_data  = []
        epoch_exp_data   = []

        epoch_rq_values  = []

        loop = tqdm(dataloader, desc=f"Train Epoch {epoch+1}", leave=False) if verbose else dataloader

        for batch_idx, batch in enumerate(loop):
            images, labels = dataset.unwrap_batch(batch)

            # Update each replicate's network
            for idx_rep, (net, opt) in enumerate(zip(nets, optimizers)):
                opt.zero_grad()
                out = net(images, store_hidden=True)
                loss_val = dataset.measure_loss(out, labels)
                loss_val.backward()
                opt.step()

                replicate_loss_sums[idx_rep]   += loss_val.item()
                replicate_loss_counts[idx_rep] += 1

                acc_val = dataset.measure_accuracy(out, labels)
                replicate_acc_sums[idx_rep]    += float(acc_val)
                replicate_acc_counts[idx_rep]  += 1

            # If alignment is enabled and it's an alignment epoch
            if do_align and (epoch % freq == 0):
                # measure alignment for this batch
                batch_align_data = []
                for net in nets:
                    layer_metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
                    batch_align_data.append(layer_metrics)

                # measure alignment distribution (histogram)
                dist_data = []
                for net_layer_list in batch_align_data:
                    layer_dists = []
                    for layer_dict in net_layer_list:
                        method_dists = {}
                        for m, val_tensor in layer_dict.items():
                            val_cpu = val_tensor.detach().cpu()
                            c, e = torch.histogram(val_cpu, bins=bins, density=True)
                            method_dists[m] = (c, e)
                        layer_dists.append(method_dists)
                    dist_data.append(layer_dists)

                # measure expected distribution if measure_expected
                exp_data = []
                if measure_expected:
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

                if aggregate:
                    # immediate storage in results
                    results["alignment"].append({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "data": batch_align_data
                    })
                    results["alignment_distribution"].append({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "data": dist_data
                    })
                    if measure_expected:
                        results["expected_distribution"].append({
                            "epoch": epoch,
                            "batch": batch_idx,
                            "data": exp_data
                        })
                else:
                    # aggregator=False => store them locally for the epoch
                    epoch_align_data.append(batch_align_data)
                    epoch_dist_data.append(dist_data)
                    if measure_expected:
                        epoch_exp_data.append(exp_data)

                # For wandb logging => if "RQ" in the first layer:
                if batch_align_data and batch_align_data[0]:
                    first_net_first_layer = batch_align_data[0][0]
                    if "RQ" in first_net_first_layer:
                        rq_val = first_net_first_layer["RQ"].mean().item()
                        epoch_rq_values.append(rq_val)

        # end of epoch => average replicate losses & accuracies
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

        # If aggregator=False, store them once at epoch end
        if do_align and (epoch % freq == 0) and not aggregate:
            if epoch_align_data:
                results["alignment"].append({
                    "epoch": epoch,
                    "batch": "aggregated",
                    "data": epoch_align_data
                })
            if epoch_dist_data:
                results["alignment_distribution"].append({
                    "epoch": epoch,
                    "batch": "aggregated",
                    "data": epoch_dist_data
                })
            if measure_expected and epoch_exp_data:
                results["expected_distribution"].append({
                    "epoch": epoch,
                    "batch": "aggregated",
                    "data": epoch_exp_data
                })

        # Optionally wandb log
        mean_loss_ep = float(torch.mean(results["loss"][:, epoch]))
        mean_acc_ep  = float(torch.mean(results["accuracy"][:, epoch]))
        mean_rq      = float(np.mean(epoch_rq_values)) if len(epoch_rq_values) > 0 else 0.0

        if use_wandb and wandb_inited:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": mean_loss_ep,
                "train_acc": mean_acc_ep,
                "train_alignment_RQ_epoch": mean_rq
            })

        # checkpointing
        if do_ckpt and (epoch % ckpt_freq == 0):
            cpy_res = deepcopy(results)
            cpy_res["epoch"]  = epoch
            cpy_res["device"] = dev
            cpy_res["prms"]   = parameters
            # reload to ensure no mismatch
            load_checkpoints(nets, optimizers, dev, None)
            save_checkpoint(nets, optimizers, cpy_res, ckpt_path)

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs} => loss={mean_loss_ep:.4f}, acc={mean_acc_ep:.2f}")

    results["loss"]     = results["loss"].detach()
    results["accuracy"] = results["accuracy"].detach()
    return results


@test_nets
def test(nets, dataset, **parameters):
    """
    A single function for testing/evaluation, possibly measuring alignment too.
    Args:
      nets: replicate networks
      dataset: dataset wrapper
      alignment (bool): if True, measure alignment
      methods (List[str]): alignment methods
      frequency (int): currently not used in test, but we keep for consistency
      measure_expected (bool): if True => measure expected distribution
      bins (int): # bins for histogram
      train_set (bool): use dataset.train_loader if True, else test_loader
      ...
    Returns:
      A dict with "loss", "accuracy", optional alignment data, alignment_distribution, expected_distribution
    """

    do_align         = parameters.get("alignment", False)
    methods          = parameters.get("methods", ["RQ"])
    measure_expected = parameters.get("measure_expected", True)
    bins             = parameters.get("bins", 50)
    train_set        = parameters.get("train_set", False)

    results = parameters.get("results", {})
    if not isinstance(results, dict):
        results = {}

    # We'll store single-pass results in 'loss' => shape (#nets,)
    # and 'accuracy' => shape (#nets,) for one pass
    num_reps = len(nets)
    device   = dataset.device
    loader   = dataset.train_loader if train_set else dataset.test_loader

    loss_vec = torch.zeros(num_reps)
    acc_vec  = torch.zeros(num_reps)

    total_correct = torch.zeros(num_reps, device=device)
    total_samples = 0
    total_loss    = torch.zeros(num_reps, device=device)

    for batch in loader:
        images, labels = dataset.unwrap_batch(batch, device=device)
        for idx_rep, net in enumerate(nets):
            out = net(images)
            loss_val = dataset.measure_loss(out, labels, reduction="sum")
            total_loss[idx_rep] += loss_val.detach()
            pred = out.argmax(dim=1)
            total_correct[idx_rep] += (pred == labels).sum()
        total_samples += labels.size(0)

    for idx_rep in range(num_reps):
        if total_samples > 0:
            loss_vec[idx_rep] = total_loss[idx_rep].cpu().item() / float(total_samples)
            acc_vec[idx_rep]  = (total_correct[idx_rep].cpu().item() / total_samples) * 100.0

    results["loss"]     = loss_vec
    results["accuracy"] = acc_vec

    # measure alignment if do_align
    if do_align:
        # We'll just do one pass alignment on entire loader => or a single batch if you prefer
        # for simplicity, we measure on the same last images from above => or re-run a small loop
        # measure alignment data
        images, labels = next(iter(loader))  # quick sample
        images, labels = dataset.unwrap_batch((images, labels), device=device)
        align_data = []
        dist_data  = []
        exp_data   = []

        for net in nets:
            metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
            align_data.append(metrics)
            # measure distribution for each layer, method
            layer_dists = []
            for layer_dict in metrics:
                m_d = {}
                for m, val_tensor in layer_dict.items():
                    val_cpu = val_tensor.detach().cpu()
                    c, e = torch.histogram(val_cpu, bins=bins, density=True)
                    m_d[m] = (c, e)
                layer_dists.append(m_d)
            dist_data.append(layer_dists)

            # measure expected distribution
            if measure_expected:
                net_inps = net.get_layer_inputs(images, precomputed=False)
                layer_exp_list = []
                for inp in net_inps:
                    if inp.ndim == 4:
                        inp = inp.flatten(start_dim=1)
                    wvals, _ = AlignmentMetrics.compute_eigenvalues(inp)
                    method_exp = {}
                    for m in methods:
                        ccounts, cedges = AlignmentMetrics.measure_expected_distribution(m, wvals, bins=bins)
                        method_exp[m] = (ccounts, cedges)
                    layer_exp_list.append(method_exp)
                exp_data.append(layer_exp_list)

        results["alignment"] = [{"epoch":"test", "batch":"all", "data": align_data}]
        results["alignment_distribution"] = [{"epoch":"test", "batch":"all", "data": dist_data}]
        if measure_expected:
            results["expected_distribution"] = [{"epoch":"test", "batch":"all", "data": exp_data}]

    return results