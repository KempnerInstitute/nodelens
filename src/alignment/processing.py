# --------------------------------------------
# processing.py
# --------------------------------------------

import os
import torch
from tqdm import tqdm

# [UNCHANGED] - same imports as before
from alignment.utils import load_checkpoints, test_nets, transpose_list, fgsm_attack, save_checkpoint
from alignment.alignment_metrics import AlignmentMetrics
from alignment import train  # or your local "train" logic, if you have a separate file

def train_networks(exp, nets, optimizers, dataset, **special_parameters):
    """
    (UNCHANGED except for the alignment approach is simplified to do_train/do_alignment)
    Orchestrates training and testing of networks with simpler toggles:
      - do_train = exp.args.training.do_train
      - do_alignment = exp.args.alignment.do_alignment
      - methods = exp.args.alignment.methods
      - measure_expected = exp.args.alignment.measure_expected
      - frequency = exp.args.alignment.frequency
    """

    # [UNCHANGED] read toggles from exp
    do_train = exp.args.training.do_train
    do_alignment = exp.args.alignment.do_alignment
    methods = exp.args.alignment.methods
    measure_freq = exp.args.alignment.frequency
    measure_expected = exp.args.alignment.measure_expected
    bins = exp.args.alignment.bins

    # Build parameters dict for internal usage
    params = dict(
        train_set=True,
        num_epochs=exp.args.training.epochs,
        alignment=do_alignment,
        methods=methods,
        frequency=measure_freq,
        run=exp.wandb_run,
    )
    params.update(**special_parameters)

    # handle checkpoint if needed
    results = None
    if exp.args.checkpointing.use_prev and os.path.isfile(exp.get_checkpoint_path()):
        nets, optimizers, oldres = load_checkpoints(nets, optimizers, exp.args.device, exp.get_checkpoint_path())
        for net in nets:
            net.train()
        params["num_complete"] = oldres["epoch"] + 1
        results = oldres
        print("loaded networks from previous checkpoint")
    else:
        results = {
            "alignment": [],
            "alignment_distribution": [],
            "expected_distribution": [],
        }

    params["results"] = results

    # if saving checkpoints
    if exp.args.checkpointing.save_checkpoints:
        params["save_checkpoints"] = (
            True,
            exp.args.checkpointing.frequency,
            exp.get_checkpoint_path(),
            exp.args.device
        )

    # [UNCHANGED] - if do_train
    if do_train:
        print("=== Training networks... ===")
        _train_loop(nets, optimizers, dataset, exp, params, measure_expected, bins)
    else:
        print("=== Skipping training (do_train=False) ===")

    # final test or inference
    print("=== Testing networks... ===")
    _test_loop(nets, dataset, exp, params, measure_expected, bins)

    return params["results"], {}  # or separate train_results, test_results if you want

def _train_loop(nets, optimizers, dataset, exp, params, measure_expected, bins):
    """
    (UNCHANGED except we keep the full logic for measuring alignment & hist.)
    Runs the training loop, measuring alignment at 'params["frequency"]' steps.
    """
    from alignment.utils import train_nets
    from copy import deepcopy

    alignment = params["alignment"]
    freq_align = params["frequency"]
    methods = params["methods"]
    run = params.get("run", None)
    num_epochs = params["num_epochs"]
    device = exp.args.device
    results = params["results"]
    save_ckpt_info = params.get("save_checkpoints", (False, 1, "", ""))
    do_ckpt, ckpt_freq, ckpt_path, dev = save_ckpt_info
    start_epoch = params.get("num_complete", 0)
    dataloader = dataset.train_loader

    @train_nets
    def train_inner_loop(nets, optimizers, dataset):
        for epoch in range(start_epoch, num_epochs):
            ep_loop = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
            for batch_idx, batch in enumerate(ep_loop):
                images, labels = dataset.unwrap_batch(batch, device=device)
                # forward/backward
                for net, opt in zip(nets, optimizers):
                    opt.zero_grad()
                    out = net(images, store_hidden=True)
                    loss_val = dataset.measure_loss(out, labels)
                    loss_val.backward()
                    opt.step()

                # measure alignment if alignment==True & freq
                if alignment and (batch_idx % freq_align == 0):
                    align_data = []
                    for net in nets:
                        layer_metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
                        align_data.append(layer_metrics)
                    results["alignment"].append({
                        "epoch": epoch,
                        "batch": batch_idx,
                        "data": align_data
                    })

                    # build observed distribution from align_data
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

            # checkpoint if needed
            if do_ckpt and (epoch % ckpt_freq == 0):
                cpy_res = deepcopy(results)
                cpy_res["epoch"] = epoch
                cpy_res["device"] = dev
                cpy_res["prms"] = params
                save_checkpoint(nets, optimizers, cpy_res, ckpt_path)

    train_inner_loop(nets, optimizers, dataset)

@test_nets
@torch.no_grad()
def _test_loop(nets, dataset, exp, params, measure_expected, bins):
    """
    (UNCHANGED except for the final alignment freq approach)
    Runs a test pass, measuring alignment, building hist, random distribution, etc.
    """
    alignment = params["alignment"]
    freq_align = params["frequency"]
    methods = params["methods"]
    run = params.get("run", None)
    device = exp.args.device
    results = params["results"]
    dataloader = dataset.test_loader

    test_batch_idx = 0
    for batch in tqdm(dataloader, desc="Testing", leave=False):
        images, labels = dataset.unwrap_batch(batch, device=device)
        # forward pass only
        for net in nets:
            out = net(images, store_hidden=True)

        if alignment and (test_batch_idx % freq_align == 0):
            align_data = []
            for net in nets:
                layer_metrics = AlignmentMetrics.measure_methods(net, images, methods=methods)
                align_data.append(layer_metrics)
            results["alignment"].append({
                "test_batch": test_batch_idx,
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
                "test_batch": test_batch_idx,
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
                    "test_batch": test_batch_idx,
                    "data": exp_data
                })

        test_batch_idx += 1

