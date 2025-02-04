from copy import deepcopy
import torch
from tqdm import tqdm

from alignment_v2.utils import test_nets, train_nets, save_checkpoint
from alignment_v2.utils import transpose_list, condense_values

@train_nets
def train(nets, optimizers, dataset, **parameters):
    """
    direct training loop
    """
    if not isinstance(nets, list):
        nets = [nets]
    if not isinstance(optimizers, list):
        optimizers = [optimizers]
    assert len(nets) == len(optimizers), "nets and opts mismatch"

    verbose = parameters.get("verbose", True)
    wandb_run = parameters.get("run", None)

    measure_alignment = parameters.get("alignment", False)
    methods = parameters.get("methods", ["RQ"])
    measure_weight_deltas = parameters.get("measure_weight_deltas", False)
    freq = parameters.get("frequency", 1)  # alignment frequency
    num_epochs = parameters["num_epochs"]
    use_train = parameters.get("train_set", True)

    dataloader = dataset.train_loader if use_train else dataset.test_loader
    num_steps = len(dataloader) * num_epochs

    results = parameters.get("results", None)
    if results is None:
        results = {
            "loss": torch.zeros((num_steps, len(nets))),
            "accuracy": torch.zeros((num_steps, len(nets))),
        }
        if measure_alignment:
            results["alignment"] = []
        if measure_weight_deltas:
            results["delta_weights"] = []
            results["init_weights"] = [net.get_alignment_weights() for net in nets]

    start_epoch = parameters.get("num_complete", 0)
    save_ckpt_info = parameters.get("save_checkpoints", (False, 1, "", ""))

    if verbose:
        print("Starting training loop...")

    step = start_epoch * len(dataloader)
    for epoch in range(start_epoch, num_epochs):
        if verbose:
            ep_loop = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        else:
            ep_loop = dataloader

        for batch_idx, batch in enumerate(ep_loop):
            images, labels = dataset.unwrap_batch(batch)
            for opt in optimizers:
                opt.zero_grad()

            outputs = [net(images, store_hidden=True) for net in nets]
            losses = [dataset.measure_loss(out, labels) for out in outputs]

            for l, opt in zip(losses, optimizers):
                l.backward()
                opt.step()

            for i, lval in enumerate(losses):
                results["loss"][step, i] = lval.item()

            for i, out in enumerate(outputs):
                acc_val = dataset.measure_accuracy(out, labels)
                results["accuracy"][step, i] = acc_val

            if measure_alignment and (batch_idx % freq == 0):
                # measure each net's alignment for each method if you want multi-metrics
                align_batch = []
                for net in nets:
                    # For multi-metric approach, you'd do net.measure_alignment_methods(images, methods=methods)
                    # or a loop. For now, we just do net.measure_alignment with 'alignment' as "RQ".
                    align_val = net.measure_alignment(images, precomputed=True, method="alignment")
                    # If you'd like multiple metrics in a single pass, you'd change net's code or do a loop
                    align_batch.append(align_val)
                results["alignment"].append(align_batch)

            if measure_weight_deltas:
                c_delta = [
                    net.compare_weights(init_w)
                    for net, init_w in zip(nets, results["init_weights"])
                ]
                results["delta_weights"].append(c_delta)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"loss_{i}": lval.item() for i, lval in enumerate(losses)
                    } | {
                        f"acc_{i}": dataset.measure_accuracy(out, labels) for i, out in enumerate(outputs)
                    } | {
                        "global_step": step,
                        "epoch": epoch,
                    }
                )

            step += 1

        # handle checkpoint
        do_ckpt, ckpt_freq, ckpt_path, dev = save_ckpt_info
        if do_ckpt and (epoch % ckpt_freq == 0):
            from copy import deepcopy
            newres = deepcopy(results)
            newres["epoch"] = epoch
            newres["device"] = dev
            newres["prms"] = parameters
            save_checkpoint(nets, optimizers, newres, ckpt_path)

    return results

@torch.no_grad()
@test_nets
def test(nets, dataset, **parameters):
    """
    direct testing
    """
    wandb_run = parameters.get("run", None)
    measure_alignment = parameters.get("alignment", False)
    methods = parameters.get("methods", ["RQ"])
    verbose = parameters.get("verbose", True)
    use_train = parameters.get("train_set", False)

    loader = dataset.train_loader if use_train else dataset.test_loader

    total_loss = torch.zeros(len(nets))
    total_acc = torch.zeros(len(nets))
    count = 0

    alignment_log = []

    if verbose:
        loader = tqdm(loader, desc="Testing")

    for batch in loader:
        images, labels = dataset.unwrap_batch(batch)
        outs = [net(images, store_hidden=True) for net in nets]
        for i, o in enumerate(outs):
            total_loss[i] += dataset.measure_loss(o, labels).item()
            total_acc[i] += dataset.measure_accuracy(o, labels)
        count += 1

        if measure_alignment:
            batch_align = []
            for net in nets:
                # multi method or single method
                align_val = net.measure_alignment(images, precomputed=True, method="alignment")
                batch_align.append(align_val)
            alignment_log.append(batch_align)

    results = {
        "loss": (total_loss / count).tolist(),
        "accuracy": (total_acc / count).tolist(),
    }

    if measure_alignment:
        results["alignment"] = condense_values(transpose_list(alignment_log))

    if wandb_run is not None and hasattr(wandb_run, "summary"):
        wandb_run.summary["test_loss"] = torch.mean(torch.tensor(results["loss"]))
        wandb_run.summary["test_accuracy"] = torch.mean(torch.tensor(results["accuracy"]))

    return results