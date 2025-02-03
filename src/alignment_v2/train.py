import torch
from tqdm import tqdm
from utils import train_nets, test_nets, save_checkpoint

@train_nets
def train(nets, opts, dataset, **params):
    verbose = params.get("verbose", True)
    num_epochs = params.get("num_epochs", 1)
    train_set = params.get("train_set", True)
    dl = dataset.train_loader if train_set else dataset.test_loader
    steps = len(dl) * num_epochs
    n_nets = len(nets)
    results = params.get("results", {"loss": torch.zeros((steps, n_nets)),
                                       "accuracy": torch.zeros((steps, n_nets))})
    num_complete = params.get("num_complete", 0)
    ckpt_params = params.get("save_checkpoints", (False, 1, "", ""))
    epoch_range = range(num_complete, num_epochs)
    if verbose:
        epoch_range = tqdm(epoch_range, desc="Epochs")
    for epoch in epoch_range:
        for i, batch in enumerate(dl):
            idx = epoch * len(dl) + i
            x, y = dataset.unwrap_batch(batch)
            for opt in opts:
                opt.zero_grad()
            outs = [net(x, store_hidden=True) for net in nets]
            loss_vals = [dataset.measure_loss(o, y) for o in outs]
            for loss, opt in zip(loss_vals, opts):
                loss.backward()
                opt.step()
            results["loss"][idx] = torch.tensor([l.item() for l in loss_vals])
            results["accuracy"][idx] = torch.tensor([dataset.measure_accuracy(o, y).item() for o in outs])
            # Alignment metrics could be added here if needed.
        if ckpt_params[0] and (epoch % ckpt_params[1] == 0):
            save_checkpoint(nets, opts, results, ckpt_params[2])
    return results

@test_nets
def test(nets, dataset, **params):
    verbose = params.get("verbose", True)
    test_set = not params.get("train_set", False)
    dl = dataset.test_loader if test_set else dataset.train_loader
    n_nets = len(nets)
    total_loss = [0]*n_nets
    total_acc = [0]*n_nets
    batches = 0
    for batch in tqdm(dl, desc="Testing") if verbose else dl:
        x, y = dataset.unwrap_batch(batch)
        outs = [net(x, store_hidden=True) for net in nets]
        for i, o in enumerate(outs):
            total_loss[i] += dataset.measure_loss(o, y).item()
            total_acc[i] += dataset.measure_accuracy(o, y).item()
        batches += 1
    return {"loss": [l/batches for l in total_loss], "accuracy": [a/batches for a in total_acc]}