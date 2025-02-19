import os
import torch
from tqdm import tqdm

from alignment.utils import load_checkpoints, test_nets
from alignment.alignment_metrics import AlignmentMetrics
from alignment.train import train, test

def parse_alignment_to_tensor(alignment_list, aggregate=True, by_layer=False):
    if len(alignment_list) == 0:
        raise ValueError("parse_alignment_to_tensor: empty alignment_list")

    # if by_layer=False => return shape (#nets, total_nodes)
    # if by_layer=True  => return shape (#layers, #nets, nodes_per_layer_for_that_layer)
    # we must ensure each record has the same #layers & same #nodes per layer if we want to stack.
    # if any mismatch occurs, we pad or unify.

    if not by_layer:
        all_records_tensors = []
        for record in alignment_list:
            netwise_tensors = []
            for net_i_data in record["data"]:
                node_tensors = []
                for layer_dict in net_i_data:
                    if "RQ" not in layer_dict:
                        raise ValueError("Expected 'RQ' in layer_dict")
                    node_tensors.append(layer_dict["RQ"].flatten())
                # cat all layers => shape (#nodes_total)
                netwise_tensors.append(torch.cat(node_tensors, dim=0))
            # stack across nets => shape (#nets, #nodes_total)
            netwise_tsr = torch.stack(netwise_tensors, dim=0)
            all_records_tensors.append(netwise_tsr)
        # shape => (#records, #nets, #nodes_total)
        bigstack = torch.stack(all_records_tensors, dim=0)
        # average if aggregator => final shape (#nets, #nodes_total)
        if aggregate:
            return bigstack.mean(dim=0)
        else:
            # or just take the mean anyway so we have a single snapshot
            return bigstack.mean(dim=0)
    else:
        # by_layer=True => produce shape => (#layers, #nets, #nodes_per_layer)
        # each record can have multiple layers; we must ensure the same #layers across records
        # also each net must have the same #layers. We'll find the max #layers and zero-pad if needed.
        all_records = []  # each entry => list of shape (#layers, #nets, #nodes_layer)

        max_layers_found = 0
        for record in alignment_list:
            # record["data"] => list of #nets
            # each net => list of layer_dicts => { "RQ": <Tensor(#nodes_layer,)> }
            # first find how many layers => from the first net
            max_layers_found = max(max_layers_found, len(record["data"][0]))

        # now parse each record
        for record in alignment_list:
            # build shape => (#layers, #nets, variable #nodes?) => we can't stack directly unless each layer's #nodes match across nets
            net_list = record["data"]
            num_nets = len(net_list)
            # net_list[i] => list of layer_dict for net i
            # we unify so each net has 'max_layers_found' layers (pad if missing)
            rec_layers = []
            for layer_idx in range(max_layers_found):
                # gather each net's RQ for this layer
                netwise = []
                for net_idx in range(num_nets):
                    net_layers = net_list[net_idx]
                    if layer_idx < len(net_layers):
                        layer_dict = net_layers[layer_idx]
                        if "RQ" not in layer_dict:
                            raise ValueError("Expected 'RQ' in layer_dict")
                        netwise.append(layer_dict["RQ"].flatten())
                    else:
                        # pad with zero if layer is missing
                        netwise.append(torch.zeros(0))
                # netwise => list of length #nets, each => shape (#nodes_layer_i,)
                # we can find max #nodes among them and pad
                max_nodes_layer = 0
                for item in netwise:
                    max_nodes_layer = max(max_nodes_layer, item.numel())
                # pad each net to max_nodes_layer
                padded_nets = []
                for item in netwise:
                    if item.numel() < max_nodes_layer:
                        pad_size = max_nodes_layer - item.numel()
                        item_padded = torch.cat([item, torch.zeros(pad_size, dtype=item.dtype, device=item.device)], dim=0)
                        padded_nets.append(item_padded)
                    else:
                        padded_nets.append(item)
                # shape => (#nets, max_nodes_layer)
                layer_tsr = torch.stack(padded_nets, dim=0)
                rec_layers.append(layer_tsr)
            # now rec_layers => list of length max_layers_found => each => (#nets, max_nodes_layer)
            # stack => shape => (#layers, #nets, max_nodes_layer)
            rec_layers_tsr = torch.stack(rec_layers, dim=0)
            all_records.append(rec_layers_tsr)

        # shape => (#records, #layers, #nets, max_nodes_layer)
        bigstack = torch.stack(all_records, dim=0)
        if aggregate:
            # => (#layers, #nets, max_nodes_layer)
            return bigstack.mean(dim=0)
        else:
            return bigstack.mean(dim=0)


def train_networks(exp, nets, optimizers, dataset, **special_parameters):
    do_alignment_train = exp.args.alignment.do_alignment
    methods = exp.args.alignment.methods
    measure_freq = exp.args.alignment.frequency

    params = dict(
        train_set=True,
        num_epochs=exp.args.training.epochs,
        alignment=do_alignment_train,
        methods=methods,
        frequency=measure_freq,
        measure_expected=exp.args.alignment.measure_expected,
        results=None,
        verbose=True,
    )
    params.update(**special_parameters)

    if exp.args.checkpointing.use_prev and os.path.isfile(exp.get_checkpoint_path()):
        nets, optimizers, results = load_checkpoints(nets, optimizers, exp.args.device, exp.get_checkpoint_path())
        for net in nets:
            net.train()
        params["num_complete"] = results["epoch"] + 1
        params["results"] = results
        print("loaded networks from previous checkpoint")

    if exp.args.checkpointing.save_checkpoints:
        params["save_checkpoints"] = (
            True,
            exp.args.checkpointing.frequency,
            exp.get_checkpoint_path(),
            exp.args.device,
        )

    print("training networks...")
    train_results = train(nets, optimizers, dataset, **params)

    do_alignment_infer = exp.args.alignment.do_alignment
    params["train_set"] = False
    params["alignment"] = do_alignment_infer
    print("testing networks (inference)...")
    test_results = test(nets, dataset, **params)

    return train_results, test_results

def test_networks(exp, nets, dataset):
    do_align = exp.args.alignment.do_alignment
    methods = exp.args.alignment.methods
    freq = exp.args.alignment.frequency

    test_params = dict(
        train_set=False,
        alignment=do_align,
        methods=methods,
        frequency=freq,
        measure_expected=exp.args.alignment.measure_expected,
        bins=exp.args.alignment.bins,
        results=None,
        verbose=True
    )
    print("testing networks (no training)...")
    test_results = test(nets, dataset, **test_params)
    return test_results

@test_nets
@torch.no_grad()
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    if not isinstance(nets, list):
        nets = [nets]
    if alignment is None:
        alignment = test(nets, dataset, alignment=True, methods=["RQ"], **parameters)["alignment"]
    aggregator = parameters.get("aggregate_alignment", False)
    by_layer = parameters.get("by_layer", False)
    parsed = parse_alignment_to_tensor(alignment, aggregate=aggregator, by_layer=by_layer)

    if not by_layer:
        # shape => (#nets, #nodes_total)
        idx_alignment = [torch.argsort(parsed, dim=1)]
        num_snapshots = 1
        num_nets_ = parsed.shape[0]
    else:
        # shape => (#layers, #nets, max_nodes_layer)
        num_layers, num_nets_, max_nodes_layer = parsed.shape
        idx_alignment = []
        for layer_i in range(num_layers):
            idx_alignment.append(torch.argsort(parsed[layer_i], dim=1))
        num_snapshots = num_layers

    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    print("Progressive Dropout:")

    if by_layer:
        progdrop_loss_high = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_loss_low  = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_loss_rand = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_acc_high  = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_acc_low   = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_acc_rand  = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
    else:
        progdrop_loss_high = torch.zeros((num_nets_, num_drops, 1), device="cpu")
        progdrop_loss_low  = torch.zeros((num_nets_, num_drops, 1), device="cpu")
        progdrop_loss_rand = torch.zeros((num_nets_, num_drops, 1), device="cpu")
        progdrop_acc_high  = torch.zeros((num_nets_, num_drops, 1), device="cpu")
        progdrop_acc_low   = torch.zeros((num_nets_, num_drops, 1), device="cpu")
        progdrop_acc_rand  = torch.zeros((num_nets_, num_drops, 1), device="cpu")

    num_batches = 0
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for dropidx, fraction in enumerate(drop_fraction):
            if by_layer:
                for layer_i in range(num_layers):
                    layer_idx_sorted = idx_alignment[layer_i]
                    device_of_idx = layer_idx_sorted.device
                    node_count = layer_idx_sorted.size(1)
                    drop_num = int(node_count * fraction)
                    if drop_num > 0:
                        hi = torch.index_select(layer_idx_sorted, dim=1,
                            index=torch.arange(node_count - drop_num, node_count, device=device_of_idx))
                        lo = torch.index_select(layer_idx_sorted, dim=1,
                            index=torch.arange(drop_num, device=device_of_idx))
                        rr = []
                        for i_net in range(num_nets_):
                            perm = torch.randperm(node_count, device=device_of_idx)
                            rr.append(perm[:drop_num])
                        rr = torch.stack(rr, dim=0)
                    else:
                        hi = layer_idx_sorted[:, :0]
                        lo = layer_idx_sorted[:, :0]
                        rr = layer_idx_sorted[:, :0]
                    out_high, out_low, out_rand_ = [], [], []
                    for i_net, net in enumerate(nets):
                        oh, _ = net.forward_targeted_dropout(images, [hi[i_net]], [layer_i])
                        ol, _ = net.forward_targeted_dropout(images, [lo[i_net]], [layer_i])
                        or_, _= net.forward_targeted_dropout(images, [rr[i_net]], [layer_i])
                        out_high.append(oh)
                        out_low.append(ol)
                        out_rand_.append(or_)
                    lh, ll, lr = [], [], []
                    ah, al, ar = [], [], []
                    for idxn in range(num_nets_):
                        lv_h = float(dataset.measure_loss(out_high[idxn], labels).detach().cpu())
                        lv_l = float(dataset.measure_loss(out_low[idxn], labels).detach().cpu())
                        lv_r = float(dataset.measure_loss(out_rand_[idxn], labels).detach().cpu())
                        lh.append(lv_h)
                        ll.append(lv_l)
                        lr.append(lv_r)
                        av_h = float(dataset.measure_accuracy(out_high[idxn], labels).detach().cpu())
                        av_l = float(dataset.measure_accuracy(out_low[idxn], labels).detach().cpu())
                        av_r = float(dataset.measure_accuracy(out_rand_[idxn], labels).detach().cpu())
                        ah.append(av_h)
                        al.append(av_l)
                        ar.append(av_r)
                    progdrop_loss_high[:, dropidx, layer_i] += torch.tensor(lh, device="cpu")
                    progdrop_loss_low[:, dropidx, layer_i]  += torch.tensor(ll, device="cpu")
                    progdrop_loss_rand[:, dropidx, layer_i] += torch.tensor(lr, device="cpu")
                    progdrop_acc_high[:, dropidx, layer_i]  += torch.tensor(ah, device="cpu")
                    progdrop_acc_low[:, dropidx, layer_i]   += torch.tensor(al, device="cpu")
                    progdrop_acc_rand[:, dropidx, layer_i]  += torch.tensor(ar, device="cpu")
            else:
                layer_idx_sorted = idx_alignment[0]
                device_of_idx = layer_idx_sorted.device
                node_count = layer_idx_sorted.size(1)
                drop_num = int(node_count * fraction)
                if drop_num > 0:
                    hi = torch.index_select(layer_idx_sorted, dim=1,
                        index=torch.arange(node_count - drop_num, node_count, device=device_of_idx))
                    lo = torch.index_select(layer_idx_sorted, dim=1,
                        index=torch.arange(drop_num, device=device_of_idx))
                    rr = []
                    for i_net in range(num_nets_):
                        perm = torch.randperm(node_count, device=device_of_idx)
                        rr.append(perm[:drop_num])
                    rr = torch.stack(rr, dim=0)
                else:
                    hi = layer_idx_sorted[:, :0]
                    lo = layer_idx_sorted[:, :0]
                    rr = layer_idx_sorted[:, :0]
                out_high, out_low, out_rand_ = [], [], []
                for i_net, net in enumerate(nets):
                    oh, _ = net.forward_targeted_dropout(images, [hi[i_net]], [0])
                    ol, _ = net.forward_targeted_dropout(images, [lo[i_net]], [0])
                    or_, _= net.forward_targeted_dropout(images, [rr[i_net]], [0])
                    out_high.append(oh)
                    out_low.append(ol)
                    out_rand_.append(or_)
                lh, ll, lr = [], [], []
                ah, al, ar = [], [], []
                for idxn in range(num_nets_):
                    lv_h = float(dataset.measure_loss(out_high[idxn], labels).detach().cpu())
                    lv_l = float(dataset.measure_loss(out_low[idxn], labels).detach().cpu())
                    lv_r = float(dataset.measure_loss(out_rand_[idxn], labels).detach().cpu())
                    lh.append(lv_h)
                    ll.append(lv_l)
                    lr.append(lv_r)
                    av_h = float(dataset.measure_accuracy(out_high[idxn], labels).detach().cpu())
                    av_l = float(dataset.measure_accuracy(out_low[idxn], labels).detach().cpu())
                    av_r = float(dataset.measure_accuracy(out_rand_[idxn], labels).detach().cpu())
                    ah.append(av_h)
                    al.append(av_l)
                    ar.append(av_r)
                progdrop_loss_high[:, dropidx, 0] += torch.tensor(lh, device="cpu")
                progdrop_loss_low[:, dropidx, 0]  += torch.tensor(ll, device="cpu")
                progdrop_loss_rand[:, dropidx, 0] += torch.tensor(lr, device="cpu")
                progdrop_acc_high[:, dropidx, 0]  += torch.tensor(ah, device="cpu")
                progdrop_acc_low[:, dropidx, 0]   += torch.tensor(al, device="cpu")
                progdrop_acc_rand[:, dropidx, 0]  += torch.tensor(ar, device="cpu")

    progdrop_loss_high /= num_batches
    progdrop_loss_low  /= num_batches
    progdrop_loss_rand /= num_batches
    progdrop_acc_high  /= num_batches
    progdrop_acc_low   /= num_batches
    progdrop_acc_rand  /= num_batches

    results = {
        "progdrop_loss_high": progdrop_loss_high,
        "progdrop_loss_low":  progdrop_loss_low,
        "progdrop_loss_rand": progdrop_loss_rand,
        "progdrop_acc_high":  progdrop_acc_high,
        "progdrop_acc_low":   progdrop_acc_low,
        "progdrop_acc_rand":  progdrop_acc_rand,
        "dropout_fraction":   drop_fraction,
        "by_layer":           by_layer,
        "idx_dropout_layers": list(range(num_snapshots)),
    }
    return results

def get_dropout_indices(idx_alignment, fraction):
    idx_high_list = []
    idx_low_list = []
    idx_rand_list = []
    for idx_sorted in idx_alignment:
        device_of_idx = idx_sorted.device
        num_nets, num_nodes = idx_sorted.shape
        drop_num = int(num_nodes * fraction)
        if drop_num > 0:
            hi = torch.index_select(
                idx_sorted,
                dim=1,
                index=torch.arange(num_nodes - drop_num, num_nodes, device=device_of_idx)
            )
            lo = torch.index_select(
                idx_sorted,
                dim=1,
                index=torch.arange(drop_num, device=device_of_idx)
            )
            rand = []
            for i_net in range(num_nets):
                perm = torch.randperm(num_nodes, device=device_of_idx)
                rand.append(perm[:drop_num])
            rand = torch.stack(rand, dim=0)
        else:
            hi = idx_sorted[:, :0]
            lo = idx_sorted[:, :0]
            rand = idx_sorted[:, :0]
        idx_high_list.append(hi)
        idx_low_list.append(lo)
        idx_rand_list.append(rand)
    return idx_high_list, idx_low_list, idx_rand_list

def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    print("performing targeted dropout...")
    dropout_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
        aggregate_alignment=exp.args.extra.aggregate_alignment
    )
    dropout_results = progressive_dropout(nets, dataset, alignment=alignment, **dropout_params)
    return dropout_results, dropout_params

def measure_eigenfeatures(exp, nets, dataset, train_set=False):
    from tqdm import tqdm
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

@test_nets
@torch.no_grad()
def eigenvector_dropout(nets, dataset, eigenvalues, eigenvectors, **parameters):
    num_nets = len(nets) if isinstance(nets, list) else 1
    if not isinstance(nets, list):
        nets = [nets]
    align_layer_indices = list(range(len(nets[0].alignment_layers)))
    by_layer = parameters.get("by_layer", False)
    num_layers = len(align_layer_indices) if by_layer else 1
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    idx_eigenvalue = []
    for net_i in range(num_nets):
        layer_idxs = []
        for evec_j in eigenvectors[net_i]:
            dim = evec_j.size(1)
            layer_idxs.append(torch.arange(dim - 1, -1, -1).unsqueeze(0))
        idx_eigenvalue.append(layer_idxs)
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low   = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand  = torch.zeros((num_nets, num_drops, num_layers))
    use_test = not parameters.get("train_set", True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader
    num_batches = 0
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_eigenvalue, fraction)
            for layer_i in range(num_layers):
                if layer_i >= len(idx_high):
                    break
                if by_layer:
                    drop_layer = [align_layer_indices[layer_i]]
                    drop_high = [idx_high[layer_i]]
                    drop_low  = [idx_low[layer_i]]
                    drop_rand = [idx_rand[layer_i]]
                else:
                    drop_layer = align_layer_indices
                    drop_high = idx_high
                    drop_low  = idx_low
                    drop_rand = idx_rand
                out_high, out_low, out_rand_ = [], [], []
                for i_net, net in enumerate(nets):
                    high_idxs = [dr[i_net, :] for dr in drop_high]
                    low_idxs  = [dr[i_net, :] for dr in drop_low]
                    rand_idxs = [dr[i_net, :] for dr in drop_rand]
                    oh, _ = net.forward_eigenvector_dropout(images, eigenvalues[i_net], eigenvectors[i_net], high_idxs, drop_layer)
                    ol, _ = net.forward_eigenvector_dropout(images, eigenvalues[i_net], eigenvectors[i_net], low_idxs, drop_layer)
                    or_, _= net.forward_eigenvector_dropout(images, eigenvalues[i_net], eigenvectors[i_net], rand_idxs, drop_layer)
                    out_high.append(oh)
                    out_low.append(ol)
                    out_rand_.append(or_)
                loss_high = []
                loss_low = []
                loss_rand = []
                acc_high = []
                acc_low = []
                acc_rand = []
                for idxn in range(num_nets):
                    lv_h = float(dataset.measure_loss(out_high[idxn], labels).detach().cpu())
                    lv_l = float(dataset.measure_loss(out_low[idxn], labels).detach().cpu())
                    lv_r = float(dataset.measure_loss(out_rand_[idxn], labels).detach().cpu())
                    loss_high.append(lv_h)
                    loss_low.append(lv_l)
                    loss_rand.append(lv_r)
                    av_h = float(dataset.measure_accuracy(out_high[idxn], labels).detach().cpu())
                    av_l = float(dataset.measure_accuracy(out_low[idxn], labels).detach().cpu())
                    av_r = float(dataset.measure_accuracy(out_rand_[idxn], labels).detach().cpu())
                    acc_high.append(av_h)
                    acc_low.append(av_l)
                    acc_rand.append(av_r)
                progdrop_loss_high[:, dropidx, layer_i] += torch.tensor(loss_high)
                progdrop_loss_low[:, dropidx, layer_i]  += torch.tensor(loss_low)
                progdrop_loss_rand[:, dropidx, layer_i] += torch.tensor(loss_rand)
                progdrop_acc_high[:, dropidx, layer_i]  += torch.tensor(acc_high)
                progdrop_acc_low[:, dropidx, layer_i]   += torch.tensor(acc_low)
                progdrop_acc_rand[:, dropidx, layer_i]  += torch.tensor(acc_rand)
    progdrop_loss_high /= num_batches
    progdrop_loss_low  /= num_batches
    progdrop_loss_rand /= num_batches
    progdrop_acc_high  /= num_batches
    progdrop_acc_low   /= num_batches
    progdrop_acc_rand  /= num_batches
    results = {
        "progdrop_loss_high": progdrop_loss_high,
        "progdrop_loss_low":  progdrop_loss_low,
        "progdrop_loss_rand": progdrop_loss_rand,
        "progdrop_acc_high":  progdrop_acc_high,
        "progdrop_acc_low":   progdrop_acc_low,
        "progdrop_acc_rand":  progdrop_acc_rand,
        "dropout_fraction":   drop_fraction,
        "by_layer":           by_layer,
    }
    return results

def eigenvector_dropout_experiment(exp, nets, dataset, eigen_results, train_set=False):
    evec_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
    )
    evec_dropout_results = eigenvector_dropout(
        nets,
        dataset,
        eigen_results["eigvals"],
        eigen_results["eigvecs"],
        **evec_params
    )
    return evec_dropout_results, evec_params

def evaluate_pretrained_model(net, dataset):
    net.eval()
    device = next(net.parameters()).device
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in dataset.test_loader:
            images, labels = dataset.unwrap_batch(batch, device=device)
            outputs = net(images)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Pretrained Model Accuracy: {accuracy:.2f}%")
    return accuracy