import os
import torch
from tqdm import tqdm

from alignment.utils import load_checkpoints, test_nets
from alignment.alignment_metrics import AlignmentMetrics
from alignment.train import train, test


def parse_alignment_to_tensor(alignment_list, aggregate=True, by_layer=False):
    """
    Convert a list of alignment records (each with "data" => per-net/per-layer Tensors) into
    a unified Tensor for dropout. If by_layer=False => (num_nets, total_nodes).
    If by_layer=True => (num_layers, num_nets, global_max_nodes).

    We do heavy zero-padding if by_layer=True and different layers have different # of nodes.

    alignment_list is typically results["alignment"], a list of dicts:
      [ { "epoch":..., "batch":..., "data":[ [ { "RQ":...}, ...], [ { "RQ":...}, ...], ... ] }, ... ]
    """
    if len(alignment_list) == 0:
        raise ValueError("parse_alignment_to_tensor: empty alignment_list")

    if not by_layer:
        # Same as before: cat all layers for each net => shape (#nets, total_nodes)
        # Then we can stack across multiple records => (#records, #nets, total_nodes)
        # and optionally average if aggregate=True
        all_records_tensors = []
        for record in alignment_list:
            netwise_tensors = []
            for net_i_data in record["data"]:
                node_tensors = []
                for layer_dict in net_i_data:
                    if "RQ" not in layer_dict:
                        raise ValueError("Expected 'RQ' in layer_dict")
                    node_tensors.append(layer_dict["RQ"].flatten())
                # cat => shape (sum_of_layer_nodes,)
                netwise_tensors.append(torch.cat(node_tensors, dim=0))  # (total_nodes,)
            netwise_tsr = torch.stack(netwise_tensors, dim=0)  # (#nets, total_nodes)
            all_records_tensors.append(netwise_tsr)
        bigstack = torch.stack(all_records_tensors, dim=0)     # (#records, #nets, total_nodes)
        if aggregate:
            return bigstack.mean(dim=0)                       # => (#nets, total_nodes)
        else:
            return bigstack.mean(dim=0)                       # or just pick last record, etc.
    else:
        # by_layer=True => each layer can have a different # of nodes. We unify all records.
        # We pick the global max # nodes among *all layers, all records* to allow stacking.

        # 1) Find max number of layers overall:
        max_layers_found = 0
        # 2) Find global_max_nodes => single int for ALL layers:
        global_max_nodes = 0

        for record in alignment_list:
            net_list = record["data"]  # shape => (#nets)
            for net_i_data in net_list:
                # net_i_data => list of layer_dicts
                num_layers_net = len(net_i_data)
                max_layers_found = max(max_layers_found, num_layers_net)
                # check node counts
                for layer_idx, layer_dict in enumerate(net_i_data):
                    if "RQ" not in layer_dict:
                        raise ValueError("Expected 'RQ' in layer_dict")
                    ncount = layer_dict["RQ"].numel()
                    if ncount > global_max_nodes:
                        global_max_nodes = ncount

        # 3) Build array => for each record => shape (#layers= max_layers_found, #nets, global_max_nodes)
        all_record_arrays = []
        for record in alignment_list:
            net_list = record["data"]
            num_nets = len(net_list)
            # for this record, build a list => each layer => (#nets, global_max_nodes) => zero-padded
            layer_arrays = []
            for layer_i in range(max_layers_found):
                # gather each net's data for layer_i
                net_arrays = []
                for net_i_data in net_list:
                    if layer_i < len(net_i_data):
                        lay_dict = net_i_data[layer_i]
                        rq_vals = lay_dict["RQ"].flatten()
                    else:
                        rq_vals = torch.zeros(0)
                    node_count = rq_vals.numel()
                    if node_count < global_max_nodes:
                        pad_size = global_max_nodes - node_count
                        rq_vals = torch.cat([rq_vals, torch.zeros(pad_size, dtype=rq_vals.dtype, device=rq_vals.device)], dim=0)
                    net_arrays.append(rq_vals)  # shape => (global_max_nodes,)
                layer_tensor = torch.stack(net_arrays, dim=0)  # (#nets, global_max_nodes)
                layer_arrays.append(layer_tensor)
            # stack => (#layers_found, #nets, global_max_nodes)
            record_3d = torch.stack(layer_arrays, dim=0)  
            all_record_arrays.append(record_3d)

        # shape => (#records, max_layers_found, #nets, global_max_nodes)
        bigstack = torch.stack(all_record_arrays, dim=0)
        if aggregate:
            final_3d = bigstack.mean(dim=0)  # => (max_layers_found, #nets, global_max_nodes)
        else:
            final_3d = bigstack.mean(dim=0)
        return final_3d


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
        idx_alignment = [torch.argsort(parsed, dim=1)]
        num_snapshots = 1
        num_nets_ = parsed.shape[0]
    else:
        num_layers, num_nets_, global_max_nodes = parsed.shape
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
                for layer_i in range(num_snapshots):
                    layer_idx_sorted = idx_alignment[layer_i]
                    dev = layer_idx_sorted.device
                    node_count = layer_idx_sorted.size(1)
                    dn = int(node_count * fraction)
                    if dn > 0:
                        hi = torch.index_select(layer_idx_sorted, dim=1,
                              index=torch.arange(node_count - dn, node_count, device=dev))
                        lo = torch.index_select(layer_idx_sorted, dim=1,
                              index=torch.arange(dn, device=dev))
                        rr = []
                        for i_net in range(num_nets_):
                            perm = torch.randperm(node_count, device=dev)
                            rr.append(perm[:dn])
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
                dev = idx_alignment[0].device
                node_count = idx_alignment[0].size(1)
                dn = int(node_count * fraction)
                if dn > 0:
                    hi = torch.index_select(idx_alignment[0], dim=1,
                         index=torch.arange(node_count - dn, node_count, device=dev))
                    lo = torch.index_select(idx_alignment[0], dim=1,
                         index=torch.arange(dn, device=dev))
                    rr = []
                    for i_net in range(num_nets_):
                        perm = torch.randperm(node_count, device=dev)
                        rr.append(perm[:dn])
                    rr = torch.stack(rr, dim=0)
                else:
                    hi = idx_alignment[0][:, :0]
                    lo = idx_alignment[0][:, :0]
                    rr = idx_alignment[0][:, :0]

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