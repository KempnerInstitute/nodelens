import os
import torch
from tqdm import tqdm

from alignment.utils import load_checkpoints, test_nets
from alignment.alignment_metrics import AlignmentMetrics
from alignment.train import train, test

def parse_alignment_to_tensor(alignment_list, aggregate=True, by_layer=False):
    if len(alignment_list) == 0:
        raise ValueError("parse_alignment_to_tensor: empty alignment_list")
    all_records_tensors = []
    if by_layer:
        # return a 3D shape => (#records, #layers, #nets * #nodes_per_layer)
        # we collect each layer separately, so each layer is a "snapshot"
        layer_tensors_list = []
        max_layers = 0
        for record in alignment_list:
            netwise_layers = []
            for net_i_data in record["data"]:
                layer_list = []
                for layer_dict in net_i_data:
                    if "RQ" not in layer_dict:
                        raise ValueError("Expected 'RQ' in layer_dict")
                    layer_list.append(layer_dict["RQ"].flatten())
                netwise_layers.append(layer_list)
            max_layers = max(max_layers, len(netwise_layers[0]))
            layer_tensors_list.append(netwise_layers)
        # shape => (#records, #nets, #layers, ?)
        # unify them
        rec_level = []
        for rec_idx, rec_data in enumerate(layer_tensors_list):
            # rec_data => list of #nets, each is a list of #layers
            # we want shape => (#layers, #nets, total_nodes_this_layer)
            # first find how many layers from the first net
            rec_layers = []
            num_nets = len(rec_data)
            for layer_i in range(max_layers):
                per_net = []
                for net_idx in range(num_nets):
                    net_layers = rec_data[net_idx]
                    if layer_i < len(net_layers):
                        per_net.append(net_layers[layer_i])
                    else:
                        per_net.append(torch.zeros(0))
                # cat each net's flattened RQ
                per_net_cat = []
                for item in per_net:
                    per_net_cat.append(item)
                per_net_cat = torch.stack(per_net_cat, dim=0)
                rec_layers.append(per_net_cat)
            rec_level.append(rec_layers)
        # shape => (#records, #layers, #nets, #nodes?)
        # we keep it that way
        # if aggregate => average on #records
        # we want final => (#layers, #nets, #nodes)
        rec_level_tsr = []
        for rec_idx, rec_data in enumerate(rec_level):
            # rec_data => list of #layers => each is (#nets, #nodes_layer)
            rec_data_stack = []
            for layer in rec_data:
                rec_data_stack.append(layer)
            rec_level_tsr.append(rec_data_stack)
        # now rec_level_tsr => (#records, #layers, (#nets, #nodes))
        # stack
        # shape => (#records, #layers, #nets, #nodes)
        out_list = []
        for rec_data_stack in rec_level_tsr:
            out_list.append(torch.stack(rec_data_stack, dim=0))
        bigstack = torch.stack(out_list, dim=0)
        if aggregate:
            # shape => (#records, #layers, #nets, #nodes)
            return bigstack.mean(dim=0)
        else:
            return bigstack.mean(dim=0)
    else:
        for record in alignment_list:
            netwise_tensors = []
            for net_i_data in record["data"]:
                node_tensors = []
                for layer_dict in net_i_data:
                    if "RQ" not in layer_dict:
                        raise ValueError("Expected 'RQ' in layer_dict")
                    node_tensors.append(layer_dict["RQ"].flatten())
                netwise_tensors.append(torch.cat(node_tensors, dim=0))
            netwise_tsr = torch.stack(netwise_tensors, dim=0)
            all_records_tensors.append(netwise_tsr)
        bigstack = torch.stack(all_records_tensors, dim=0)
        if aggregate:
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
    if by_layer:
        # shape => (#layers, #nets, #nodes)
        num_layers, num_nets, num_nodes = parsed.shape
        num_snapshots = num_layers
        idx_alignment = []
        for layer_i in range(num_layers):
            idx_alignment.append(torch.argsort(parsed[layer_i], dim=1))
    else:
        # shape => (#nets, #nodes)
        num_snapshots = 1
        idx_alignment = [torch.argsort(parsed, dim=1)]
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    print("Progressive Dropout:")
    progdrop_loss_high = []
    progdrop_loss_low = []
    progdrop_loss_rand = []
    progdrop_acc_high = []
    progdrop_acc_low = []
    progdrop_acc_rand = []
    if by_layer:
        num_nets_ = parsed.shape[1]
        progdrop_loss_high = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_loss_low  = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_loss_rand = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_acc_high  = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_acc_low   = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
        progdrop_acc_rand  = torch.zeros((num_nets_, num_drops, num_layers), device="cpu")
    else:
        num_nets_ = parsed.shape[0]
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
            drop_num = None
            if by_layer:
                drop_high_list, drop_low_list, drop_rand_list = [], [], []
                for layer_i in range(num_layers):
                    device_of_idx = idx_alignment[layer_i].device
                    num_nodes = idx_alignment[layer_i].size(1)
                    dn = int(num_nodes * fraction)
                    if dn > 0:
                        hi = torch.index_select(idx_alignment[layer_i], dim=1, index=torch.arange(num_nodes - dn, num_nodes, device=device_of_idx))
                        lo = torch.index_select(idx_alignment[layer_i], dim=1, index=torch.arange(dn, device=device_of_idx))
                        rr = []
                        for i_net in range(num_nets_):
                            perm = torch.randperm(num_nodes, device=device_of_idx)
                            rr.append(perm[:dn])
                        rr = torch.stack(rr, dim=0)
                    else:
                        hi = idx_alignment[layer_i][:, :0]
                        lo = idx_alignment[layer_i][:, :0]
                        rr = idx_alignment[layer_i][:, :0]
                    drop_high_list.append(hi)
                    drop_low_list.append(lo)
                    drop_rand_list.append(rr)
                for layer_i in range(num_layers):
                    out_high, out_low, out_rand_ = [], [], []
                    for i_net, net in enumerate(nets):
                        oh, _ = net.forward_targeted_dropout(images, [drop_high_list[layer_i][i_net]], [layer_i])
                        ol, _ = net.forward_targeted_dropout(images, [drop_low_list[layer_i][i_net]], [layer_i])
                        or_, _= net.forward_targeted_dropout(images, [drop_rand_list[layer_i][i_net]], [layer_i])
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
                device_of_idx = idx_alignment[0].device
                num_nodes = idx_alignment[0].size(1)
                dn = int(num_nodes * fraction)
                if dn > 0:
                    hi = torch.index_select(idx_alignment[0], dim=1, index=torch.arange(num_nodes - dn, num_nodes, device=device_of_idx))
                    lo = torch.index_select(idx_alignment[0], dim=1, index=torch.arange(dn, device=device_of_idx))
                    rr = []
                    for i_net in range(num_nets_):
                        perm = torch.randperm(num_nodes, device=device_of_idx)
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
    if by_layer:
        progdrop_loss_high /= num_batches
        progdrop_loss_low  /= num_batches
        progdrop_loss_rand /= num_batches
        progdrop_acc_high  /= num_batches
        progdrop_acc_low   /= num_batches
        progdrop_acc_rand  /= num_batches
    else:
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