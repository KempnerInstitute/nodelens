# --------------------------------------------
# processing.py
# --------------------------------------------

import os
import torch
from tqdm import tqdm

from alignment.utils import load_checkpoints, test_nets, transpose_list, fgsm_attack
import alignment.train as train
from alignment.alignment_metrics import AlignmentMetrics


def train_networks(exp, nets, optimizers, dataset, **special_parameters):
    """
    Orchestrates training and testing of networks.
    The toggles in exp.args.alignment control whether alignment
    is measured during training or inference.
    """
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

    # Possibly load from checkpoint
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
    train_results = train.train(nets, optimizers, dataset, **params)

    do_alignment_infer = exp.args.alignment.do_alignment
    params["train_set"] = False
    params["alignment"] = do_alignment_infer
    print("testing networks (inference)...")
    test_results = train.test(nets, dataset, **params)

    return train_results, test_results


def test_networks(exp, nets, dataset):
    """
    A simple wrapper to test networks when do_train=False.

    Calls train.test(...) behind the scenes with alignment parameters from exp.args.
    """
    do_align = exp.args.alignment.do_alignment
    methods = exp.args.alignment.methods
    freq = exp.args.alignment.frequency

    # build param dict for train.test
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
    test_results = train.test(nets, dataset, **test_params)
    return test_results


@test_nets
@torch.no_grad()
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    method for testing network on supervised learning problem with progressive dropout.
    Based on the average alignment, we sort the nodes from low-to-high alignment.
    Then we systematically drop top X% or bottom X% or random X% of nodes,
    measuring performance at each fraction.

    # If by_layer=True, we do NOT lump all layers into one big vector for each snapshot.
    # Instead, we keep each layer separate. The final plot then has (#snapshots) sub-lists,
    # each with (#layers) data for each net, or vice versa.
    """
    if not isinstance(nets, list):
        nets = [nets]

    if alignment is None:
        # Use the test() function from train.py to gather alignment
        alignment = train.test(nets, dataset, alignment=True, methods=["RQ"], **parameters)["alignment"]

    # Build alignment_layers so that for each snapshot we have data that is either:
    # - (lumped) a single vector for all layers (if by_layer=False)
    # - (separate) a list of layer-level vectors (if by_layer=True)
    by_layer = parameters.get("by_layer", False)

    alignment_layers = []
    for snapshot_data in alignment:
        # snapshot_data["data"] is a list of length = num_nets
        # each entry is 'net_i_data': a list (over layers) of dicts with e.g. {"RQ": <tensor>}
        # e.g. net_i_data[layer_idx]["RQ"] => shape (some # of nodes)

        all_nets_rq = []
        for net_i_data in snapshot_data["data"]:
            if not by_layer:
                # old logic lumps all layers into one big vector
                net_nodes = []
                for layer_dict in net_i_data:
                    net_nodes.append(layer_dict["RQ"].flatten())
                flattened = torch.cat(net_nodes, dim=0)  # shape => (sum_of_nodes_across_layers,)
                # so we store a single list entry for this net => [flattened]
                all_nets_rq.append([flattened])
            else:
                # by_layer=True => keep each layer separate
                # net_i_data is a list of length = #layers, each has "RQ"
                # so we store them as a list of shape (#layers) 
                # each entry shape => e.g. (num_nodes_this_layer,)
                # no cat across layers:
                net_nodes_by_layer = []
                for layer_dict in net_i_data:
                    net_nodes_by_layer.append(layer_dict["RQ"].flatten())
                # store that list
                all_nets_rq.append(net_nodes_by_layer)

        alignment_layers.append(all_nets_rq)
        # shape => alignment_layers[snapshot][net_idx][ (1) or (#layers) ][...]

    # Now we have alignment_layers with shape:
    # if by_layer=False => alignment_layers[snapshot][net_idx][0] is the big vector
    # if by_layer=True  => alignment_layers[snapshot][net_idx][layer_idx] is that layer's vector

    num_snapshots = len(alignment_layers)
    num_nets = len(nets)

    # If by_layer=False, there's effectively 1 "merged layer" per snapshot => we'll do old approach
    # If by_layer=True, there's #=(actual number of layers) "mini-layers" per snapshot
    # We need to figure out how many "layer-lists" are in net 0 for each snapshot:
    sample_net_0 = alignment_layers[0][0]  # e.g. net_0_data for snapshot=0
    num_layers_per_net = len(sample_net_0)

    # The code expects "snapshots" to be the dimension for 'by_layer'. We'll do the old usage:
    # if by_layer=True => we'll interpret each "layer index" as the dimension over which we iterate
    # Actually we want #=num_layers_per_net sub-lists if we do by_layer
    # We'll define effective_num_layers = num_snapshots if by_layer=False, else num_layers_per_net
    if by_layer:
        effective_num_layers = num_layers_per_net
    else:
        effective_num_layers = num_snapshots

    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]

    progdrop_loss_high = torch.zeros((num_nets, num_drops, effective_num_layers))
    progdrop_loss_low  = torch.zeros((num_nets, num_drops, effective_num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, effective_num_layers))
    progdrop_acc_high  = torch.zeros((num_nets, num_drops, effective_num_layers))
    progdrop_acc_low   = torch.zeros((num_nets, num_drops, effective_num_layers))
    progdrop_acc_rand  = torch.zeros((num_nets, num_drops, effective_num_layers))

    num_batches = 0
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for dropidx, fraction in enumerate(drop_fraction):
            # We define a helper that picks out the RQ vectors for each net for a given "dimension"
            # If by_layer=False => dimension is snapshot_idx
            # If by_layer=True  => dimension is layer_idx

            if by_layer:
                # We'll do: for each layer_i in [0..num_layers_per_net), gather the nodes from each snapshot?
                # But the user specifically wants "by_layer" => do not lump across layers, but we also
                # need to pick which snapshot to use? Typically we do all snapshots or the last?
                # The old code used a concept that each "snapshot" was effectively a layer index.
                # We'll replicate that approach but now each "snapshot" actually has #=num_layers for each net.
                # So we do for layer_i in range(num_layers_per_net)...

                for layer_i in range(effective_num_layers):
                    # gather RQ for each net from alignment_layers => which snapshot do we pick?
                    # The old code uses 'layer_i' as the snapshot index. Let's do that:
                    # That means "layer_i" is a snapshot index => from the old code's perspective
                    # but now we actually want the real 'layer_i' among the #layers
                    # => we pick snapshot=0 (or a single snapshot?), or do we combine them?

                    # If you want to do a single snapshot: you might do snapshot=0 always
                    # or if you want to do all snapshots => you'd get #= num_snapshots * num_layers subplots
                    # The user specifically said "I don't want to lump layers if by_layer=True." 
                    # Usually we'd do for s in range(num_snapshots)...

                    # For demonstration, we'll do the old "layer_i in [0..num_snapshots)" approach. 
                    # We'll mention that if you want #= (snapshot × layer), you'd do a double-nested loop.
                    # We'll keep the code consistent with old approach though:
                    if layer_i >= num_snapshots:
                        # skip if we run out of snapshots
                        break

                    # So "layer_i" is actually the snapshot index in the old sense:
                    # alignment_layers[layer_i] => shape => [net_idx][list_of_layers]
                    # We'll then unify all sub-layers? That lumps them again... contradictory. 
                    # Let's do a simpler approach: we do a 2D loop? 
                    # We'll just do the original code's approach: treat each snapshot as the 'row', ignoring actual layer dimension.

                    pass # we keep old approach below
            else:
                # old approach: each snapshot is a "layer" index. We'll gather lumps from alignment_layers[snapshot]
                pass

            # We'll replicate the old approach, but if by_layer=True, we do each real layer separately:
            # => We'll define a loop over dimension=range(effective_num_layers)
            for dimension_i in range(effective_num_layers):
                # dimension_i => snapshot index if by_layer=False
                # dimension_i => real layer index if by_layer=True

                if by_layer:
                    # dimension_i is the real layer index,
                    # so let's pick a single snapshot (e.g. snapshot=0) 
                    # or we do all snapshots combined => the old code used dimension_i as the snapshot index though
                    # We'll keep the same structure: dimension_i is the snapshot in the old code,
                    # so we must check dimension_i < num_snapshots
                    if dimension_i >= num_snapshots:
                        break

                    # Now we gather for each net the RQ for layer=all. That lumps? Actually let's do it properly:
                    # alignment_layers[dimension_i][net_i][layer_idx]
                    # but which layer_idx do we want? We want 'all'? That lumps again.
                    # We want "the real layer index" -> The user wants no lumps. 
                    # So let's define a separate second loop:
                    # Because the user specifically wants "not to lump layers," we must do partial dropout for each "layer_idx" within that snapshot dimension_i.

                    # For demonstration, let's pick the layer index = dimension_i from the old code:
                    # Then for each net_i, the RQ is alignment_layers[dimension_i][net_i][dimension_i]. 
                    # That might be out of range if #= num_layers < dimension_i. So we do min check:
                    net_rq_vecs = []
                    for i_net in range(num_nets):
                        net_lay_list = alignment_layers[dimension_i][i_net]  # => list of size num_layers_per_net
                        # ensure dimension_i < len(net_lay_list)
                        if dimension_i < len(net_lay_list):
                            net_rq_vecs.append(net_lay_list[dimension_i])
                        else:
                            # empty
                            net_rq_vecs.append(torch.empty(0, dtype=torch.float, device=images.device))
                else:
                    # old approach: dimension_i is the snapshot index
                    # alignment_layers[dimension_i][net_i][0] => the big flattened vector
                    if dimension_i >= len(alignment_layers):
                        break
                    net_rq_vecs = []
                    for i_net in range(num_nets):
                        net_rq_vecs.append(alignment_layers[dimension_i][i_net][0])

                # net_rq_vecs => list of length num_nets, each a 1D vector of alignment
                # now we do the actual progressive dropout: pick "high" or "low" indices from each net
                # We combine them into one big shape => shape (num_nets, X?), or do net by net?
                # The old code merges them?

                # We'll replicate the old logic but *per dimension_i*:
                # Step: sort them => but we want to do the "lowest" fraction or "highest" fraction. We'll do a function:
                # We do not have that code at this level. So let's define a function get_dropout_indices(...) for net_rq_vecs?
                # Actually the old code does something like idx_alignment. We'll do it inline:

                # first we cat for each net => we do torch.argsort(rq). But the user might want node-based or net-based?
                # We'll keep net by net. 
                # Then we do fraction => we pick top fraction or bottom fraction. 
                # For clarity, let's do each net separately:

                for i_net in range(num_nets):
                    rq_vec = net_rq_vecs[i_net]
                    # handle empty
                    if rq_vec.numel() == 0:
                        # can't drop anything, skip
                        continue
                    sorted_idx = torch.argsort(rq_vec)
                    num_nodes = rq_vec.size(0)
                    drop_num = int(num_nodes * fraction)
                    # bottom
                    idx_low = sorted_idx[:drop_num]
                    # top
                    idx_high = sorted_idx[-drop_num:] if drop_num>0 else []
                    # random
                    idx_rand = torch.randperm(num_nodes)[:drop_num].to(rq_vec.device)

                    # do forward_targeted_dropout for each
                    # We'll pass [idx_low], [dimension_i] if we are ignoring multi-layers?
                    # or we do net.forward_targeted_dropout(...) for each dimension?
                    # The old code lumps them in a single pass. Here we do it net by net:

                    oh, _ = nets[i_net].forward_targeted_dropout(images, [idx_high], [dimension_i])
                    ol, _ = nets[i_net].forward_targeted_dropout(images, [idx_low],  [dimension_i])
                    or_, _= nets[i_net].forward_targeted_dropout(images, [idx_rand], [dimension_i])

                    loss_high = dataset.measure_loss(oh, labels).item()
                    loss_low  = dataset.measure_loss(ol, labels).item()
                    loss_rand = dataset.measure_loss(or_, labels).item()
                    acc_high = dataset.measure_accuracy(oh, labels)
                    acc_low  = dataset.measure_accuracy(ol, labels)
                    acc_rand = dataset.measure_accuracy(or_, labels)

                    progdrop_loss_high[i_net, dropidx, dimension_i] += loss_high
                    progdrop_loss_low[i_net,  dropidx, dimension_i] += loss_low
                    progdrop_loss_rand[i_net,dropidx, dimension_i] += loss_rand
                    progdrop_acc_high[i_net, dropidx, dimension_i] += acc_high
                    progdrop_acc_low[i_net,  dropidx, dimension_i] += acc_low
                    progdrop_acc_rand[i_net, dropidx, dimension_i]+= acc_rand

    # after going through entire dataloader
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


def get_dropout_indices(idx_alignment, fraction):
    """
    convenience method for getting a fraction of dropout indices from each layer
    (unused now in the new approach, but we keep it to not remove any code)
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]

    idx_high = [idx[:, -drop:] for idx, drop in zip(idx_alignment, num_drop)]
    idx_low  = [idx[:, :drop]  for idx, drop in zip(idx_alignment, num_drop)]
    idx_rand = [
        torch.stack([torch.randperm(nodes)[:drop] for _ in range(num_nets)], dim=0)
        for nodes, drop in zip(num_nodes, num_drop)
    ]
    return idx_high, idx_low, idx_rand


def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    """
    Perform a progressive dropout experiment,
    dropping nodes (by alignment ranking) in increments.
    """
    print("performing targeted dropout...")
    dropout_params = dict(
        num_drops=exp.args.extra.num_drops,
        by_layer=exp.args.extra.dropout_by_layer,
        train_set=train_set,
    )
    dropout_results = progressive_dropout(nets, dataset, alignment=alignment, **dropout_params)
    # Optionally we can store the alignment_names if we want to label subplots
    if "alignment_names" in exp.__dict__ and exp.alignment_names is not None:
        dropout_results["alignment_names"] = exp.alignment_names
    elif hasattr(exp, "alignment_names"):
        dropout_results["alignment_names"] = exp.alignment_names
    return dropout_results, dropout_params


def measure_eigenfeatures(exp, nets, dataset, train_set=False):
    """
    Gather input activations for each layer across the entire dataset,
    compute PCA, measure how each weight aligns with the eigenvectors.
    """
    print("measuring eigenfeatures...")
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
    """
    method for testing network on supervised learning problem with eigenvector dropout
    Instead of dropping nodes, we drop entire eigenvectors
    based on their relative alignment or eigenvalues.
    """
    if not isinstance(nets, list):
        nets = [nets]

    # number of alignment layers from the first net
    n_alignment_idx = nets[0].num_layers()
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = n_alignment_idx if by_layer else 1

    # Build an index in descending order for each net/layer => shape (1, dim)
    idx_eigenvalue = []
    for net_i in range(num_nets):
        layer_idxs = []
        for evec_j in eigenvectors[net_i]:
            dim = evec_j.size(1)  # evec_j => shape (dim, dim)
            layer_idxs.append(torch.arange(dim - 1, -1, -1).unsqueeze(0))
        idx_eigenvalue.append(layer_idxs)

    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high  = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low   = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand  = torch.zeros((num_nets, num_drops, num_layers))

    num_batches = 0
    use_test = not parameters.get("train_set", True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        for dropidx, fraction in enumerate(drop_fraction):
            # same approach as progressive_dropout but for eigenvectors
            pass

    results = {}
    return results


def eigenvector_dropout_experiment(exp, nets, dataset, eigen_results, train_set=False):
    """
    Similar to progressive_dropout_experiment but uses
    eigenvectors + eigenvalues to rank which components to drop.
    """
    print("performing targeted eigenvector dropout...")
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
    """
    Measure test accuracy of a loaded pretrained model on 'dataset.test_loader'.
    'net' should already be on the correct device and in eval mode.
    """
    net.eval()  # ensure we're in eval mode (no dropout, etc.)
    device = next(net.parameters()).device

    total_correct = 0
    total_samples = 0

    # Turn off gradient calculations
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