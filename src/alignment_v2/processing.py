import os

import torch
from tqdm import tqdm

from alignment_v2 import train
from alignment_v2.utils import load_checkpoints, test_nets, transpose_list, fgsm_attack

def train_networks(exp, nets, optimizers, dataset, **special_parameters):
    """train and test networks"""
    # alignment is only computed if compute_alignment == True and compute_during_training == True
    do_alignment = exp.args.alignment.compute_alignment and exp.args.alignment.compute_during_training

    parameters = dict(
        train_set=True,
        num_epochs=exp.args.training.epochs,
        alignment=do_alignment,
        measure_weight_deltas=exp.args.alignment.measure_weight_deltas,
        alignment_eval_frequency=exp.args.alignment.alignment_eval_frequency,
        methods=exp.args.alignment.methods,
        run=exp.run,
    )

    parameters.update(**special_parameters)

    if exp.args.checkpointing.use_prev & os.path.isfile(exp.get_checkpoint_path()):
        nets, optimizers, results = load_checkpoints(nets, optimizers, exp.args.device, exp.get_checkpoint_path())
        for net in nets:
            net.train()

        parameters["num_complete"] = results["epoch"] + 1
        parameters["results"] = results
        print("loaded networks from previous checkpoint")

    if exp.args.checkpointing.save_checkpoints:
        parameters["save_checkpoints"] = (True, exp.args.checkpointing.frequency, exp.get_checkpoint_path(), exp.args.device)

    print("training networks...")
    train_results = train.train(nets, optimizers, dataset, **parameters)

    # do testing loop if we want alignment during inference or we simply want final evaluation
    do_alignment_inference = exp.args.alignment.compute_alignment and exp.args.alignment.compute_during_inference

    parameters["train_set"] = False
    parameters["alignment"] = do_alignment_inference

    print("testing networks (inference)...")
    test_results = train.test(nets, dataset, **parameters)

    return train_results, test_results

def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    """
    perform a progressive dropout (of nodes) experiment
    alignment is optional, but will be recomputed if you've already measured it.
    """
    print("performing targeted dropout...")
    dropout_parameters = dict(num_drops=exp.args.extra.num_drops, by_layer=exp.args.extra.dropout_by_layer, train_set=train_set)
    dropout_results = train.progressive_dropout(nets, dataset, alignment=alignment, **dropout_parameters)
    return dropout_results, dropout_parameters

def measure_eigenfeatures(exp, nets, dataset, train_set=False):
    """
    measure eigenfeatures for each net
    """
    print("measuring eigenfeatures...")
    from alignment.core.utils import transpose_list
    beta, eigvals, eigvecs, class_betas = [], [], [], []
    for net in tqdm(nets):
        inputs, labels = net._process_collect_activity(
            dataset,
            train_set=train_set,
            with_updates=False,
            use_training_mode=False,
        )
        eigenfeatures = net.measure_eigenfeatures(inputs, with_updates=False)
        beta_by_class = net.measure_class_eigenfeatures(inputs, labels, eigenfeatures[2], rms=False, with_updates=False)
        beta.append(eigenfeatures[0])
        eigvals.append(eigenfeatures[1])
        eigvecs.append(eigenfeatures[2])
        class_betas.append(beta_by_class)

    class_names = getattr(dataset.train_loader if train_set else dataset.test_loader, "dataset").classes
    return dict(
        beta=beta,
        eigvals=eigvals,
        eigvecs=eigvecs,
        class_betas=class_betas,
        class_names=class_names,
    )

def eigenvector_dropout(exp, nets, dataset, eigen_results, train_set=False):
    """
    do targeted eigenvector dropout with precomputed eigenfeatures
    """
    print("performing targeted eigenvector dropout...")
    evec_dropout_parameters = dict(num_drops=exp.args.extra.num_drops, by_layer=exp.args.extra.dropout_by_layer, train_set=train_set)
    evec_dropout_results = train.eigenvector_dropout(nets, dataset, eigen_results["eigvals"], eigen_results["eigvecs"], **evec_dropout_parameters)
    return evec_dropout_results, evec_dropout_parameters

@test_nets
def measure_adversarial_attacks(nets, dataset, exp, eigen_results, train_set=False, **parameters):
    """
    do adversarial attack and measure structure with regards to eigenfeatures
    """

    def get_beta(inputs, eigenvectors):
        return [input.cpu() @ evec for input, evec in zip(inputs, eigenvectors)]

    epsilons = parameters.get("epsilons")
    use_sign = parameters.get("use_sign")
    fgsm_transform = parameters.get("fgsm_transform", lambda x: x)

    eigenvectors = eigen_results["eigvecs"]

    num_eps = len(epsilons)
    num_nets = len(nets)
    accuracy = torch.zeros((num_nets, num_eps))
    examples = [[[] for _ in range(num_eps)] for _ in range(num_nets)]
    betas = [[torch.zeros((num_nets, evec.size(0))) for evec in eigenvectors[0]] for _ in range(num_eps)]

    dataloader = dataset.train_loader if train_set else dataset.test_loader

    for batch in tqdm(dataloader):
        input, labels = dataset.unwrap_batch(batch)

        inputs = [input.clone() for _ in range(num_nets)]
        for inp in inputs:
            inp.requires_grad = True

        outputs = [net(inp, store_hidden=True) for net, inp in zip(nets, inputs)]
        input_to_layers = [net.get_layer_inputs(inp, precomputed=True) for net, inp in zip(nets, inputs)]
        init_preds = [torch.argmax(output, axis=1) for output in outputs]
        least_likely = [torch.argmin(output, axis=1) for output in outputs]

        c_betas = transpose_list([get_beta(inp, evec) for inp, evec in zip(input_to_layers, eigenvectors)])
        s_betas = [torch.stack(cb) for cb in c_betas]

        loss = [dataset.measure_loss(output, labels) for output in outputs]

        for net in nets:
            net.zero_grad()

        for l in loss:
            l.backward()

        data_grads = [inp.grad.data for inp in inputs]

        for epsidx, eps in enumerate(epsilons):
            perturbed_inputs = [fgsm_attack(inp, eps, data_grad, fgsm_transform, use_sign) for inp, data_grad in zip(inputs, data_grads)]
            outputs = [net(perturbed_input, store_hidden=True) for net, perturbed_input in zip(nets, perturbed_inputs)]
            input_to_layers = [net.get_layer_inputs(perturbed_input, precomputed=True) for net, perturbed_input in zip(nets, perturbed_inputs)]
            c_eps_betas = transpose_list([get_beta(inp, evec) for inp, evec in zip(input_to_layers, eigenvectors)])
            s_eps_betas = [torch.stack(ceb) for ceb in c_eps_betas]
            d_eps_betas = [sebeta - sbeta for sebeta, sbeta in zip(s_eps_betas, s_betas)]
            rms_betas = [torch.sqrt(torch.mean(db**2, dim=1)) for db in d_eps_betas]

            for ii, rbeta in enumerate(rms_betas):
                betas[epsidx][ii] += rbeta.detach()

            final_preds = [torch.argmax(output, axis=1) for output in outputs]
            accuracy[:, epsidx] += torch.tensor([sum(final_pred == labels).cpu() for final_pred in final_preds])

            idx_success = [
                torch.where((init_pred == labels) & (final_pred != labels))[0].cpu() for init_pred, final_pred in zip(init_preds, final_preds)
            ]

            adv_exs = [perturbed_input.detach().cpu().numpy() for perturbed_input in perturbed_inputs]
            for ii, (adv_ex, idx, init_pred, final_pred) in enumerate(zip(adv_exs, idx_success, init_preds, final_preds)):
                examples[ii][epsidx].append((init_pred[idx], final_pred[idx], adv_ex[idx]))

    accuracy = accuracy / float(len(dataloader.dataset))
    from alignment.core.utils import transpose_list
    betas = transpose_list([[cb / float(len(dataloader.dataset)) for cb in beta] for beta in betas])
    return dict(accuracy=accuracy, betas=betas, examples=examples, epsilons=epsilons, use_sign=use_sign)