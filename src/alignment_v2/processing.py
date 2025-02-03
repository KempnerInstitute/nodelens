import os
import torch
from tqdm import tqdm
import train
from utils import load_checkpoints, test_nets, transpose_list, fgsm_attack

def train_networks(exp,nets,opts,dataset,**special_parameters):
    prms=dict(
        train_set=True,
        num_epochs=exp.args.training.epochs,
        alignment=not exp.args.alignment.no_alignment,
        delta_weights=exp.args.alignment.delta_weights,
        frequency=exp.args.alignment.frequency,
        run=exp.run
    )
    prms.update(**special_parameters)
    if exp.args.checkpointing.use_prev and os.path.isfile(exp.get_checkpoint_path()):
        nets,opts,res=load_checkpoints(nets,opts,exp.args.device,exp.get_checkpoint_path())
        for n in nets:
            n.train()
        prms["num_complete"]=res["epoch"]+1
        prms["results"]=res
        print("Loaded from checkpoint")
    if exp.args.checkpointing.save_checkpoints:
        prms["save_checkpoints"]=(True,exp.args.checkpointing.frequency,exp.get_checkpoint_path(),exp.args.device)
    print("training...")
    tr=train.train(nets,opts,dataset,**prms)
    print("testing...")
    prms["train_set"]=False
    te=train.test(nets,dataset,**prms)
    return tr,te

def progressive_dropout_experiment(exp,nets,dataset,alignment=None,train_set=False):
    print("targeted dropout...")
    pars=dict(num_drops=exp.args.extra.num_drops,by_layer=exp.args.extra.dropout_by_layer,train_set=train_set)
    out=train.progressive_dropout(nets,dataset,alignment=alignment,**pars)
    return out,pars

def measure_eigenfeatures(exp,nets,dataset,train_set=False):
    print("measuring eigenfeatures...")
    from tqdm import tqdm
    beta,eigvals,eigvecs,class_betas=[],[],[],[]
    for net in tqdm(nets):
        inp,lab=net._process_collect_activity(dataset,train_set=train_set,with_updates=False,use_training_mode=False)
        ef=net.measure_eigenfeatures(inp,with_updates=False)
        c=net.measure_class_eigenfeatures(inp,lab,ef[2],rms=False,with_updates=False)
        beta.append(ef[0]); eigvals.append(ef[1]); eigvecs.append(ef[2]); class_betas.append(c)
    ds=dataset.train_loader if train_set else dataset.test_loader
    cn=ds.dataset.classes
    return dict(beta=beta,eigvals=eigvals,eigvecs=eigvecs,class_betas=class_betas,class_names=cn)

def eigenvector_dropout(exp,nets,dataset,e_res,train_set=False):
    print("eigenvector dropout...")
    pars=dict(num_drops=exp.args.extra.num_drops,by_layer=exp.args.extra.dropout_by_layer,train_set=train_set)
    out=train.eigenvector_dropout(nets,dataset,e_res["eigvals"],e_res["eigvecs"],**pars)
    return out,pars

@test_nets
def measure_adversarial_attacks(nets,dataset,exp,eigen_results,train_set=False,**parameters):
    def get_beta(inputs,eigenvectors):
        return [i.cpu()@e for i,e in zip(inputs,eigenvectors)]
    epsilons=parameters.get("epsilons")
    use_sign=parameters.get("use_sign")
    fgsm_transform=parameters.get("fgsm_transform",lambda x:x)
    evecs=eigen_results["eigvecs"]
    num_eps=len(epsilons)
    num_nets=len(nets)
    acc=torch.zeros((num_nets,num_eps))
    examples=[[[] for _ in range(num_eps)] for _ in range(num_nets)]
    betas=[[torch.zeros((num_nets,e.size(0))) for e in evecs[0]] for _ in range(num_eps)]
    dl=dataset.train_loader if train_set else dataset.test_loader
    from tqdm import tqdm
    for batch in tqdm(dl):
        inp,lab=dataset.unwrap_batch(batch)
        inputs=[inp.clone() for _ in range(num_nets)]
        for i2 in inputs:
            i2.requires_grad=True
        outs=[n(i2,store_hidden=True) for n,i2 in zip(nets,inputs)]
        l_inputs=[n.get_layer_inputs(i2,precomputed=True) for n,i2 in zip(nets,inputs)]
        init_preds=[torch.argmax(o,axis=1) for o in outs]
        c_betas=transpose_list([get_beta(x,e) for x,e in zip(l_inputs,evecs)])
        s_betas=[torch.stack(cb) for cb in c_betas]
        loss=[dataset.measure_loss(o,lab) for o in outs]
        for lz in nets:
            lz.zero_grad()
        for lz in loss:
            lz.backward()
        grads=[i2.grad.data for i2 in inputs]
        for ei, eps in enumerate(epsilons):
            perturbed=[fgsm_attack(i2,eps,g,fgsm_transform,use_sign) for i2,g in zip(inputs,grads)]
            outs2=[n(p,store_hidden=True) for n,p in zip(nets,perturbed)]
            l_inp2=[n.get_layer_inputs(p,precomputed=True) for n,p in zip(nets,perturbed)]
            c_eps=transpose_list([get_beta(x,e) for x,e in zip(l_inp2,evecs)])
            s_eps=[torch.stack(ce) for ce in c_eps]
            d_eps=[se - sb for se,sb in zip(s_eps,s_betas)]
            r_eps=[torch.sqrt(torch.mean(de**2,dim=1)) for de in d_eps]
            for ii,rb in enumerate(r_eps):
                betas[ei][ii]+=rb.detach()
            final_preds=[torch.argmax(o,axis=1) for o in outs2]
            acc[:,ei]+=torch.tensor([sum(fp==lab).cpu() for fp in final_preds])
            # storing examples if needed
    acc=acc/float(len(dl.dataset))
    betas=transpose_list([[cb/float(len(dl.dataset)) for cb in b] for b in betas])
    return dict(accuracy=acc,betas=betas,examples=examples,epsilons=epsilons,use_sign=use_sign)