import torch
from tqdm import tqdm
from utils import train_nets, test_nets, transpose_list, condense_values, save_checkpoint, smart_pca, expected_alignment_distribution

@train_nets
def train(nets, opts, dataset, **params):
    verbose=params.get("verbose",True)
    alignment_flag=params.get("alignment",True)
    delta_w=params.get("delta_weights",False)
    freq=params.get("frequency",1)
    run=params.get("run",None)
    measure_delta_alignment=params.get("delta_alignment",False)
    compare_expected=params.get("compare_expected",False)
    num_epochs=params.get("num_epochs",1)
    use_train=params.get("train_set",True)
    dl=dataset.train_loader if use_train else dataset.test_loader
    steps=len(dl)*num_epochs
    n_nets=len(nets)
    results=params.get("results",False)
    if not results:
        results=dict(
            loss=torch.zeros((steps,n_nets)),
            accuracy=torch.zeros((steps,n_nets))
        )
        if alignment_flag:
            results["alignment"]=[]
        if delta_w:
            results["delta_weights"]=[]
            results["init_weights"]=[n.get_alignment_weights() for n in nets]
        if measure_delta_alignment:
            if "init_weights" not in results:
                results["init_weights"]=[n.get_alignment_weights() for n in nets]
            results["delta_alignment"]=[]
        if compare_expected:
            results["compare_alignment_bins"]=torch.linspace(0,1,301)
            results["compare_alignment_expected"]=[]
            results["compare_alignment_observed"]=[]
            if measure_delta_alignment:
                results["compare_delta_alignment_observed"]=[]
    num_complete=params.get("num_complete",0)
    save_ckpt, freq_ckpt, path_ckpt, dev = params.get("save_checkpoints",(False,1,"",""))
    if verbose:
        print(f"Starting from epoch {num_complete} of {num_epochs}")

    idx0=num_complete*len(dl)
    e_loop=range(num_complete,num_epochs)
    if verbose:
        e_loop=tqdm(e_loop,desc="Epochs")
    for ep in e_loop:
        if verbose:
            b_loop=tqdm(dl,desc=f"Epoch {ep}",leave=False)
        else:
            b_loop=dl
        for i,batch in enumerate(b_loop):
            cidx=ep*len(dl)+i
            x,y=dataset.unwrap_batch(batch)
            for op in opts:
                op.zero_grad()
            outs=[n(x,store_hidden=True) for n in nets]
            ls=[dataset.measure_loss(o,y) for o in outs]
            for lz,op in zip(ls,opts):
                lz.backward()
                op.step()
            results["loss"][cidx]=torch.tensor([lz.item() for lz in ls])
            results["accuracy"][cidx]=torch.tensor([dataset.measure_accuracy(o,y).cpu() for o in outs])

            if (i%freq)==0:
                if alignment_flag:
                    results["alignment"].append([n.measure_alignment(x,precomputed=True) for n in nets])
                if delta_w or measure_delta_alignment:
                    d_w=[n.compare_weights(iw) for n,iw in zip(nets,results["init_weights"])]
                    if delta_w:
                        results["delta_weights"].append(d_w)
                    if measure_delta_alignment:
                        d_al=[n.measure_alignment_weights(x,w,precomputed=True) for n,w in zip(nets,d_w)]
                        results["delta_alignment"].append(d_al)
                if compare_expected:
                    pass
            if run:
                pass
        if save_ckpt and (ep%freq_ckpt==0):
            save_checkpoint(nets,opts,results|dict(epoch=ep,device=dev,prms=params),path_ckpt)
    return results

@test_nets
def test(nets,dataset,**params):
    verbose=params.get("verbose",True)
    run=params.get("run",None)
    alignment_flag=params.get("alignment",True)
    use_train=params.get("train_set",False)
    dl=dataset.train_loader if use_train else dataset.test_loader
    n_nets=len(nets)
    total_l=[0]*n_nets
    total_c=[0]*n_nets
    nbatch=0
    align=[]
    if verbose:
        dl=tqdm(dl,desc="Testing")
    for b in dl:
        x,y=dataset.unwrap_batch(b)
        outs=[n(x,store_hidden=True) for n in nets]
        for i,o in enumerate(outs):
            total_l[i]+=dataset.measure_loss(o,y).item()
            total_c[i]+=dataset.measure_accuracy(o,y).item()
        nbatch+=1
        if alignment_flag:
            align.append([n.measure_alignment(x,precomputed=True) for n in nets])
    results=dict(
        loss=[tl/nbatch for tl in total_l],
        accuracy=[tc/nbatch for tc in total_c]
    )
    if alignment_flag:
        from utils import condense_values, transpose_list
        results["alignment"]=condense_values(transpose_list(align))
    if run:
        pass
    return results

@test_nets
def progressive_dropout(nets,dataset,alignment=None,**params):
    pass

@test_nets
def eigenvector_dropout(nets,dataset,evals,evecs,**params):
    pass