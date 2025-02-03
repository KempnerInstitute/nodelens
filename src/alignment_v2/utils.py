import torch
import numpy as np
from functools import wraps
from contextlib import contextmanager
from natsort import natsorted
from warnings import warn
import zipfile,os,math
from scipy.linalg import null_space
from sklearn.decomposition import IncrementalPCA

def set_net_mode(net,training=True):
    old=net.training
    net.train() if training else net.eval()
    return old

def test_nets(func):
    @wraps(func)
    def inner(nets,*args,**kwargs):
        modes=[set_net_mode(n,False) for n in nets]
        out=func(nets,*args,**kwargs)
        for m,n in zip(modes,nets):
            set_net_mode(n,m)
        return out
    return inner

def train_nets(func):
    @wraps(func)
    def inner(nets,*args,**kwargs):
        modes=[set_net_mode(n,True) for n in nets]
        out=func(nets,*args,**kwargs)
        for m,n in zip(modes,nets):
            set_net_mode(n,m)
        return out
    return inner

def get_device(x):
    if isinstance(x,torch.nn.Module):
        return next(x.parameters()).device.type
    elif isinstance(x,torch.Tensor):
        return "cuda" if x.is_cuda else "cpu"
    else:
        raise ValueError("No device found")

def check_iterable(v):
    try:
        iter(v)
        return True
    except:
        return False

def remove_by_idx(inp,idx,dim):
    keep=[i for i in range(inp.size(dim)) if i not in idx]
    return torch.index_select(inp,dim,torch.tensor(keep,device=inp.device))

def alignment(x,w,method="alignment",relative=True):
    if method=="alignment":
        c=torch.cov(x.T)
    elif method=="similarity":
        c=smartcorr(x.T)
    else:
        raise ValueError("Unknown method")
    rq=torch.sum((w@c)*w,dim=1)/torch.sum(w*w,dim=1)
    if relative:
        return rq/torch.trace(c)
    return rq

def smartcorr(x):
    z=(torch.var(x,dim=1)==0)
    cc=torch.corrcoef(x)
    cc[z,:]=0
    cc[:,z]=0
    return cc

def fractional_histogram(*args,**kwargs):
    c,b=np.histogram(*args,**kwargs)
    c=c/np.sum(c)
    return c,b

def edge2center(e):
    return e[:-1]+np.diff(e)/2

def fast_rank(x):
    if x.size(-2)<x.size(-1):
        x=x.transpose(-2,-1)
    return int(torch.linalg.matrix_rank(x))

def sklearn_pca(inp,use_rank=True,rank=None):
    num_samp,num_feat=inp.shape
    if use_rank:
        if rank is None:
            rank=fast_rank(inp)
    else:
        rank=None
    ipca=IncrementalPCA(n_components=rank).fit(inp)
    v=ipca.components_
    w=ipca.singular_values_**2/num_samp
    if v.shape[0]<num_feat:
        vs=null_space(v).T
        v=np.vstack((v,vs))
        w=np.concatenate((w,np.zeros(vs.shape[0])))
    return torch.tensor(w,dtype=torch.float),torch.tensor(v,dtype=torch.float).T

def eigendecomposition(C,use_rank=True):
    try:
        w,v=torch.linalg.eigh(C)
    except torch._C._LinAlgError:
        return sklearn_pca(C,use_rank=use_rank)
    idx=torch.argsort(-w)
    w=w[idx]; v=v[:,idx]
    if use_rank:
        r=torch.linalg.matrix_rank(C)
        w[r:]=0
    return w,v

def batch_cov(x, centered=True, correction=True):
    nd=x.ndim
    no_batch=(nd==2)
    if no_batch:
        x=x.unsqueeze(0)
    s=x.size(2)
    if centered:
        x=x-x.mean(dim=2,keepdim=True)
    bcov=torch.bmm(x,x.transpose(1,2))
    bcov/=(s-1.0*correction)
    if no_batch:
        bcov=bcov.squeeze(0)
    return bcov

def smart_pca(x,centered=True,use_rank=True,correction=True):
    if x.ndim==2:
        no_batch=True
        x=x.unsqueeze(0)
    else:
        no_batch=False
    _,D,S=x.size()
    if D>S:
        v,w,_ = named_transpose([torch.linalg.svd(xi) for xi in x])
        w=[(ww**2)/(S-1.0*correction) for ww in w]
        w=[torch.cat((ww,torch.zeros(D-ww.size(0),device=ww.device))) for ww in w]
    else:
        bc=batch_cov(x,centered,correction)
        w,v=named_transpose([eigendecomposition(b,use_rank) for b in bc])
    w=torch.stack(w)
    v=torch.stack(v)
    if no_batch:
        w=w.squeeze(0)
        v=v.squeeze(0)
    return w,v

def transpose_list(lol):
    return list(map(list,zip(*lol)))

def named_transpose(lol,reduction=None):
    if reduction:
        return map(reduction,zip(*lol))
    return map(list,zip(*lol))

def ptp(x,dim=None,keepdim=False):
    if dim is None:
        return x.max()-x.min()
    return x.max(dim,keepdim).values - x.min(dim,keepdim).values

def compute_stats_by_type(x,num_types,dim,method="var"):
    ondim=x.size(dim)
    per_type=ondim//num_types
    y=x.unsqueeze(dim)
    shp=list(y.shape)
    shp[dim+1]=per_type
    shp[dim]=num_types
    y=y.view(shp)
    means=torch.mean(y,dim=dim+1)
    if method=="var":
        dev=torch.var(y,dim=dim+1)
    elif method=="std":
        dev=torch.std(y,dim=dim+1)
    elif method=="se":
        dev=torch.std(y,dim=dim+1)/np.sqrt(per_type)
    elif method=="range":
        dev=ptp(y,dim=dim+1)
    else:
        raise ValueError("Unknown method")
    return means,dev

def rms(x,dim=None,keepdim=False):
    if dim is None:
        return torch.sqrt(torch.mean(x**2))
    return torch.sqrt(torch.mean(x**2,dim=dim,keepdim=keepdim))

def weighted_average(data,weights,dim,keepdim=False,ignore_nan=False):
    if ignore_nan:
        s=torch.nansum
    else:
        s=torch.sum
    num=s(data*weights,dim=dim,keepdim=keepdim)
    den=s(weights,dim=dim,keepdim=keepdim)
    return num/den

def fgsm_attack(img,eps,grad,transform,sign):
    warn("fgsm_attack is only for testing")
    if sign:
        grad=grad.sign()
    else:
        grad=grad.clone()
    pim=img+eps*grad
    pim=transform(pim)
    return pim

def condense_values(full):
    num_layers=len(full[0][0])
    def vbl(lol,layer):
        return torch.cat([f[layer].view(1,-1) for f in lol],dim=0).cpu()
    splitted=transpose_list(full)
    out=[]
    for layer in range(num_layers):
        out.append(torch.stack([vbl(f,layer) for f in splitted]))
    return out

def save_checkpoint(nets,opts,results,path):
    multi_model={f"model_state_dict_{i}":n.state_dict() for i,n in enumerate(nets)}
    multi_opt={f"optimizer_state_dict_{i}":o.state_dict() for i,o in enumerate(opts)}
    ch=results|multi_model|multi_opt
    torch.save(ch,path)

def load_checkpoints(nets,opts,device,path):
    ch=torch.load(path,map_location=device)
    net_ids=[k for k in ch if k.startswith("model_state_dict")]
    from natsort import natsorted
    net_ids=natsorted(net_ids)
    opt_ids=[k for k in ch if k.startswith("optimizer_state_dict")]
    opt_ids=natsorted(opt_ids)
    for netid,optid,net,op in zip(net_ids,opt_ids,nets,opts):
        net.load_state_dict(ch.pop(netid))
        op.load_state_dict(ch.pop(optid))
    if device=="cuda":
        for n in nets:
            n.to(device)
    return nets,opts,ch

def match_git(path):
    return ".git" in path

def compress_directory(output_path,directory_path=None):
    if directory_path is None:
        directory_path=os.path.dirname(os.path.abspath(__file__))+"/../.."
    from gitignore_parser import parse_gitignore
    gitignore_path=os.path.join(directory_path,".gitignore")
    matches=parse_gitignore(gitignore_path)
    files_to_copy=[]
    archive_names=[]
    for dirpath,dirnames,files in os.walk(directory_path):
        if matches(dirpath) or match_git(dirpath):
            dirnames[:]=[]
        else:
            keep=[f for f in files if not matches(f) and not match_git(f)]
            full=[os.path.join(dirpath,f) for f in keep]
            for file in full:
                files_to_copy.append(file)
                archive_names.append(os.path.relpath(file,directory_path))
    with zipfile.ZipFile(output_path,"w",zipfile.ZIP_DEFLATED) as zipf:
        for f,n in zip(files_to_copy,archive_names):
            zipf.write(f,arcname=n)