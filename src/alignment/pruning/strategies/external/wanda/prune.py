import time 
import heapq 
import torch 
import torch.nn as nn 
from .layerwrapper import WrappedGPT
from .data import get_loaders 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prepare_calibration_input(model, dataloader, device, seqlen):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if hasattr(model, 'hf_device_map') and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, sparsity_ratio=None):
    if sparsity_ratio is None:
        sparsity_ratio = args.sparsity_ratio
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    # Get sequence length from tokenizer or use default
    seqlen = getattr(tokenizer, 'model_max_length', None)
    if seqlen is None or seqlen > 10000:  # Some tokenizers have very large max_length
        seqlen = 2048  # Default sequence length

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, seqlen)

    layers = model.model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        for j in range(args.nsamples):
            with torch.no_grad():
                # Generate position_ids if they are None
                seq_len = inps[j].shape[0]
                pos_ids = torch.arange(seq_len, dtype=torch.long, device=inps[j].device).unsqueeze(0)
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            # if prune_n != 0:
            #     # structured n:m sparsity
            #     for ii in range(W_metric.shape[1]):
            #         if ii % prune_m == 0:
            #             tmp = W_metric[:,ii:(ii+prune_m)].float()
            #             W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            # else:
            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            #     if args.use_variant:
            #         # wanda variant 
            #         tmp_metric = torch.cumsum(sort_res[0], dim=1)
            #         sum_before = W_metric.sum(dim=1)

            #         alpha = 0.4
            #         alpha_hist = [0., 0.8]
            #         W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            #         while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
            #             if cur_sparsity > args.sparsity_ratio:
            #                 alpha_new = (alpha + alpha_hist[0]) / 2.0
            #                 alpha_hist[1] = alpha
            #             else:
            #                 alpha_new = (alpha + alpha_hist[1]) / 2.0
            #                 alpha_hist[0] = alpha

            #             alpha = alpha_new 
            #             W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
            #         print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
            #     else:
                    # unstructured pruning
            indices = sort_res[1][:,:int(W_metric.shape[1]*sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                seq_len = inps[j].shape[0]
                pos_ids = torch.arange(seq_len, dtype=torch.long, device=inps[j].device).unsqueeze(0)
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()