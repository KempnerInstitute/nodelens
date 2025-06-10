#!/usr/bin/env python

import os
import sys
import random
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from alignment_v2.datasets import get_dataset
from alignment_v2.models.registry import get_model
from alignment_v2 import train
from alignment_v2.train import progressive_dropout
from alignment_v2 import plotting

##############################################################################
# 1. Worker function for each process
##############################################################################
def ddp_worker(rank, world_size, port, epochs, batch_size, lr, dropout_rate):

    try:
        master_addr = "127.0.0.1"

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        print(f"\n[DDP_RANK={rank}] world_size={world_size}, port={port}, device={device}")

        # Build model
        model_name   = "MLP" #"AlexNet"
        dataset_name = "MNIST" #"ImageNet"  
        net = get_model(
            model_name,
            build=True,
            dataset=dataset_name,
            dropout=dropout_rate,
            ignore_flag=False
        )
        net.to(device)

        ddp_net = DDP(net, device_ids=[rank], output_device=rank)

        loader_params = dict(
            batch_size=batch_size,
            shuffle=False,    
            num_workers=4
        )
        dataset = get_dataset(
            dataset_name,
            build=True,
            transform_parameters=ddp_net.module, 
            loader_parameters=loader_params,
            device="cuda",    
            distributed=True  
        )

        print(f"[DDP_RANK={rank}] Dataset ready. Creating optimizer...")

        optimizer = torch.optim.Adam(ddp_net.parameters(), lr=lr, weight_decay=0)

        train_params = dict(
            num_epochs=epochs,
            alignment=True,
            alignment_expansion=False,
            compare_expected=False,
            frequency=1,
            delta_alignment=False,
        )
        nets = [ddp_net]
        optimizers = [optimizer]

        print(f"[DDP_RANK={rank}] Starting training for {epochs} epochs...")
        results = train.train(nets, optimizers, dataset, **train_params)
        print(f"[DDP_RANK={rank}] Done training. Results keys={list(results.keys())}")

        if rank == 0:
            print("[DDP_RANK=0] Doing progressive dropout experiment.")
            if "alignment" not in results:
                print("[DDP_RANK=0] 'alignment' not found => skipping dropout.")
            else:
                dropout_params = dict(
                    num_drops=3,
                    by_layer=False,
                    train_set=False,
                )
                drop_res = progressive_dropout(nets, dataset, alignment=results["alignment"], **dropout_params)
                print("[DDP_RANK=0] dropout results keys:", list(drop_res.keys()))

                plotting.plot_dropout_results(
                    exp=None,
                    dropout_results=drop_res,
                    dropout_parameters=dropout_params,
                    prms={
                        "vals": [model_name],
                        "name": "ModelType",
                        "dataset": dataset_name,
                        "dropout": dropout_rate,
                        "lr": lr,
                        "weight_decay": 0
                    },
                    dropout_type="alignment-based",
                )

    except Exception as e:
        print(f"[DDP_RANK={rank}] ERROR => {e}")
        traceback.print_exc()
        raise e
    finally:
        dist.destroy_process_group()
        print(f"[DDP_RANK={rank}] Exiting rank {rank}...")

##############################################################################
# 2. Main function: spawns local processes
##############################################################################
def main():
    world_size  = 4  # number of GPUs on this node
    epochs      = 5
    batch_size  = 64
    lr          = 1e-3
    dropout_rate= 0.0

    import random
    port = random.randint(20000, 30000)

    print(f"Launching {world_size} processes with tcp://127.0.0.1:{port} for DDP.")
    torch.multiprocessing.set_start_method("spawn", force=True)

    # spawn processes
    mp.spawn(
        ddp_worker,
        nprocs=world_size,
        args=(world_size, port, epochs, batch_size, lr, dropout_rate),
        join=True
    )
    print("All DDP processes done successfully.")

if __name__ == "__main__":
    main()