#!/usr/bin/env python

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# Insert your alignment_v2 path if needed:
# sys.path.insert(0, "/path/to/alignment_v2")

from alignment_v2.datasets import get_dataset
from alignment_v2.models.registry import get_model
from alignment_v2 import train
from alignment_v2.train import progressive_dropout
from alignment_v2 import plotting

def main():
    ###############################################################
    # 1. Parse environment variables from Slurm
    ###############################################################
    rank = int(os.environ["SLURM_PROCID"])       # global rank among all tasks
    world_size = int(os.environ["WORLD_SIZE"])   # total tasks
    local_rank = int(os.environ["SLURM_LOCALID"])# rank on this node [0..3]
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # GPU device assignment (1 GPU per task)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank={rank}] local_rank={local_rank}, device={device}, MASTER={master_addr}:{master_port}")

    ###############################################################
    # 2. Initialize Distributed Process Group
    ###############################################################
    # Use NCCL backend for multi-GPU
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank
    )

    # (Optional) Check if init worked
    assert dist.is_initialized(), f"[Rank {rank}] dist not initialized?"

    ###############################################################
    # 3. Build Model + Dataset
    ###############################################################
    model_name = "AlexNet"
    dataset_name = "ImageNet"
    dropout_rate = 0.0
    learning_rate = 1e-3
    weight_decay = 0
    epochs = 2
    batch_size = 64  # per GPU

    print(f"[Rank={rank}] Building model: {model_name}, dataset: {dataset_name}...")

    # Build the AlexNet model for ImageNet
    net = get_model(
        model_name,
        build=True,
        dataset=dataset_name,
        dropout=dropout_rate,
        ignore_flag=False
    )
    net.to(device)

    # Wrap in DDP
    ddp_net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    # Build dataset with distributed=True => uses DistributedSampler
    loader_params = dict(
        batch_size=batch_size,
        shuffle=False,   # Sampler does the shuffle
        num_workers=4
    )
    dataset = get_dataset(
        dataset_name,
        build=True,
        transform_parameters=ddp_net.module,  # alignment transforms
        loader_parameters=loader_params,
        device="cpu",        # Transforms on CPU
        distributed=True     # so it uses DistributedSampler
    )

    print(f"[Rank={rank}] Dataset built. Creating optimizer...")

    # Create optimizer
    optimizer = torch.optim.Adam(ddp_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ###############################################################
    # 4. Training
    ###############################################################
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

    print(f"[Rank={rank}] Starting training for {epochs} epochs.")
    results = train.train(nets, optimizers, dataset, **train_params)

    print(f"[Rank={rank}] Training complete. Results keys = {list(results.keys())}")

    ###############################################################
    # 5. Progressive Dropout (only rank=0 for final experiment)
    ###############################################################
    # If we want to measure alignment-based dropout:
    if rank == 0:
        print("[Rank=0] Performing progressive dropout experiment.")
        if "alignment" not in results:
            print("[Rank=0] 'alignment' not in results => skipping dropout.")
        else:
            dropout_params = {
                "num_drops": 3,
                "by_layer": False,
                "train_set": False,
            }
            drop_res = progressive_dropout(nets, dataset, alignment=results["alignment"], **dropout_params)
            print("[Rank=0] Dropout results keys:", list(drop_res.keys()))

            # Optionally plot
            plotting.plot_dropout_results(
                exp=None,
                dropout_results=drop_res,
                dropout_parameters=dropout_params,
                prms={
                    "vals": [model_name],
                    "name": "ModelType",
                    "dataset": dataset_name,
                    "dropout": dropout_rate,
                    "lr": learning_rate,
                    "weight_decay": weight_decay
                },
                dropout_type="alignment-based",
            )

    # Cleanup
    dist.destroy_process_group()
    print(f"[Rank={rank}] Done. Exiting.")


if __name__ == "__main__":
    main()